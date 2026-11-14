"""Baseline chord recognition: beat-synchronous chroma + triad templates + Viterbi smoothing.

Inputs are separated stems: `harmonic` (guitars/keys, ideally without drums and vocals),
optional `bass` (helps pick the root) and optional `drums` (used only to find the beats).
Output: list of (start_s, end_s, chord) plus key and tempo.
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

SR = 22050
HOP = 512
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

QUALITIES = {  # suffix -> (intervals, weights)
    "": ((0, 4, 7), (1.0, 0.8, 0.9)),
    "m": ((0, 3, 7), (1.0, 0.8, 0.9)),
}
SEVENTHS = {
    "7": ((0, 4, 7, 10), (1.0, 0.8, 0.9, 0.6)),
    "m7": ((0, 3, 7, 10), (1.0, 0.8, 0.9, 0.6)),
    "maj7": ((0, 4, 7, 11), (1.0, 0.8, 0.9, 0.6)),
}

_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])  # Krumhansl
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class ChordResult:
    segments: list  # [(start_s, end_s, chord)]
    key: str
    bpm: float
    beats: np.ndarray


def _templates(sevenths: bool):
    quals = dict(QUALITIES)
    if sevenths:
        quals |= SEVENTHS
    names, mats, roots = [], [], []
    for root in range(12):
        for suffix, (ivs, ws) in quals.items():
            v = np.zeros(12)
            for iv, w in zip(ivs, ws):
                v[(root + iv) % 12] = w
            names.append(NOTES[root] + suffix)
            mats.append(v / np.linalg.norm(v))
            roots.append(root)
    return names, np.array(mats), np.array(roots)


def _chroma(y, tuning):
    harm = librosa.effects.harmonic(y, margin=4)
    c = librosa.feature.chroma_cqt(y=harm, sr=SR, hop_length=HOP, tuning=tuning, n_chroma=12)
    return c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-9)


def _bass_chroma(y, tuning):
    c = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("C1"),
                                   n_octaves=3, bins_per_octave=36, tuning=tuning, n_chroma=12)
    return c / (c.sum(axis=0, keepdims=True) + 1e-9)


def estimate_key(chroma: np.ndarray) -> str:
    prof = chroma.mean(axis=1)
    best, name = -2, "C major"
    for tonic in range(12):
        for mode, p in (("major", _MAJOR), ("minor", _MINOR)):
            r = np.corrcoef(prof, np.roll(p, tonic))[0, 1]
            if r > best:
                best, name = r, f"{NOTES[tonic]} {mode}"
    return name


def diatonic_mask(names, roots, key: str) -> np.ndarray:
    """1 for chords that belong to the key (triad quality decides), 0 otherwise."""
    tonic, mode = key.split()
    t = NOTES.index(tonic)
    if mode == "major":
        allowed = {(0, "maj"), (2, "min"), (4, "min"), (5, "maj"), (7, "maj"), (9, "min")}
    else:  # natural minor + the common major V and VII
        allowed = {(0, "min"), (3, "maj"), (5, "min"), (7, "min"), (8, "maj"), (10, "maj"), (7, "maj")}
    out = []
    for name, root in zip(names, roots):
        suffix = name[len(NOTES[root]):]
        qual = "min" if suffix.startswith("m") and not suffix.startswith("maj") else "maj"
        out.append(1.0 if ((root - t) % 12, qual) in allowed else 0.0)
    return np.array(out)


def repeat_smooth(sync: np.ndarray, mix: float = 0.6, context: int = 4, min_lag: int = 8) -> np.ndarray:
    """Average each beat's chroma with the beats that sound like it elsewhere in the song
    (repeated choruses/verses). This is a vote across repetitions."""
    feat = librosa.feature.stack_memory(sync, n_steps=context, mode="edge")
    rec = librosa.segment.recurrence_matrix(feat, mode="affinity", metric="cosine", width=min_lag, sym=True)
    smooth = librosa.decompose.nn_filter(sync, rec=rec, aggregate=np.average)
    out = (1 - mix) * sync + mix * smooth
    return out / (np.linalg.norm(out, axis=0, keepdims=True) + 1e-9)


def detect_chords(harmonic_path, bass_path=None, drums_path=None, sevenths=True,
                  stay=0.85, bass_weight=0.15, sharpness=12.0,
                  use_repeats=True, key_prior=0.05, seventh_penalty=0.06) -> ChordResult:
    y = librosa.load(harmonic_path, sr=SR, mono=True)[0]
    tuning = float(librosa.estimate_tuning(y=y, sr=SR))
    chroma = _chroma(y, tuning)

    ref = librosa.load(drums_path, sr=SR, mono=True)[0] if drums_path else y
    tempo, beats = librosa.beat.beat_track(y=ref, sr=SR, hop_length=HOP)
    bpm = float(np.atleast_1d(tempo)[0])
    beats = librosa.util.fix_frames(beats, x_min=0, x_max=chroma.shape[1] - 1)
    sync = librosa.util.sync(chroma, beats, aggregate=np.median)  # [12, n_segments]
    if use_repeats:
        sync = repeat_smooth(sync)

    names, T, roots = _templates(sevenths)
    key = estimate_key(chroma)
    scores = T @ sync  # [n_chords, n_segments] cosine similarity
    if key_prior:
        scores = scores + key_prior * diatonic_mask(names, roots, key)[:, None]
    if sevenths and seventh_penalty:
        # 4-note templates partially match almost anything; only use them when clearly better
        is7 = np.array([n[len(NOTES[r]):] in SEVENTHS for n, r in zip(names, roots)], dtype=float)
        scores = scores - seventh_penalty * is7[:, None]
    if bass_path:
        by = librosa.load(bass_path, sr=SR, mono=True)[0]
        bsync = librosa.util.sync(_bass_chroma(by, tuning)[:, : chroma.shape[1]], beats, aggregate=np.median)
        scores = scores + bass_weight * bsync[roots]
    prob = np.exp(sharpness * (scores - scores.max(axis=0, keepdims=True)))
    prob /= prob.sum(axis=0, keepdims=True)
    path = librosa.sequence.viterbi(prob, librosa.sequence.transition_loop(len(names), stay))

    times = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    times = np.append(times, len(y) / SR)
    segs = []
    for i, k in enumerate(path):
        t0, t1 = float(times[i]), float(times[i + 1])
        if segs and segs[-1][2] == names[k]:
            segs[-1] = (segs[-1][0], t1, names[k])
        else:
            segs.append((t0, t1, names[k]))
    return ChordResult(segs, key, bpm, librosa.frames_to_time(beats, sr=SR, hop_length=HOP))
