"""Chord recognition from separated stems, in two phases:

1. `extract`: audio -> beat-synchronous features (the slow part: decoding audio, CQT, beat tracking).
   The result can be cached on disk, so trying decoding settings takes a fraction of a second.
2. `decode`: features -> chords: triad templates scored per beat, key prior, Viterbi smoothing,
   and slash chords from the bass stem.

Inputs are separated stems: `harmonic` (a path or a list of paths that are summed; guitars/keys + bass works
best), optional `bass` (root and slash chords) and optional `drums` (used only to find the beats).
Default settings were chosen with bench/sweep.py against a reference chart.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

SR = 22050
HOP = 512
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
FLAT_KEYS = {"F major", "A# major", "D# major", "G# major", "C# major", "F# major",
             "D minor", "G minor", "C minor", "F minor", "A# minor", "D# minor"}

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

DEFAULTS = dict(sevenths=True, sharpness=20.0, change=0.15, bass_weight=0.25, slash=True, slash_run=2, slash_share=0.45, rebase_margin=0.08,
                use_repeats=True, key_prior=0.05, seventh_penalty=0.06, key_from_chords=True)


@dataclass
class ChordResult:
    segments: list  # [(start_s, end_s, chord)]
    key: str
    bpm: float
    beats: np.ndarray


@dataclass
class Features:
    chroma: np.ndarray         # [12, n_beats] harmonic chroma (CQT), L2-normalized per beat
    bass: np.ndarray | None    # [12, n_beats] bass chroma, sum-normalized per beat
    times: np.ndarray          # [n_beats + 1] beat start times plus the end of the audio
    key: str
    bpm: float

    def save(self, path: Path):
        np.savez(path, chroma=self.chroma, bass=self.bass if self.bass is not None else np.zeros(0), times=self.times,
                 key=self.key, bpm=self.bpm)

    @classmethod
    def load(cls, path: Path) -> "Features":
        z = np.load(path)
        opt = lambda a: a if a.size else None  # noqa: E731
        return cls(z["chroma"], opt(z["bass"]), z["times"], str(z["key"]), float(z["bpm"]))


# ------------------------------------------------------------------ naming
def spell(chord: str, key: str) -> str:
    """Chord names are built with sharps; in flat keys (C minor, F major...) write flats (Ab, Bb)."""
    if key not in FLAT_KEYS or chord == "N":
        return chord
    main, _, bass = chord.partition("/")
    root = main[:2] if len(main) > 1 and main[1] == "#" else main[:1]
    out = FLAT_NAMES[NOTES.index(root)] + main[len(root):]
    return out + ("/" + FLAT_NAMES[NOTES.index(bass)] if bass else "")


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


# ------------------------------------------------------------------ features
def _norm(c, axis=0):
    return c / (np.linalg.norm(c, axis=axis, keepdims=True) + 1e-9)


def _load(paths) -> np.ndarray:
    paths = paths if isinstance(paths, (list, tuple)) else [paths]
    ys = [librosa.load(p, sr=SR, mono=True)[0] for p in paths]
    n = min(len(y) for y in ys)
    return np.sum([y[:n] for y in ys], axis=0)


def _chroma(y, tuning):
    harm = librosa.effects.harmonic(y, margin=4)
    return _norm(librosa.feature.chroma_cqt(y=harm, sr=SR, hop_length=HOP, tuning=tuning, n_chroma=12))


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


def fill_beat_gaps(beats: np.ndarray, max_ratio: float = 1.6) -> np.ndarray:
    """The beat tracker works on the drums, so a drumless stretch (an intro, a break) comes out with no
    beats and its chords collapse into one long chord. Fill every gap longer than max_ratio beats with an
    even grid at the song's own tempo."""
    if len(beats) < 3:
        return beats
    med = float(np.median(np.diff(beats)))
    out = [beats[0]]
    for a, b in zip(beats[:-1], beats[1:]):
        gap = b - a
        if gap > max_ratio * med:
            n = int(round(gap / med))
            out += [a + gap * k / n for k in range(1, n)]
        out.append(b)
    return np.array(out)


def extract(harmonic, bass_path=None, drums_path=None, cache_dir: Path | None = None, subdivide: int = 2) -> Features:
    """Audio -> beat-synchronous features. With cache_dir, reuses a previous extraction of the same files.

    Each tracked beat is split in `subdivide` parts: beat trackers often lock to half the real tempo (84 for a
    168 BPM song), and then chords could only change every other beat, landing on 2 and 4 instead of 1."""
    srcs = [str(p) for p in (harmonic if isinstance(harmonic, (list, tuple)) else [harmonic])]
    cache = None
    if cache_dir:
        ident = "|".join(srcs + [str(bass_path), str(drums_path), f"sub{subdivide}"] +
                         [f"{Path(p).stat().st_mtime_ns}" for p in srcs + [bass_path, drums_path] if p])
        cache = Path(cache_dir) / f"chords_{hashlib.sha1(ident.encode()).hexdigest()[:16]}.npz"
        if cache.exists():
            return Features.load(cache)

    y = _load(srcs)
    tuning = float(librosa.estimate_tuning(y=y, sr=SR))
    frame_chroma = _chroma(y, tuning)
    n = frame_chroma.shape[1]

    ref = librosa.load(drums_path, sr=SR, mono=True)[0] if drums_path else y
    tempo, beats = librosa.beat.beat_track(y=ref, sr=SR, hop_length=HOP)
    beats = librosa.util.fix_frames(beats, x_min=0, x_max=n - 1)
    beats = fill_beat_gaps(beats.astype(float))
    if subdivide > 1 and len(beats) > 1:
        beats = np.concatenate([beats[:-1, None] + np.diff(beats)[:, None] * np.arange(subdivide) / subdivide,
                                beats[-1:, None]], axis=None)
    beats = np.unique(np.round(beats).astype(int))

    def sync(c):
        return librosa.util.sync(c[:, :n], beats, aggregate=np.median)

    bass = None
    if bass_path:
        bass = sync(_bass_chroma(librosa.load(bass_path, sr=SR, mono=True)[0], tuning))
    times = np.append(librosa.frames_to_time(beats, sr=SR, hop_length=HOP), len(y) / SR)
    f = Features(_norm(sync(frame_chroma)), bass, times,
                 estimate_key(frame_chroma), float(np.atleast_1d(tempo)[0]))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        f.save(cache)
    return f


# ------------------------------------------------------------------ decoding
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


def diatonic_seventh(name: str, key: str) -> bool:
    """True if every note of the seventh chord is in the key's scale (in minor, V7 with the leading tone too)."""
    tonic, mode = key.split()
    t = NOTES.index(tonic)
    scale = {(t + i) % 12 for i in ((0, 2, 4, 5, 7, 9, 11) if mode == "major" else (0, 2, 3, 5, 7, 8, 10))}
    root = NOTES.index(name[:2] if len(name) > 1 and name[1] == "#" else name[:1])
    suffix = name[len(NOTES[root]):]
    notes = {(root + iv) % 12 for iv in SEVENTHS[suffix][0]}
    if mode == "minor" and root == (t + 7) % 12 and suffix == "7":
        return notes - {(t + 11) % 12} <= scale
    return notes <= scale


def repeat_smooth(sync: np.ndarray, mix: float = 0.6, context: int = 4, min_lag: int = 8) -> np.ndarray:
    """Average each beat's chroma with the beats that sound like it elsewhere in the song
    (repeated choruses/verses). This is a vote across repetitions."""
    if sync.shape[1] <= 2 * min_lag + context:
        return sync
    feat = librosa.feature.stack_memory(sync, n_steps=context, mode="edge")
    rec = librosa.segment.recurrence_matrix(feat, mode="affinity", metric="cosine", width=min_lag, sym=True)
    smooth = librosa.decompose.nn_filter(sync, rec=rec, aggregate=np.average)
    return _norm((1 - mix) * sync + mix * smooth)


def _viterbi(logp: np.ndarray, change: np.ndarray) -> np.ndarray:
    """Most likely state path. logp [S, N] log emissions; change[t] = probability of leaving the current
    state on step t (spread evenly over the other states)."""
    S, N = logp.shape
    v = logp[:, 0].copy()
    back = np.zeros((S, N), dtype=int)
    idx = np.arange(S)
    for t in range(1, N):
        p = float(np.clip(change[t], 1e-6, 1 - 1e-6))
        stay, move = np.log(1 - p), np.log(p / (S - 1))
        order = np.argsort(v)
        top, second = order[-1], order[-2]
        best_other = np.where(idx == top, second, top)  # best previous state other than j itself
        cand_stay, cand_move = v + stay, v[best_other] + move
        take_move = cand_move > cand_stay
        back[:, t] = np.where(take_move, best_other, idx)
        v = np.where(take_move, cand_move, cand_stay) + logp[:, t]
    path = np.zeros(N, dtype=int)
    path[-1] = int(v.argmax())
    for t in range(N - 1, 0, -1):
        path[t - 1] = back[path[t], t]
    return path


def bass_runs(bass_pc: np.ndarray, share: np.ndarray, root: int, i0: int, i1: int, min_run: int, min_share: float):
    """Runs (start, end, pitch class) of beats in [i0, i1) where the bass holds one note that is not the root."""
    runs, i = [], i0
    while i < i1:
        j = i
        while j < i1 and bass_pc[j] == bass_pc[i]:
            j += 1
        if bass_pc[i] != root and j - i >= min_run and share[i:j].mean() >= min_share:
            runs.append((i, j, int(bass_pc[i])))
        i = j
    return runs


def _path(chroma, bass, key: str, p: dict):
    """Best chord per beat for a given key -> (names, roots, scores, path)."""
    names, T, roots = _templates(p["sevenths"])
    if p["sevenths"]:  # only sevenths built from the key's own notes (Abmaj7 in C minor, never Bbmaj7)
        keep = [i for i, n in enumerate(names) if n[len(NOTES[roots[i]]):] not in SEVENTHS or diatonic_seventh(n, key)]
        names, T, roots = [names[i] for i in keep], T[keep], roots[keep]
    scores = T @ chroma  # [n_chords, n_beats] cosine similarity
    if p["key_prior"]:
        scores = scores + p["key_prior"] * diatonic_mask(names, roots, key)[:, None]
    if p["sevenths"] and p["seventh_penalty"]:
        # 4-note templates partially match almost anything; only use them when clearly better
        is7 = np.array([n[len(NOTES[r]):] in SEVENTHS for n, r in zip(names, roots)], dtype=float)
        scores = scores - p["seventh_penalty"] * is7[:, None]
    if bass is not None:
        scores = scores + p["bass_weight"] * bass[roots]
    z = p["sharpness"] * (scores - scores.max(axis=0, keepdims=True))
    logp = z - np.log(np.exp(z).sum(axis=0, keepdims=True))
    return names, roots, scores, _viterbi(logp, np.full(scores.shape[1], p["change"]))


def key_from_chords(names, roots, path, durations, tonic_weight: float = 1.0) -> str:
    """The key whose own chords sound the longest, plus a bonus for the time on its tonic chord.
    More reliable than the average chroma, which mistakes E minor for E major when the third is quiet."""
    dur = np.zeros(len(names))
    np.add.at(dur, path, durations)
    best, best_key = -1.0, "C major"
    for t in range(12):
        for mode in ("major", "minor"):
            key = f"{NOTES[t]} {mode}"
            tonic = NOTES[t] + ("" if mode == "major" else "m")
            s = float(dur @ diatonic_mask(names, roots, key)) + tonic_weight * float(dur[names.index(tonic)])
            if s > best:
                best, best_key = s, key
    return best_key


def decode(f: Features, **params) -> ChordResult:
    p = {**DEFAULTS, **params}
    chroma = f.chroma
    if p["use_repeats"]:
        chroma = repeat_smooth(chroma)
    key = f.key
    if p["key_from_chords"]:  # first pass on triads with no key prior, only to find the key
        n0, r0, _, path0 = _path(chroma, f.bass, key, {**p, "key_prior": 0.0, "sevenths": False})
        key = key_from_chords(n0, r0, path0, np.diff(f.times))
    names, roots, scores, path = _path(chroma, f.bass, key, p)

    labels = [names[k] for k in path]
    if p["slash"] and f.bass is not None:  # a bass note held under a chord that is not its root -> slash chord
        bass_pc, share = f.bass.argmax(axis=0), f.bass.max(axis=0)
        i = 0
        while i < len(path):
            j = i
            while j < len(path) and path[j] == path[i]:
                j += 1
            for s, e, pc in bass_runs(bass_pc, share, int(roots[path[i]]), i, j, p["slash_run"], p["slash_share"]):
                # the bass note may simply be the real root (Cm vs Ab share two notes): if a chord rooted on it
                # fits almost as well, it is that chord; otherwise it is a slash chord
                rooted = np.where(roots == pc)[0]
                k_best = rooted[scores[rooted, s:e].mean(axis=1).argmax()]
                use_root = scores[k_best, s:e].mean() >= scores[path[i], s:e].mean() - p["rebase_margin"]
                for k in range(s, e):
                    labels[k] = names[k_best] if use_root else f"{names[path[i]]}/{NOTES[pc]}"
            i = j
    labels = [spell(x, key) for x in labels]

    segs = []
    for i, name in enumerate(labels):
        t0, t1 = float(f.times[i]), float(f.times[i + 1])
        if segs and segs[-1][2] == name:
            segs[-1] = (segs[-1][0], t1, name)
        else:
            segs.append((t0, t1, name))
    return ChordResult(segs, key, f.bpm, f.times[:-1])


def detect_chords(harmonic_path, bass_path=None, drums_path=None, cache_dir: Path | None = None,
                  **params) -> ChordResult:
    return decode(extract(harmonic_path, bass_path, drums_path, cache_dir), **params)
