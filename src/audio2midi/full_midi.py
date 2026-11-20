"""One multitrack MIDI for the DAW: drums + piano chords on a clean tempo, plus the song aligned to it.

The per-beat tempo map of the quantized drum MIDI follows the beat tracker, jitter included (hundreds of
tempo changes), which DAWs import badly. Songs rendered by tools like Suno keep a fixed tempo, and the rest
drift slowly. So the tempo is fitted again from the exact drum hits:

- stable songs get ONE tempo (the one that best puts every hit on a sixteenth grid);
- songs that speed up or slow down get a smooth tempo map that follows the drift, not the jitter.

Bar 1 starts on the first downbeat at or before the start of the audio, so the MIDI begins a fraction of a
bar before the song. The song is written again with that much silence in front (<name>_alineada.wav):
dropping both files at the start of a DAW project lines the audio up with the bar grid.
"""
from __future__ import annotations

from pathlib import Path

import mido
import numpy as np

from .chord_midi import PPQ, TempoMap, voicing

DRUM_CH, PIANO_CH = 9, 0
STABLE_SPREAD_BPM = 0.6   # local tempos within this range -> one fixed tempo...
MAX_FIXED_DRIFT_S = 0.025  # ...unless the hits stray this far from the fixed grid somewhere (~ what the ear notices)


# ------------------------------------------------------------------ reading MIDI notes in seconds
def _notes_in_seconds(path: Path):
    """[(start_s, pitch, velocity)] of a MIDI file, through its own tempo map."""
    m = mido.MidiFile(str(path))
    changes, notes = [], []
    for tr in m.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "set_tempo":
                changes.append((t, msg.tempo))
            elif msg.type == "note_on" and msg.velocity > 0:
                notes.append((t, msg.note, msg.velocity))
    tmap = TempoMap(changes, m.ticks_per_beat)
    secs = np.array(tmap.secs)
    out = []
    for tick, pitch, vel in notes:  # ticks -> seconds (inverse of TempoMap.tick)
        i = max(0, int(np.searchsorted(tmap.ticks, tick, side="right")) - 1)
        out.append((secs[i] + (tick - tmap.ticks[i]) * tmap.tempos[i] / 1e6 / tmap.ppq, pitch, vel))
    return sorted(out)


# ------------------------------------------------------------------ tempo fitting
def _fit(hits: np.ndarray, lo: float, hi: float, res: float):
    """Fixed tempo whose sixteenth grid best fits the hits -> (bpm, grid offset in s)."""
    best = (-1.0, lo, 0.0)
    for bpm in np.arange(lo, hi, res):
        step = 60.0 / bpm / 4
        z = np.exp(2j * np.pi * hits / step).mean()
        if abs(z) > best[0]:
            best = (abs(z), bpm, np.angle(z) / (2 * np.pi) * step)
    return best[1], best[2]


def _drift(hits, beats, window=10.0) -> float:
    """Largest median offset (s) of the hits from the grid over any `window` seconds."""
    r = _residual(hits, beats)
    meds = [np.median(r[(hits >= t) & (hits < t + window)]) for t in np.arange(0, hits.max(), window)
            if np.sum((hits >= t) & (hits < t + window)) >= 6]
    return float(np.max(np.abs(meds))) if meds else 0.0


def _residual(hits, beats):
    """Signed distance (s) of each hit to the nearest sixteenth of a beat grid."""
    idx = np.interp(hits, beats, np.arange(len(beats))) * 4
    return (idx - np.round(idx)) * np.interp(hits, beats[1:], np.diff(beats)) / 4


def fit_beats(hits: np.ndarray, duration: float, approx_bpm: float):
    """Beat times covering [-1 bar, duration + 1 bar] -> (beats, fixed_bpm or None, local tempos)."""
    bpm, off = _fit(hits, approx_bpm - 4, approx_bpm + 4, 0.005)
    local = []
    for c in np.arange(15, max(16, hits.max() - 14), 8.0):
        w = hits[(hits > c - 15) & (hits < c + 15)]
        if len(w) >= 24:
            local.append((c, _fit(w, bpm - 2, bpm + 2, 0.02)[0]))
    local = np.array(local) if local else np.array([[0.0, bpm]])
    T = 60.0 / bpm
    if np.ptp(local[:, 1]) < STABLE_SPREAD_BPM:
        n0 = int(np.ceil((off + 2 * 4 * T) / T))
        beats = off - n0 * T + T * np.arange(int((duration + 2 * 4 * T - off) / T) + n0 + 1)
        if _drift(hits, beats) <= MAX_FIXED_DRIFT_S:
            return beats, bpm, local
    # variable tempo: integrate the smoothed local tempo curve, then pull the grid onto the hits
    curve = np.convolve(local[:, 1], np.ones(3) / 3, mode="same") if len(local) >= 3 else local[:, 1]
    curve[0], curve[-1] = local[0, 1], local[-1, 1]
    beats = [off]
    while beats[-1] < duration + 8 * T:
        beats.append(beats[-1] + 60.0 / np.interp(beats[-1], local[:, 0], curve))
    while beats[0] > -8 * T:
        beats.insert(0, beats[0] - 60.0 / np.interp(beats[0], local[:, 0], curve))
    beats = np.array(beats)
    for _ in range(3):
        r = _residual(hits, beats)
        # smooth correction: median residual in 20 s windows, interpolated at every beat
        cs = np.arange(0, hits.max() + 10, 5.0)
        med = np.array([np.median(r[np.abs(hits - c) < 10]) if np.any(np.abs(hits - c) < 10) else 0.0 for c in cs])
        beats = beats + np.interp(beats, cs, med)
    return beats, None, local


def _beat_phase(hits_by_pitch: dict, beats: np.ndarray) -> np.ndarray:
    """The sixteenth grid has four candidate 'beat' positions; kicks and snares mostly land on the beat."""
    if not len(beats):
        return beats
    main = np.concatenate([hits_by_pitch.get(36, []), hits_by_pitch.get(38, [])])
    if not len(main):
        return beats
    pos = np.round(np.interp(main, beats, np.arange(len(beats))) * 4).astype(int) % 4
    shift = int(np.bincount(pos, minlength=4).argmax())
    if shift == 0:
        return beats
    return beats + np.interp(beats, beats[1:], np.diff(beats)) * shift / 4


def _downbeat(hits_by_pitch: dict, chord_starts, beats: np.ndarray) -> int:
    """Which beat (mod 4) opens the bar: kicks and chord changes favour it, snares (on 2 and 4) do not."""
    score = np.zeros(4)
    for times, w in ((hits_by_pitch.get(36, []), 1.0), (chord_starts, 3.0), (hits_by_pitch.get(38, []), -1.0)):
        if len(times):
            k = np.round(np.interp(times, beats, np.arange(len(beats)))).astype(int) % 4
            score += w * np.bincount(k, minlength=4)
    return int(score.argmax())


# ------------------------------------------------------------------ writing
def _track(name: str, events) -> mido.MidiTrack:
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("track_name", name=name, time=0))
    last = 0
    for tick, msg in sorted(events, key=lambda e: (e[0], e[1].type == "note_on")):
        tr.append(msg.copy(time=tick - last))
        last = tick
    return tr


def write_full_midi(out: Path, free_midi: Path, drums_midi: Path, segments, duration: float,
                    song_audio: Path | None = None, aligned_audio: Path | None = None) -> dict:
    """Write the multitrack MIDI (and the aligned song if asked). Returns a summary for the log."""
    exact = _notes_in_seconds(free_midi)
    by_pitch = {p: np.array([t for t, q, _ in exact if q == p]) for p in (36, 38, 42)}
    hits = np.array(sorted(t for t, _, _ in exact))
    q = mido.MidiFile(str(drums_midi))
    tempos = [m.tempo for tr in q.tracks for m in tr if m.type == "set_tempo"]
    approx = 60e6 / float(np.median(tempos)) if tempos else 120.0

    beats, fixed, local = fit_beats(hits, duration, approx)
    beats = _beat_phase(by_pitch, beats)
    down = _downbeat(by_pitch, [s for s, _, c in segments if voicing(c)], beats)
    # bar 1 = the last downbeat at or before t=0
    cand = [i for i in range(len(beats)) if i % 4 == down and beats[i] <= 0.0]
    first = cand[-1] if cand else down
    lead = -float(beats[first])
    beats = beats[first:]

    def tick(t: float) -> float:
        return float(np.interp(t, beats, np.arange(len(beats)))) * PPQ

    def snap(t: float, grid: int) -> int:
        return int(round(tick(t) / grid)) * grid

    meta = [(0, mido.MetaMessage("time_signature", numerator=4, denominator=4))]
    if fixed:
        meta.append((0, mido.MetaMessage("set_tempo", tempo=int(round(60e6 / fixed)))))
    else:
        meta += [(i * PPQ, mido.MetaMessage("set_tempo", tempo=int(round(d * 1e6))))
                 for i, d in enumerate(np.diff(beats))]

    drum_events, last_on = [], {}
    for t, pitch, vel in _notes_in_seconds(drums_midi):
        a = snap(t, PPQ // 4)
        if a < 0:
            continue
        drum_events.append((a, mido.Message("note_on", channel=DRUM_CH, note=pitch, velocity=vel)))
        last_on.setdefault(pitch, []).append(a)
    for pitch, starts in last_on.items():
        starts = sorted(set(starts))
        length = PPQ // 2 if pitch == 46 else PPQ // 4
        for i, a in enumerate(starts):
            end = min(a + length - 1, (starts[i + 1] - 1) if i + 1 < len(starts) else a + length - 1)
            drum_events.append((max(end, a + 1), mido.Message("note_off", channel=DRUM_CH, note=pitch, velocity=0)))

    chord_events = [(0, mido.Message("program_change", channel=PIANO_CH, program=0))]
    for t0, t1, chord in segments:
        notes = voicing(chord)
        # nearest beat: the chord detector's beats can sit up to a sixteenth off this grid (an eighth would
        # send those changes to the off-beat)
        a, b = snap(t0, PPQ), snap(t1, PPQ) - 1
        if not notes or b <= a:
            continue
        for n in notes:
            chord_events.append((a, mido.Message("note_on", channel=PIANO_CH, note=n, velocity=80)))
            chord_events.append((b, mido.Message("note_off", channel=PIANO_CH, note=n, velocity=0)))

    mid = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    mid.tracks += [_track("Tempo", meta), _track("Bateria", drum_events), _track("Acordes", chord_events)]
    out.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(out))

    if song_audio and aligned_audio:
        import librosa
        import soundfile as sf
        y, sr = librosa.load(song_audio, sr=None, mono=False)
        y = np.atleast_2d(y)
        pad = np.zeros((y.shape[0], int(round(lead * sr))), dtype=y.dtype)
        sf.write(aligned_audio, np.concatenate([pad, y], axis=1).T, sr, subtype="PCM_24")

    return {"bpm": fixed, "bpm_range": (float(local[:, 1].min()), float(local[:, 1].max())), "lead": lead,
            "beats": beats, "error_ms": float(1000 * np.mean(np.abs(_residual(hits, beats)))),
            "drift_ms": 1000 * _drift(hits, beats)}
