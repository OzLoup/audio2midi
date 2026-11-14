"""Baseline DSP drum transcriber: kick / snare / hi-hat (all cymbals) -> MIDI.

Baseline only (no ML). Pipeline:
  1. per-band spectral flux -> peak picking (kick / snare / hat)
  2. refine each onset time on the band-filtered waveform
  3. beat tracking -> variable tempo map that follows the audio
  4. quantize hits to a 16th grid built on the tracked beats
  5. write a .mid whose tempo map keeps it in sync with the source audio
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import mido
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

SR = 44100
N_FFT = 2048
HOP = 256  # ~5.8 ms
PPQ = 480

NOTE = {"kick": 36, "snare": 38, "hat": 42}  # General MIDI drum map
BANDS = {"kick": (30, 120), "snare": (180, 2500), "hat": (6000, 16000)}
CFG = {"kick": (0.06, 0.30), "snare": (0.08, 0.30), "hat": (0.04, 0.25)}  # min gap s, rel threshold


# ----------------------------------------------------------------- detection
def band_flux(S_db, freqs, lo, hi):
    band = S_db[(freqs >= lo) & (freqs <= hi)]
    diff = np.maximum(band[:, 1:] - band[:, :-1], 0.0)
    return np.concatenate([[0.0], diff.mean(axis=0)])


def pick_onsets(flux, hop_s, min_gap_s, rel_thresh):
    f = np.convolve(flux, np.ones(3) / 3, mode="same")
    win = int(0.5 / hop_s)
    local = np.array([np.median(f[max(0, i - win): i + win + 1]) for i in range(len(f))])
    thresh = local + rel_thresh * np.percentile(f, 99)
    peaks, _ = find_peaks(f, height=thresh, distance=max(1, int(min_gap_s / hop_s)))
    return peaks, f[peaks]


def band_envelope(y, lo, hi):
    sos = butter(4, [lo, min(hi, SR / 2 - 100)], btype="band", fs=SR, output="sos")
    env = np.abs(sosfiltfilt(sos, y))
    k = int(0.002 * SR)  # 2 ms smoothing
    return np.convolve(env, np.ones(k) / k, mode="same")


def refine_time(env, t, back=0.05):
    """Move a flux-peak time back to where the band envelope starts rising."""
    c = int(t * SR)
    lo, hi = max(0, c - int(0.03 * SR)), min(len(env), c + int(0.03 * SR))
    if hi <= lo:
        return t
    peak = lo + int(np.argmax(env[lo:hi]))
    level = 0.3 * env[peak]
    i, limit = peak, max(0, peak - int(back * SR))
    while i > limit and env[i] > level:
        i -= 1
    return i / SR


def detect_hits(y):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    S_db = librosa.amplitude_to_db(S, ref=np.max, top_db=80.0)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    hop_s = HOP / SR
    mid = band_flux(S_db, freqs, 2000, 8000)
    hits = {}
    for name, (lo, hi) in BANDS.items():
        peaks, strength = pick_onsets(band_flux(S_db, freqs, lo, hi), hop_s, *CFG[name])
        if name == "snare":  # real snares also have noisy 2-8 kHz energy
            keep = [i for i, p in enumerate(peaks) if mid[max(0, p - 1): p + 2].max() > 0.5]
            peaks, strength = peaks[keep], strength[keep]
        env = band_envelope(y, lo, hi)
        times = np.array([refine_time(env, p * hop_s) for p in peaks])
        hits[name] = (times, strength)
    return hits


# --------------------------------------------------------------- beat / grid
def track_beats(y, start_bpm):
    oenv = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    _, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=SR, hop_length=HOP,
                                       start_bpm=start_bpm, tightness=150)
    bt = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    # median-smooth inter-beat intervals so single glitches don't create tempo spikes
    ibi = np.diff(bt)
    k = 5
    pad = np.pad(ibi, k // 2, mode="edge")
    ibi_s = np.array([np.median(pad[i:i + k]) for i in range(len(ibi))])
    return bt, ibi_s


def build_beat_grid(bt, ibi, duration):
    """Beat times covering [0, duration]: extrapolate with the edge intervals."""
    beats = list(bt)
    first, last = ibi[0], ibi[-1]
    while beats[0] - first > 0:
        beats.insert(0, beats[0] - first)
    beats.insert(0, beats[0] - first)
    while beats[-1] < duration + last:
        beats.append(beats[-1] + last)
    return np.array(beats)


def pin_to_zero(beats):
    """Make tick 0 == audio t=0: replace every beat <= 0 by a single beat at 0."""
    ibi0 = np.median(np.diff(beats))
    rest = beats[beats >= ibi0]  # first interval is 1-2 beats long, never a tempo spike
    return np.concatenate([[0.0], rest])


def quantize(hits, beats, subdivision, normalize=True):
    """Return {class: {slot: strength}} where slot is in 1/subdivision-beat units from beats[0]."""
    idx = np.arange(len(beats), dtype=float)

    def to_slots(times):
        pos = np.interp(times, beats, idx) * subdivision
        return pos

    # global phase correction: mean offset of hits from the grid, in slot units
    allpos = np.concatenate([to_slots(t) for t, _ in hits.values() if len(t)])
    off = np.angle(np.mean(np.exp(2j * np.pi * allpos))) / (2 * np.pi)  # in slots, -0.5..0.5
    out, stats = {}, {}
    for name, (times, strength) in hits.items():
        pos = to_slots(times) - off
        slots = np.round(pos).astype(int)
        err_ms = np.abs(pos - slots) * np.interp(times, beats[:-1], np.diff(beats)) / subdivision * 1000
        s_norm = strength / np.percentile(strength, 95) if normalize else strength
        d = {}
        for s, v in zip(slots, s_norm):
            d[s] = max(d.get(s, 0.0), float(v))
        out[name] = d
        stats[name] = (len(times), len(d), float(err_ms.mean()))
    return out, off, stats


# ------------------------------------------------------------------ MIDI out
def write_midi(path, beats, quantized, subdivision, phase_slots=0.0, direct_velocity=False):
    mid = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    tempo_track, drum_track = mido.MidiTrack(), mido.MidiTrack()
    mid.tracks += [tempo_track, drum_track]

    # tempo map: one tempo event per beat so the MIDI follows the audio
    ibi = np.diff(beats)
    events = [(0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(60.0 / ibi[0]), time=0))]
    for i, d in enumerate(ibi):
        events.append((i * PPQ, mido.MetaMessage("set_tempo", tempo=int(round(d * 1e6)))))
    _fill(tempo_track, events)  # beats[0] == 0, so tick 0 is audio t=0

    dur = PPQ // subdivision // 2
    notes = []
    for name, d in quantized.items():
        for slot, s in d.items():
            tick = int(round(slot * PPQ / subdivision))
            if tick < 0:
                continue
            vel = int(np.clip(s, 1, 127)) if direct_velocity else int(np.clip(45 + 70 * min(s, 1.2) / 1.2, 1, 127))
            notes.append((tick, 1, mido.Message("note_on", channel=9, note=NOTE[name], velocity=vel)))
            notes.append((tick + dur, 0, mido.Message("note_off", channel=9, note=NOTE[name], velocity=0)))
    notes.sort(key=lambda x: (x[0], x[1]))
    _fill(drum_track, [(t, m) for t, _, m in notes])
    mid.save(path)


def _fill(track, events):
    events = sorted(events, key=lambda e: e[0])
    last = 0
    for tick, msg in events:
        msg = msg.copy(time=tick - last)
        track.append(msg)
        last = tick
    track.append(mido.MetaMessage("end_of_track", time=0))


# ----------------------------------------------------------------------- CLI
def transcribe(path: Path, out: Path, start_bpm: float, subdivision: int = 4):
    y, _ = librosa.load(path, sr=SR, mono=True)
    dur = len(y) / SR
    hits = detect_hits(y)
    bt, ibi = track_beats(y, start_bpm)
    beats = build_beat_grid(bt, ibi, dur)
    # 1st pass only measures the global phase between hits and grid; shift the grid by it
    _, off, _ = quantize(hits, beats, subdivision)
    local_step = np.append(np.diff(beats), ibi[-1]) / subdivision
    beats = pin_to_zero(beats + off * local_step)
    quantized, off2, stats = quantize(hits, beats, subdivision)

    write_midi(out, beats, quantized, subdivision, off2)

    bpms = 60.0 / np.diff(bt)
    print(f"duration {dur:.1f}s | tracked beats {len(bt)} | tempo mean {bpms.mean():.1f} "
          f"min {bpms.min():.1f} max {bpms.max():.1f} BPM")
    step_ms = 60.0 / bpms.mean() / subdivision * 1000
    print(f"grid step ~{step_ms:.0f} ms (random-placement error would be ~{step_ms/4:.1f} ms)")
    for n, (raw, q, err) in stats.items():
        print(f"  {n:6s} detected {raw:5d}  after quantize {q:5d}  mean grid error {err:.1f} ms")
    return beats, quantized, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--bpm", type=float, required=True, help="approximate tempo (hint for the beat tracker)")
    ap.add_argument("--subdivision", type=int, default=4, help="grid steps per beat (4 = 16ths)")
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args()
    out = a.out or Path("output") / (a.audio.stem + ".mid")
    out.parent.mkdir(parents=True, exist_ok=True)
    transcribe(a.audio, out, a.bpm, a.subdivision)
    print("wrote", out)


if __name__ == "__main__":
    main()
