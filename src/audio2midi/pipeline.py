"""Drum audio -> MIDI (kick / snare / hi-hat), built on the pretrained ADTOF model.

Improvements over the plain ADTOF run:
  1. per-class thresholds (CLI) + a sensitivity table showing how many notes depend on them
  2. test-time augmentation: average the model's activations over a few audio variants
  3. velocity estimated from the audio energy of each hit
  4. quantization on a tempo map that follows the audio (smoothed beat tracking)

Outputs:  <stem>_quant.mid  (tempo map + 16th grid, in sync with the audio)
          <stem>_free.mid   (120 BPM, exact hit times, no quantization)
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import torch
from adtof_pytorch import (FRAME_RNN_THRESHOLDS, PeakPicker, calculate_n_bins,
                           create_frame_rnn_model, get_default_weights_path,
                           load_pytorch_weights)
from adtof_pytorch.audio import create_adtof_processor
from scipy.signal import savgol_filter

from . import transcribe as T

SR = 44100
FPS = 100
IDX = {"kick": 0, "snare": 1, "tom": 2, "hat": 3, "cym": 4}  # ADTOF output channels
GROUP = {"kick": ["kick"], "snare": ["snare"], "hat": ["tom", "hat", "cym"]}
VEL_BANDS = {"kick": (30, 120), "snare": (180, 2500), "hat": (6000, 16000)}


# --------------------------------------------------------------- activations
def load_model(device):
    model = create_frame_rnn_model(calculate_n_bins())
    model.eval()
    model = load_pytorch_weights(model, get_default_weights_path(), strict=False)
    return model.to(device)


def activations(model, y, device):
    proc = create_adtof_processor()
    feats = proc.apply_filterbank(proc.compute_stft(y)).T[:, :, None]
    x = torch.from_numpy(np.ascontiguousarray(feats)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x).cpu().numpy()[0]  # [frames, 5]


def tta_variants(y):
    yield "orig", y
    yield "gain-4dB", y * 0.63
    yield "gain+4dB", y * 1.58
    yield "pitch-1", librosa.effects.pitch_shift(y, sr=SR, n_steps=-1)
    yield "pitch+1", librosa.effects.pitch_shift(y, sr=SR, n_steps=1)


def averaged_activations(model, y, device, tta=True):
    acts = []
    for name, v in (tta_variants(y) if tta else [("orig", y)]):
        a = activations(model, v.astype(np.float32), device)
        acts.append(a)
        print(f"  pass {name:9s} frames={len(a)}")
    n = min(len(a) for a in acts)
    return np.mean([a[:n] for a in acts], axis=0), acts[0][:n]


# ----------------------------------------------------------------- peak pick
def pick(act, thresholds):
    """-> {'kick': (times, act_value), 'snare': ..., 'hat': ...}; toms/cymbals folded into hat."""
    picker = PeakPicker(thresholds=thresholds, fps=FPS)
    peaks = picker.pick(act, labels=list(range(5)))[0]
    per = {n: np.array(peaks[i]) for n, i in IDX.items()}
    out = {}
    for cls, members in GROUP.items():
        ts, vs = [], []
        for m in members:
            for t in per[m]:
                ts.append(t)
                vs.append(act[min(int(round(t * FPS)), len(act) - 1), IDX[m]])
        order = np.argsort(ts)
        ts, vs = np.array(ts)[order], np.array(vs)[order]
        keep_t, keep_v = [], []
        for t, v in zip(ts, vs):  # merge hits of the same group closer than 30 ms
            if keep_t and t - keep_t[-1] < 0.03:
                if v > keep_v[-1]:
                    keep_t[-1], keep_v[-1] = t, v
            else:
                keep_t.append(t)
                keep_v.append(v)
        out[cls] = (np.array(keep_t), np.array(keep_v))
    return out


def sensitivity(act, thresholds):
    print("Sensibilidad (cuantas notas hay con cada umbral):")
    print("  umbral        kick  snare   hat")
    for d in (-0.08, -0.04, 0.0, 0.04, 0.08):
        th = [float(np.clip(t + d, 0.02, 0.95)) for t in thresholds]
        h = pick(act, th)
        print(f"  {d:+.2f}  {'(actual)' if d == 0 else '        '}  {len(h['kick'][0]):5d} {len(h['snare'][0]):6d} {len(h['hat'][0]):5d}")


# ------------------------------------------------------------------ velocity
def velocities(y, hits):
    out = {}
    for cls, (times, _) in hits.items():
        lo, hi = VEL_BANDS[cls]
        env = T.band_envelope(y, lo, hi)
        peaks = np.array([env[max(0, int((t - 0.02) * SR)): int((t + 0.04) * SR) + 1].max() for t in times])
        db = 20 * np.log10(peaks + 1e-6)
        p5, p95 = np.percentile(db, [5, 95])
        out[cls] = np.clip(40 + 80 * (db - p5) / max(p95 - p5, 1e-6), 25, 127)
    return out


# --------------------------------------------------------------- tempo / grid
def choose_beats(act, dur, start_bpm, hits, subdivision):
    """Beat-track on kick+snare activations, then smooth the beat times as much as the
    hits' grid fit allows."""
    onset_env = act[:, 0] + act[:, 1] + 0.3 * act[:, 3]
    _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=SR, hop_length=SR // FPS,
                                       start_bpm=start_bpm, tightness=150)
    bt = beats / FPS
    results = []
    for win in (1, 5, 9, 15, 25, 41):
        if win == 1:
            sm = bt
        elif win < len(bt):
            sm = savgol_filter(bt, win, 2)
        else:
            continue
        ibi = np.diff(sm)
        grid = T.build_beat_grid(sm, ibi, dur)
        _, off, _ = T.quantize(hits, grid, subdivision)
        step = np.append(np.diff(grid), ibi[-1]) / subdivision
        grid = T.pin_to_zero(grid + off * step)
        _, _, st = T.quantize(hits, grid, subdivision)
        err = np.average([s[2] for s in st.values()], weights=[s[0] for s in st.values()])
        bpm = 60 / ibi
        results.append((win, err, bpm.min(), bpm.max(), sm, ibi))
    best = min(r[1] for r in results)
    print("  suavizado   error de rejilla   tempo min-max")
    for r in results:
        print(f"  {r[0]:5d}      {r[1]:6.1f} ms        {r[2]:5.1f} - {r[3]:5.1f} BPM")
    chosen = [r for r in results if r[1] <= best + 1.0][-1]  # smoothest within 1 ms of the best fit
    print(f"  -> elegido suavizado {chosen[0]}")
    return chosen[4], chosen[5]


# ----------------------------------------------------------------------- main
def run(audio: Path, outdir: Path, bpm: float | None, thresholds, tta=True, subdivision=4, device="cuda",
        render=True, name: str | None = None):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        device = "cpu"
    name = name or audio.stem
    y, _ = librosa.load(audio, sr=SR, mono=True)
    dur = len(y) / SR
    if not bpm:
        bpm = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=SR)[0])[0])
        print(f"tempo estimado: {bpm:.1f} BPM")
    model = load_model(device)

    print("1) Modelo")
    act, act_single = averaged_activations(model, y, device, tta)
    hits_single = pick(act_single, thresholds)
    hits = pick(act, thresholds)
    print("  notas sin TTA:", {k: len(v[0]) for k, v in hits_single.items()}, "| con TTA:", {k: len(v[0]) for k, v in hits.items()})
    for k in hits:  # how stable: share of TTA hits that also exist in the single pass (within 30 ms)
        a, b = hits[k][0], hits_single[k][0]
        if len(a) and len(b):
            shared = np.mean([np.abs(b - t).min() < 0.03 for t in a])
            print(f"  {k:6s} {shared*100:.0f}% de las notas coinciden con la pasada simple")
    sensitivity(act, thresholds)

    print("2) Velocity")
    vel = velocities(y, hits)
    for k, v in vel.items():
        print(f"  {k:6s} velocity p5={np.percentile(v,5):.0f} mediana={np.median(v):.0f} p95={np.percentile(v,95):.0f}")

    print("3) Tempo y rejilla")
    sm, ibi = choose_beats(act, dur, bpm, {k: (hits[k][0], vel[k]) for k in hits}, subdivision)
    grid = T.build_beat_grid(sm, ibi, dur)
    vhits = {k: (hits[k][0], vel[k]) for k in hits}
    _, off, _ = T.quantize(vhits, grid, subdivision, normalize=False)
    step = np.append(np.diff(grid), ibi[-1]) / subdivision
    grid = T.pin_to_zero(grid + off * step)
    quant, off2, stats = T.quantize(vhits, grid, subdivision, normalize=False)
    for n, (raw, q, err) in stats.items():
        print(f"  {n:6s} detectadas {raw:5d}  tras cuantizar {q:5d}  error medio {err:.1f} ms")

    outdir.mkdir(parents=True, exist_ok=True)
    qpath = outdir / f"{name}_quant.mid"
    T.write_midi(qpath, grid, quant, subdivision, direct_velocity=True)

    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    d = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    for cls, (times, _) in hits.items():
        for t, v in zip(times, vel[cls]):
            d.notes.append(pretty_midi.Note(int(v), T.NOTE[cls], float(t), float(t) + 0.05))
    pm.instruments.append(d)
    fpath = outdir / f"{name}_free.mid"
    pm.write(str(fpath))
    print("escrito:", qpath.name, "|", fpath.name)

    if render:
        from .render import render as render_audio
        render_audio(qpath, audio, dur)
        print("audicion:", qpath.stem + "_mix.wav")
    return qpath, fpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--bpm", type=float, default=None, help="tempo aproximado (pista para el beat tracker); se estima si falta")
    ap.add_argument("--thresholds", type=str, help="5 umbrales: kick,snare,tom,hat,cymbal")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--name", help="nombre base de los archivos de salida")
    ap.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    a = ap.parse_args()
    th = [float(x) for x in a.thresholds.split(",")] if a.thresholds else list(FRAME_RNN_THRESHOLDS)
    run(a.audio, a.out, a.bpm, th, tta=not a.no_tta, render=not a.no_render, device=a.device, name=a.name)


if __name__ == "__main__":
    main()
