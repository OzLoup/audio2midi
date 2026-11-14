"""Restore high-frequency content that a separator removed from the instrumental without justification.

For each time-frequency bin above `fc`, if the vocal estimate holds little energy compared to the
original, but the separated instrumental holds much less than the original, we blend the original back.
Where the vocal estimate is clearly present we keep the separator's instrumental.

Usage: python -m audio2midi.restore_highs original.wav instrumental.wav vocals.wav -o out.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter

N_FFT, HOP = 4096, 1024


def restore(orig, inst, voc, sr, f_lo=5000.0, f_hi=8000.0, ratio_max=0.25, n_fft=N_FFT, hop=HOP):
    """orig/inst/voc: arrays [samples, channels]. Returns restored instrumental.

    Short windows (n_fft=1024, hop=256) keep cymbal/hi-hat transients sharp; long ones smear them."""
    n = min(len(orig), len(inst), len(voc))
    out = np.zeros((n, orig.shape[1]), dtype=np.float32)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    wf = np.clip((freqs - f_lo) / (f_hi - f_lo), 0.0, 1.0)[:, None]  # 0 below f_lo, 1 above f_hi
    for ch in range(orig.shape[1]):
        O = librosa.stft(orig[:n, ch], n_fft=n_fft, hop_length=hop)
        I = librosa.stft(inst[:n, ch], n_fft=n_fft, hop_length=hop)
        V = librosa.stft(voc[:n, ch], n_fft=n_fft, hop_length=hop)
        po, pv = np.abs(O) ** 2, np.abs(V) ** 2
        # smooth the vocal-presence ratio so the mask does not flutter
        r = uniform_filter(pv, size=(5, 3)) / (uniform_filter(po, size=(5, 3)) + 1e-12)
        w_mask = np.clip(1.0 - r / ratio_max, 0.0, 1.0)  # 1 = no vocal there, restore fully
        w = wf * w_mask
        mix = w * O + (1.0 - w) * I
        out[:, ch] = librosa.istft(mix, hop_length=hop, length=n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original", type=Path)
    ap.add_argument("instrumental", type=Path)
    ap.add_argument("vocals", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--f-lo", type=float, default=5000.0)
    ap.add_argument("--f-hi", type=float, default=8000.0)
    ap.add_argument("--ratio-max", type=float, default=0.25)
    a = ap.parse_args()
    orig, sr = sf.read(a.original, dtype="float32", always_2d=True)
    inst, _ = sf.read(a.instrumental, dtype="float32", always_2d=True)
    voc, _ = sf.read(a.vocals, dtype="float32", always_2d=True)
    out = restore(orig, inst, voc, sr, a.f_lo, a.f_hi, a.ratio_max)
    peak = np.abs(out).max()
    if peak > 0.98:
        out *= 0.98 / peak
    sf.write(a.out, out, sr, subtype="PCM_24")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
