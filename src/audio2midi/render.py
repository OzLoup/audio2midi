"""Audition helper: render a drum MIDI to audio with simple synthetic sounds.

Uses the MIDI's own tempo map (via pretty_midi), so it never depends on a DAW's tempo import.
Writes two files next to the MIDI: *_midi.wav (MIDI only) and *_mix.wav (original left-ish + MIDI).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 44100
rng = np.random.default_rng(0)


def _kick():
    t = np.arange(int(0.25 * SR)) / SR
    f = 45 + 90 * np.exp(-t * 35)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 14)


def _snare():
    t = np.arange(int(0.2 * SR)) / SR
    noise = sosfilt(butter(2, [1500, 9000], "band", fs=SR, output="sos"), rng.standard_normal(len(t)))
    return 0.6 * noise * np.exp(-t * 22) + 0.5 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30)


def _hat():
    t = np.arange(int(0.07 * SR)) / SR
    noise = sosfilt(butter(2, 7000, "high", fs=SR, output="sos"), rng.standard_normal(len(t)))
    return 0.35 * noise * np.exp(-t * 70)


SOUNDS = {36: _kick(), 38: _snare(), 42: _hat()}


def render(midi_path: Path, audio_path: Path | None, length_s: float):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    out = np.zeros(int(length_s * SR) + SR)
    for n in pm.instruments[0].notes:
        snd = SOUNDS.get(n.pitch)
        if snd is None:
            continue
        i = int(n.start * SR)
        out[i:i + len(snd)] += snd * (n.velocity / 127.0)
    out = out / max(1e-9, np.abs(out).max()) * 0.8
    sf.write(midi_path.with_name(midi_path.stem + "_midi.wav"), out, SR, subtype="PCM_16")
    if audio_path is not None:
        y, _ = librosa.load(audio_path, sr=SR, mono=True)
        mix = np.zeros_like(out)
        mix[:len(y)] += y / max(1e-9, np.abs(y).max()) * 0.5
        mix += out * 0.6
        sf.write(midi_path.with_name(midi_path.stem + "_mix.wav"), mix / np.abs(mix).max() * 0.9, SR, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("midi", type=Path)
    ap.add_argument("--audio", type=Path)
    a = ap.parse_args()
    length = librosa.get_duration(path=a.audio) if a.audio else pretty_midi.PrettyMIDI(str(a.midi)).get_end_time()
    render(a.midi, a.audio, length)
    print("ok")


if __name__ == "__main__":
    main()
