"""Transcribe a drum track with the pretrained ADTOF model and fold it to kick / snare / hi-hat.

ADTOF classes: 35 kick, 38 snare, 47 toms, 42 hi-hat, 49 cymbals.
Folding: toms and cymbals -> hi-hat (42), as requested for the first version.
Output is written at a constant 120 BPM, so note times in seconds equal audio time.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pretty_midi
from adtof_pytorch import transcribe_to_midi

FOLD = {35: 36, 38: 38, 47: 42, 42: 42, 49: 42}
LABEL = {36: "kick", 38: "snare", 42: "hat"}


def run(audio: Path, out: Path, device: str = "cuda") -> Path:
    raw = out.with_name(out.stem + "_raw5.mid")
    transcribe_to_midi(audio, raw, device=device)
    src = pretty_midi.PrettyMIDI(str(raw))
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    seen = set()
    for n in sorted(src.instruments[0].notes, key=lambda n: n.start):
        pitch = FOLD.get(n.pitch)
        if pitch is None:
            continue
        key = (pitch, round(n.start, 3))  # drop duplicates created by folding (e.g. hat + cymbal)
        if key in seen:
            continue
        seen.add(key)
        drums.notes.append(pretty_midi.Note(100, pitch, n.start, n.start + 0.05))
    pm.instruments.append(drums)
    out.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out))
    counts = Counter(LABEL[n.pitch] for n in drums.notes)
    print(dict(counts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = a.out or Path("output") / (a.audio.stem + "_adtof.mid")
    run(a.audio, out, a.device)
    print("wrote", out)


if __name__ == "__main__":
    main()
