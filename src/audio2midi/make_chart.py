"""Build the chord chart (HTML, PDF, ChordPro, text) from separated stems and, if present, Whisper lyrics.

Usage: python -m audio2midi.make_chart --other O.wav --bass B.wav --drums D.wav --vocals V.wav
                                       [--lyrics whisper.json] --title "Song" --out folder --name stem
Without --lyrics the chart shows only the chords, one per bar, eight bars per line.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .chart import Line, Song, Token, save_all, song_chords, suggest_capo
from .chords import detect_chords
from .compose import build_song


def chords_only(result, title: str) -> Song:
    song = Song(title=title, key=result.key, bpm=result.bpm)
    bar = 4 * 60.0 / result.bpm
    events = []
    for t0, t1, c in result.segments:  # a long chord is repeated once per bar
        n = max(1, round((t1 - t0) / bar))
        events += [(t0 + i * bar, c) for i in range(n)]
    for i in range(0, len(events), 8):
        chunk = events[i:i + 8]
        m, s = divmod(int(chunk[0][0]), 60)
        song.lines.append(Line([Token("", c) for _, c in chunk], label=f"{m}:{s:02d}"))
    return song


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--other", type=Path, required=True, help="harmonic stem (never mixed with the bass)")
    ap.add_argument("--bass", type=Path)
    ap.add_argument("--drums", type=Path)
    ap.add_argument("--vocals", type=Path)
    ap.add_argument("--lyrics", type=Path)
    ap.add_argument("--title", default="Untitled")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    print("detectando acordes...", flush=True)
    r = detect_chords(a.other, a.bass, a.drums)
    print(f"tonalidad {r.key} | ~{r.bpm:.0f} BPM | {len(r.segments)} cambios de acorde", flush=True)
    if a.lyrics and a.lyrics.exists() and a.vocals:
        song, used, dropped = build_song(r.segments, a.lyrics, a.vocals, title=a.title, key=r.key, bpm=r.bpm)
        print(f"letra: {used} palabras usadas, {dropped} descartadas por caer en silencio", flush=True)
    else:
        song = chords_only(r, a.title)
        print("sin letra: hoja solo con acordes", flush=True)
    capo, score, base = suggest_capo(song_chords(song))
    song.capo = capo
    print(f"cejilla sugerida: traste {capo} ({score*100:.0f}% formas abiertas, sin cejilla {base*100:.0f}%)", flush=True)
    folder, pdf = save_all(song, a.out, a.name)
    print("escrito:", ", ".join(sorted(p.name for p in folder.glob(a.name + ".*"))), flush=True)


if __name__ == "__main__":
    main()
