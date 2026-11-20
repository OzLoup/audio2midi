"""Build the chord chart (HTML, PDF, ChordPro, text) from separated stems and, if present, Whisper lyrics.

Usage: python -m audio2midi.make_chart --other O.wav --bass B.wav --drums D.wav --vocals V.wav
                                       [--lyrics whisper.json] [--tempo-midi drums_quant.mid]
                                       --title "Song" --out folder --name stem
Also writes <name>.mid: the chords as a piano track to play along with in a DAW.
Without --lyrics the chart shows only the chords, one per bar, eight bars per line.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .chart import Line, Song, Token, save_all, song_chords, suggest_capo
from .chord_midi import write_chord_midi
from .full_midi import write_full_midi
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


def write_full(a, r):
    """Drums (with the filled hi-hats if they exist) + chords in one MIDI, and the song aligned to it."""
    import librosa
    quant = a.tempo_midi
    free = quant.with_name(quant.name.replace("_quant.mid", "_free.mid"))
    hats = quant.with_name(quant.name.replace("_quant.mid", "_quant_hihats.mid"))
    if not free.exists():
        print("MIDI completo: falta el MIDI sin cuantizar de la batería, no se genera", flush=True)
        return
    duration = librosa.get_duration(path=a.song) if a.song else r.segments[-1][1]
    info = write_full_midi(a.full_midi, free, hats if hats.exists() else quant, r.segments, duration,
                           a.song, a.aligned_audio if a.song else None)
    tempo = (f"tempo fijo {info['bpm']:.2f} BPM" if info["bpm"] else
             f"tempo variable {info['bpm_range'][0]:.1f}-{info['bpm_range'][1]:.1f} BPM (la canción acelera o frena)")
    print(f"MIDI completo: batería + acordes, {tempo}; golpes a {info['error_ms']:.0f} ms de la rejilla de media, "
          f"{info['drift_ms']:.0f} ms como máximo", flush=True)
    print(f"  el compás 1 empieza {info['lead']:.3f} s antes de la canción"
          + ("; la canción alineada lleva ese silencio delante: suelta los dos al inicio del proyecto" if a.song else ""),
          flush=True)


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
    ap.add_argument("--tempo-midi", type=Path,
                    help="MIDI whose tempo map the chord MIDI reuses (the quantized drums), so both share one grid")
    ap.add_argument("--full-midi", type=Path,
                    help="also write drums + chords in one MIDI on a clean tempo (needs --tempo-midi and its _free.mid)")
    ap.add_argument("--song", type=Path, help="the original song: written again aligned to bar 1 of --full-midi")
    ap.add_argument("--aligned-audio", type=Path, help="where to write the aligned song")
    a = ap.parse_args()

    print("detectando acordes...", flush=True)
    # bass summed into the harmonic input: measured clearly better than the guitars/keys stem alone
    r = detect_chords([a.other, a.bass] if a.bass else a.other, a.bass, a.drums)
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
    n = write_chord_midi(r.segments, folder / f"{a.name}.mid", a.tempo_midi, r.beats)
    grid = "tempo de la batería" if a.tempo_midi and a.tempo_midi.exists() else "pulso propio"
    print(f"MIDI de acordes: {n} acordes en piano ({grid})", flush=True)
    if a.full_midi and a.tempo_midi and a.tempo_midi.exists():
        write_full(a, r)
    print("escrito:", ", ".join(sorted(p.name for p in folder.glob(a.name + ".*"))), flush=True)


if __name__ == "__main__":
    main()
