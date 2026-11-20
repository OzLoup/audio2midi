"""Rellena los huecos de hi-hat en el MIDI de bateria.

Pone un hi-hat (cerrado = nota 42, abierto = nota 46) en cada negra, o cada corchea, de los compases que
tienen redoblante. No duplica golpes de hi-hat que ya existan y no toca bombo, redoblante, tempo ni los
compases sin redoblante (arranques, silencios). Los hi-hats duran 1/16 (cerrado) o 1/8 (abierto) sin
solaparse con el siguiente golpe del mismo plato.

  python -m audio2midi.fill_hats entrada.mid salida.mid [--open] [--8ths]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mido

CLOSED, OPEN, SNARE = 42, 46, 38
HATS = (CLOSED, OPEN)
VELOCITY = 85


def fill(src: Path, dst: Path, open_hat: bool = False, eighths: bool = False) -> int:
    """Escribe `dst` con los hi-hats añadidos y devuelve cuantos se añadieron."""
    m = mido.MidiFile(str(src))
    T = m.ticks_per_beat
    bar, step = 4 * T, T // (2 if eighths else 1)
    idx = next((i for i, tr in enumerate(m.tracks) if any(x.type == "note_on" for x in tr)), None)
    if idx is None:
        raise ValueError("el MIDI no tiene notas")
    tr = m.tracks[idx]
    ev, t = [], 0
    for x in tr:
        t += x.time
        if x.type != "end_of_track":
            ev.append([t, x])
    end = t
    on = lambda e, n: e[1].type == "note_on" and e[1].velocity > 0 and e[1].note == n  # noqa: E731
    channel = next(e[1].channel for e in ev if e[1].type == "note_on")
    note = OPEN if open_hat else CLOSED
    snare_bars = {e[0] // bar for e in ev if on(e, SNARE)}
    existing = {e[0] for e in ev if any(on(e, n) for n in HATS)}
    near = lambda s: any(abs(s - h) < T // 8 for h in existing)  # noqa: E731
    added = 0
    for b in sorted(snare_bars):
        for s in range(b * bar, (b + 1) * bar, step):
            if not near(s):
                ev.append([s, mido.Message("note_on", note=note, velocity=VELOCITY, channel=channel)])
                existing.add(s)
                added += 1
    # los fines de nota de los hi-hats se recalculan: duracion fija sin solaparse con el siguiente golpe
    ev = [e for e in ev if not (e[1].type in ("note_on", "note_off") and e[1].note in HATS
                                and (e[1].type == "note_off" or e[1].velocity == 0))]
    for n in HATS:
        starts = sorted(e[0] for e in ev if on(e, n))
        length = (T // 2 if n == OPEN else T // 4) - 1
        for i, s in enumerate(starts):
            nxt = starts[i + 1] if i + 1 < len(starts) else 1 << 40
            ev.append([min(s + length, nxt - 1), mido.Message("note_off", note=n, velocity=0, channel=channel)])
    ev.sort(key=lambda e: (e[0], e[1].type != "note_off"))
    out, prev = mido.MidiTrack(), 0
    out.name = tr.name
    for s, x in ev:
        out.append(x.copy(time=s - prev))
        prev = s
    out.append(mido.MetaMessage("end_of_track", time=max(0, end - prev)))
    m.tracks[idx] = out
    dst.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(dst))
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--open", action="store_true", help="hi-hat abierto (nota 46) en vez de cerrado (42)")
    ap.add_argument("--8ths", dest="eighths", action="store_true", help="a corcheas en vez de a negras")
    a = ap.parse_args()
    print(f"hi-hats nuevos: {fill(a.src, a.dst, a.open, a.eighths)} -> {a.dst.name}")


if __name__ == "__main__":
    main()
