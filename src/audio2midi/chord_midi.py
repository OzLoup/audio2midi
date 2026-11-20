"""Chords -> a simple piano MIDI track to guide playing along in a DAW.

One block chord per detected chord, held until the next change: the triad (or seventh) in close position
around middle C plus the bass note an octave below (the slash note for chords like Cm/G).

To line up with the audio in the DAW the MIDI needs a tempo map that follows the song. When the
quantized drum MIDI exists its tempo map is reused, so drums and chords share one grid; otherwise one
is built from the chord detector's own beats.
"""
from __future__ import annotations

import bisect
import re
from pathlib import Path

import mido
import numpy as np

PPQ = 480
PITCH = {n: i for i, n in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])} | \
        {n: i for i, n in enumerate(["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"])}
INTERVALS = {"": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10), "m7": (0, 3, 7, 10), "maj7": (0, 4, 7, 11),
             "5": (0, 7), "sus4": (0, 5, 7), "sus2": (0, 2, 7), "dim": (0, 3, 6), "aug": (0, 4, 8)}
_CHORD = re.compile(r"^([A-G][#b]?)(.*?)(?:/([A-G][#b]?))?$")
LOW_ROOT = 53     # chord roots go from F3 (53) to E4 (64): close voicings around middle C
BASS_OCTAVE = 36  # bass note in C2..B2
VELOCITY = 80
SNAP = PPQ // 2   # chord changes snap to the nearest eighth: on the beat, or anticipated by half a beat


def voicing(chord: str) -> list[int] | None:
    """MIDI notes for a chord name, or None if it cannot be read (e.g. 'N' = no chord)."""
    m = _CHORD.match(chord)
    if not m or m.group(2) not in INTERVALS:
        return None
    root = PITCH[m.group(1)]
    bass = PITCH[m.group(3)] if m.group(3) else root
    base = LOW_ROOT + (root - LOW_ROOT) % 12
    return [BASS_OCTAVE + bass] + [base + iv for iv in INTERVALS[m.group(2)]]


class TempoMap:
    """Seconds <-> ticks for a list of (tick, microseconds per beat) tempo changes."""

    def __init__(self, changes: list[tuple[int, int]], ppq: int):
        changes = sorted(changes)
        if not changes or changes[0][0] != 0:
            changes.insert(0, (0, 500000))
        self.ppq, self.ticks, self.tempos, self.secs = ppq, [], [], []
        sec, prev_tick, prev_tempo = 0.0, 0, changes[0][1]
        for tick, tempo in changes:
            sec += (tick - prev_tick) * prev_tempo / 1e6 / ppq
            self.ticks.append(tick)
            self.tempos.append(tempo)
            self.secs.append(sec)
            prev_tick, prev_tempo = tick, tempo

    def tick(self, t: float) -> int:
        i = max(0, bisect.bisect_right(self.secs, t) - 1)
        return int(round(self.ticks[i] + (t - self.secs[i]) * 1e6 / self.tempos[i] * self.ppq))


def snap(tick: int) -> int:
    return int(round(tick / SNAP)) * SNAP


def _tempo_from_midi(path: Path) -> tuple[list[mido.MetaMessage], TempoMap]:
    src = mido.MidiFile(str(path))
    metas, changes = [], []
    for tr in src.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type in ("set_tempo", "time_signature"):
                metas.append((t, msg))
                if msg.type == "set_tempo":
                    changes.append((t, msg.tempo))
    # rescale to our resolution so the copied tempo events and the chord notes share one tick grid
    scale = PPQ / src.ticks_per_beat
    metas = [(int(round(t * scale)), m) for t, m in metas]
    return [m.copy(time=t) for t, m in sorted(metas, key=lambda x: x[0])], \
        TempoMap([(int(round(t * scale)), v) for t, v in changes], PPQ)


def _tempo_from_beats(beats: np.ndarray) -> tuple[list[mido.MetaMessage], TempoMap]:
    """One tempo change per beat; beats[0] is taken as tick 0 = audio time 0."""
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 2:
        changes = [(0, 500000)]
    else:
        changes = [(i * PPQ, int(round(d * 1e6))) for i, d in enumerate(np.diff(beats)) if d > 0]
    metas = [mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0)]
    metas += [mido.MetaMessage("set_tempo", tempo=v, time=t) for t, v in changes]
    return metas, TempoMap(changes, PPQ)


def write_chord_midi(segments, path: Path, tempo_midi: Path | None = None, beats=None) -> int:
    """Write the chords as a piano track. Returns the number of chords written."""
    metas, tmap = (_tempo_from_midi(tempo_midi) if tempo_midi and Path(tempo_midi).exists()
                   else _tempo_from_beats(beats if beats is not None else [0.0]))
    events = []
    for t0, t1, chord in segments:
        notes = voicing(chord)
        if not notes:
            continue
        # the chord detector's beats sit a few ms off the drum grid: snap so chords land on the grid lines
        a, b = snap(tmap.tick(t0)), snap(tmap.tick(t1)) - 1
        if b <= a:
            continue
        for n in notes:
            events.append((a, 1, mido.Message("note_on", note=n, velocity=VELOCITY, channel=0)))
            events.append((b, 0, mido.Message("note_off", note=n, velocity=0, channel=0)))
    events.sort(key=lambda e: (e[0], e[1]))

    mid = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    tempo_track, chord_track = mido.MidiTrack(), mido.MidiTrack()
    mid.tracks += [tempo_track, chord_track]
    last = 0
    for m in sorted(metas, key=lambda m: m.time):  # metas carry absolute ticks in .time
        tempo_track.append(m.copy(time=m.time - last))
        last = m.time
    chord_track.append(mido.MetaMessage("track_name", name="Acordes", time=0))
    chord_track.append(mido.Message("program_change", program=0, channel=0, time=0))  # acoustic grand piano
    last = 0
    for tick, _, msg in events:
        chord_track.append(msg.copy(time=tick - last))
        last = tick
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))
    return sum(1 for s in segments if voicing(s[2]))
