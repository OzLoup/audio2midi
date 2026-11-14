"""Combine chord segments and timed lyrics into a chart.Song.

- drops words that fall on silence in the vocal stem (Whisper hallucinations such as "Gracias.")
- groups words into lines using Whisper's phrases (splitting long ones at pauses)
- puts each chord above the word where it starts to sound
- labels repeated blocks as Chorus, other blocks as Verse, and wordless stretches as Intro/Instrumental
"""
from __future__ import annotations

import difflib
import json
import re

import numpy as np
import soundfile as sf

from .chart import Line, Song, Token

MIN_VOICE_DB = -50.0   # a word over a vocal stem quieter than this is treated as hallucinated
SPLIT_GAP = 0.9        # seconds of pause that split a long Whisper phrase into two lines
MAX_LINE_S = 7.5
BLOCK_GAP = 2.5        # a pause longer than this starts a new section block
CHORD_LEAD = 0.30      # a chord that starts up to this much before a word still belongs to that word


def chord_at(segments, t):
    for t0, t1, c in segments:
        if t0 <= t < t1:
            return c
    return segments[-1][2] if segments and t >= segments[-1][1] else (segments[0][2] if segments else None)


def load_words(asr_json, vocals_path):
    data = json.load(open(asr_json, encoding="utf-8"))
    x, sr = sf.read(vocals_path, dtype="float32", always_2d=True)
    m = x.mean(axis=1)
    keep, dropped = [], []
    for w in data["words"]:
        seg = m[int(w["start"] * sr): max(int(w["end"] * sr), int(w["start"] * sr) + 1)]
        db = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9) if len(seg) else -120
        (keep if db > MIN_VOICE_DB else dropped).append(w)
    return keep, dropped, data


def _starts_sentence(word: str) -> bool:
    w = word.lstrip("¿¡\"'(")
    return bool(w) and w[0].isupper()


def group_lines(words):
    """Split the word stream into lines at pauses, sentence starts, or when a line gets too long."""
    lines, cur = [], []
    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            too_long = w["end"] - cur[0]["start"] > MAX_LINE_S
            ends_phrase = re.search(r"[.?!…,]$", cur[-1]["w"]) is not None and gap > 0.4
            new_sentence = _starts_sentence(w["w"]) and len(cur) >= 3 and gap > 0.1
            if gap > SPLIT_GAP or too_long or ends_phrase or new_sentence:
                lines.append(cur)
                cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    return _merge_orphans(lines)


def _merge_orphans(lines, min_words=3, max_join_gap=1.2, max_len_s=10.5):
    """Attach 1-2 word lines to a neighbour so they don't sit alone ("vencer", "mí")."""
    out = []
    for ln in lines:
        if out and len(ln) < min_words:
            prev = out[-1]
            if ln[0]["start"] - prev[-1]["end"] < max_join_gap and ln[-1]["end"] - prev[0]["start"] < max_len_s:
                prev.extend(ln)
                continue
        out.append(list(ln))
    # a short line that could not join backwards joins the next line instead
    i = 0
    while i < len(out) - 1:
        if len(out[i]) < min_words and out[i + 1][0]["start"] - out[i][-1]["end"] < max_join_gap:
            out[i + 1] = out[i] + out[i + 1]
            del out[i]
        else:
            i += 1
    return out


def _norm(text):
    return re.sub(r"[^a-záéíóúñü ]", "", text.lower()).strip()


def _chorus_flags(lines_text):
    """A line is chorus-like if a similar line shows up elsewhere in the song."""
    norm = [_norm(t) for t in lines_text]
    flags = []
    for i, a in enumerate(norm):
        rep = any(j != i and difflib.SequenceMatcher(None, a, b).ratio() >= 0.72
                  for j, b in enumerate(norm) if len(a) > 8 and len(b) > 8)
        flags.append(rep)
    return flags


def _instrumental_lines(segments, t0, t1, per_line=8):
    """Chord-only lines for a wordless stretch: one chord per change."""
    chords = [(a, c) for a, b, c in segments if b > t0 and a < t1]
    chords = [(max(a, t0), c) for a, c in chords]
    out = []
    for i in range(0, len(chords), per_line):
        out.append(Line([Token("", c) for _, c in chords[i:i + per_line]]))
    return out


def build_song(segments, asr_json, vocals_path, title, key="", bpm=None, artist=""):
    words, dropped, _ = load_words(asr_json, vocals_path)
    wlines = group_lines(words)
    texts = [" ".join(w["w"] for w in ln) for ln in wlines]
    chorus = _chorus_flags(texts)

    song = Song(title=title, artist=artist, key=key, bpm=bpm)

    # blocks = runs of lines separated by a pause; a block is a Chorus if most of its lines repeat elsewhere
    blocks, cur = [], []
    for i, ln in enumerate(wlines):
        if cur and ln[0]["start"] - wlines[cur[-1]][-1]["end"] > BLOCK_GAP:
            blocks.append(cur)
            cur = []
        cur.append(i)
    if cur:
        blocks.append(cur)
    labels = {}
    verse_n = 0
    for b in blocks:
        if sum(chorus[i] for i in b) >= len(b) / 2:
            labels[b[0]] = "Chorus"
        else:
            verse_n += 1
            labels[b[0]] = f"Verse {verse_n}"

    prev_end = 0.0
    first = True
    for idx, (ln, text) in enumerate(zip(wlines, texts)):
        start, end = ln[0]["start"], ln[-1]["end"]
        # wordless stretch before this line -> chord-only lines
        if start - prev_end > 4.0:
            label = "Intro" if first else "Instrumental"
            inst = _instrumental_lines(segments, prev_end if not first else 0.0, start - 0.3)
            if inst:
                inst[0].label = label
                song.lines += inst
        label = labels.get(idx)

        tokens, last_chord = [], None
        for i, w in enumerate(ln):
            # chords whose start falls inside this word (allowing a small lead)
            here = [c for a, b, c in segments if w["start"] - CHORD_LEAD <= a < w["end"] - 0.05]
            chord = here[0] if here else None
            if i == 0:
                chord = chord_at(segments, w["start"] + 0.1)  # every line opens with the sounding chord
            if chord == last_chord and i != 0:
                chord = None
            if chord:
                last_chord = chord
            tokens.append(Token(w["w"] + (" " if i < len(ln) - 1 else ""), chord))
        song.lines.append(Line(tokens, label=label))
        prev_end = end
        first = False
    return song, len(words), len(dropped)
