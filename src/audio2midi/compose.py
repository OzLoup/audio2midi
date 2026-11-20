"""Combine chord segments and timed lyrics into a chart.Song.

- drops words that fall on silence in the vocal stem (Whisper hallucinations such as "Gracias.")
- splits the words into verse lines (break model + typical line length, see group_lines)
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
BLOCK_GAP = 2.5        # a pause longer than this starts a new section block
CHORD_LEAD = 0.30      # a chord that starts up to this much before a word still belongs to that word
# phrases Whisper is known to invent over music (learned from video subtitles), dropped whole
HALLUCINATION = re.compile(r"subt[ií]tul|amara\.org|gracias por ver|suscr[ií]b|like y comparte|"
                           r"thanks for watching|nos vemos en el pr[oó]ximo", re.I)


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
    phrases = data.get("segments", [])
    phrase_starts = np.array([s["start"] for s in phrases])
    junk = [(s["start"] - 0.05, s["end"] + 0.05) for s in phrases if HALLUCINATION.search(s["text"])]
    for w in data["words"]:
        w["phrase"] = bool(len(phrase_starts)) and bool(np.min(np.abs(phrase_starts - w["start"])) < 0.05)
        seg = m[int(w["start"] * sr): max(int(w["end"] * sr), int(w["start"] * sr) + 1)]
        db = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9) if len(seg) else -120
        fake = any(a <= w["start"] and w["end"] <= b for a, b in junk)
        (keep if db > MIN_VOICE_DB and not fake else dropped).append(w)
    return keep, dropped, data


def _starts_sentence(word: str) -> bool:
    w = word.lstrip("¿¡\"'(")
    return bool(w) and w[0].isupper()


# ------------------------------------------------------------------ verse line breaks
# Whisper's word times are glued together in sung lyrics (0 s between "nombre" and "Si"), so pauses alone
# can't find the end of a verse. What does: Whisper's own phrases, the capital letter it puts on a new
# sentence, punctuation, a held last word, and that verse lines in one song last about the same time.
# Each gap between two words gets a break probability (logistic model trained on bench/lines.py against
# reference lyrics), and a dynamic programme picks the breaks that best agree with those probabilities
# while keeping lines close to the song's typical line length (or half of it).
LINE_FEATURES = ("bias", "phrase", "capital", "phrase_and_capital", "punct", "gap", "held")
LINE_WEIGHTS = (-4.06, 1.82, 1.77, 1.90, 1.72, 1.86, 1.08)
LINE_PARAMS = dict(length_weight=3.0, half_cost=0.5, hard_gap=2.0, max_words=24)


def gap_features(words) -> np.ndarray:
    """One row per gap between word i and i+1, columns as LINE_FEATURES."""
    rows = []
    for a, b in zip(words, words[1:]):
        phrase = bool(b.get("phrase"))
        cap = _starts_sentence(b["w"])
        rows.append((1.0, phrase, cap, phrase and cap, re.search(r"[.?!…,;:]$", a["w"]) is not None,
                     min(max(b["start"] - a["end"], 0.0), 1.0), min(a["end"] - a["start"], 2.0) / 2))
    return np.array(rows, dtype=float).reshape(-1, len(LINE_FEATURES))


def break_probs(words, weights=LINE_WEIGHTS) -> np.ndarray:
    if len(words) < 2:
        return np.zeros(0)
    return 1.0 / (1.0 + np.exp(-gap_features(words) @ np.asarray(weights)))


def _typical_length(words, p) -> float:
    """Typical line length (s, start to start) from the confident breaks; 5 s if there are too few."""
    starts = [words[0]["start"]] + [words[i + 1]["start"] for i in np.flatnonzero(p > 0.8)]
    d = np.diff(starts)
    d = d[(d > 2.0) & (d < 12.0)]
    return float(np.median(d)) if len(d) >= 4 else 5.0


def group_lines(words, weights=LINE_WEIGHTS, **params):
    """Split the word stream into verse lines (see the comment above)."""
    if len(words) < 2:
        return [list(words)] if words else []
    q = {**LINE_PARAMS, **params}
    p = np.clip(break_probs(words, weights), 1e-4, 1 - 1e-4)
    gaps = np.array([b["start"] - a["end"] for a, b in zip(words, words[1:])])
    p[gaps > q["hard_gap"]] = 1 - 1e-4
    period = _typical_length(words, p)
    lp, l1p = np.log(p), np.log1p(-p)
    c1p = np.concatenate([[0.0], np.cumsum(l1p)])
    hard = np.concatenate([[0], np.cumsum(gaps > q["hard_gap"])])
    n = len(words)
    best = np.full(n + 1, -np.inf)
    best[0], back = 0.0, np.zeros(n + 1, dtype=int)
    for j in range(1, n + 1):          # line = words[i:j]
        for i in range(max(0, j - q["max_words"]), j):
            if best[i] == -np.inf or hard[j - 1] - hard[i] > 0:  # never join across a long silence
                continue
            # the line runs until the next one starts (a held last word belongs to it)
            end = words[j]["start"] if j < n else words[-1]["end"]
            dur = max(end - words[i]["start"], 0.3)
            dev = min(np.log(dur / period) ** 2, np.log(dur / (period / 2)) ** 2 + q["half_cost"])
            if j < n and gaps[j - 1] > q["hard_gap"]:
                dev = min(dev, max(0.0, np.log(dur / period)) ** 2)  # a line before a silence may be short
            s = best[i] + (c1p[j - 1] - c1p[i]) + (lp[j - 1] if j < n else 0.0) - q["length_weight"] * dev
            if s > best[j]:
                best[j], back[j] = s, i
    lines, j = [], n
    while j > 0:
        lines.append(list(words[back[j]:j]))
        j = back[j]
    return lines[::-1]


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
