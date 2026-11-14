"""Chord chart model + renderers (ChordPro, plain text, HTML, PDF) with guitar helpers (capo).

A song is a list of Lines. A Line is a list of Tokens; a Token is a piece of lyric text with an
optional chord placed above its first character. Instrumental lines have tokens with empty text.
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
PITCH = {n: i for i, n in enumerate(SHARP)} | {n: i for i, n in enumerate(FLAT)}
_CHORD_RE = re.compile(r"^([A-G][#b]?)(.*?)(?:/([A-G][#b]?))?$")

# Chords a beginner can play without a barre. Used to choose a capo position.
OPEN_SHAPES = {"C", "A", "G", "E", "D", "Am", "Em", "Dm", "A7", "E7", "D7", "G7", "C7", "B7",
               "Cmaj7", "Amaj7", "Dmaj7", "Fmaj7", "Gmaj7", "Am7", "Em7", "Dm7", "Asus4", "Dsus4", "Esus4",
               "Asus2", "Dsus2", "Cadd9", "Gsus4"}


@dataclass
class Token:
    text: str = ""
    chord: str | None = None


@dataclass
class Line:
    tokens: list[Token] = field(default_factory=list)
    label: str | None = None  # section name shown before the line ("Intro", "Verse 1", ...)

    @property
    def instrumental(self) -> bool:
        return all(not t.text.strip() for t in self.tokens)


@dataclass
class Song:
    title: str = "Untitled"
    artist: str = ""
    key: str = ""
    bpm: float | None = None
    capo: int = 0  # chords are stored as *sounding* chords; capo is applied when rendering
    lines: list[Line] = field(default_factory=list)


# ------------------------------------------------------------------ chord helpers
def parse_chord(chord: str):
    m = _CHORD_RE.match(chord)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def transpose_chord(chord: str, semitones: int, prefer_flats: bool = False) -> str:
    p = parse_chord(chord)
    if p is None:
        return chord
    root, qual, bass = p
    names = FLAT if prefer_flats else SHARP
    out = names[(PITCH[root] + semitones) % 12] + qual
    if bass:
        out += "/" + names[(PITCH[bass] + semitones) % 12]
    return out


def shape_score(chords: list[str]) -> float:
    """Share of chords (weighted by count) that are open, barre-free shapes."""
    if not chords:
        return 0.0
    return sum(1 for c in chords if c.split("/")[0] in OPEN_SHAPES) / len(chords)


def suggest_capo(chords: list[str], max_capo: int = 7):
    """Pick the capo fret that makes the most chords open shapes. Ties -> lower capo.
    Returns (capo, shapes_score, score_without_capo)."""
    base = shape_score(chords)
    best, best_score = 0, base
    for capo in range(1, max_capo + 1):
        s = shape_score([transpose_chord(c, -capo) for c in chords])
        if s > best_score + 0.05:  # require a clear gain before recommending a capo
            best, best_score = capo, s
    return best, best_score, base


def song_chords(song: Song) -> list[str]:
    return [t.chord for ln in song.lines for t in ln.tokens if t.chord and t.chord != "N"]


def _shown(song: Song, chord: str) -> str:
    """Chord as it should be *played* (capo applied)."""
    return transpose_chord(chord, -song.capo) if song.capo else chord


# ------------------------------------------------------------------ ChordPro
def to_chordpro(song: Song) -> str:
    out = [f"{{title: {song.title}}}"]
    if song.artist:
        out.append(f"{{artist: {song.artist}}}")
    if song.key:
        out.append(f"{{key: {song.key}}}")
    if song.bpm:
        out.append(f"{{tempo: {round(song.bpm)}}}")
    if song.capo:
        out.append(f"{{capo: {song.capo}}}")
    out.append("")
    for ln in song.lines:
        if ln.label:
            out.append(f"{{comment: {ln.label}}}")
        row = ""
        for t in ln.tokens:
            row += (f"[{_shown(song, t.chord)}]" if t.chord and t.chord != "N" else "") + t.text
            if ln.instrumental and t.chord:
                row += " "
        out.append(row.rstrip())
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ plain text
def to_text(song: Song) -> str:
    out = [song.title + (f" - {song.artist}" if song.artist else "")]
    meta = [f"Key: {song.key}" if song.key else "", f"Tempo: {round(song.bpm)} BPM" if song.bpm else "",
            f"Capo: {song.capo}" if song.capo else ""]
    out += [" | ".join(m for m in meta if m), ""]
    for ln in song.lines:
        if ln.label:
            out.append(f"[{ln.label}]")
        chord_row, text_row = "", ""
        for t in ln.tokens:
            c = _shown(song, t.chord) if t.chord and t.chord != "N" else ""
            width = max(len(t.text), len(c) + 1 if c else 0)
            chord_row = chord_row.ljust(len(text_row)) + c
            text_row += t.text.ljust(width)
            chord_row = chord_row.ljust(len(text_row))
        if chord_row.strip():
            out.append(chord_row.rstrip())
        if text_row.strip():
            out.append(text_row.rstrip())
        out.append("")
    return "\n".join(out)


# ------------------------------------------------------------------ HTML
_CSS = """
:root{--bg:#fff;--fg:#1c1c1e;--muted:#6b6b70;--chord:#b3261e;--rule:#e4e4e7}
@media (prefers-color-scheme:dark){:root:not([data-print]){--bg:#17171a;--fg:#ececf0;--muted:#9a9aa3;--chord:#ff8a80;--rule:#2c2c31}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.35 system-ui,Segoe UI,sans-serif}
main{max-width:760px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;margin:0 0 4px}.artist{color:var(--muted);margin:0 0 12px}
.meta{display:flex;flex-wrap:wrap;gap:8px 18px;color:var(--muted);font-size:14px;padding-bottom:14px;border-bottom:1px solid var(--rule);margin-bottom:18px}
.meta b{color:var(--fg);font-weight:600}
.label{font-weight:700;font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:22px 0 6px}
.line{display:flex;flex-wrap:wrap;align-items:flex-end;margin:0 0 6px}
.tok{display:inline-flex;flex-direction:column}
.c{color:var(--chord);font:700 15px/1.2 ui-monospace,Consolas,monospace;min-height:1.2em;padding-right:6px}
.t{white-space:pre}
.inst .tok{margin-right:14px}
@media print{@page{margin:14mm}body{font-size:14px}main{max-width:none;padding:0}.label{break-after:avoid}.line{break-inside:avoid}}
@media (max-width:480px){body{font-size:16px}main{padding:20px 16px 48px}}
"""


def to_html(song: Song, for_print: bool = False) -> str:
    rows = []
    for ln in song.lines:
        if ln.label:
            rows.append(f'<div class="label">{html.escape(ln.label)}</div>')
        cls = "line inst" if ln.instrumental else "line"
        toks = []
        for t in ln.tokens:
            c = _shown(song, t.chord) if t.chord and t.chord != "N" else ""
            toks.append(f'<span class="tok"><span class="c">{html.escape(c)}</span>'
                        f'<span class="t">{html.escape(t.text) or "&nbsp;"}</span></span>')
        rows.append(f'<div class="{cls}">{"".join(toks)}</div>')
    meta = []
    if song.key:
        meta.append(f"Key <b>{html.escape(song.key)}</b>")
    if song.bpm:
        meta.append(f"Tempo <b>{round(song.bpm)} BPM</b>")
    if song.capo:
        meta.append(f"Capo <b>fret {song.capo}</b>")
    attr = ' data-print="1"' if for_print else ""
    return (f'<!doctype html><html lang="en"{attr}><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{html.escape(song.title)}</title><style>{_CSS}</style></head><body><main>"
            f"<h1>{html.escape(song.title)}</h1>"
            f'<p class="artist">{html.escape(song.artist)}</p>'
            f'<div class="meta">{" ".join(f"<span>{m}</span>" for m in meta)}</div>'
            f'{"".join(rows)}</main></body></html>')


# ------------------------------------------------------------------ PDF (via Edge/Chrome headless)
def _browser() -> str | None:
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"):
        if Path(p).exists():
            return p
    return shutil.which("chrome") or shutil.which("msedge") or shutil.which("chromium")


def _pdf_qt(html: str, out: Path) -> None:
    """Print the HTML to PDF with QtWebEngine (ships with PySide6; no external browser needed)."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QEventLoop, QMarginsF, QTimer
    from PySide6.QtGui import QPageLayout, QPageSize
    from PySide6.QtWebEngineCore import QWebEnginePage
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    page = QWebEnginePage()
    loop = QEventLoop()
    result = {}
    page.loadFinished.connect(lambda ok: (result.update(ok=ok), loop.quit()))
    page.setHtml(html)
    QTimer.singleShot(60_000, loop.quit)
    loop.exec()
    if not result.get("ok"):
        raise RuntimeError("QtWebEngine could not load the chart")
    layout = QPageLayout(QPageSize(QPageSize.A4), QPageLayout.Portrait, QMarginsF(12, 12, 12, 12), QPageLayout.Millimeter)
    done = QEventLoop()
    page.pdfPrintingFinished.connect(lambda path, ok: (result.update(pdf=ok), done.quit()))
    page.printToPdf(str(out), layout)
    QTimer.singleShot(60_000, done.quit)
    done.exec()
    if not result.get("pdf"):
        raise RuntimeError("QtWebEngine could not write the PDF")


def to_pdf(song: Song, out: Path) -> Path:
    out = Path(out).resolve()
    html = to_html(song, for_print=True)
    try:
        _pdf_qt(html, out)
        return out
    except ImportError:
        pass  # no QtWebEngine: fall back to a Chromium-based browser
    exe = _browser()
    if exe is None:
        raise RuntimeError("No QtWebEngine or Chromium-based browser found to print the PDF")
    with tempfile.TemporaryDirectory() as d:
        page = Path(d) / "chart.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={out}", page.as_uri()],
                       check=True, capture_output=True, timeout=120)
    return out


def save_all(song: Song, folder: Path, stem: str):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.cho").write_text(to_chordpro(song), encoding="utf-8")
    (folder / f"{stem}.txt").write_text(to_text(song), encoding="utf-8")
    (folder / f"{stem}.html").write_text(to_html(song), encoding="utf-8")
    pdf = None
    try:
        pdf = to_pdf(song, folder / f"{stem}.pdf")
    except Exception as e:  # PDF is optional
        print("PDF no generado:", e)
    return folder, pdf
