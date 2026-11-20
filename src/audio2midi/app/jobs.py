"""What the app runs: the processing steps, their commands and where their files end up.

No Qt here. Each step runs as its own process because the engine lives in three environments:
  main  (.venv-new)  drums -> MIDI, chords, chart (and this app)
  sep   (.venv-new)  stem separation (audio-separator); the same environment as main
  asr   (.venv-asr)  lyrics (faster-whisper); apart because it needs onnxruntime, not onnxruntime-gpu
(the folder names are the development layout; see paths.py for the installed one)

Files for a song called X:
  stems/X/htdemucs_ft/      drums, bass, vocals, other (fast separation)
  stems/X/roformer_inst/    instrumental + vocals (high quality, slow)
  output/X/midi/            X_bateria_quant.mid, X_bateria_free.mid, audition wavs
  output/X/letra/           X_whisper.json
  output/X/hoja/            X_acordes.html / .pdf / .cho / .txt
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import json

from .. import paths
from ..device import Device

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
WIN = sys.platform == "win32"

DEMUCS_MODEL = "htdemucs_ft.yaml"
ROFORMER_MODEL = "melband_roformer_inst_v1e_plus.ckpt"
LANGUAGES = {"Detectar": None, "Español": "es", "Inglés": "en", "Portugués": "pt"}
PROMPTS = {"es": "Letra de una canción en español.", "en": "Lyrics of a song in English.",
           "pt": "Letra de uma canção em português."}


def venv_exe(venv: str, name: str) -> Path:
    """Executable in an environment: venv is 'main', 'sep' or 'asr' (see paths.py)."""
    return paths.env_exe(venv, name)


def process_env() -> dict:
    env = dict(os.environ)
    env["PATH"] = str(paths.BIN_DIR) + os.pathsep + env.get("PATH", "")  # portable ffmpeg
    env["PYTHONPATH"] = str(paths.SRC_DIR)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"   # never pick up packages of another Python on this PC
    env["PYTHONWARNINGS"] = "ignore"
    env["HF_HOME"] = str(paths.HF_HOME)
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    return env


# ------------------------------------------------------------------ where a song's files live
@dataclass
class SongFiles:
    audio: Path

    @property
    def name(self) -> str:
        return self.audio.stem

    @property
    def stems(self) -> Path:
        return paths.STEMS_DIR / self.name

    @property
    def out(self) -> Path:
        return paths.OUTPUT_DIR / self.name

    def demucs(self, part: str) -> Path:  # part: Drums, Bass, Vocals, Other
        return self.stems / "htdemucs_ft" / f"{self.name}_({part})_htdemucs_ft.wav"

    def _roformer(self, part: str) -> Path | None:
        # audio-separator names the stems after the model: "(vocals)" and "(other)" on some versions,
        # "(Vocals)" and "(Instrumental)" on others -> match case-insensitively, accept both for the instrumental
        names = {"instrumental": ("instrumental", "other"), "vocals": ("vocals",)}[part.lower()]
        for f in sorted((self.stems / "roformer_inst").glob("*.wav")):
            if any(f"({n})" in f.name.lower() for n in names):
                return f
        return None

    @property
    def roformer_instrumental(self) -> Path | None:
        return self._roformer("Instrumental")

    @property
    def roformer_vocals(self) -> Path | None:
        return self._roformer("Vocals")

    @property
    def vocals(self) -> Path:
        """Best vocal stem available: Roformer if it was run, else Demucs."""
        return self.roformer_vocals or self.demucs("Vocals")

    @property
    def drum_midi(self) -> Path:
        return self.out / "midi" / f"{self.name}_bateria_quant.mid"

    @property
    def hats_midi(self) -> Path:
        return self.out / "midi" / f"{self.name}_bateria_quant_hihats.mid"

    @property
    def full_midi(self) -> Path:
        return self.out / "midi" / f"{self.name}_completo.mid"

    @property
    def aligned_audio(self) -> Path:
        return self.out / "midi" / f"{self.name}_alineada.wav"

    @property
    def lyrics(self) -> Path:
        return self.out / "letra" / f"{self.name}_whisper.json"

    @property
    def chart_html(self) -> Path:
        return self.out / "hoja" / f"{self.name}_acordes.html"

    def demucs_done(self) -> bool:
        return all(self.demucs(p).exists() for p in ("Drums", "Bass", "Vocals", "Other"))

    def results(self) -> list[tuple[str, str, Path]]:
        """Every file worth showing as (group, friendly label, path), for the results view."""
        rows: list[tuple[str, str, Path]] = []

        def add(group, items):
            rows.extend((group, label, p) for label, p in items if p and p.exists())

        add("Pistas separadas (alta calidad)", [("Instrumental", self.roformer_instrumental),
                                                ("Voz", self.roformer_vocals)])
        add("Pistas separadas", [(lbl, self.demucs(p)) for lbl, p in
                                 (("Voz", "Vocals"), ("Batería", "Drums"), ("Bajo", "Bass"),
                                  ("Resto (guitarras, teclados…)", "Other"))])
        midi = self.out / "midi" / f"{self.name}_bateria"
        add("Batería en MIDI", [("MIDI cuantizado (para tu DAW)", Path(f"{midi}_quant.mid")),
                                ("MIDI cuantizado con hi-hats rellenados", self.hats_midi),
                                ("MIDI sin cuantizar (tiempos exactos)", Path(f"{midi}_free.mid")),
                                ("Escuchar: canción + MIDI encima", Path(f"{midi}_quant_mix.wav")),
                                ("Escuchar: solo el MIDI", Path(f"{midi}_quant_midi.wav"))])
        add("Para tu DAW (todo en la misma rejilla)", [("MIDI completo: batería + acordes", self.full_midi),
                                                          ("Canción alineada al compás 1", self.aligned_audio)])
        sheet = self.out / "hoja" / f"{self.name}_acordes"
        add("Hoja de acordes", [("Hoja (navegador)", Path(f"{sheet}.html")), ("Hoja (PDF)", Path(f"{sheet}.pdf")),
                                ("Acordes en MIDI (piano)", Path(f"{sheet}.mid")),
                                ("ChordPro (editable)", Path(f"{sheet}.cho")), ("Texto plano", Path(f"{sheet}.txt"))])
        add("Letra", [("Letra con tiempos (Whisper)", self.lyrics)])
        return rows


# ------------------------------------------------------------------ steps
@dataclass
class Options:
    drums: bool = True
    drums_only_input: bool = False   # the file already is a drum track: skip separation for it
    bpm: float = 0.0                 # 0 = estimate
    fill_hats: bool = False          # add the missing hi-hats to the quantized drum MIDI
    hat_open: bool = False           # ... as open hi-hat (note 46) instead of closed (42)
    hat_eighths: bool = False        # ... on every eighth instead of every quarter note
    chart: bool = True
    lyrics: bool = True
    language: str | None = None
    hq_separation: bool = False
    use_gpu: bool = True
    redo: bool = False


@dataclass
class Step:
    key: str
    title: str
    kind: str                         # what it runs on: "cuda", "mps" or "cpu"
    seconds: float                    # estimated duration for this song on this machine
    command: Callable[[], list[str]]  # built when the step starts, so it sees files made by earlier steps
    done: Callable[[], bool]
    prepare: Callable[[], None] = field(default=lambda: None)
    env: dict = field(default_factory=dict)   # extra environment variables for this step
    audio_s: float | None = None      # length of the audio it processes (to learn from the real duration)
    always: bool = False              # rebuilt on every run even if its files exist (quick, and depends on options)

    @property
    def device(self) -> str:          # shown to the user
        return "CPU" if self.kind == "cpu" else "GPU"

    @property
    def estimate(self) -> str:        # shown to the user
        return fmt_estimate(self.seconds)

    def skip(self, redo: bool) -> bool:
        return not redo and not self.always and self.done()


# ------------------------------------------------------------------ how long things take
# (fixed seconds, seconds per second of audio). Measured on a GTX 1650 SUPER (4 GB) and a Ryzen 5 8500G.
# Other machines differ: `learn` corrects the estimates with the real durations seen on each one.
RATES: dict[tuple[str, str], tuple[float, float]] = {
    ("demucs", "cuda"): (10.0, 0.36),     # 30 s clip: 21 s, 90 s clip: 42 s
    ("demucs", "cpu"): (0.0, 4.7),        # 139 s, 425 s
    ("roformer", "cuda"): (26.0, 0.9),    # 10 s clip: 35 s, 30 s clip: 53 s
    ("roformer", "cpu"): (26.0, 3.1),     # 57 s, 119 s
    ("drums", "cuda"): (4.5, 0.04),       # 5.6 s, 7.9 s
    ("drums", "cpu"): (4.0, 0.15),        # ~8 s, 15-21 s
    ("hats", "cpu"): (1.0, 0.0),          # a MIDI rewrite: instant
    ("lyrics", "cpu"): (30.0, 2.0),       # not measured: from ~8 min for a 4 min song
    ("chart", "cpu"): (15.0, 0.2),        # not measured: rough
}   # a device without its own row (Apple's GPU, for now) uses the CPU row until `learn` corrects it
DEFAULT_AUDIO_S = 240.0   # assumed when the length of a file cannot be read


def _timings_file() -> Path:
    return paths.DATA_DIR / "timings.json"


def _scales() -> dict[str, float]:
    try:
        return json.loads(_timings_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _base_seconds(key: str, kind: str, audio_s: float | None) -> float:
    fixed, per_s = RATES.get((key, kind)) or RATES[(key, "cpu")]
    return fixed + per_s * (audio_s or DEFAULT_AUDIO_S)


def estimate_s(key: str, kind: str, audio_s: float | None) -> float:
    """Expected duration of a step, corrected by what this machine took before."""
    return _base_seconds(key, kind, audio_s) * _scales().get(f"{key}/{kind}", 1.0)


def learn(step: Step, actual_s: float) -> None:
    """Remember how the real duration compared with the built-in rate (per step and device)."""
    if not step.audio_s or actual_s <= 0:
        return
    scales = _scales()
    name = f"{step.key}/{step.kind}"
    seen = min(max(actual_s / _base_seconds(step.key, step.kind, step.audio_s), 0.05), 50.0)
    scales[name] = round(0.5 * scales[name] + 0.5 * seen, 3) if name in scales else round(seen, 3)
    try:
        _timings_file().write_text(json.dumps(scales, indent=1), encoding="utf-8")
    except OSError:
        pass


def fmt_estimate(seconds: float) -> str:
    if seconds < 50:
        return "< 1 min"
    if seconds < 90 * 60:
        return f"~{max(1, round(seconds / 60))} min"
    return f"~{seconds / 3600:.1f} h".replace(".", ",")


def separator_cmd() -> list[str]:
    """The stem separator's command line, without arguments. It goes through the Python of its environment
    (not audio-separator.exe, whose launcher keeps the path of the machine that built it)."""
    return [str(venv_exe("sep", "python")), "-c", "from audio_separator.utils.cli import main; main()"]


def plan(song: SongFiles, o: Options, dev: Device | None = None) -> list[Step]:
    dev = dev or Device()
    steps: list[Step] = []
    py = str(venv_exe("main", "python"))
    sep = separator_cmd()
    models = str(paths.SEPARATOR_MODELS)
    audio_s = duration_s(song.audio)

    # the separator picks the GPU by itself; hiding it is how "do not use the GPU" reaches it
    sep_kind = dev.kind if o.use_gpu else "cpu"
    sep_env = {"CUDA_VISIBLE_DEVICES": "-1"} if dev.kind == "cuda" and not o.use_gpu else {}
    # the drum model runs on CUDA or CPU (Apple's GPU is not wired in yet)
    drum_kind = "cuda" if dev.kind == "cuda" and o.use_gpu else "cpu"

    def step(key, title, kind, command, done, prepare=lambda: None, env=None, always=False):
        steps.append(Step(key, title, kind, estimate_s(key, kind, audio_s), command, done, prepare,
                          env or {}, audio_s, always))

    needs_demucs = o.chart or (o.drums and not o.drums_only_input)
    if needs_demucs:
        out = song.stems / "htdemucs_ft"
        step("demucs", "Separar pistas (batería, bajo, voz, resto)", sep_kind,
             lambda: [*sep, str(song.audio), "-m", DEMUCS_MODEL, "--model_file_dir", models,
                      "--output_dir", str(out), "--output_format", "WAV"],
             song.demucs_done, lambda: out.mkdir(parents=True, exist_ok=True), sep_env)

    if o.hq_separation:
        out_hq = song.stems / "roformer_inst"
        step("roformer", "Instrumental y voz en alta calidad", sep_kind,
             lambda: [*sep, str(song.audio), "-m", ROFORMER_MODEL, "--model_file_dir", models,
                      "--output_dir", str(out_hq), "--output_format", "WAV", "--use_autocast", "--mdxc_overlap", "2"],
             lambda: song.roformer_instrumental is not None, lambda: out_hq.mkdir(parents=True, exist_ok=True),
             sep_env)

    if o.drums:
        def drums_cmd():
            src = song.audio if o.drums_only_input else song.demucs("Drums")
            cmd = [py, "-m", "audio2midi.pipeline", str(src), "--out", str(song.out / "midi"),
                   "--name", f"{song.name}_bateria", "--device", drum_kind]
            return cmd + (["--bpm", f"{o.bpm:g}"] if o.bpm else [])
        step("drums", "Batería a MIDI", drum_kind, drums_cmd, lambda: song.drum_midi.exists())

    if o.fill_hats and (o.drums or song.drum_midi.exists()):
        def hats_cmd():
            cmd = [py, "-m", "audio2midi.fill_hats", str(song.drum_midi), str(song.hats_midi)]
            return cmd + (["--open"] if o.hat_open else []) + (["--8ths"] if o.hat_eighths else [])
        # always rebuilt (it takes a moment and depends on the hi-hat options, so a leftover file may be stale)
        step("hats", "Rellenar hi-hats en el MIDI", "cpu", hats_cmd, lambda: song.hats_midi.exists(), always=True)

    if o.chart and o.lyrics:
        def asr_cmd():
            cmd = [str(venv_exe("asr", "python")), str(paths.SRC_DIR / "audio2midi" / "asr_worker.py"),
                   str(song.vocals), str(song.lyrics), "--model", "large-v3", "--device", "cpu", "--compute", "int8"]
            if o.language:
                cmd += ["--language", o.language, "--prompt", PROMPTS[o.language]]
            return cmd
        step("lyrics", "Letra (Whisper)", "cpu", asr_cmd, lambda: song.lyrics.exists(),
             lambda: song.lyrics.parent.mkdir(parents=True, exist_ok=True))

    if o.chart:
        def chart_cmd():
            cmd = [py, "-m", "audio2midi.make_chart", "--other", str(song.demucs("Other")),
                   "--bass", str(song.demucs("Bass")), "--drums", str(song.demucs("Drums")),
                   "--vocals", str(song.vocals), "--title", song.name.replace("_", " "),
                   "--out", str(song.out / "hoja"), "--name", f"{song.name}_acordes"]
            if o.lyrics and song.lyrics.exists():
                cmd += ["--lyrics", str(song.lyrics)]
            if song.drum_midi.exists():  # chord MIDI on the same tempo grid as the drum MIDI
                cmd += ["--tempo-midi", str(song.drum_midi), "--full-midi", str(song.full_midi),
                        "--song", str(song.audio), "--aligned-audio", str(song.aligned_audio)]
            return cmd
        # the chart is cheap and depends on the steps above, so it is always rebuilt
        step("chart", "Hoja de acordes", "cpu", chart_cmd, lambda: song.chart_html.exists(), always=True)
    return steps


# ------------------------------------------------------------------ models downloaded on first use
def _demucs_ready() -> bool:
    """htdemucs_ft is a yaml (`models: ['f7e0c4bc', ...]`) naming the first 8 characters of its checkpoints,
    which are stored as <id>-<rest>.th; every one of them must be on disk."""
    try:
        text = (paths.SEPARATOR_MODELS / DEMUCS_MODEL).read_text(encoding="utf-8")
    except OSError:
        return False
    found = re.search(r"models:\s*\[(.*?)\]", text)
    ids = re.findall(r"[0-9a-f]{8}", found.group(1)) if found else []
    return bool(ids) and all(any(paths.SEPARATOR_MODELS.glob(f"{i}-*.th")) for i in ids)


def _whisper_ready() -> bool:
    hub = paths.HF_HOME / "hub" / "models--Systran--faster-whisper-large-v3" / "snapshots"
    return any(hub.glob("*/model.bin")) if hub.exists() else False


# step key -> (what the user sees, approximate size in MB, is it already on disk?)
MODELS = {
    "demucs": ("Separación de pistas (Demucs)", 300, _demucs_ready),
    "roformer": ("Separación en alta calidad (Roformer)", 880,
                 lambda: (paths.SEPARATOR_MODELS / ROFORMER_MODEL).exists()),
    "lyrics": ("Letra (Whisper large-v3)", 3000, _whisper_ready),
}


def missing_downloads(steps: list[Step]) -> list[tuple[str, int]]:
    """Models that these steps need and that are not on disk yet: (name, MB). They download on first use."""
    out = []
    for s in steps:
        what = MODELS.get(s.key)
        if what and not what[2]():
            out.append((what[0], what[1]))
    return out


def missing_environments(steps: list[Step]) -> list[str]:
    need = {"demucs": "sep", "roformer": "sep", "lyrics": "asr"}
    envs = {"main"} | {need[s.key] for s in steps if s.key in need}
    return sorted(e for e in envs if not venv_exe(e, "python").exists())


# ------------------------------------------------------------------ helpers
def gpu_users() -> list[str]:
    """Python processes using the NVIDIA GPU, i.e. other ML work (empty if none or nvidia-smi is missing).

    On Windows every desktop app shows up as a GPU user, so only Python processes are reported."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        r = subprocess.run([exe, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10,
                           creationflags=subprocess.CREATE_NO_WINDOW if WIN else 0)
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for ln in r.stdout.splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 2 and "python" in Path(parts[1]).name.lower():
            out.append(f"{Path(parts[1]).name} (PID {parts[0]})")
    return out


def duration_s(path: Path) -> float | None:
    try:
        import soundfile as sf
        return sf.info(str(path)).duration
    except Exception:
        return None


def kill_tree(pid: int):
    """Stop a step and every process it started (the separator launcher spawns a Python child)."""
    if WIN:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        # The step shares the app's process group (QProcess), so killpg would kill the app too:
        # walk the descendants and kill them one by one, children first.
        def descendants(parent: int) -> list[int]:
            out = subprocess.run(["pgrep", "-P", str(parent)], capture_output=True, text=True).stdout
            kids = [int(x) for x in out.split()]
            return [g for k in kids for g in descendants(k)] + kids
        for target in descendants(pid) + [pid]:
            try:
                os.kill(target, 9)
            except OSError:
                pass


def open_path(path: Path):
    if WIN:
        os.startfile(str(path))  # noqa: S606 - opening the user's own output file
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


def reveal(path: Path):
    if WIN:
        subprocess.Popen(["explorer", "/select,", str(path)])
    else:
        open_path(path.parent)
