"""Where things live. No Qt, no heavy imports: every module (and every step process) can use it.

Two roots:
  APP_DIR   read-only: the code, the portable ffmpeg and the Python environments.
  DATA_DIR  writable: the user's songs, stems, results and downloaded models.

Development (the repo checkout, or AUDIO2MIDI_DATA unset with a `.venv` next to the code):
  both are the repo folder, so nothing moves.
Installed: DATA_DIR is a per-user folder (%LOCALAPPDATA%\\audio2midi, ~/Library/Application Support/audio2midi),
  because the install folder (e.g. Program Files) is not writable.
Set AUDIO2MIDI_DATA to force another data folder (for example a bigger disk).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[2]
WIN = sys.platform == "win32"

# logical environment name -> folder name in a development checkout
_DEV_ENVS = {"main": ".venv-new", "sep": ".venv-new", "asr": ".venv-asr"}   # main and sep share one environment
_INSTALLED_ENVS = {"main": "main", "sep": "main", "asr": "asr"}               # folders under runtime/


def is_dev() -> bool:
    """A checkout has the main environment (or git metadata) next to the code."""
    return (APP_DIR / ".git").exists() or (APP_DIR / _DEV_ENVS["main"]).exists()


def _user_data_dir() -> Path:
    if WIN:
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "audio2midi"


def _data_dir() -> Path:
    override = os.environ.get("AUDIO2MIDI_DATA")
    if override:
        return Path(override).expanduser()
    return APP_DIR if is_dev() else _user_data_dir()


DATA_DIR = _data_dir()

INPUT_DIR = DATA_DIR / "input"
STEMS_DIR = DATA_DIR / "stems"
OUTPUT_DIR = DATA_DIR / "output"
MODELS_DIR = DATA_DIR / "models"
SEPARATOR_MODELS = MODELS_DIR / "separator"
HF_HOME = MODELS_DIR / "hf"
BIN_DIR = APP_DIR / "bin"
SRC_DIR = APP_DIR / "src"


def env_dir(name: str) -> Path:
    """Folder of a Python environment: 'main', 'sep' or 'asr'."""
    dev = APP_DIR / _DEV_ENVS[name]
    return dev if dev.exists() or is_dev() else APP_DIR / "runtime" / _INSTALLED_ENVS[name]


def env_exe(name: str, exe: str) -> Path:
    """Executable `exe` ('python', 'audio-separator', ...) inside environment `name`."""
    folder = env_dir(name) / ("Scripts" if WIN else "bin")
    return folder / (exe + (".exe" if WIN else ""))


def ensure_data_dirs() -> None:
    for d in (INPUT_DIR, STEMS_DIR, OUTPUT_DIR, SEPARATOR_MODELS, HF_HOME):
        d.mkdir(parents=True, exist_ok=True)
