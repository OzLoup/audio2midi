"""Starts the desktop app without needing PYTHONPATH or a working directory.

The installer's shortcut runs:  runtime\main\pythonw.exe launch.pyw
(.pyw = no console window).
"""
import site
import sys
from pathlib import Path

# The app brings everything it needs: ignore packages from any Python installed on this PC (user site)
_user = site.getusersitepackages()
sys.path[:] = [p for p in sys.path if not p.startswith(_user)]
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from audio2midi import paths  # noqa: E402
from audio2midi.app.main_window import main  # noqa: E402

paths.ensure_data_dirs()
main()
