@echo off
rem Abre la app de escritorio de audio2midi
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
start "" "%~dp0.venv-new\Scripts\pythonw.exe" -m audio2midi.app
