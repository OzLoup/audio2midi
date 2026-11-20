"""Desktop window: pick a song, choose what to get, run it, then listen to and open the results."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QSettings, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPalette
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QPlainTextEdit, QPushButton, QSlider, QSplitter, QTabWidget,
                               QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from . import jobs
from .. import device, paths
from .jobs import Options, SongFiles

STATE_TEXT = {"pending": "Pendiente", "reuse": "Ya hecho", "run": "En curso…", "ok": "Listo",
              "error": "Error", "stopped": "Detenido", "skipped": "Omitido"}
STATE_COLOR = {"ok": "#3ecf8e", "reuse": "#3ecf8e", "error": "#ff5c6c", "run": "#4c90ff", "stopped": "#f0b44c"}
ASSETS = Path(__file__).parent / "assets"


def icon(name: str) -> QIcon:
    return QIcon(str(ASSETS / f"{name}.svg"))
PLAYABLE = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def fmt_time(s: float | None) -> str:
    if s is None:
        return ""
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


# ------------------------------------------------------------------ runs the steps one after another
class Runner(QObject):
    step_state = Signal(int, str, str)   # step index, state key, detail
    line = Signal(str)                   # full output line (for the log)
    progress = Signal(str)               # progress-bar style line that keeps rewriting itself
    finished = Signal(bool)              # True when every step ended well

    def __init__(self, steps: list[jobs.Step], redo: bool):
        super().__init__()
        self.steps, self.redo = steps, redo
        self.i = -1
        self.proc: QProcess | None = None
        self.stopping = False
        self.started_at = 0.0
        self._buf = ""

    def start(self):
        self._next()

    def stop(self):
        self.stopping = True
        if self.proc and self.proc.state() != QProcess.NotRunning:
            jobs.kill_tree(int(self.proc.processId()))

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def _next(self):
        self.i += 1
        if self.i >= len(self.steps):
            self.finished.emit(True)
            return
        step = self.steps[self.i]
        if step.skip(self.redo):
            self.step_state.emit(self.i, "reuse", "")
            self._next()
            return
        step.prepare()
        cmd = step.command()
        self.line.emit(f"\n=== {step.title}\n$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        p = QProcess(self)
        env = QProcessEnvironment()
        for k, v in {**jobs.process_env(), **step.env}.items():
            env.insert(k, v)
        p.setProcessEnvironment(env)
        p.setWorkingDirectory(str(paths.DATA_DIR))
        p.setProcessChannelMode(QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(self._read)
        p.finished.connect(self._done)
        p.errorOccurred.connect(self._error)
        self.proc, self._buf, self.started_at = p, "", time.time()
        self.step_state.emit(self.i, "run", "")
        p.start(cmd[0], cmd[1:])

    def _read(self):
        self._buf += bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        # Windows ends lines with \r\n; a lone \r is a progress bar rewriting its line
        self._buf = self._buf.replace("\r\n", "\n")
        hold = self._buf.endswith("\r")  # might be the first half of a \r\n split across reads
        if hold:
            self._buf = self._buf[:-1]
        while True:
            cut = min((k for k in (self._buf.find("\n"), self._buf.find("\r")) if k >= 0), default=-1)
            if cut < 0:
                break
            text, sep, self._buf = self._buf[:cut], self._buf[cut], self._buf[cut + 1:]
            text = text.strip()
            if not text:
                continue
            if sep == "\r":
                self.progress.emit(text)   # tqdm bars rewrite the same line with \r
            else:
                self.line.emit(text)
                self.progress.emit(text)
        if hold:
            self._buf += "\r"

    def _error(self, err):
        if err == QProcess.FailedToStart:
            self.line.emit(f"No se pudo iniciar: {self.proc.program()}")
            self.step_state.emit(self.i, "error", "no se pudo iniciar")
            self.finished.emit(False)

    def _done(self, code, _status):
        if self._buf.strip():
            self.line.emit(self._buf.strip())
        step = self.steps[self.i]
        secs = fmt_time(self.elapsed())
        if self.stopping:
            self.step_state.emit(self.i, "stopped", secs)
            self.finished.emit(False)
        elif code != 0:
            self.step_state.emit(self.i, "error", f"{secs} · código {code}")
            self.finished.emit(False)
        elif step.key != "chart" and not step.done():
            self.line.emit("El paso terminó pero no produjo los archivos esperados.")
            self.step_state.emit(self.i, "error", f"{secs} · faltan archivos")
            self.finished.emit(False)
        else:
            jobs.learn(step, self.elapsed())   # next estimates on this machine use the real duration
            self.step_state.emit(self.i, "ok", secs)
            self._next()


# ------------------------------------------------------------------ bottom audio player
class Player(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("player")
        self.audio = QAudioOutput()
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio)
        self.play_btn = QToolButton(icon=icon("play"), toolTip="Reproducir / pausa (espacio)")
        self.play_btn.setObjectName("play")
        self.stop_btn = QToolButton(icon=icon("stop"), toolTip="Detener")
        self.name = QLabel("Elige una pista en Resultados para escucharla")
        self.name.setMinimumWidth(200)
        self.slider = QSlider(Qt.Horizontal)
        self.time = QLabel("0:00 / 0:00")
        self.volume = QSlider(Qt.Horizontal, maximum=100, value=80, toolTip="Volumen")
        self.volume.setFixedWidth(90)
        self.audio.setVolume(0.8)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(10)
        for w in (self.play_btn, self.stop_btn, self.name):
            lay.addWidget(w)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.time)
        vol = QLabel("Vol")
        vol.setObjectName("hint")
        lay.addWidget(vol)
        lay.addWidget(self.volume)
        self.play_btn.clicked.connect(self.toggle)
        self.stop_btn.clicked.connect(self.player.stop)
        self.volume.valueChanged.connect(lambda v: self.audio.setVolume(v / 100))
        self.slider.sliderMoved.connect(self.player.setPosition)
        self.player.positionChanged.connect(self._pos)
        self.player.durationChanged.connect(lambda d: self.slider.setRange(0, d))
        self.player.playbackStateChanged.connect(self._state)
        self._enable(False)

    def _enable(self, on):
        for w in (self.play_btn, self.stop_btn, self.slider):
            w.setEnabled(on)

    def load(self, path: Path, autoplay=True):
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.name.setText(path.name)
        self.name.setToolTip(str(path))
        self._enable(True)
        if autoplay:
            self.player.play()

    def toggle(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        elif not self.player.source().isEmpty():
            self.player.play()

    def _pos(self, ms):
        if not self.slider.isSliderDown():
            self.slider.setValue(ms)
        self.time.setText(f"{fmt_time(ms / 1000)} / {fmt_time(self.player.duration() / 1000)}")

    def _state(self, s):
        self.play_btn.setIcon(icon("pause" if s == QMediaPlayer.PlayingState else "play"))


# ------------------------------------------------------------------ main window
class MainWindow(QMainWindow):
    device_found = Signal(object)   # emitted from the probing thread with a device.Device

    def __init__(self):
        super().__init__()
        self.setWindowTitle("audio2midi")
        self.resize(1200, 860)
        self.setAcceptDrops(True)
        self.settings = QSettings("audio2midi", "audio2midi")
        self.song: SongFiles | None = None
        self.runner: Runner | None = None
        self.steps: list[jobs.Step] = []
        self.device: device.Device | None = None   # None until the probe answers

        # left: songs
        self.songs = QListWidget()
        self.songs.setObjectName("songs")
        self.songs.setSelectionMode(QAbstractItemView.SingleSelection)
        self.songs.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.songs.setTextElideMode(Qt.ElideRight)
        self.songs.currentItemChanged.connect(self._song_selected)
        add = QPushButton("Agregar canción…")
        add.clicked.connect(self.add_dialog)
        hint = QLabel("o arrastra un archivo de audio a la ventana")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        left = QWidget()
        left.setObjectName("sidebar")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(16, 18, 12, 14)
        ll.setSpacing(8)
        brand = QLabel('audio<span style="color:#2f7df6">2</span>midi')
        brand.setObjectName("brand")
        ll.addWidget(brand)
        ll.addSpacing(10)
        cap = QLabel("CANCIONES")
        cap.setObjectName("section")
        ll.addWidget(cap)
        ll.addWidget(self.songs, 1)
        ll.addWidget(add)
        ll.addWidget(hint)

        # right: header + tabs
        self.title = QLabel("Ninguna canción")
        self.title.setObjectName("title")
        self.subtitle = QLabel("Agrega una canción para empezar")
        self.subtitle.setObjectName("hint")
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_process_tab(), "Procesar")
        self.tabs.addTab(self._build_results_tab(), "Resultados")
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(24, 18, 24, 14)
        rl.setSpacing(4)
        rl.addWidget(self.title)
        rl.addWidget(self.subtitle)
        rl.addSpacing(8)
        rl.addWidget(self.tabs, 1)

        split = QSplitter()
        split.setHandleWidth(1)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([270, 910])
        self.player = Player()
        central = QWidget()
        central.setObjectName("central")
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(split, 1)
        cl.addWidget(self.player)
        self.setCentralWidget(central)

        space = QAction(self, shortcut=QKeySequence(Qt.Key_Space))
        space.triggered.connect(self.player.toggle)
        self.addAction(space)
        self.clock = QTimer(self, interval=1000)
        self.clock.timeout.connect(self._tick)

        self._load_options()
        self._load_songs()
        self.device_found.connect(self._on_device)
        threading.Thread(target=lambda: self.device_found.emit(
            device.probe(str(jobs.venv_exe("main", "python")))), daemon=True).start()

    # ---------------------------------------------------------------- device
    def _on_device(self, dev: device.Device):
        """The probe answered: show what the machine has and redo the plan with real estimates."""
        self.device = dev
        self.device_label.setText(dev.label)
        self._update_enabled()
        if self.runner is None:
            self._show_plan()

    # ---------------------------------------------------------------- process tab
    def _build_process_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        what = QGroupBox("¿Qué quieres obtener?")
        f = QVBoxLayout(what)
        self.opt_drums = QCheckBox("Batería en MIDI  (kick, snare y hi-hat, cuantizada)")
        self.opt_drums_only = QCheckBox("El archivo ya es solo la batería (no separar)")
        self.opt_bpm = QDoubleSpinBox(minimum=0, maximum=300, decimals=1, singleStep=1, suffix=" BPM")
        self.opt_bpm.setSpecialValueText("automático")
        self.opt_bpm.setToolTip("Tempo aproximado. Ayuda a encontrar el pulso; con 'automático' se estima solo.")
        self.opt_hats = QCheckBox("Rellenar los huecos de hi-hat en el MIDI  (también guarda una copia con ellos)")
        self.opt_hats.setToolTip("Añade un hi-hat en cada golpe en los compases con redoblante. No toca bombo, "
                                 "redoblante ni los compases sin redoblante (arranques y silencios).")
        self.opt_hat_kind = QComboBox()
        self.opt_hat_kind.addItems(["cerrado", "abierto"])
        self.opt_hat_dens = QComboBox()
        self.opt_hat_dens.addItems(["a negras", "a corcheas"])
        self.opt_chart = QCheckBox("Hoja de acordes para guitarra  (HTML, PDF y ChordPro)")
        self.opt_lyrics = QCheckBox("Con la letra encima de los acordes (Whisper, ~8 min en CPU)")
        self.opt_lang = QComboBox()
        self.opt_lang.addItems(list(jobs.LANGUAGES))
        self.opt_hq = QCheckBox("Instrumental y voz en alta calidad  (Roformer, lento)")
        f.addWidget(self.opt_drums)
        sub = QHBoxLayout()
        sub.setContentsMargins(26, 0, 0, 6)
        sub.addWidget(self.opt_drums_only)
        sub.addSpacing(16)
        sub.addWidget(QLabel("Tempo:"))
        sub.addWidget(self.opt_bpm)
        sub.addStretch()
        f.addLayout(sub)
        sub_h = QHBoxLayout()
        sub_h.setContentsMargins(26, 0, 0, 6)
        sub_h.addWidget(self.opt_hats)
        sub_h.addSpacing(16)
        sub_h.addWidget(QLabel("Hi-hat:"))
        sub_h.addWidget(self.opt_hat_kind)
        sub_h.addWidget(self.opt_hat_dens)
        sub_h.addStretch()
        f.addLayout(sub_h)
        f.addWidget(self.opt_chart)
        sub2 = QHBoxLayout()
        sub2.setContentsMargins(26, 0, 0, 6)
        sub2.addWidget(self.opt_lyrics)
        sub2.addSpacing(16)
        sub2.addWidget(QLabel("Idioma:"))
        sub2.addWidget(self.opt_lang)
        sub2.addStretch()
        f.addLayout(sub2)
        f.addWidget(self.opt_hq)

        more = QGroupBox("Opciones")
        g = QHBoxLayout(more)
        self.opt_gpu = QCheckBox("Usar la GPU")
        self.opt_gpu.setToolTip("Para la separación de pistas y la batería. Apagada, todo corre en la CPU (más lento).")
        self.opt_redo = QCheckBox("Rehacer lo que ya está hecho")
        self.opt_redo.setToolTip("Si está apagado, los pasos que ya tienen sus archivos se reutilizan.")
        self.device_label = QLabel("Detectando el equipo…")
        self.device_label.setObjectName("hint")
        g.addWidget(self.opt_gpu)
        g.addWidget(self.opt_redo)
        g.addStretch()
        g.addWidget(self.device_label)

        self.run_btn = QPushButton("Procesar")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("Detener")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        self.status = QLabel("")
        self.status.setObjectName("hint")
        btns = QHBoxLayout()
        btns.addWidget(self.run_btn)
        btns.addWidget(self.stop_btn)
        btns.addSpacing(12)
        btns.addWidget(self.status, 1)

        self.step_view = QTreeWidget()
        self.step_view.setHeaderLabels(["Paso", "Estado", "Dónde corre", "Tiempo"])
        self.step_view.setRootIsDecorated(False)
        self.step_view.setSelectionMode(QAbstractItemView.NoSelection)
        hdr = self.step_view.header()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeToContents)

        self.progress_line = QLabel("")
        self.progress_line.setObjectName("mono")
        self.progress_line.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.log_btn = QToolButton(text="Ver detalles", checkable=True)
        self.step_view.setObjectName("steps")
        self.step_view.setMinimumHeight(150)
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(5000)
        self.log.setObjectName("mono")
        self.log.setVisible(False)
        self.log_btn.toggled.connect(self.log.setVisible)
        self.log_btn.toggled.connect(lambda on: self.log_btn.setText("Ocultar detalles" if on else "Ver detalles"))

        lay.addWidget(what)
        lay.addWidget(more)
        lay.addLayout(btns)
        lay.addWidget(self.step_view, 1)
        lay.addWidget(self.progress_line)
        lay.addWidget(self.log_btn, 0, Qt.AlignLeft)
        lay.addWidget(self.log, 2)

        for cb in (self.opt_drums, self.opt_drums_only, self.opt_chart, self.opt_lyrics, self.opt_hq,
                   self.opt_gpu, self.opt_redo):
            cb.toggled.connect(self._options_changed)
        self.opt_lang.currentIndexChanged.connect(self._options_changed)
        self.opt_hats.toggled.connect(self._options_changed)
        self.opt_hat_kind.currentIndexChanged.connect(self._options_changed)
        self.opt_hat_dens.currentIndexChanged.connect(self._options_changed)
        self.opt_bpm.valueChanged.connect(self._options_changed)
        return w

    def _load_options(self):
        s = self.settings
        b = lambda k, d: s.value(k, d) in (True, "true")  # noqa: E731
        self.opt_drums.setChecked(b("opt/drums", True))
        self.opt_hats.setChecked(b("opt/hats", False))
        self.opt_hat_kind.setCurrentText(s.value("opt/hat_kind", "cerrado"))
        self.opt_hat_dens.setCurrentText(s.value("opt/hat_dens", "a negras"))
        self.opt_chart.setChecked(b("opt/chart", True))
        self.opt_lyrics.setChecked(b("opt/lyrics", True))
        self.opt_hq.setChecked(b("opt/hq", False))
        self.opt_gpu.setChecked(b("opt/gpu", True))
        self.opt_lang.setCurrentText(s.value("opt/lang", "Detectar"))

    def _save_options(self):
        s = self.settings
        s.setValue("opt/drums", self.opt_drums.isChecked())
        s.setValue("opt/hats", self.opt_hats.isChecked())
        s.setValue("opt/hat_kind", self.opt_hat_kind.currentText())
        s.setValue("opt/hat_dens", self.opt_hat_dens.currentText())
        s.setValue("opt/chart", self.opt_chart.isChecked())
        s.setValue("opt/lyrics", self.opt_lyrics.isChecked())
        s.setValue("opt/hq", self.opt_hq.isChecked())
        s.setValue("opt/gpu", self.opt_gpu.isChecked())
        s.setValue("opt/lang", self.opt_lang.currentText())

    def _options_changed(self, *_):
        self._update_enabled()
        if self.runner is None:
            self._show_plan()

    def _update_enabled(self):
        self.opt_drums_only.setEnabled(self.opt_drums.isChecked())
        self.opt_bpm.setEnabled(self.opt_drums.isChecked())
        self.opt_hat_kind.setEnabled(self.opt_hats.isChecked())
        self.opt_hat_dens.setEnabled(self.opt_hats.isChecked())
        self.opt_lyrics.setEnabled(self.opt_chart.isChecked())
        self.opt_lang.setEnabled(self.opt_chart.isChecked() and self.opt_lyrics.isChecked())
        has_gpu = self.device is not None and self.device.gpu
        self.opt_gpu.setEnabled(has_gpu)
        if self.device is not None and not has_gpu:
            self.opt_gpu.setToolTip("No se encontró una GPU compatible en este equipo: todo corre en la CPU.")

    def options(self) -> Options:
        has_gpu = self.device is not None and self.device.gpu
        return Options(drums=self.opt_drums.isChecked(), drums_only_input=self.opt_drums_only.isChecked(),
                       bpm=self.opt_bpm.value(),
                       fill_hats=self.opt_hats.isChecked(), hat_open=self.opt_hat_kind.currentText() == "abierto",
                       hat_eighths=self.opt_hat_dens.currentText() == "a corcheas", chart=self.opt_chart.isChecked(),
                       lyrics=self.opt_lyrics.isChecked(), language=jobs.LANGUAGES[self.opt_lang.currentText()],
                       hq_separation=self.opt_hq.isChecked(), use_gpu=has_gpu and self.opt_gpu.isChecked(),
                       redo=self.opt_redo.isChecked())

    def _show_plan(self):
        """Preview the steps that 'Procesar' would run, marking the ones already done."""
        self.step_view.clear()
        self.steps = jobs.plan(self.song, self.options(), self.device) if self.song else []
        o = self.options()
        for s in self.steps:
            # until the probe answers the device is unknown, so neither it nor the time is shown
            it = QTreeWidgetItem([s.title, "", s.device if self.device else "…", s.estimate if self.device else "…"])
            self.step_view.addTopLevelItem(it)
            self._paint(it, "reuse" if s.skip(o.redo) else "pending")
        self.run_btn.setEnabled(bool(self.steps) and self.runner is None)
        if self.song and not self.steps:
            self.status.setText("Marca al menos una cosa para obtener.")
        elif self.song:
            todo = [s for s in self.steps if not s.skip(o.redo)]
            self.status.setText("Todo está hecho; solo se rehará la hoja." if len(todo) <= 1 and todo and todo[0].key == "chart"
                                else f"{len(todo)} paso(s) por hacer.")

    def _paint(self, item: QTreeWidgetItem, state: str, detail: str = ""):
        item.setText(1, STATE_TEXT[state])
        if detail:
            item.setText(3, detail)
        color = STATE_COLOR.get(state)
        item.setForeground(1, QColor(color) if color else self.palette().color(self.foregroundRole()))
        f = item.font(1)
        f.setBold(state == "run")
        item.setFont(1, f)

    # ---------------------------------------------------------------- results tab
    def _build_results_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.results = QTreeWidget()
        self.results.setHeaderLabels(["Resultado", "Archivo", "Tamaño", "Modificado"])
        hdr = self.results.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.results.itemDoubleClicked.connect(lambda it, _c: self._activate(it))
        self.results.currentItemChanged.connect(self._result_selected)
        self.play_res = QPushButton("Escuchar")
        self.open_res = QPushButton("Abrir")
        self.show_res = QPushButton("Mostrar en la carpeta")
        self.play_res.clicked.connect(lambda: self._activate(self.results.currentItem(), play=True))
        self.open_res.clicked.connect(lambda: self._open_selected())
        self.show_res.clicked.connect(lambda: self._reveal_selected())
        row = QHBoxLayout()
        for b in (self.play_res, self.open_res, self.show_res):
            row.addWidget(b)
        row.addStretch()
        self.empty_results = QLabel("Todavía no hay resultados para esta canción. Ve a «Procesar».")
        self.empty_results.setObjectName("hint")
        lay.addWidget(self.empty_results)
        lay.addWidget(self.results, 1)
        lay.addLayout(row)
        self._result_selected(None)
        return w

    def _refresh_results(self):
        self.results.clear()
        groups = {}
        for group, label, path in (self.song.results() if self.song else []):
            if group not in groups:
                g = QTreeWidgetItem([group])
                f = g.font(0)
                f.setBold(True)
                g.setFont(0, f)
                g.setFlags(g.flags() & ~Qt.ItemIsSelectable)
                self.results.addTopLevelItem(g)
                groups[group] = g
            st = path.stat()
            it = QTreeWidgetItem([label, path.name, f"{st.st_size / 1e6:.1f} MB" if st.st_size > 1e5 else f"{st.st_size / 1e3:.0f} KB",
                                  time.strftime("%d/%m %H:%M", time.localtime(st.st_mtime))])
            it.setData(0, Qt.UserRole, str(path))
            it.setToolTip(0, str(path))
            it.setToolTip(1, str(path))
            it.setForeground(1, self.palette().color(QPalette.PlaceholderText))
            groups[group].addChild(it)
        self.results.expandAll()
        self.empty_results.setVisible(not groups)
        self.results.setVisible(bool(groups))

    def _selected_path(self) -> Path | None:
        it = self.results.currentItem()
        p = it.data(0, Qt.UserRole) if it else None
        return Path(p) if p else None

    def _result_selected(self, *_):
        p = self._selected_path()
        self.play_res.setEnabled(bool(p) and p.suffix.lower() in PLAYABLE)
        self.open_res.setEnabled(bool(p))
        self.show_res.setEnabled(bool(p))

    def _activate(self, item, play=False):
        p = Path(item.data(0, Qt.UserRole)) if item and item.data(0, Qt.UserRole) else None
        if not p:
            return
        if p.suffix.lower() in PLAYABLE:
            self.player.load(p)
        elif not play:
            jobs.open_path(p)

    def _open_selected(self):
        p = self._selected_path()
        if p:
            jobs.open_path(p)

    def _reveal_selected(self):
        p = self._selected_path()
        if p:
            jobs.reveal(p)

    # ---------------------------------------------------------------- songs
    def _load_songs(self):
        found = [str(p) for p in sorted(paths.INPUT_DIR.glob("*")) if p.suffix.lower() in jobs.AUDIO_EXTS]
        for p in self.settings.value("songs", []) or []:
            if p not in found and Path(p).exists():
                found.append(p)
        for p in found:
            self._add_item(Path(p))
        last = self.settings.value("last_song")
        items = [self.songs.item(i) for i in range(self.songs.count())]
        pick = next((it for it in items if it.data(Qt.UserRole) == last), items[0] if items else None)
        if pick:
            self.songs.setCurrentItem(pick)
        else:
            self._show_plan()

    def _add_item(self, path: Path) -> QListWidgetItem:
        for i in range(self.songs.count()):
            if self.songs.item(i).data(Qt.UserRole) == str(path):
                return self.songs.item(i)
        it = QListWidgetItem(path.stem.replace("_", " "))
        it.setData(Qt.UserRole, str(path))
        it.setToolTip(str(path))
        self.songs.addItem(it)
        return it

    def add_song(self, path: Path):
        it = self._add_item(path)
        saved = list(self.settings.value("songs", []) or [])
        if str(path) not in saved and path.parent != paths.INPUT_DIR:
            self.settings.setValue("songs", saved + [str(path)])
        self.songs.setCurrentItem(it)

    def add_dialog(self):
        exts = " ".join("*" + e for e in sorted(jobs.AUDIO_EXTS))
        start = self.settings.value("last_dir", str(paths.INPUT_DIR))
        path, _ = QFileDialog.getOpenFileName(self, "Elige una canción", start, f"Audio ({exts})")
        if path:
            self.settings.setValue("last_dir", str(Path(path).parent))
            self.add_song(Path(path))

    def _song_selected(self, item, _prev=None):
        if item is None:
            return
        self.song = SongFiles(Path(item.data(Qt.UserRole)))
        self.settings.setValue("last_song", str(self.song.audio))
        self.title.setText(self.song.name.replace("_", " "))
        dur = jobs.duration_s(self.song.audio)
        self.subtitle.setText(" · ".join(x for x in (fmt_time(dur), str(self.song.audio)) if x))
        self.log.clear()
        self.progress_line.clear()
        self._options_changed()
        self._refresh_results()

    def dragEnterEvent(self, e):
        if any(Path(u.toLocalFile()).suffix.lower() in jobs.AUDIO_EXTS for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = Path(u.toLocalFile())
            if p.suffix.lower() in jobs.AUDIO_EXTS:
                self.add_song(p)

    # ---------------------------------------------------------------- running
    def start(self):
        if not self.song or self.runner:
            return
        if self.device is None:
            self.status.setText("Todavía estoy detectando el equipo; prueba de nuevo en unos segundos.")
            return
        o = self.options()
        self._show_plan()
        todo = [s for s in self.steps if not s.skip(o.redo)]
        missing = jobs.missing_environments(todo)
        if missing:
            QMessageBox.critical(self, "Falta un entorno", "No encuentro estos entornos de Python en el proyecto:\n"
                                 + "\n".join(missing))
            return
        downloads = jobs.missing_downloads(todo)
        if downloads:
            mb = sum(m for _, m in downloads)
            size = f"{mb / 1000:.1f} GB".replace(".", ",") if mb >= 1000 else f"{mb} MB"
            lines = "\n".join(f"  • {n}: ~{m} MB" for n, m in downloads)
            if QMessageBox.question(
                    self, "Descarga de modelos",
                    f"Faltan modelos que se descargan una sola vez (~{size} en total, necesita internet):\n\n{lines}"
                    f"\n\nSe guardan en:\n{paths.MODELS_DIR}\n\n(Sin internet, copia ahí la carpeta de modelos "
                    f"de otra instalación.)\n\n¿Continuar?") != QMessageBox.Yes:
                return
        total = sum(s.seconds for s in todo)
        slow =[s for s in todo if s.seconds >= 10 * 60]
        if slow:
            where = ("ocupa la GPU todo ese tiempo" if any(s.kind != "cpu" for s in slow)
                     else "corre en la CPU porque no hay GPU disponible, y el equipo irá lento mientras tanto")
            lines = "\n".join(f"  • {s.title}: {s.estimate}" for s in slow)
            if QMessageBox.question(self, "Paso largo", f"Esto tarda bastante y {where}:\n\n{lines}\n\n"
                                    f"En total, {jobs.fmt_estimate(total)}. ¿Lanzarlo ahora?") != QMessageBox.Yes:
                return
        if any(s.kind == "cuda" for s in todo):
            users = jobs.gpu_users()
            mem = f"Con {self.device.vram_gb:g} GB de memoria" if self.device.vram_gb else "Con la memoria de la GPU"
            if users and QMessageBox.question(
                    self, "La GPU está en uso",
                    "Hay otro proceso de Python usando la GPU (¿otra sesión de trabajo?):\n\n" + "\n".join(users)
                    + f"\n\n{mem} pueden chocar. ¿Continuar igual?") != QMessageBox.Yes:
                return
        self._save_options()
        self.log.clear()
        self.runner = Runner(self.steps, o.redo)
        self.runner.step_state.connect(self._on_step)
        self.runner.line.connect(self.log.appendPlainText)
        self.runner.progress.connect(lambda t: self.progress_line.setText(t[:160]))
        self.runner.finished.connect(self._on_finished)
        self._set_running(True)
        self.runner.start()

    def stop(self):
        if self.runner:
            self.status.setText("Deteniendo…")
            self.runner.stop()

    def _set_running(self, on: bool):
        self.run_btn.setEnabled(not on)
        self.stop_btn.setEnabled(on)
        self.songs.setEnabled(not on)
        for w in (self.opt_drums, self.opt_drums_only, self.opt_bpm, self.opt_chart, self.opt_lyrics,
                  self.opt_lang, self.opt_hq, self.opt_gpu, self.opt_redo):
            w.setEnabled(not on)
        if on:
            self.clock.start()
        else:
            self.clock.stop()
            self._update_enabled()  # keeps the final step states on screen (no plan redraw)
            self.run_btn.setEnabled(bool(self.steps))

    def _tick(self):
        if self.runner and 0 <= self.runner.i < len(self.steps):
            step = self.steps[self.runner.i]
            self.status.setText(f"{step.title}: {fmt_time(self.runner.elapsed())} (estimado {step.estimate})")

    def _on_step(self, i, state, detail):
        it = self.step_view.topLevelItem(i)
        if it:
            self._paint(it, state, detail)
        if state in ("ok", "reuse"):
            self._refresh_results()

    def _on_finished(self, ok):
        runner, self.runner = self.runner, None
        last = self.steps[runner.i] if 0 <= runner.i < len(self.steps) else None
        self._set_running(False)
        # keep the final states on screen instead of redrawing the preview
        for j in range(runner.i + 1, len(self.steps)):
            it = self.step_view.topLevelItem(j)
            if it and not ok:
                self._paint(it, "skipped")
        self._refresh_results()
        if ok:
            self.status.setText("Terminado. Los archivos están en la pestaña Resultados.")
            self.progress_line.clear()
            self.tabs.setCurrentIndex(1)
        elif runner.stopping:
            self.status.setText("Detenido. Lo que ya terminó se conserva.")
        else:
            self.status.setText(f"Falló «{last.title if last else ''}». Abre «Ver detalles» para ver el error.")
            self.log_btn.setChecked(True)

    def closeEvent(self, e):
        if self.runner:
            if QMessageBox.question(self, "Hay un proceso en curso",
                                    "Si cierras, se detiene el paso actual. ¿Cerrar igual?") != QMessageBox.Yes:
                e.ignore()
                return
            self.runner.finished.disconnect()
            self.runner.stop()
        self._save_options()
        e.accept()


# ------------------------------------------------------------------ look and feel: black base, blue accent
BG, SURFACE, SURFACE2 = "#07080b", "#0f1117", "#161922"
BORDER, BORDER2 = "#222634", "#2e3446"
TEXT, MUTED, DIM = "#e8ebf2", "#8a93a8", "#4f586c"
ACCENT, ACCENT_HOVER, ACCENT_PRESSED = "#2f7df6", "#4c90ff", "#2366d1"
ACCENT_SOFT = "#14223b"   # the accent at 16% over the panels, opaque so overlapping layers do not stack

STYLE = """
QWidget { color: TEXT; font-size: 13px; }
QMainWindow, QWidget#central { background: BG; }
QWidget#sidebar { background: SURFACE; }
QSplitter::handle { background: BORDER; }

QLabel#brand { font-size: 19px; font-weight: 800; }
QLabel#title { font-size: 26px; font-weight: 700; }
QLabel#section { color: MUTED; font-size: 11px; font-weight: 700; }
QLabel#hint { color: MUTED; }
QLabel#mono, QPlainTextEdit#mono { font-family: Menlo, Consolas, monospace; font-size: 11px; color: MUTED; }

QListWidget#songs { background: transparent; border: none; outline: 0; }
QListWidget#songs::item { padding: 9px 10px; border-radius: 8px; margin: 1px 0; color: TEXT; }
QListWidget#songs::item:hover { background: SURFACE2; }
QListWidget#songs::item:selected { background: ACCENT_SOFT; color: #ffffff; border-left: 3px solid ACCENT; }

QTabWidget::pane { border: none; }
QTabBar::tab { background: transparent; color: MUTED; padding: 8px 18px; margin-right: 6px;
               border: none; border-bottom: 2px solid transparent; font-weight: 600; }
QTabBar::tab:selected { color: TEXT; border-bottom: 2px solid ACCENT; }
QTabBar::tab:hover:!selected { color: TEXT; }

QGroupBox { background: SURFACE; border: 1px solid BORDER; border-radius: 12px;
            margin-top: 20px; padding: 12px 14px 8px 14px; font-weight: 700; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 4px; top: 2px;
                   color: MUTED; padding: 0 2px; }
QGroupBox QCheckBox, QGroupBox QLabel { font-weight: normal; }

QCheckBox { spacing: 9px; padding: 1px 0; }
QCheckBox:disabled { color: DIM; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 5px; border: 1px solid BORDER2; background: SURFACE2; }
QCheckBox::indicator:hover { border-color: ACCENT; }
QCheckBox::indicator:checked { background: ACCENT; border-color: ACCENT; image: url(ASSETS/check.svg); }
QCheckBox::indicator:disabled { background: #10131a; border-color: BORDER; }
QCheckBox::indicator:checked:disabled { background: #1d3157; border-color: #1d3157; }

QPushButton { background: SURFACE2; border: 1px solid BORDER2; border-radius: 9px; padding: 8px 16px; }
QPushButton:hover { border-color: ACCENT; }
QPushButton:pressed { background: #1c2030; }
QPushButton:disabled { color: DIM; border-color: BORDER; background: #10131a; }
QPushButton#primary { background: ACCENT; border: none; color: #ffffff; font-weight: 700; padding: 9px 28px; }
QPushButton#primary:hover { background: ACCENT_HOVER; }
QPushButton#primary:pressed { background: ACCENT_PRESSED; }
QPushButton#primary:disabled { background: #172644; color: #5f6f8f; }

QComboBox, QDoubleSpinBox { background: SURFACE2; border: 1px solid BORDER2; border-radius: 8px;
                            padding: 4px 10px; min-height: 20px; }
QComboBox:hover, QDoubleSpinBox:hover { border-color: ACCENT; }
QComboBox:disabled, QDoubleSpinBox:disabled { color: DIM; border-color: BORDER; background: #10131a; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow { image: url(ASSETS/chevron-down.svg); width: 10px; height: 10px; }
QComboBox QAbstractItemView { background: SURFACE2; border: 1px solid BORDER2; outline: 0; padding: 4px;
                              selection-background-color: ACCENT; selection-color: #ffffff; }
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 18px; border: none; background: transparent; }
QDoubleSpinBox::up-arrow { image: url(ASSETS/chevron-up.svg); width: 9px; height: 9px; }
QDoubleSpinBox::down-arrow { image: url(ASSETS/chevron-down.svg); width: 9px; height: 9px; }

QTreeWidget { background: SURFACE; border: 1px solid BORDER; border-radius: 12px; padding: 6px; outline: 0;
              selection-background-color: ACCENT_SOFT; selection-color: TEXT; show-decoration-selected: 1; }
QTreeWidget::item { padding: 7px 4px; border: none; }
QTreeWidget::item:hover { background: #151924; }
QTreeWidget::item:selected { background: ACCENT_SOFT; color: TEXT; }
QTreeWidget::branch { background: transparent; }
QTreeWidget::branch:selected { background: ACCENT_SOFT; }
QHeaderView { background: transparent; }
QHeaderView::section { background: transparent; color: MUTED; border: none; border-bottom: 1px solid BORDER;
                       padding: 6px 8px; font-size: 12px; font-weight: 600; }

QPlainTextEdit#mono { background: #0a0c11; border: 1px solid BORDER; border-radius: 10px; padding: 8px; }
QToolButton { background: transparent; border: 1px solid BORDER2; border-radius: 8px; padding: 5px 12px; color: MUTED; }
QToolButton:hover { border-color: ACCENT; color: TEXT; }
QToolButton:checked { background: ACCENT_SOFT; border-color: ACCENT; color: TEXT; }

QFrame#player { background: SURFACE; border-top: 1px solid BORDER; }
QFrame#player QToolButton { border-radius: 17px; min-width: 34px; max-width: 34px; min-height: 34px; max-height: 34px; padding: 0; }
QFrame#player QToolButton#play { background: ACCENT; border: none; }
QFrame#player QToolButton#play:hover { background: ACCENT_HOVER; }
QFrame#player QToolButton#play:disabled { background: #172644; }
QSlider::groove:horizontal { height: 4px; background: BORDER2; border-radius: 2px; }
QSlider::sub-page:horizontal { background: ACCENT; border-radius: 2px; }
QSlider::handle:horizontal { background: #ffffff; width: 12px; height: 12px; margin: -4px 0; border-radius: 6px; }
QSlider::handle:horizontal:hover { background: ACCENT_HOVER; }
QSlider::sub-page:horizontal:disabled { background: BORDER2; }
QSlider::handle:horizontal:disabled { background: DIM; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle { background: BORDER2; border-radius: 4px; min-height: 30px; min-width: 30px; }
QScrollBar::handle:hover { background: #3a4256; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

QToolTip { background: SURFACE2; color: TEXT; border: 1px solid BORDER2; border-radius: 6px; padding: 6px; }
QMessageBox { background: SURFACE; }
"""


def stylesheet() -> str:
    s = STYLE.replace("ASSETS", ASSETS.as_posix())
    for name in ("ACCENT_PRESSED", "ACCENT_HOVER", "ACCENT_SOFT", "ACCENT", "SURFACE2", "SURFACE", "BORDER2", "BORDER",
                 "TEXT", "MUTED", "DIM", "BG"):  # longest names first so ACCENT does not eat ACCENT_HOVER
        s = s.replace(name, globals()[name])
    return s


def setup_app(app: QApplication):
    """Dark theme: Fusion + a matching palette (for what the stylesheet does not reach) + the stylesheet."""
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in ((QPalette.Window, BG), (QPalette.WindowText, TEXT), (QPalette.Base, SURFACE2),
                        (QPalette.AlternateBase, SURFACE), (QPalette.Text, TEXT), (QPalette.Button, SURFACE2),
                        (QPalette.ButtonText, TEXT), (QPalette.Highlight, ACCENT), (QPalette.HighlightedText, "#ffffff"),
                        (QPalette.PlaceholderText, MUTED), (QPalette.ToolTipBase, SURFACE2), (QPalette.ToolTipText, TEXT),
                        (QPalette.Link, ACCENT_HOVER), (QPalette.Mid, BORDER)):
        pal.setColor(role, QColor(color))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(DIM))
    app.setPalette(pal)
    app.setStyleSheet(stylesheet())


def main():
    import sys
    app = QApplication(sys.argv)
    app.setApplicationName("audio2midi")
    setup_app(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
