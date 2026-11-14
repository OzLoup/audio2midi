"""Desktop window: pick a song, choose what to get, run it, then listen to and open the results."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QSettings, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QKeySequence, QPalette
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QPlainTextEdit, QPushButton, QSlider, QSplitter, QStyle, QTabWidget,
                               QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from . import jobs
from .. import device, paths
from .jobs import Options, SongFiles

STATE_TEXT = {"pending": "Pendiente", "reuse": "Ya hecho", "run": "En curso…", "ok": "Listo",
              "error": "Error", "stopped": "Detenido", "skipped": "Omitido"}
STATE_COLOR = {"ok": "#2e9d5b", "reuse": "#2e9d5b", "error": "#d04545", "run": "#3b7dd8", "stopped": "#c28a1e"}
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
        st = self.style()
        self.play_btn = QToolButton(icon=st.standardIcon(QStyle.SP_MediaPlay), toolTip="Reproducir / pausa (espacio)")
        self.stop_btn = QToolButton(icon=st.standardIcon(QStyle.SP_MediaStop), toolTip="Detener")
        self.name = QLabel("Elige una pista en Resultados para escucharla")
        self.name.setMinimumWidth(200)
        self.slider = QSlider(Qt.Horizontal)
        self.time = QLabel("0:00 / 0:00")
        self.volume = QSlider(Qt.Horizontal, maximum=100, value=80, toolTip="Volumen")
        self.volume.setFixedWidth(90)
        self.audio.setVolume(0.8)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 12, 6)
        for w in (self.play_btn, self.stop_btn, self.name):
            lay.addWidget(w)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.time)
        lay.addWidget(QLabel("Vol"))
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
        icon = QStyle.SP_MediaPause if s == QMediaPlayer.PlayingState else QStyle.SP_MediaPlay
        self.play_btn.setIcon(self.style().standardIcon(icon))


# ------------------------------------------------------------------ main window
class MainWindow(QMainWindow):
    device_found = Signal(object)   # emitted from the probing thread with a device.Device

    def __init__(self):
        super().__init__()
        self.setWindowTitle("audio2midi")
        self.resize(1180, 760)
        self.setAcceptDrops(True)
        self.settings = QSettings("audio2midi", "audio2midi")
        self.song: SongFiles | None = None
        self.runner: Runner | None = None
        self.steps: list[jobs.Step] = []
        self.device: device.Device | None = None   # None until the probe answers

        # left: songs
        self.songs = QListWidget()
        self.songs.setSelectionMode(QAbstractItemView.SingleSelection)
        self.songs.currentItemChanged.connect(self._song_selected)
        add = QPushButton("Agregar canción…")
        add.clicked.connect(self.add_dialog)
        hint = QLabel("o arrastra un archivo de audio a la ventana")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(12, 12, 6, 12)
        cap = QLabel("Canciones")
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
        rl.setContentsMargins(6, 12, 12, 12)
        rl.addWidget(self.title)
        rl.addWidget(self.subtitle)
        rl.addWidget(self.tabs, 1)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 920])
        self.player = Player()
        central = QWidget()
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
        self.opt_bpm.valueChanged.connect(self._options_changed)
        return w

    def _load_options(self):
        s = self.settings
        b = lambda k, d: s.value(k, d) in (True, "true")  # noqa: E731
        self.opt_drums.setChecked(b("opt/drums", True))
        self.opt_chart.setChecked(b("opt/chart", True))
        self.opt_lyrics.setChecked(b("opt/lyrics", True))
        self.opt_hq.setChecked(b("opt/hq", False))
        self.opt_gpu.setChecked(b("opt/gpu", True))
        self.opt_lang.setCurrentText(s.value("opt/lang", "Detectar"))

    def _save_options(self):
        s = self.settings
        s.setValue("opt/drums", self.opt_drums.isChecked())
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
        self.opt_lyrics.setEnabled(self.opt_chart.isChecked())
        self.opt_lang.setEnabled(self.opt_chart.isChecked() and self.opt_lyrics.isChecked())
        has_gpu = self.device is not None and self.device.gpu
        self.opt_gpu.setEnabled(has_gpu)
        if self.device is not None and not has_gpu:
            self.opt_gpu.setToolTip("No se encontró una GPU compatible en este equipo: todo corre en la CPU.")

    def options(self) -> Options:
        has_gpu = self.device is not None and self.device.gpu
        return Options(drums=self.opt_drums.isChecked(), drums_only_input=self.opt_drums_only.isChecked(),
                       bpm=self.opt_bpm.value(), chart=self.opt_chart.isChecked(),
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


STYLE = """
QLabel#title { font-size: 20px; font-weight: 600; }
QLabel#section { font-weight: 600; }
QLabel#hint { color: palette(placeholder-text); }
QLabel#mono, QPlainTextEdit#mono { font-family: Consolas, Menlo, monospace; font-size: 11px; }
QGroupBox { font-weight: 600; margin-top: 10px; padding-top: 8px; }
QGroupBox QCheckBox, QGroupBox QLabel { font-weight: normal; }
QPushButton { padding: 6px 14px; }
QPushButton#primary { font-weight: 600; padding: 6px 22px; }
QFrame#player { border-top: 1px solid palette(mid); }
"""


def main():
    import sys
    app = QApplication(sys.argv)
    app.setApplicationName("audio2midi")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    f = app.font()
    f.setPointSize(max(f.pointSize(), 10))
    app.setFont(f)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
