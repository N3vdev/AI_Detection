"""
Conveyor Belt Operator UI
=========================
PyQt6 desktop app — live camera feeds, trigger detection, inspection results.

Run:
    python run_conveyor_ui.py [--session SESSION_ID] [--max N]

Camera setup (phones via USB + IP Webcam):
    Run setup_cameras.bat first to ADB-forward the ports, then set
    CAMERA_INDICES in live/conveyor_config.py.
"""

import sys
import os
import time
import datetime
import threading
import argparse
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QGridLayout, QHBoxLayout, QVBoxLayout, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QImage, QPixmap, QFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live import conveyor_config as config
from live.conveyor_main import ConveyorSystem
from conveyor_ui.widgets import CameraWidget, ResultBar

NUM_CAMS = 4


# ── Frame dispatcher — reads buffers at 10fps, emits QImages ─────────────────

class FrameDispatcher(QThread):
    frame_ready  = pyqtSignal(int, QImage)   # cam_idx, image
    cam_status   = pyqtSignal(int, bool)     # cam_idx, connected

    def __init__(self, buffers, parent=None):
        super().__init__(parent)
        self._buffers  = buffers
        self._running  = True
        self._last_ok  = [False] * NUM_CAMS

    def run(self):
        while self._running:
            now = time.monotonic()
            for i, buf in enumerate(self._buffers):
                frame = buf.get_closest(now, window_ms=500)
                connected = frame is not None
                if connected != self._last_ok[i]:
                    self._last_ok[i] = connected
                    self.cam_status.emit(i, connected)
                if frame is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                    self.frame_ready.emit(i, qimg)
            time.sleep(0.10)  # 10 fps

    def stop(self):
        self._running = False
        self.wait()


# ── Thread-safe event bridge (worker thread → Qt signals) ────────────────────

class EventBridge(QObject):
    trigger_fired = pyqtSignal(int)   # cam_idx
    result_ready  = pyqtSignal(dict)

    def on_trigger(self, cam_idx):
        self.trigger_fired.emit(cam_idx)

    def on_result(self, result):
        self.result_ready.emit(result)


# ── Main window ───────────────────────────────────────────────────────────────

class ConveyorUIApp(QMainWindow):
    def __init__(self, session_id, max_products):
        super().__init__()
        self._session_id   = session_id
        self._max_products = max_products
        self._system       = None
        self._dispatcher   = None
        self._bridge       = EventBridge()

        self.setWindowTitle("AI Product Inspector — Conveyor")
        self.setMinimumSize(1100, 720)
        self._load_style()
        self._build_ui()
        self._start_system()

    # ── Style ─────────────────────────────────────────────────────────────────

    def _load_style(self):
        qss_path = os.path.join(os.path.dirname(__file__), "conveyor_ui", "style.qss")
        if os.path.exists(qss_path):
            with open(qss_path, encoding="utf-8") as f:
                self.setStyleSheet(f.read())
        self.setStyleSheet(self.styleSheet() + "QMainWindow { background: #0d0d0d; }")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        root.setStyleSheet("background: #0d0d0d;")
        self.setCentralWidget(root)

        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header
        main_layout.addWidget(self._build_header())

        # Camera grid
        grid_widget = QWidget()
        grid_widget.setStyleSheet("background: #0d0d0d;")
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setSpacing(10)

        self._cam_widgets = []
        for i in range(NUM_CAMS):
            w = CameraWidget(i if i < len(config.CAMERA_INDICES) else i)
            grid.addWidget(w, i // 2, i % 2)
            self._cam_widgets.append(w)
            if i >= len(config.CAMERA_INDICES):
                w.set_disconnected()

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        main_layout.addWidget(grid_widget, stretch=1)

        # Result bar
        self._result_bar = ResultBar()
        main_layout.addWidget(self._result_bar)

    def _build_header(self):
        header = QWidget()
        header.setObjectName("header")
        header.setFixedHeight(48)
        header.setStyleSheet("background: #111111; border-bottom: 1px solid #1e1e1e;")

        title = QLabel("AI PRODUCT INSPECTOR")
        title.setObjectName("app_title")
        title.setStyleSheet("color: #e0e0e0; font-size: 14px; font-weight: 600; letter-spacing: 1px;")

        self._session_lbl = QLabel(f"session: {self._session_id}")
        self._session_lbl.setObjectName("session_label")
        self._session_lbl.setStyleSheet("color: #444; font-size: 11px;")

        self._btn_stop = QPushButton("STOP")
        self._btn_stop.setObjectName("btn_stop")
        self._btn_stop.setFixedSize(72, 30)
        self._btn_stop.setStyleSheet(
            "QPushButton { background: #1a1a1a; color: #888; border: 1px solid #2a2a2a; "
            "border-radius: 6px; font-size: 12px; font-weight: 500; }"
            "QPushButton:hover { background: #2a0a0a; color: #f87171; border-color: #5c1a1a; }"
        )
        self._btn_stop.clicked.connect(self.close)

        row = QHBoxLayout(header)
        row.setContentsMargins(16, 0, 16, 0)
        row.addWidget(title)
        row.addSpacing(20)
        row.addWidget(self._session_lbl)
        row.addStretch()
        row.addWidget(self._btn_stop)

        return header

    # ── System startup ────────────────────────────────────────────────────────

    def _start_system(self):
        self._bridge.trigger_fired.connect(self._on_trigger)
        self._bridge.result_ready.connect(self._on_result)

        def run():
            self._system = ConveyorSystem(
                on_trigger=self._bridge.on_trigger,
                on_result=self._bridge.on_result,
            )
            self._system.start(self._session_id)

            # Start frame dispatcher once buffers exist
            self._dispatcher = FrameDispatcher(self._system._buffers)
            self._dispatcher.frame_ready.connect(self._on_frame)
            self._dispatcher.cam_status.connect(self._on_cam_status)
            self._dispatcher.start()

            self._system.run_session(self._max_products)

        t = threading.Thread(target=run, daemon=True, name="ConveyorSession")
        t.start()

    # ── Qt slots ──────────────────────────────────────────────────────────────

    def _on_frame(self, cam_idx: int, qimage: QImage):
        if cam_idx < len(self._cam_widgets):
            self._cam_widgets[cam_idx].set_frame(qimage)

    def _on_cam_status(self, cam_idx: int, connected: bool):
        if cam_idx < len(self._cam_widgets):
            if not connected:
                self._cam_widgets[cam_idx].set_disconnected()

    def _on_trigger(self, cam_idx: int):
        # Flash trigger camera, show scanning on all active cameras
        if cam_idx < len(self._cam_widgets):
            self._cam_widgets[cam_idx].flash_trigger()
        for w in self._cam_widgets:
            w.show_scanning()

    def _on_result(self, result: dict):
        for w in self._cam_widgets:
            w.hide_scanning()
        self._result_bar.update_result(result)

    # ── Clean shutdown ────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._dispatcher:
            self._dispatcher.stop()
        if self._system:
            threading.Thread(target=self._system.stop, daemon=True).start()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Conveyor Belt Operator UI")
    parser.add_argument("--session", default=None, help="Session ID (auto-generated if omitted)")
    parser.add_argument("--max", type=int, default=config.MAX_PRODUCTS, dest="max_products")
    args = parser.parse_args()

    session_id = args.session or f"truck_{datetime.date.today()}_{os.urandom(3).hex()}"

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Global dark palette base
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor("#0d0d0d"))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor("#e0e0e0"))
    palette.setColor(QPalette.ColorRole.Base,            QColor("#111111"))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor("#1a1a1a"))
    palette.setColor(QPalette.ColorRole.Text,            QColor("#e0e0e0"))
    palette.setColor(QPalette.ColorRole.Button,          QColor("#1a1a1a"))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor("#e0e0e0"))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor("#22c55e"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#000000"))
    app.setPalette(palette)

    window = ConveyorUIApp(session_id, args.max_products)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
