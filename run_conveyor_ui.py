"""
Conveyor Belt Operator UI
=========================
PyQt6 desktop app — camera selector, live feeds, trigger detection, results.

Run:
    python run_conveyor_ui.py [--session ID] [--max N]
"""

import sys
import os
import time
import datetime
import threading
import argparse
import json
import cv2

PREFS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conveyor_ui", "camera_prefs.json")

def _load_prefs() -> dict:
    try:
        with open(PREFS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_prefs(prefs: dict):
    try:
        with open(PREFS_PATH, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=2)
    except Exception:
        pass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QGridLayout, QHBoxLayout, QVBoxLayout, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live import conveyor_config as config
from live.conveyor_main import ConveyorSystem
from conveyor_ui.widgets import CameraWidget, ResultBar
from conveyor_ui.camera_scanner import CameraScanner
from conveyor_ui.multi_detector import MultiCameraDetector

NUM_CAMS = 4


# ── Frame dispatcher — reads buffers at 15fps, emits QImages ─────────────────

class FrameDispatcher(QThread):
    frame_ready = pyqtSignal(int, QImage)
    cam_status  = pyqtSignal(int, bool)

    def __init__(self, buffers, detector=None, parent=None):
        super().__init__(parent)
        self._buffers  = buffers
        self._detector = detector
        self._running  = True
        self._last_ok  = [False] * len(buffers)

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
                    display = frame.copy()
                    if self._detector:
                        boxes = self._detector.get_boxes(i)
                        if boxes:
                            display = self._draw_boxes(display, boxes)
                    rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                    self.frame_ready.emit(i, qimg)
            time.sleep(0.067)   # ~15 fps

    @staticmethod
    def _draw_boxes(frame, boxes):
        h, w = frame.shape[:2]
        color = (0, 220, 80)
        for (x1n, y1n, x2n, y2n) in boxes:
            x1, y1 = int(x1n * w), int(y1n * h)
            x2, y2 = int(x2n * w), int(y2n * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            seg   = max(12, min(28, (x2 - x1) // 5))
            thick = 3
            for px, py, sx, sy in [
                (x1, y1,  seg,  0), (x1, y1,  0,  seg),
                (x2, y1, -seg,  0), (x2, y1,  0,  seg),
                (x1, y2,  seg,  0), (x1, y2,  0, -seg),
                (x2, y2, -seg,  0), (x2, y2,  0, -seg),
            ]:
                cv2.line(frame, (px, py), (px + sx, py + sy), color, thick)
            label = "PRODUCT"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        return frame

    def stop(self):
        self._running = False
        self.wait()


# ── Thread-safe bridge: worker thread → Qt signals ───────────────────────────

class EventBridge(QObject):
    trigger_fired    = pyqtSignal(int)
    result_ready     = pyqtSignal(dict)
    system_ready     = pyqtSignal()
    loading_progress = pyqtSignal(str)

    def on_trigger(self, cam_idx):
        self.trigger_fired.emit(cam_idx)

    def on_result(self, result):
        self.result_ready.emit(result)

    def on_system_ready(self):
        self.system_ready.emit()

    def on_progress(self, msg):
        self.loading_progress.emit(msg)


# ── Main window ───────────────────────────────────────────────────────────────

class ConveyorUIApp(QMainWindow):
    def __init__(self, session_id, max_products):
        super().__init__()
        self._session_id   = session_id
        self._max_products = max_products
        self._system       = None
        self._dispatcher   = None
        self._detector     = None
        self._bridge       = EventBridge()
        self._session_running = False
        self._prefs        = _load_prefs()
        self._scan_cooldown = False

        self.setWindowTitle("AI Product Inspector")
        self.setMinimumSize(1120, 760)
        self.resize(1280, 860)
        self._apply_style()
        self._build_ui()
        self._start_scanner()

    # ── Style ─────────────────────────────────────────────────────────────────

    def _apply_style(self):
        qss_path = os.path.join(os.path.dirname(__file__), "conveyor_ui", "style.qss")
        if os.path.exists(qss_path):
            with open(qss_path, encoding="utf-8") as f:
                self.setStyleSheet(f.read())
        else:
            self.setStyleSheet("QMainWindow, QWidget { background: #0d0d0d; }")

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        main = QVBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        main.addWidget(self._build_header())

        grid_w = QWidget()
        grid_w.setStyleSheet("background: #090909;")
        grid = QGridLayout(grid_w)
        grid.setContentsMargins(10, 10, 10, 8)
        grid.setSpacing(8)

        self._cam_widgets: list[CameraWidget] = []
        for i in range(NUM_CAMS):
            w = CameraWidget(i)
            w.source_changed.connect(self._on_source_changed)
            grid.addWidget(w, i // 2, i % 2)
            self._cam_widgets.append(w)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        main.addWidget(grid_w, stretch=1)

        self._result_bar = ResultBar()
        main.addWidget(self._result_bar)

    def _build_header(self):
        header = QWidget()
        header.setObjectName("app_header")
        header.setFixedHeight(56)

        badge = QLabel("AI")
        badge.setFixedSize(28, 22)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            "color: #22c55e; background: #061a0e; border: 1px solid #0d3d1a;"
            "border-radius: 4px; font-size: 10px; font-weight: 800; letter-spacing: 1px;"
        )

        title = QLabel("PRODUCT INSPECTOR")
        title.setStyleSheet(
            "color: #484848; font-size: 11px; font-weight: 700; letter-spacing: 2.5px;"
        )

        div = QLabel("|")
        div.setStyleSheet("color: #1c1c1c; font-size: 16px; padding: 0 4px;")

        self._session_lbl = QLabel(self._session_id)
        self._session_lbl.setStyleSheet(
            "color: #282828; font-size: 9px; letter-spacing: 0.5px;"
            "font-family: 'Consolas', 'Courier New', monospace;"
        )

        self._scan_status = QLabel("Scanning for cameras...")
        self._scan_status.setStyleSheet(
            "color: #353535; font-size: 11px; font-style: italic;"
        )

        _btn_h = 30

        self._btn_start = QPushButton("START SESSION")
        self._btn_start.setFixedSize(128, _btn_h)
        self._btn_start.setEnabled(False)
        self._btn_start.setStyleSheet(
            "QPushButton {"
            "  background: #061a0e; color: #22c55e; border: 1px solid #0d3d1a;"
            "  border-radius: 5px; font-size: 10px; font-weight: 700; letter-spacing: 1.2px;"
            "}"
            "QPushButton:enabled:hover {"
            "  background: #0a2d15; color: #4ade80; border-color: #166534;"
            "}"
            "QPushButton:disabled {"
            "  background: #0e0e0e; color: #202020; border-color: #181818;"
            "}"
            "QPushButton:pressed { background: #041208; }"
        )
        self._btn_start.clicked.connect(self._start_session)

        self._btn_stop = QPushButton("STOP")
        self._btn_stop.setFixedSize(68, _btn_h)
        self._btn_stop.hide()
        self._btn_stop.setStyleSheet(
            "QPushButton {"
            "  background: #111; color: #484848; border: 1px solid #1e1e1e;"
            "  border-radius: 5px; font-size: 10px; font-weight: 600; letter-spacing: 0.5px;"
            "}"
            "QPushButton:hover {"
            "  background: #1e0808; color: #f87171; border-color: #4d1212;"
            "}"
            "QPushButton:pressed { background: #130606; }"
        )
        self._btn_stop.clicked.connect(self.close)

        self._btn_scan = QPushButton("⊕  SCAN")
        self._btn_scan.setFixedSize(84, _btn_h)
        self._btn_scan.hide()
        self._btn_scan.setStyleSheet(
            "QPushButton {"
            "  background: #060e1e; color: #38bdf8; border: 1px solid #0d2a50;"
            "  border-radius: 5px; font-size: 10px; font-weight: 700; letter-spacing: 0.5px;"
            "}"
            "QPushButton:hover {"
            "  background: #0a1a30; color: #7dd3fc; border-color: #155ea0;"
            "}"
            "QPushButton:disabled {"
            "  background: #0e0e0e; color: #181818; border-color: #181818;"
            "}"
            "QPushButton:pressed { background: #050c16; }"
        )
        self._btn_scan.clicked.connect(self._do_manual_scan)

        row = QHBoxLayout(header)
        row.setContentsMargins(16, 0, 14, 0)
        row.setSpacing(10)
        row.addWidget(badge)
        row.addWidget(title)
        row.addWidget(div)
        row.addWidget(self._session_lbl)
        row.addSpacing(8)
        row.addWidget(self._scan_status)
        row.addStretch()
        row.addWidget(self._btn_scan)
        row.addSpacing(4)
        row.addWidget(self._btn_start)
        row.addWidget(self._btn_stop)

        return header

    # ── Camera scanner ────────────────────────────────────────────────────────

    def _start_scanner(self):
        for w in self._cam_widgets:
            w.populate_start()

        extra_urls = [u for u in config.CAMERA_INDICES if isinstance(u, str)]
        self._scanner = CameraScanner(extra_urls=extra_urls)
        self._scanner.camera_found.connect(self._on_camera_found)
        self._scanner.scan_complete.connect(self._on_scan_complete)
        self._scanner.start()

    def _on_camera_found(self, label: str, source):
        for w in self._cam_widgets:
            w.add_camera_option(label, source)
        self._btn_start.setEnabled(True)

    def _on_source_changed(self, cam_idx: int, source):
        key = f"cam_{cam_idx}"
        if source is None:
            self._prefs.pop(key, None)
        else:
            self._prefs[key] = source
        _save_prefs(self._prefs)

    def _on_scan_complete(self):
        count = self._cam_widgets[0]._combo.count() - 1
        if count == 0:
            self._scan_status.setText("No cameras found")
            self._scan_status.setStyleSheet("color: #f87171; font-size: 11px; font-style: normal;")
        else:
            self._scan_status.setText(
                f"{count} camera{'s' if count != 1 else ''} available"
            )
            self._scan_status.setStyleSheet(
                "color: #22c55e; font-size: 11px; font-style: normal;"
            )
        for w in self._cam_widgets:
            w.scan_complete()
        for i, w in enumerate(self._cam_widgets):
            saved = self._prefs.get(f"cam_{i}")
            if saved is not None:
                w.restore_selection(saved)

    # ── Session start ─────────────────────────────────────────────────────────

    def _start_session(self):
        sources = [w._selected_source for w in self._cam_widgets]

        if not any(s is not None for s in sources):
            self._scan_status.setText("Select at least one camera first")
            self._scan_status.setStyleSheet("color: #f59e0b; font-size: 11px;")
            return

        self._session_running = True
        self._btn_start.hide()
        self._btn_stop.show()
        self._scan_status.setText("Starting up...")
        self._scan_status.setStyleSheet("color: #444; font-size: 11px; font-style: italic;")

        for w in self._cam_widgets:
            if w._selected_source is not None:
                w.prepare_for_session()
            else:
                w._combo.setEnabled(False)

        self._bridge.trigger_fired.connect(self._on_trigger)
        self._bridge.result_ready.connect(self._on_result)
        self._bridge.system_ready.connect(self._on_system_ready)
        self._bridge.loading_progress.connect(self._on_loading_progress)

        def run():
            self._system = ConveyorSystem(
                camera_indices=sources,
                on_trigger=self._bridge.on_trigger,
                on_result=self._bridge.on_result,
                on_progress=self._bridge.on_progress,
            )
            self._system.start(self._session_id)
            self._bridge.on_system_ready()
            self._system.run_session(self._max_products)

        threading.Thread(target=run, daemon=True, name="ConveyorSession").start()

    # ── Qt slots ──────────────────────────────────────────────────────────────

    def _on_loading_progress(self, msg: str):
        # Truncate long messages for the header label
        display = msg if len(msg) <= 52 else msg[:50] + "..."
        self._scan_status.setText(display)
        self._scan_status.setStyleSheet(
            "color: #3a3a3a; font-size: 10px; font-style: italic;"
        )
        for w in self._cam_widgets:
            w.set_loading_step(msg)

    def _on_frame(self, cam_idx: int, qimage: QImage):
        if cam_idx < len(self._cam_widgets):
            self._cam_widgets[cam_idx].set_frame(qimage)

    def _on_cam_status(self, cam_idx: int, connected: bool):
        if cam_idx < len(self._cam_widgets) and not connected:
            self._cam_widgets[cam_idx].set_disconnected()

    def _on_system_ready(self):
        self._scan_status.setText("● SESSION ACTIVE")
        self._scan_status.setStyleSheet(
            "color: #22c55e; font-size: 10px; font-weight: 700;"
            "letter-spacing: 0.5px; font-style: normal;"
        )

        self._detector = MultiCameraDetector(
            self._system._buffers,
            trigger=self._system._trigger,
            conf=config.TRIGGER_CONFIDENCE_THRESHOLD,
            min_box_area=config.TRIGGER_MIN_BOX_AREA,
            classes=getattr(config, 'TRIGGER_CLASSES', None),
        )
        self._detector.start()

        self._dispatcher = FrameDispatcher(
            self._system._buffers,
            detector=self._detector,
        )
        self._dispatcher.frame_ready.connect(self._on_frame)
        self._dispatcher.cam_status.connect(self._on_cam_status)
        self._dispatcher.start()

        for w in self._cam_widgets:
            w.on_session_start()

        self._btn_scan.show()

    def _do_manual_scan(self):
        if not self._system or self._scan_cooldown:
            return
        # Respond instantly — snap happens in background thread
        self._scan_cooldown = True
        self._btn_scan.setEnabled(False)
        for w in self._cam_widgets:
            w.flash_trigger()
            w.show_scanning()
        QTimer.singleShot(1200, self._scan_ready)
        threading.Thread(target=self._system.manual_snap, daemon=True).start()

    def _scan_ready(self):
        self._scan_cooldown = False
        self._btn_scan.setEnabled(True)

    def _on_trigger(self, cam_idx: int):
        if cam_idx < len(self._cam_widgets):
            self._cam_widgets[cam_idx].flash_trigger()
        for w in self._cam_widgets:
            w.show_scanning()

    def _on_result(self, result: dict):
        for w in self._cam_widgets:
            w.hide_scanning()
        self._result_bar.update_result(result)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._detector:
            self._detector.stop()
        if self._dispatcher:
            self._dispatcher.stop()
        if self._system:
            threading.Thread(target=self._system.stop, daemon=True).start()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default=None)
    parser.add_argument("--max", type=int, default=config.MAX_PRODUCTS, dest="max_products")
    args = parser.parse_args()

    session_id = args.session or f"truck_{datetime.date.today()}_{os.urandom(3).hex()}"

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,           QColor("#0d0d0d"))
    palette.setColor(QPalette.ColorRole.WindowText,       QColor("#e0e0e0"))
    palette.setColor(QPalette.ColorRole.Base,             QColor("#111111"))
    palette.setColor(QPalette.ColorRole.Text,             QColor("#e0e0e0"))
    palette.setColor(QPalette.ColorRole.Button,           QColor("#1a1a1a"))
    palette.setColor(QPalette.ColorRole.ButtonText,       QColor("#e0e0e0"))
    palette.setColor(QPalette.ColorRole.Highlight,        QColor("#22c55e"))
    palette.setColor(QPalette.ColorRole.HighlightedText,  QColor("#000000"))
    app.setPalette(palette)

    window = ConveyorUIApp(session_id, args.max_products)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
