import time
import cv2
from PyQt6.QtWidgets import (
    QFrame, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
    QSizePolicy, QComboBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap


CAM_LABELS = ["FRONT", "BACK", "LEFT", "RIGHT"]


class CameraWidget(QFrame):
    source_changed = pyqtSignal(int, object)   # cam_idx, source (or None)

    def __init__(self, cam_idx: int, parent=None):
        super().__init__(parent)
        self.cam_idx = cam_idx
        self._selected_source = None   # int index or URL string; None = unassigned
        self._preview_cap = None       # cv2.VideoCapture used for pre-session preview
        self._session_active = False

        self.setObjectName("cam_widget")
        self.setMinimumSize(300, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # ── Header row ────────────────────────────────────────────────────────
        self._dot = QLabel("●")
        self._dot.setFixedWidth(16)
        self._dot.setStyleSheet("color: #333; font-size: 16px;")

        self._header = QLabel(f"CAM {cam_idx + 1}  —  {CAM_LABELS[cam_idx]}")
        self._header.setStyleSheet(
            "color: #666; font-size: 10px; font-weight: 600; letter-spacing: 1px;"
            "padding: 8px 0 4px 0;"
        )

        header_row = QHBoxLayout()
        header_row.setContentsMargins(10, 0, 10, 0)
        header_row.setSpacing(6)
        header_row.addWidget(self._dot)
        header_row.addWidget(self._header)
        header_row.addStretch()

        # ── Camera selector dropdown ───────────────────────────────────────────
        self._combo = QComboBox()
        self._combo.addItem("Scanning for cameras...", None)
        self._combo.setEnabled(False)
        self._combo.setStyleSheet(
            "QComboBox { background: #1a1a1a; color: #888; border: 1px solid #2a2a2a; "
            "border-radius: 5px; padding: 4px 8px; font-size: 11px; }"
            "QComboBox:enabled { color: #ccc; border-color: #333; }"
            "QComboBox::drop-down { border: none; width: 20px; }"
            "QComboBox QAbstractItemView { background: #1a1a1a; color: #ccc; "
            "selection-background-color: #22c55e; selection-color: #000; border: 1px solid #333; }"
        )
        self._combo.currentIndexChanged.connect(self._on_combo_changed)

        # ── Feed / placeholder ────────────────────────────────────────────────
        self._feed = QLabel()
        self._feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._feed.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._feed.setMinimumHeight(150)
        self._feed.setStyleSheet("background: #0a0a0a; border-radius: 6px;")

        self._placeholder = QLabel("NO CAMERA SELECTED")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._placeholder.setStyleSheet("color: #2a2a2a; font-size: 12px; font-weight: 500; letter-spacing: 2px;")

        # ── Scanning overlay (during AI pipeline) ─────────────────────────────
        self._scanning = QLabel("SCANNING...")
        self._scanning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scanning.setStyleSheet(
            "background: rgba(0,0,0,160); color: #22c55e; font-size: 14px; "
            "font-weight: 700; letter-spacing: 2px; border-radius: 6px;"
        )
        self._scanning.hide()

        # ── Layout ────────────────────────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(header_row)
        layout.addWidget(self._combo)
        layout.addWidget(self._placeholder, stretch=1)
        layout.addWidget(self._feed, stretch=1)
        layout.addWidget(self._scanning)

        self._feed.hide()

        # Preview timer — grabs a frame every 150ms before session starts
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._grab_preview_frame)

        # Trigger flash timer
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._clear_flash)

    # ── Dropdown population (called by scanner) ───────────────────────────────

    def populate_start(self):
        """Called once scanning begins — clears 'scanning...' and readies the box."""
        self._combo.clear()
        self._combo.addItem("-- Select camera --", None)
        self._combo.setEnabled(True)

    def add_camera_option(self, label: str, source):
        self._combo.addItem(label, source)

    def scan_complete(self):
        if self._combo.count() == 1:  # only the placeholder
            self._combo.setItemText(0, "No cameras found")

    # ── Dropdown change ───────────────────────────────────────────────────────

    def restore_selection(self, source):
        """Called on startup to reselect a previously saved camera source."""
        for i in range(self._combo.count()):
            if self._combo.itemData(i) == source:
                self._combo.setCurrentIndex(i)
                return
        # Source not found in list (camera not connected) — add a ghost entry
        label = f"USB Camera {source}  (index {source})" if isinstance(source, int) \
            else f"IP Camera  {source}  (offline?)"
        self._combo.addItem(f"{label}  ← last used", source)
        self._combo.setCurrentIndex(self._combo.count() - 1)

    def _on_combo_changed(self, index):
        source = self._combo.itemData(index)
        self._selected_source = source
        self._stop_preview()
        self.source_changed.emit(self.cam_idx, source)
        if source is not None:
            self._start_preview(source)
        else:
            self._feed.hide()
            self._placeholder.show()
            self._set_dot(False)

    def _start_preview(self, source):
        self._preview_cap = cv2.VideoCapture(
            source,
            cv2.CAP_DSHOW if isinstance(source, int) else cv2.CAP_ANY,
        )
        if self._preview_cap.isOpened():
            self._set_dot(True)
            self._placeholder.hide()
            self._feed.show()
            self._preview_timer.start(150)
        else:
            self._preview_cap.release()
            self._preview_cap = None
            self._set_dot(False)

    def _stop_preview(self):
        self._preview_timer.stop()
        if self._preview_cap:
            self._preview_cap.release()
            self._preview_cap = None

    def _grab_preview_frame(self):
        if self._session_active or self._preview_cap is None:
            return
        ok, frame = self._preview_cap.read()
        if ok:
            self._display_frame(frame)

    # ── Session-mode frame updates (from FrameDispatcher) ─────────────────────

    def set_frame(self, qimage: QImage):
        self._feed.show()
        self._placeholder.hide()
        pixmap = QPixmap.fromImage(qimage).scaled(
            self._feed.width(), self._feed.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._feed.setPixmap(pixmap)

    def set_disconnected(self):
        self._set_dot(False)
        self._feed.hide()
        self._placeholder.setText("NO SIGNAL")
        self._placeholder.show()
        self._scanning.hide()

    def on_session_start(self):
        """Lock the dropdown and stop the preview capture — session owns the camera now."""
        self._session_active = True
        self._stop_preview()
        self._combo.setEnabled(False)

    # ── Trigger flash & scanning overlay ──────────────────────────────────────

    def flash_trigger(self):
        self.setStyleSheet(
            "QFrame#cam_widget { border: 2px solid #22c55e; border-radius: 10px; "
            "background: #0d1a0d; }"
        )
        self._flash_timer.start(500)

    def show_scanning(self):
        self._scanning.show()

    def hide_scanning(self):
        self._scanning.hide()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _display_frame(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg).scaled(
            self._feed.width(), self._feed.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._feed.setPixmap(pixmap)

    def _set_dot(self, live: bool):
        self._dot.setStyleSheet(
            f"color: {'#22c55e' if live else '#333'}; font-size: 16px;"
        )

    def _clear_flash(self):
        self.setStyleSheet("")


class ResultBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("result_bar")
        self.setFixedHeight(52)
        self.setStyleSheet("background: #111111; border-top: 1px solid #1e1e1e;")

        self._seq     = QLabel("")
        self._seq.setStyleSheet("color: #444; font-size: 11px; font-weight: 600; min-width: 32px;")

        self._brand   = QLabel("")
        self._brand.setStyleSheet("color: #22c55e; font-size: 13px; font-weight: 700;")

        self._product = QLabel("")
        self._product.setStyleSheet("color: #aaa; font-size: 12px;")

        self._idle = QLabel("WAITING FOR PRODUCT")
        self._idle.setStyleSheet("color: #2a2a2a; font-size: 12px; font-weight: 500; letter-spacing: 1px;")
        self._idle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._fields_widget = QWidget()
        fields_row = QHBoxLayout(self._fields_widget)
        fields_row.setContentsMargins(0, 0, 0, 0)
        fields_row.setSpacing(0)

        self._expiry  = self._field("Exp: —")
        self._mfg     = self._field("MFG: —")
        self._batch   = self._field("Batch: —")
        self._status  = self._field("—")
        self._ms      = self._field("—")
        self._ms.setStyleSheet(self._ms.styleSheet() + " color: #333;")

        for w in [self._expiry, self._mfg, self._batch, self._status, self._ms]:
            fields_row.addWidget(w)
        fields_row.addStretch()

        left = QHBoxLayout()
        left.setSpacing(8)
        left.addWidget(self._seq)
        left.addWidget(self._brand)
        left.addWidget(self._product)

        main = QHBoxLayout(self)
        main.setContentsMargins(16, 0, 16, 0)
        main.setSpacing(16)
        main.addLayout(left)
        main.addWidget(self._fields_widget)
        main.addStretch()
        main.addWidget(self._idle)

        self._fields_widget.hide()

    def _field(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "color: #666; font-size: 11px; padding: 0 8px; border-left: 1px solid #222;"
        )
        return lbl

    def update_result(self, result: dict):
        seq    = result.get("seq_number", "")
        brand  = result.get("brand") or result.get("barcode") or "—"
        status = result.get("status", "—")
        ms     = result.get("processing_ms")

        self._seq.setText(f"#{seq}")
        self._brand.setText(brand)
        self._product.setText(result.get("product_name") or "")
        self._expiry.setText(f"Exp: {result.get('expiry_date') or '—'}")
        self._mfg.setText(f"MFG: {result.get('manufacture_date') or '—'}")
        self._batch.setText(f"Batch: {result.get('batch_number') or '—'}")
        self._ms.setText(f"{ms/1000:.1f}s" if ms else "—")
        self._status.setText(status)

        if "Barcode" in status:
            color = "#0ea5e9"
        elif "Incomplete" in status or "Error" in status:
            color = "#f59e0b"
        else:
            color = "#22c55e"
        self._status.setStyleSheet(
            f"color: {color}; font-size: 11px; padding: 0 8px; border-left: 1px solid #222;"
        )

        self._idle.hide()
        self._fields_widget.show()
