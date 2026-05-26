import time
import cv2
import numpy as np
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QSizePolicy
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QImage, QPixmap, QColor


CAM_LABELS = ["FRONT", "BACK", "LEFT", "RIGHT"]


class CameraWidget(QFrame):
    def __init__(self, cam_idx: int, parent=None):
        super().__init__(parent)
        self.cam_idx = cam_idx
        self.setObjectName("cam_widget")
        self.setMinimumSize(300, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # ── Header row ────────────────────────────────────────────────────────
        self._dot = QLabel("●")
        self._dot.setObjectName("dot_dead")
        self._dot.setFixedWidth(16)

        self._header = QLabel(f"CAM {cam_idx + 1} — {CAM_LABELS[cam_idx]}")
        self._header.setObjectName("cam_header")

        header_row = QHBoxLayout()
        header_row.setContentsMargins(10, 8, 10, 0)
        header_row.setSpacing(6)
        header_row.addWidget(self._dot)
        header_row.addWidget(self._header)
        header_row.addStretch()

        # ── Feed area ─────────────────────────────────────────────────────────
        self._feed = QLabel()
        self._feed.setObjectName("feed")
        self._feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._feed.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._feed.setMinimumHeight(160)

        self._no_signal = QLabel("NO SIGNAL")
        self._no_signal.setObjectName("no_signal")
        self._no_signal.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_signal.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # ── Scanning overlay ──────────────────────────────────────────────────
        self._scanning = QLabel("SCANNING...")
        self._scanning.setObjectName("scanning_overlay")
        self._scanning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scanning.hide()
        self._scan_timer = QTimer(self)
        self._scan_timer.timeout.connect(self._hide_scanning)

        # ── Layout ────────────────────────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 8)
        layout.setSpacing(4)
        layout.addLayout(header_row)
        layout.addWidget(self._no_signal)
        layout.addWidget(self._feed)
        layout.addWidget(self._scanning)

        self._feed.hide()
        self._connected = False

        # Trigger flash timer
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._clear_flash)

    # ── Public slots ──────────────────────────────────────────────────────────

    def set_frame(self, qimage: QImage):
        if not self._connected:
            self._connected = True
            self._dot.setObjectName("dot_live")
            self._dot.setStyleSheet("color: #22c55e;")
            self._no_signal.hide()
            self._feed.show()

        pixmap = QPixmap.fromImage(qimage).scaled(
            self._feed.width(), self._feed.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._feed.setPixmap(pixmap)

    def set_disconnected(self):
        self._connected = False
        self._dot.setObjectName("dot_dead")
        self._dot.setStyleSheet("color: #333;")
        self._feed.hide()
        self._no_signal.show()
        self._scanning.hide()

    def flash_trigger(self):
        self.setProperty("triggered", True)
        self.setStyleSheet("QFrame#cam_widget { border: 2px solid #22c55e; border-radius: 10px; }")
        self._flash_timer.start(500)

    def show_scanning(self):
        self._scanning.show()
        self._scan_timer.start(15000)  # safety hide after 15s

    def hide_scanning(self):
        self._scanning.hide()
        self._scan_timer.stop()

    # ── Private ───────────────────────────────────────────────────────────────

    def _clear_flash(self):
        self.setStyleSheet("")

    def _hide_scanning(self):
        self._scanning.hide()


class ResultBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("result_bar")
        self.setFixedHeight(52)

        self._seq   = QLabel("")
        self._seq.setObjectName("result_seq")

        self._brand = QLabel("")
        self._brand.setObjectName("result_brand")

        self._product = QLabel("")
        self._product.setObjectName("result_product")

        self._idle = QLabel("WAITING FOR PRODUCT")
        self._idle.setObjectName("idle_hint")

        self._fields_widget = QWidget()
        fields_row = QHBoxLayout(self._fields_widget)
        fields_row.setContentsMargins(0, 0, 0, 0)
        fields_row.setSpacing(0)

        self._expiry   = self._make_field("Exp: —")
        self._mfg      = self._make_field("MFG: —")
        self._batch    = self._make_field("Batch: —")
        self._status   = self._make_field("—")
        self._status.setObjectName("result_status_ok")
        self._ms       = self._make_field("—")
        self._ms.setObjectName("result_ms")

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

    def _make_field(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("result_field")
        return lbl

    def update_result(self, result: dict):
        seq = result.get("seq_number", "")
        self._seq.setText(f"#{seq}")
        self._brand.setText(result.get("brand") or result.get("barcode") or "—")
        self._product.setText(result.get("product_name") or "")

        exp   = result.get("expiry_date")   or "—"
        mfg   = result.get("manufacture_date") or "—"
        batch = result.get("batch_number")  or "—"
        ms    = result.get("processing_ms")
        status = result.get("status", "—")

        self._expiry.setText(f"Exp: {exp}")
        self._mfg.setText(f"MFG: {mfg}")
        self._batch.setText(f"Batch: {batch}")
        self._ms.setText(f"{ms/1000:.1f}s" if ms else "—")
        self._status.setText(status)

        if "Barcode" in status:
            self._status.setObjectName("result_status_barcode")
        elif "Incomplete" in status or "Error" in status:
            self._status.setObjectName("result_status_incomplete")
        else:
            self._status.setObjectName("result_status_ok")
        self._status.setStyle(self._status.style())  # force re-apply QSS

        self._idle.hide()
        self._fields_widget.show()
