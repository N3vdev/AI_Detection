import cv2
from PyQt6.QtWidgets import (
    QFrame, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
    QSizePolicy, QComboBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap


CAM_LABELS = ["FRONT", "BACK", "LEFT", "RIGHT"]

_SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# ── Base style restored after flash ──────────────────────────────────────────
_CAM_BASE_STYLE = (
    "QFrame#cam_widget { background: #111111; border: 1px solid #1c1c1c; border-radius: 8px; }"
)
_CAM_FLASH_STYLE = (
    "QFrame#cam_widget { background: #0a1a0a; border: 2px solid #22c55e; border-radius: 8px; }"
)


class CameraWidget(QFrame):
    source_changed = pyqtSignal(int, object)   # cam_idx, source (or None)

    def __init__(self, cam_idx: int, parent=None):
        super().__init__(parent)
        self.cam_idx = cam_idx
        self._selected_source = None
        self._preview_cap = None
        self._session_active = False

        self.setObjectName("cam_widget")
        self.setMinimumSize(300, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(_CAM_BASE_STYLE)

        # ── Header row: dot + label + combo ───────────────────────────────────
        self._dot = QLabel("●")
        self._dot.setFixedWidth(14)
        self._dot.setStyleSheet("color: #2a2a2a; font-size: 13px; padding-top: 1px;")

        self._header = QLabel(f"CAM {cam_idx + 1}  —  {CAM_LABELS[cam_idx]}")
        self._header.setStyleSheet(
            "color: #555; font-size: 9px; font-weight: 700; letter-spacing: 1.5px;"
        )

        self._combo = QComboBox()
        self._combo.addItem("Scanning...", None)
        self._combo.setEnabled(False)
        self._combo.currentIndexChanged.connect(self._on_combo_changed)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(10, 6, 8, 4)
        header_row.setSpacing(6)
        header_row.addWidget(self._dot)
        header_row.addWidget(self._header)
        header_row.addStretch()
        header_row.addWidget(self._combo)

        # ── Feed / placeholder ────────────────────────────────────────────────
        self._feed = QLabel()
        self._feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._feed.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._feed.setMinimumHeight(120)
        self._feed.setStyleSheet("background: #0a0a0a; border-radius: 5px;")

        self._placeholder = QLabel("NO CAMERA SELECTED")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._placeholder.setStyleSheet(
            "color: #1e1e1e; font-size: 11px; font-weight: 600; letter-spacing: 2.5px;"
        )

        # ── Layout ────────────────────────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 6)
        layout.setSpacing(4)
        layout.addLayout(header_row)
        layout.addWidget(self._placeholder, stretch=1)
        layout.addWidget(self._feed, stretch=1)

        self._feed.hide()

        # ── Floating overlays (not in layout — float over the feed area) ──────
        self._loading_overlay = QLabel(self)
        self._loading_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_overlay.setStyleSheet("background: #0a0a0a; border-radius: 5px;")
        self._loading_overlay.hide()

        self._scanning_overlay = QLabel(self)
        self._scanning_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scanning_overlay.setStyleSheet(
            "background: rgba(0, 12, 0, 210); border: 1px solid #1a3d1a; "
            "border-radius: 5px; color: #22c55e; font-size: 13px; "
            "font-weight: 700; letter-spacing: 3px;"
        )
        self._scanning_overlay.setText("SCANNING")
        self._scanning_overlay.hide()

        # ── Timers ────────────────────────────────────────────────────────────
        self._loading_step = 0
        self._loading_anim_timer = QTimer(self)
        self._loading_anim_timer.timeout.connect(self._tick_loading)

        self._scan_step = 0
        self._scan_pulse_timer = QTimer(self)
        self._scan_pulse_timer.timeout.connect(self._tick_scan_pulse)

        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._grab_preview_frame)

        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._clear_flash)

    # ── Resize — reposition floating overlays ────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_overlays()

    def _reposition_overlays(self):
        ref = self._feed if self._feed.isVisible() else self._placeholder
        geo = ref.geometry()
        if geo.isValid():
            for overlay in (self._loading_overlay, self._scanning_overlay):
                overlay.setGeometry(geo)
                overlay.raise_()

    # ── Loading animation ─────────────────────────────────────────────────────

    def show_loading(self):
        self._loading_step = 0
        self._update_loading_text()
        self._reposition_overlays()
        self._loading_overlay.show()
        self._loading_overlay.raise_()
        self._loading_anim_timer.start(100)

    def hide_loading(self):
        self._loading_anim_timer.stop()
        self._loading_overlay.hide()

    def _tick_loading(self):
        self._loading_step += 1
        self._update_loading_text()

    def _update_loading_text(self):
        spin = _SPIN[self._loading_step % len(_SPIN)]
        self._loading_overlay.setText(
            f"<div style='text-align:center; line-height:2;'>"
            f"<span style='color:#222; font-size:9px; letter-spacing:3px;'>AI MODELS</span><br>"
            f"<span style='color:#22c55e; font-size:22px;'>{spin}</span>"
            f"</div>"
        )

    # ── Dropdown population ───────────────────────────────────────────────────

    def populate_start(self):
        self._combo.clear()
        self._combo.addItem("— Select camera —", None)
        self._combo.setEnabled(True)

    def add_camera_option(self, label: str, source):
        self._combo.addItem(label, source)

    def scan_complete(self):
        if self._combo.count() == 1:
            self._combo.setItemText(0, "No cameras found")

    def restore_selection(self, source):
        for i in range(self._combo.count()):
            if self._combo.itemData(i) == source:
                self._combo.setCurrentIndex(i)
                return
        label = (
            f"USB Camera {source}  (index {source})" if isinstance(source, int)
            else f"IP Camera  {source}  (offline?)"
        )
        self._combo.addItem(f"{label}  ← last used", source)
        self._combo.setCurrentIndex(self._combo.count() - 1)

    # ── Combo change ──────────────────────────────────────────────────────────

    def _on_combo_changed(self, index):
        source = self._combo.itemData(index)
        self._selected_source = source
        self._stop_preview()
        self.source_changed.emit(self.cam_idx, source)
        if source is not None:
            self._start_preview(source)
        else:
            self._feed.hide()
            self._placeholder.setText("NO CAMERA SELECTED")
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
            self._preview_timer.start(100)
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

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def prepare_for_session(self):
        self._combo.setEnabled(False)
        self._feed.show()
        self._placeholder.hide()
        self.show_loading()

    def on_session_start(self):
        self._session_active = True
        self._stop_preview()
        self.hide_loading()

    # ── Frame updates (from FrameDispatcher) ──────────────────────────────────

    def set_frame(self, qimage: QImage):
        self._feed.show()
        self._placeholder.hide()
        self._feed.setPixmap(self._cover_pixmap(QPixmap.fromImage(qimage)))

    def set_disconnected(self):
        self._set_dot(False)
        self._feed.hide()
        self._placeholder.setText("NO SIGNAL")
        self._placeholder.show()
        self.hide_scanning()

    # ── Trigger flash ─────────────────────────────────────────────────────────

    def flash_trigger(self):
        self.setStyleSheet(_CAM_FLASH_STYLE)
        self._flash_timer.start(500)

    def _clear_flash(self):
        self.setStyleSheet(_CAM_BASE_STYLE)

    # ── Scanning overlay ──────────────────────────────────────────────────────

    def show_scanning(self):
        self._scan_step = 0
        self._reposition_overlays()
        self._scanning_overlay.show()
        self._scanning_overlay.raise_()
        self._scan_pulse_timer.start(600)

    def hide_scanning(self):
        self._scan_pulse_timer.stop()
        self._scanning_overlay.hide()

    def _tick_scan_pulse(self):
        self._scan_step = (self._scan_step + 1) % 2
        colors = ("#22c55e", "#166534")
        self._scanning_overlay.setStyleSheet(
            f"background: rgba(0,12,0,210); border: 1px solid #1a3d1a; "
            f"border-radius: 5px; color: {colors[self._scan_step]}; "
            f"font-size: 13px; font-weight: 700; letter-spacing: 3px;"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _display_frame(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self._feed.setPixmap(self._cover_pixmap(QPixmap.fromImage(qimg)))

    def _cover_pixmap(self, pixmap: QPixmap) -> QPixmap:
        tw, th = self._feed.width(), self._feed.height()
        if tw <= 0 or th <= 0:
            return pixmap
        scaled = pixmap.scaled(tw, th,
                               Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                               Qt.TransformationMode.SmoothTransformation)
        x = (scaled.width()  - tw) // 2
        y = (scaled.height() - th) // 2
        return scaled.copy(x, y, tw, th)

    def _set_dot(self, live: bool):
        self._dot.setStyleSheet(
            f"color: {'#22c55e' if live else '#2a2a2a'}; font-size: 13px; padding-top: 1px;"
        )


# ── Result bar ────────────────────────────────────────────────────────────────

class ResultBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("result_bar")
        self.setFixedHeight(72)

        # ── Row 1: sequence · brand · product ─────────────────────────────────
        self._seq = QLabel("")
        self._seq.setStyleSheet(
            "color: #333; font-size: 10px; font-weight: 600; min-width: 28px;"
        )

        self._brand = QLabel("")
        self._brand.setStyleSheet(
            "color: #22c55e; font-size: 14px; font-weight: 700; letter-spacing: 0.3px;"
        )

        self._product = QLabel("")
        self._product.setStyleSheet("color: #888; font-size: 12px;")
        self._product.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(8)
        row1.addWidget(self._seq)
        row1.addWidget(self._brand)
        row1.addWidget(self._product, stretch=1)

        # ── Row 2: fields ──────────────────────────────────────────────────────
        self._expiry = self._field_label("Exp: —")
        self._mfg    = self._field_label("MFG: —")
        self._batch  = self._field_label("Batch: —")
        self._time   = self._field_label("—", dim=True)

        self._status = QLabel("—")
        self._status.setStyleSheet(
            "color: #22c55e; background: #0a1a0a; border: 1px solid #1a3d1a; "
            "border-radius: 4px; font-size: 9px; font-weight: 700; "
            "letter-spacing: 0.5px; padding: 1px 7px;"
        )

        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(0)
        for w in (self._expiry, self._mfg, self._batch):
            row2.addWidget(w)
        row2.addSpacing(10)
        row2.addWidget(self._status)
        row2.addSpacing(10)
        row2.addWidget(self._time)
        row2.addStretch()

        # ── Idle message ───────────────────────────────────────────────────────
        self._idle = QLabel("WAITING FOR PRODUCT")
        self._idle.setStyleSheet(
            "color: #1e1e1e; font-size: 11px; font-weight: 600; letter-spacing: 2px;"
        )
        self._idle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # ── Content widget (hidden until first result) ─────────────────────────
        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(4)
        content_layout.addLayout(row1)
        content_layout.addLayout(row2)

        # ── Main layout ────────────────────────────────────────────────────────
        main = QHBoxLayout(self)
        main.setContentsMargins(18, 0, 18, 0)
        main.setSpacing(0)
        main.addWidget(self._content, stretch=1)
        main.addWidget(self._idle, stretch=1)

        self._content.hide()

    @staticmethod
    def _field_label(text, dim=False):
        lbl = QLabel(text)
        color = "#333" if dim else "#555"
        lbl.setStyleSheet(
            f"color: {color}; font-size: 10px; padding: 0 10px 0 0; "
        )
        return lbl

    def update_result(self, result: dict):
        seq    = result.get("seq_number", "")
        brand  = result.get("brand") or result.get("barcode") or "—"
        product = result.get("product_name") or ""
        status = result.get("status", "—")
        ms     = result.get("processing_ms")

        self._seq.setText(f"#{seq}")
        self._brand.setText(brand)

        # Truncate long product names
        if len(product) > 60:
            product = product[:58] + "…"
        self._product.setText(product)

        self._expiry.setText(f"Exp: {result.get('expiry_date') or '—'}")
        self._mfg.setText(f"MFG: {result.get('manufacture_date') or '—'}")
        self._batch.setText(f"Batch: {result.get('batch_number') or '—'}")
        self._time.setText(f"{ms/1000:.1f}s" if ms else "—")

        # Status badge color
        if "Barcode" in status:
            bg, border, fg = "#071e2e", "#0e3d5c", "#38bdf8"
        elif "Incomplete" in status or "Error" in status:
            bg, border, fg = "#1f1200", "#3d2500", "#f59e0b"
        else:
            bg, border, fg = "#071a0e", "#0f3d1e", "#22c55e"

        short_status = status.replace("Complete ", "").replace("(", "").replace(")", "").strip()
        self._status.setText(short_status.upper())
        self._status.setStyleSheet(
            f"color: {fg}; background: {bg}; border: 1px solid {border}; "
            "border-radius: 4px; font-size: 9px; font-weight: 700; "
            "letter-spacing: 0.5px; padding: 1px 7px;"
        )

        # Update brand color to match status
        self._brand.setStyleSheet(
            f"color: {fg}; font-size: 14px; font-weight: 700; letter-spacing: 0.3px;"
        )

        # Show result, hide idle
        self._idle.hide()
        self._content.show()

        # Brief flash: highlight background then fade back
        self.setStyleSheet("#result_bar { background: #0d1a0d; border-top: 1px solid #1e3a1e; }")
        QTimer.singleShot(350, lambda: self.setStyleSheet(
            "#result_bar { background: #0e0e0e; border-top: 1px solid #1e1e1e; }"
        ))
