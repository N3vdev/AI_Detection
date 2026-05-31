import cv2
from PyQt6.QtWidgets import (
    QFrame, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
    QSizePolicy, QComboBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap


CAM_LABELS = ["FRONT", "BACK", "LEFT", "RIGHT"]

_SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

_CAM_BASE_STYLE = (
    "QFrame#cam_widget {"
    "  background: #0d0d0d;"
    "  border: 1px solid #1c1c1c;"
    "  border-radius: 10px;"
    "}"
)
_CAM_FLASH_STYLE = (
    "QFrame#cam_widget {"
    "  background: #050e07;"
    "  border: 2px solid #22c55e;"
    "  border-radius: 10px;"
    "}"
)
_FEED_STYLE = (
    "background: #080808;"
    "border-bottom-left-radius: 9px;"
    "border-bottom-right-radius: 9px;"
)


class CameraWidget(QFrame):
    source_changed = pyqtSignal(int, object)

    def __init__(self, cam_idx: int, parent=None):
        super().__init__(parent)
        self.cam_idx = cam_idx
        self._selected_source = None
        self._preview_cap = None
        self._session_active = False
        self._loading_step_text = "Loading models..."

        self.setObjectName("cam_widget")
        self.setMinimumSize(300, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(_CAM_BASE_STYLE)

        # ── Header ────────────────────────────────────────────────────────────
        self._dot = QLabel("●")
        self._dot.setFixedWidth(12)
        self._dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._dot.setStyleSheet("color: #333; font-size: 9px; padding-top: 1px;")

        self._cam_num = QLabel(f"CAM {cam_idx + 1}")
        self._cam_num.setStyleSheet(
            "color: #585858; font-size: 10px; font-weight: 700; letter-spacing: 1.5px;"
        )

        self._cam_label = QLabel(CAM_LABELS[cam_idx])
        self._cam_label.setStyleSheet(
            "color: #3a3a3a; font-size: 9px; letter-spacing: 2.5px; padding-left: 7px;"
        )

        self._combo = QComboBox()
        self._combo.addItem("Scanning...", None)
        self._combo.setEnabled(False)
        self._combo.currentIndexChanged.connect(self._on_combo_changed)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(10, 8, 8, 8)
        header_row.setSpacing(4)
        header_row.addWidget(self._dot)
        header_row.addWidget(self._cam_num)
        header_row.addWidget(self._cam_label)
        header_row.addStretch()
        header_row.addWidget(self._combo)

        _header_w = QWidget()
        _header_w.setObjectName("cam_header")
        _header_w.setLayout(header_row)
        _header_w.setStyleSheet(
            "QWidget#cam_header { background: transparent; border-bottom: 1px solid #141414; }"
        )

        # ── Feed / placeholder ────────────────────────────────────────────────
        self._feed = QLabel()
        self._feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._feed.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._feed.setStyleSheet(_FEED_STYLE)

        self._placeholder = QLabel()
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._placeholder.setStyleSheet(_FEED_STYLE)
        self._set_placeholder_text("NO INPUT")

        # ── Main layout ────────────────────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(_header_w)
        layout.addWidget(self._placeholder, stretch=1)
        layout.addWidget(self._feed, stretch=1)
        self._feed.hide()

        # ── Floating overlays ─────────────────────────────────────────────────
        self._loading_overlay = QLabel(self)
        self._loading_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_overlay.setStyleSheet(
            "background: rgba(7, 7, 7, 248);"
            "border-bottom-left-radius: 9px;"
            "border-bottom-right-radius: 9px;"
        )
        self._loading_overlay.hide()

        self._scanning_overlay = QLabel(self)
        self._scanning_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scanning_overlay.setText("SCANNING")
        self._scanning_overlay.hide()
        self._scan_bright = True

        # ── Timers ────────────────────────────────────────────────────────────
        self._loading_step = 0
        self._loading_anim_timer = QTimer(self)
        self._loading_anim_timer.timeout.connect(self._tick_loading)

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

    def _set_placeholder_text(self, state="NO INPUT"):
        icon = "⊘" if state == "NO SIGNAL" else "⊟"
        self._placeholder.setText(
            "<center>"
            f"<p style='color:#303030; font-size:22px; margin:0 0 8px 0; line-height:1;'>{icon}</p>"
            f"<p style='color:#404040; font-size:8px; letter-spacing:4px; font-weight:700; margin:0;'>{state}</p>"
            "</center>"
        )

    # ── Loading animation ─────────────────────────────────────────────────────

    def set_loading_step(self, text: str):
        """Called from UI thread via signal to show the current loading step."""
        self._loading_step_text = text
        self._update_loading_text()

    def show_loading(self):
        self._loading_step = 0
        self._loading_step_text = "Loading models..."
        self._update_loading_text()
        self._reposition_overlays()
        self._loading_overlay.show()
        self._loading_overlay.raise_()
        self._loading_anim_timer.start(90)

    def hide_loading(self):
        self._loading_anim_timer.stop()
        self._loading_overlay.hide()

    def _tick_loading(self):
        self._loading_step += 1
        self._update_loading_text()

    def _update_loading_text(self):
        spin = _SPIN[self._loading_step % len(_SPIN)]
        step = self._loading_step_text
        # Wrap long text at ~34 chars
        if len(step) > 34:
            step = step[:32] + "…"
        self._loading_overlay.setText(
            "<div style='text-align:center; padding: 8px;'>"
            f"<p style='color:#22c55e; font-size:22px; margin:0 0 12px 0;'>{spin}</p>"
            "<p style='color:#666; font-size:8px; letter-spacing:3px; "
            "font-weight:700; margin:0 0 8px 0;'>LOADING</p>"
            f"<p style='color:#888; font-size:9px; margin:0; "
            f"font-family: Consolas, monospace;'>{step}</p>"
            "</div>"
        )

    # ── Dropdown population ───────────────────────────────────────────────────

    def populate_start(self):
        self._combo.clear()
        self._combo.addItem("— select camera —", None)
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
            f"USB {source}  (idx {source})" if isinstance(source, int)
            else f"IP  {source}  (offline?)"
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
            self._set_placeholder_text("NO INPUT")
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

    def restore_after_session(self):
        self._session_active = False
        self.hide_loading()
        self.hide_scanning()
        self.setStyleSheet(_CAM_BASE_STYLE)
        self._combo.setEnabled(True)
        if self._selected_source is not None:
            # Delay restart so the camera device has time to be released
            QTimer.singleShot(800, lambda: self._start_preview(self._selected_source))
        else:
            self._feed.hide()
            self._set_placeholder_text("NO INPUT")
            self._placeholder.show()
            self._set_dot(False)

    # ── Frame updates (from FrameDispatcher) ──────────────────────────────────

    def set_frame(self, qimage: QImage):
        self._feed.show()
        self._placeholder.hide()
        self._feed.setPixmap(self._cover_pixmap(QPixmap.fromImage(qimage)))

    def set_disconnected(self):
        self._set_dot(False)
        self._feed.hide()
        self._set_placeholder_text("NO SIGNAL")
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
        self._scan_bright = True
        self._apply_scan_style()
        self._reposition_overlays()
        self._scanning_overlay.show()
        self._scanning_overlay.raise_()
        self._scan_pulse_timer.start(520)

    def hide_scanning(self):
        self._scan_pulse_timer.stop()
        self._scanning_overlay.hide()

    def _tick_scan_pulse(self):
        self._scan_bright = not self._scan_bright
        self._apply_scan_style()

    def _apply_scan_style(self):
        if self._scan_bright:
            color, border = "#22c55e", "#0e3318"
        else:
            color, border = "#166534", "#0a1e10"
        self._scanning_overlay.setStyleSheet(
            f"background: rgba(3, 9, 4, 222);"
            f"border: 1px solid {border};"
            "border-bottom-left-radius: 9px;"
            "border-bottom-right-radius: 9px;"
            f"color: {color};"
            "font-size: 11px; font-weight: 700; letter-spacing: 5px;"
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
        # FastTransformation — no filter, GPU-friendly, eliminates UI lag
        scaled = pixmap.scaled(
            tw, th,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.FastTransformation,
        )
        x = (scaled.width()  - tw) // 2
        y = (scaled.height() - th) // 2
        return scaled.copy(x, y, tw, th)

    def _set_dot(self, live: bool):
        if live:
            self._dot.setStyleSheet("color: #22c55e; font-size: 9px; padding-top: 1px;")
            self._cam_num.setStyleSheet(
                "color: #a0a0a0; font-size: 10px; font-weight: 700; letter-spacing: 1.5px;"
            )
            self._cam_label.setStyleSheet(
                "color: #606060; font-size: 9px; letter-spacing: 2.5px; padding-left: 7px;"
            )
        else:
            self._dot.setStyleSheet("color: #333; font-size: 9px; padding-top: 1px;")
            self._cam_num.setStyleSheet(
                "color: #585858; font-size: 10px; font-weight: 700; letter-spacing: 1.5px;"
            )
            self._cam_label.setStyleSheet(
                "color: #3a3a3a; font-size: 9px; letter-spacing: 2.5px; padding-left: 7px;"
            )


# ── Result bar ────────────────────────────────────────────────────────────────

class ResultBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("result_bar")
        self.setFixedHeight(84)

        self._accent = QWidget()
        self._accent.setFixedWidth(3)
        self._accent.setFixedHeight(52)
        self._accent.setStyleSheet("background: #1e1e1e; border-radius: 2px;")

        self._seq = QLabel()
        self._seq.setFixedWidth(34)
        self._seq.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._seq.setStyleSheet(
            "color: #505050; font-size: 13px; font-weight: 700;"
            "font-family: 'Consolas', 'Courier New', monospace;"
        )

        self._brand = QLabel()
        self._brand.setStyleSheet(
            "color: #22c55e; font-size: 15px; font-weight: 700; letter-spacing: 0.2px;"
        )

        self._product = QLabel()
        self._product.setStyleSheet("color: #909090; font-size: 11px;")
        self._product.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._status = QLabel()
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setFixedHeight(20)
        self._status.setStyleSheet(
            "color: #22c55e; background: #061a0e; border: 1px solid #0d3d1a;"
            "border-radius: 4px; font-size: 9px; font-weight: 700;"
            "letter-spacing: 0.8px; padding: 0 8px;"
        )

        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(10)
        row1.addWidget(self._seq)
        row1.addWidget(self._brand)
        row1.addWidget(self._product, stretch=1)
        row1.addWidget(self._status)

        self._expiry = self._make_chip()
        self._mfg    = self._make_chip()
        self._batch  = self._make_chip()

        self._time = QLabel()
        self._time.setStyleSheet(
            "color: #606060; font-size: 9px;"
            "font-family: 'Consolas', 'Courier New', monospace;"
        )

        row2 = QHBoxLayout()
        row2.setContentsMargins(44, 0, 0, 0)
        row2.setSpacing(5)
        row2.addWidget(self._expiry)
        row2.addWidget(self._mfg)
        row2.addWidget(self._batch)
        row2.addStretch()
        row2.addWidget(self._time)

        content_vbox = QVBoxLayout()
        content_vbox.setContentsMargins(0, 0, 0, 0)
        content_vbox.setSpacing(8)
        content_vbox.addLayout(row1)
        content_vbox.addLayout(row2)

        self._content = QWidget()
        cl = QVBoxLayout(self._content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addLayout(content_vbox)

        self._idle = QLabel("WAITING FOR PRODUCT")
        self._idle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._idle.setStyleSheet(
            "color: #404040; font-size: 10px; font-weight: 600; letter-spacing: 3.5px;"
        )

        main = QHBoxLayout(self)
        main.setContentsMargins(14, 0, 18, 0)
        main.setSpacing(0)
        main.addWidget(self._accent, alignment=Qt.AlignmentFlag.AlignVCenter)
        main.addSpacing(16)
        main.addWidget(self._content, stretch=1)
        main.addWidget(self._idle, stretch=1)

        self._content.hide()

    @staticmethod
    def _make_chip() -> QLabel:
        w = QLabel()
        w.setStyleSheet(
            "color: #505050; background: #0c0c0c; border: 1px solid #1e1e1e;"
            "border-radius: 4px; font-size: 9px; padding: 2px 8px;"
        )
        return w

    def _set_chip(self, chip: QLabel, label: str, value):
        has_value = bool(value) and str(value) not in ('—', 'None')
        if has_value:
            chip.setText(
                f"<span style='color:#707070;'>{label}:</span>"
                f"<span style='color:#b0b0b0;'> {value}</span>"
            )
            chip.setStyleSheet(
                "background: #111; border: 1px solid #252525;"
                "border-radius: 4px; font-size: 9px; padding: 2px 8px;"
            )
        else:
            chip.setText(
                f"<span style='color:#383838;'>{label}:</span>"
                f"<span style='color:#383838;'> —</span>"
            )
            chip.setStyleSheet(
                "background: #0c0c0c; border: 1px solid #1a1a1a;"
                "border-radius: 4px; font-size: 9px; padding: 2px 8px;"
            )

    def update_result(self, result: dict):
        seq     = result.get("seq_number", "")
        brand   = result.get("brand") or result.get("barcode") or "—"
        product = result.get("product_name") or ""
        status  = result.get("status", "—")
        ms      = result.get("processing_ms")

        self._seq.setText(f"#{seq:02d}" if isinstance(seq, int) else f"#{seq}")

        if len(product) > 55:
            product = product[:53] + "…"
        self._brand.setText(brand)
        self._product.setText(product)

        self._set_chip(self._expiry, "Exp",   result.get("expiry_date"))
        self._set_chip(self._mfg,    "MFG",   result.get("manufacture_date"))
        self._set_chip(self._batch,  "Batch", result.get("batch_number"))

        self._time.setText(f"{ms/1000:.1f}s" if ms else "")

        if "Barcode" in status:
            fg, bg, bd = "#38bdf8", "#060f1a", "#0d2844"
        elif "Incomplete" in status or "Error" in status:
            fg, bg, bd = "#f59e0b", "#140900", "#3a2000"
        else:
            fg, bg, bd = "#22c55e", "#061a0e", "#0e3d1a"

        short = (
            status.replace("Complete ", "").replace("(", "").replace(")", "").strip().upper()
        )
        self._status.setText(short)
        self._status.setStyleSheet(
            f"color: {fg}; background: {bg}; border: 1px solid {bd};"
            "border-radius: 4px; font-size: 9px; font-weight: 700;"
            "letter-spacing: 0.8px; padding: 0 8px; min-height: 20px;"
        )
        self._brand.setStyleSheet(
            f"color: {fg}; font-size: 15px; font-weight: 700; letter-spacing: 0.2px;"
        )
        self._accent.setStyleSheet(f"background: {fg}; border-radius: 2px;")

        self._idle.hide()
        self._content.show()

        self.setStyleSheet(f"#result_bar {{ background: {bg}; border-top: 1px solid {bd}; }}")
        QTimer.singleShot(380, lambda: self.setStyleSheet(
            "#result_bar { background: #0b0b0b; border-top: 1px solid #181818; }"
        ))
