"""
Enumerates available cameras by probing USB indices and configured HTTP URLs.
Runs in a QThread so it never blocks the UI.
"""
import cv2
from PyQt6.QtCore import QThread, pyqtSignal


MAX_USB_INDEX = 10  # probe 0..9


def _try_open(source) -> bool:
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW if isinstance(source, int) else cv2.CAP_ANY)
    ok = cap.isOpened()
    cap.release()
    return ok


class CameraScanner(QThread):
    """
    Emits camera_found(label, source) for every available camera found.
    label  — human-readable string shown in the dropdown
    source — int index or URL string passed to cv2.VideoCapture
    """
    camera_found  = pyqtSignal(str, object)   # label, source
    scan_complete = pyqtSignal()

    def __init__(self, extra_urls=None, parent=None):
        super().__init__(parent)
        self._extra_urls = extra_urls or []

    def run(self):
        seen = set()

        # Probe USB indices
        for i in range(MAX_USB_INDEX):
            if _try_open(i):
                label = f"USB Camera {i}  (index {i})"
                self.camera_found.emit(label, i)
                seen.add(i)

        # Probe any HTTP/RTSP URLs from config
        for url in self._extra_urls:
            if url not in seen and isinstance(url, str) and _try_open(url):
                label = f"IP Camera  {url}"
                self.camera_found.emit(label, url)
                seen.add(url)

        self.scan_complete.emit()
