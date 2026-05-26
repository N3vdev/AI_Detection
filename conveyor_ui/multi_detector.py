"""
Reuses the TriggerDetector's YOLO model to show detection boxes on all camera feeds.
No second model load — just borrows the model with a lock.
Processes all cameras sequentially at ~3fps per camera so total CPU impact is low.
"""
import time
import threading
import cv2
from PyQt6.QtCore import QThread


class MultiCameraDetector(QThread):
    def __init__(self, buffers, trigger, conf=0.45, min_box_area=0.04, classes=None, parent=None):
        super().__init__(parent)
        self._buffers      = buffers
        self._model        = trigger.model   # reuse already-loaded model
        self._model_lock   = threading.Lock()
        self._conf         = conf
        self._min_box_area = min_box_area
        self._classes      = classes
        self._running      = True
        self._lock         = threading.Lock()
        self._boxes        = [[] for _ in buffers]

    def run(self):
        while self._running:
            now = time.monotonic()
            for i, buf in enumerate(self._buffers):
                if not self._running:
                    break
                frame = buf.get_closest(now, window_ms=500)
                if frame is not None:
                    boxes = self._detect(frame)
                    with self._lock:
                        self._boxes[i] = boxes
                time.sleep(0.05)  # small gap between cameras so we don't slam CPU

            time.sleep(0.28)   # ~3 fps per camera (0.28 + n*0.05 per cycle)

    def get_boxes(self, cam_idx: int) -> list:
        with self._lock:
            if cam_idx < len(self._boxes):
                return list(self._boxes[cam_idx])
            return []

    def _detect(self, frame):
        small = cv2.resize(frame, (640, 640))
        with self._model_lock:
            results = self._model(small, verbose=False, conf=self._conf, classes=self._classes)
        valid = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            area = (x2 - x1) * (y2 - y1) / (640 * 640)
            if area >= self._min_box_area:
                valid.append([x1 / 640, y1 / 640, x2 / 640, y2 / 640])
        return valid

    def stop(self):
        self._running = False
        self.wait()
