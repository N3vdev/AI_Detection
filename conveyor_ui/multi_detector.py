"""
Runs YOLO detection on all camera buffers at ~5fps.
Completely independent of the trigger detector — this is purely for visualising
detection boxes on all camera feeds. Loads its own model instance.
"""
import time
import threading
import cv2
from ultralytics import YOLO
from PyQt6.QtCore import QThread


class MultiCameraDetector(QThread):
    def __init__(self, buffers, model_path, conf=0.30, min_box_area=0.05, parent=None):
        super().__init__(parent)
        self._buffers      = buffers
        self._model_path   = model_path
        self._conf         = conf
        self._min_box_area = min_box_area
        self._running      = True
        self._model        = None
        self._lock         = threading.Lock()
        # latest detected boxes per camera slot: list of [x1n, y1n, x2n, y2n]
        self._boxes = [[] for _ in buffers]

    def run(self):
        print("[Detector] Loading YOLO for multi-camera display...")
        self._model = YOLO(self._model_path)
        self._model.to("cpu")
        print("[Detector] Ready.")

        while self._running:
            now = time.monotonic()
            for i, buf in enumerate(self._buffers):
                frame = buf.get_closest(now, window_ms=500)
                if frame is not None:
                    boxes = self._detect(frame)
                    with self._lock:
                        self._boxes[i] = boxes
            time.sleep(0.20)  # 5 fps — enough for smooth box display

    def get_boxes(self, cam_idx: int) -> list:
        with self._lock:
            if cam_idx < len(self._boxes):
                return list(self._boxes[cam_idx])
            return []

    def _detect(self, frame):
        small = cv2.resize(frame, (640, 640))
        results = self._model(small, verbose=False, conf=self._conf)
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
