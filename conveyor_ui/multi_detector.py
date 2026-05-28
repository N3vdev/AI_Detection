"""
Reuses the TriggerDetector's YOLO model to show detection boxes on all camera feeds.

Smooth tracking strategy:
  - EMA blending: new detections slide toward new position rather than jumping
  - Velocity extrapolation: get_boxes() projects forward by (now - last_detection) × velocity
    so the box follows the moving product at display rate (15fps), not detection rate (~3fps)
  - TTL persistence: box stays visible for _BOX_TTL seconds after last detection — no flicker
"""
import time
import threading
import cv2
from PyQt6.QtCore import QThread


class MultiCameraDetector(QThread):
    _SMOOTH_ALPHA = 0.45   # blend weight toward new detection (higher = more responsive)
    _BOX_TTL      = 0.55   # seconds to keep box visible after last detection

    def __init__(self, buffers, trigger, conf=0.45, min_box_area=0.04, classes=None, parent=None):
        super().__init__(parent)
        self._buffers      = buffers
        self._model        = trigger.model   # reuse already-loaded model — no second load
        self._model_lock   = threading.Lock()
        self._conf         = conf
        self._min_box_area = min_box_area
        self._classes      = classes
        self._running      = True
        self._lock         = threading.Lock()
        n = len(buffers)
        # Per-camera tracking state
        self._smooth    = [None] * n        # EMA-smoothed [x1,y1,x2,y2] or None
        self._velocity  = [[0.0]*4] * n     # [dx1,dy1,dx2,dy2] in coords/sec
        self._prev_raw  = [None] * n        # raw detection from previous frame (velocity source)
        self._prev_t    = [0.0]  * n        # time of previous detection
        self._last_det  = [0.0]  * n        # monotonic time of most recent detection

    def run(self):
        while self._running:
            cycle_start = time.monotonic()
            for i, buf in enumerate(self._buffers):
                if not self._running:
                    break
                frame = buf.get_closest(cycle_start, window_ms=500)
                if frame is not None:
                    raw = self._detect(frame)
                    t = time.monotonic()
                    with self._lock:
                        if raw:
                            # Use the largest box — most likely the product on the conveyor
                            best = max(raw, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                            prev = self._prev_raw[i]
                            pt   = self._prev_t[i]
                            # Velocity from last two detections (normalized coords / second)
                            if prev is not None and (t - pt) > 0.01:
                                dt = t - pt
                                self._velocity[i] = [
                                    (best[j] - prev[j]) / dt for j in range(4)
                                ]
                            # EMA blend: slide toward new detected position
                            s = self._smooth[i]
                            if s is not None:
                                a = self._SMOOTH_ALPHA
                                self._smooth[i] = [
                                    a * best[j] + (1 - a) * s[j] for j in range(4)
                                ]
                            else:
                                self._smooth[i] = list(best)   # first detection — snap
                            self._prev_raw[i] = list(best)
                            self._prev_t[i]   = t
                            self._last_det[i] = t

            # Target ~8fps cycle; inference time naturally paces most of it
            elapsed = time.monotonic() - cycle_start
            gap = 0.12 - elapsed
            if gap > 0:
                time.sleep(gap)

    def get_boxes(self, cam_idx: int) -> list:
        with self._lock:
            age = time.monotonic() - self._last_det[cam_idx]
            if age > self._BOX_TTL or self._smooth[cam_idx] is None:
                return []
            s   = self._smooth[cam_idx]
            vel = self._velocity[cam_idx]
            # Extrapolate: project box forward by velocity × age to match product's current position.
            # This compensates for the gap between last detection and the current display frame.
            box = [max(0.0, min(1.0, s[j] + vel[j] * age)) for j in range(4)]
            return [box]

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
