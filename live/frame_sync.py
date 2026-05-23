import cv2
import threading
import numpy as np
from collections import deque


class FrameSyncBuffer:
    def __init__(self, maxlen=90):
        self._buf = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, frame, timestamp):
        with self._lock:
            self._buf.append((timestamp, frame))

    def _candidates(self, target_ts, window_ms):
        half = window_ms / 1000.0
        with self._lock:
            return [
                (ts, f) for ts, f in self._buf
                if abs(ts - target_ts) <= half
            ]

    def get_closest(self, target_ts, window_ms=200):
        candidates = self._candidates(target_ts, window_ms)
        if not candidates:
            return None
        return min(candidates, key=lambda x: abs(x[0] - target_ts))[1]

    def get_sharpest(self, target_ts, window_ms=200, min_variance=50.0):
        candidates = self._candidates(target_ts, window_ms)
        if not candidates:
            return None
        scored = []
        for ts, frame in candidates:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            scored.append((score, frame))
        best_score, best_frame = max(scored, key=lambda x: x[0])
        if best_score < min_variance:
            return None
        return best_frame


class FrameAssembler:
    def __init__(self, buffers, window_ms=200, min_variance=50.0):
        """
        buffers: list of FrameSyncBuffer, one per camera
        """
        self.buffers = buffers
        self.window_ms = window_ms
        self.min_variance = min_variance

    def collect_snapshot(self, trigger_ts):
        """
        Returns a list of frames (one per camera).
        Uses sharpest frame within the sync window; falls back to closest if all blurry.
        Returns None for cameras that have no frames in the window.
        """
        frames = []
        for buf in self.buffers:
            frame = buf.get_sharpest(trigger_ts, self.window_ms, self.min_variance)
            if frame is None:
                frame = buf.get_closest(trigger_ts, self.window_ms)
            frames.append(frame)
        return frames
