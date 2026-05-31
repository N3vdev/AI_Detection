import cv2
import threading
import time


class CameraThread(threading.Thread):
    def __init__(self, camera_index, resolution, fps, frame_buffer, rotate=None):
        super().__init__(daemon=True, name=f"Camera-{camera_index}")
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.buffer = frame_buffer
        self.rotate = rotate          # e.g. cv2.ROTATE_90_CLOCKWISE
        self._stop_event = threading.Event()
        self._cap = None

    def _open(self):
        cap = cv2.VideoCapture(self.camera_index)
        if isinstance(self.camera_index, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            try:
                cap.set(cv2.CAP_PROP_FPS, self.fps)
            except Exception:
                pass  # some USB cameras don't support FPS control
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            # IP/MJPEG stream — FPS is controlled by the server, not settable via OpenCV
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return cap

    def run(self):
        self._cap = self._open()
        if not self._cap.isOpened():
            print(f"[Camera-{self.camera_index}] Failed to open — check device index.")
            return

        print(f"[Camera-{self.camera_index}] Started.")
        consecutive_failures = 0

        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print(f"[Camera-{self.camera_index}] Reconnecting...")
                    self._cap.release()
                    time.sleep(0.5)
                    self._cap = self._open()
                    consecutive_failures = 0
                continue

            consecutive_failures = 0
            if self.rotate is not None:
                frame = cv2.rotate(frame, self.rotate)
            self.buffer.push(frame, time.monotonic())

        self._cap.release()
        print(f"[Camera-{self.camera_index}] Stopped.")

    def stop(self):
        self._stop_event.set()
