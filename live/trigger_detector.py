import cv2
from ultralytics import YOLO


class TriggerDetector:
    def __init__(
        self,
        yolo_model_path="yolov8n.pt",
        confidence_threshold=0.20,
        enter_frames=3,    # consecutive detections to confirm product entered
        leave_frames=8,    # consecutive empty frames to confirm product left
    ):
        print("[Trigger] Loading YOLOv8n on CPU...")
        self.model = YOLO(yolo_model_path)
        self.model.to("cpu")
        print("[Trigger] Ready.")

        self.conf = confidence_threshold
        self.enter_frames = enter_frames
        self.leave_frames = leave_frames

        self._product_in_frame = False
        self._presence_count = 0   # consecutive frames WITH object
        self._absence_count = 0    # consecutive frames WITHOUT object

    def process_frame(self, frame):
        has_object = self._detect(frame)

        if not self._product_in_frame:
            # Waiting for a product to enter
            if has_object:
                self._presence_count += 1
                self._absence_count = 0
                if self._presence_count >= self.enter_frames:
                    # Confirmed — product is in frame
                    self._product_in_frame = True
                    self._presence_count = 0
                    return True   # ← TRIGGER
            else:
                self._presence_count = 0

        else:
            # Product is in frame — wait for it to leave
            if has_object:
                self._absence_count = 0   # still here
            else:
                self._absence_count += 1
                if self._absence_count >= self.leave_frames:
                    # Product left — ready for next one
                    self._product_in_frame = False
                    self._absence_count = 0

        return False

    def _detect(self, frame):
        small = cv2.resize(frame, (640, 640))
        results = self.model(small, verbose=False, conf=self.conf)
        return len(results[0].boxes) > 0

    @property
    def product_in_frame(self):
        return self._product_in_frame

    def reset(self):
        self._product_in_frame = False
        self._presence_count = 0
        self._absence_count = 0
