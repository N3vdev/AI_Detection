import cv2
from ultralytics import YOLO


class TriggerDetector:
    def __init__(
        self,
        yolo_model_path="yolov8n.pt",
        confidence_threshold=0.45,
        min_box_area=0.04,
        enter_frames=3,
        leave_frames=8,
        classes=None,   # list of COCO class IDs to detect; None = all 80 classes
    ):
        print(f"[Trigger] Loading {yolo_model_path} on CPU...")
        self.model = YOLO(yolo_model_path)
        self.model.to("cpu")
        print("[Trigger] Ready.")

        self.conf = confidence_threshold
        self.min_box_area = min_box_area
        self.enter_frames = enter_frames
        self.leave_frames = leave_frames
        self.classes = classes

        self._product_in_frame = False
        self._presence_count = 0
        self._absence_count = 0
        self._post_detect_streak = 0   # consecutive detections after trigger — 2+ needed to reset absence
        self.last_boxes = []   # normalized [x1,y1,x2,y2] for drawing on preview

    def process_frame(self, frame):
        boxes = self._detect(frame)
        has_object = len(boxes) > 0
        self.last_boxes = boxes

        if not self._product_in_frame:
            if has_object:
                self._presence_count += 1
                self._absence_count = 0
                if self._presence_count >= self.enter_frames:
                    self._product_in_frame = True
                    self._presence_count = 0
                    return True   # ← TRIGGER
            else:
                self._presence_count = 0
        else:
            if has_object:
                self._post_detect_streak += 1
                if self._post_detect_streak >= 2:
                    self._absence_count = 0  # 2 consecutive detections = product genuinely still present
            else:
                self._post_detect_streak = 0
                self._absence_count += 1
                if self._absence_count >= self.leave_frames:
                    self._product_in_frame = False
                    self._absence_count = 0
                    self._post_detect_streak = 0

        return False

    def _detect(self, frame):
        """Returns list of normalized [x1,y1,x2,y2] for boxes that pass the size filter."""
        small = cv2.resize(frame, (640, 640))
        results = self.model(small, verbose=False, conf=self.conf, classes=self.classes)

        valid = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            box_area = (x2 - x1) * (y2 - y1) / (640 * 640)
            if box_area >= self.min_box_area:
                # Normalize to 0–1
                valid.append([x1 / 640, y1 / 640, x2 / 640, y2 / 640])

        return valid

    @property
    def product_in_frame(self):
        return self._product_in_frame

    def reset(self):
        self._product_in_frame = False
        self._presence_count = 0
        self._absence_count = 0
        self._post_detect_streak = 0
        self.last_boxes = []
