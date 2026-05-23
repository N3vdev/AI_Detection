import cv2
from ultralytics import YOLO


class TriggerDetector:
    def __init__(
        self,
        yolo_model_path="yolov8n.pt",
        roi_y_band=(0.1, 0.9),
        roi_x_center_band=(0.1, 0.9),
        confidence_threshold=0.3,
        check_every_n_frames=2,
        min_gap_frames=20,
    ):
        print("[Trigger] Loading YOLOv8n on CPU...")
        self.model = YOLO(yolo_model_path)
        self.model.to("cpu")
        print("[Trigger] YOLOv8n ready.")

        self.roi_y_band = roi_y_band
        self.roi_x_center_band = roi_x_center_band
        self.conf_thresh = confidence_threshold
        self.check_every_n = check_every_n_frames
        self.min_gap = min_gap_frames

        self._triggered = False   # True = already fired, waiting for product to leave
        self._clear_count = 0     # consecutive empty-zone checks before re-arming

    def process_frame(self, frame, frame_count):
        """
        Returns True exactly once per product.
        Re-arms only after the zone has been clear for min_gap consecutive checks.
        """
        if frame_count % self.check_every_n != 0:
            return False

        has_object = self._detect_in_zone(frame)

        if not self._triggered:
            if has_object:
                self._triggered = True
                self._clear_count = 0
                return True              # ← fire once
        else:
            if has_object:
                self._clear_count = 0   # product still present, keep waiting
            else:
                self._clear_count += 1
                if self._clear_count >= self.min_gap:
                    self._triggered = False   # zone clear — re-arm for next product
                    self._clear_count = 0

        return False

    def _detect_in_zone(self, frame):
        small = cv2.resize(frame, (640, 480))
        results = self.model(small, verbose=False, conf=self.conf_thresh)

        y_lo, y_hi = self.roi_y_band
        x_lo, x_hi = self.roi_x_center_band

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            cx = ((x1 + x2) / 2) / 640
            cy = ((y1 + y2) / 2) / 480
            if x_lo <= cx <= x_hi and y_lo <= cy <= y_hi:
                return True

        return False

    def reset(self):
        self._triggered = False
        self._clear_count = 0
