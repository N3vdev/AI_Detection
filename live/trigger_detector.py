import cv2
from ultralytics import YOLO

# States for the product state machine
_IDLE = "IDLE"
_ENTERING = "ENTERING"
_CENTERED = "CENTERED"
_EXITING = "EXITING"


class TriggerDetector:
    def __init__(
        self,
        yolo_model_path="yolov8n.pt",
        roi_y_band=(0.25, 0.75),
        roi_x_center_band=(0.3, 0.7),
        confidence_threshold=0.55,
        check_every_n_frames=3,
        min_gap_frames=15,
    ):
        # Run on CPU — GPU is reserved for the inspection worker
        print("[Trigger] Loading YOLOv8n on CPU...")
        self.model = YOLO(yolo_model_path)
        self.model.to("cpu")
        print("[Trigger] YOLOv8n ready.")

        self.roi_y_band = roi_y_band
        self.roi_x_center_band = roi_x_center_band
        self.conf_thresh = confidence_threshold
        self.check_every_n = check_every_n_frames
        self.min_gap_frames = min_gap_frames

        self._state = _IDLE
        self._cooldown = 0

    def process_frame(self, frame, frame_count):
        """
        Call this on every frame from the trigger camera.
        Returns True exactly once per product (when it is centered).
        """
        if self._cooldown > 0:
            self._cooldown -= 1
            return False

        if frame_count % self.check_every_n != 0:
            return False

        centroid = self._detect_centroid(frame)
        return self._step_state(centroid)

    def _detect_centroid(self, frame):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (640, 480))
        results = self.model(small, verbose=False, conf=self.conf_thresh)

        y_lo, y_hi = self.roi_y_band
        boxes_in_roi = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            # Normalize to 0–1
            cx = ((x1 + x2) / 2) / 640
            cy = ((y1 + y2) / 2) / 480
            if y_lo <= cy <= y_hi:
                boxes_in_roi.append((cx, cy))

        if not boxes_in_roi:
            return None
        avg_cx = sum(b[0] for b in boxes_in_roi) / len(boxes_in_roi)
        avg_cy = sum(b[1] for b in boxes_in_roi) / len(boxes_in_roi)
        return (avg_cx, avg_cy)

    def _step_state(self, centroid):
        x_lo, x_hi = self.roi_x_center_band
        fired = False

        if self._state == _IDLE:
            if centroid is not None:
                self._state = _ENTERING

        elif self._state == _ENTERING:
            if centroid is None:
                self._state = _IDLE
            elif x_lo <= centroid[0] <= x_hi:
                self._state = _CENTERED
                fired = True  # Product is centered — take snapshot now

        elif self._state == _CENTERED:
            if centroid is None or centroid[0] > x_hi:
                self._state = _EXITING

        elif self._state == _EXITING:
            if centroid is None:
                self._state = _IDLE
                self._cooldown = self.min_gap_frames

        return fired

    def reset(self):
        self._state = _IDLE
        self._cooldown = 0
