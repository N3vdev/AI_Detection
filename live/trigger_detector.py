import cv2
from ultralytics import YOLO


class TriggerDetector:
    def __init__(
        self,
        yolo_model_path="yolov8n.pt",
        roi_y_band=(0.0, 1.0),
        roi_x_center_band=(0.0, 1.0),
        confidence_threshold=0.25,
        check_every_n_frames=1,
        # confirm/clear_frames no longer needed — tracker handles this
    ):
        print("[Trigger] Loading YOLOv8n with ByteTrack on CPU...")
        self.model = YOLO(yolo_model_path)
        self.model.to("cpu")
        print("[Trigger] Ready.")

        self.roi_y_band = roi_y_band
        self.roi_x_center_band = roi_x_center_band
        self.conf_thresh = confidence_threshold
        self.check_every_n = check_every_n_frames

        self._fired_ids = set()   # track IDs we already fired for this session
        self._active_ids = set()  # track IDs currently visible in frame

    def process_frame(self, frame, frame_count):
        if frame_count % self.check_every_n != 0:
            return False

        small = cv2.resize(frame, (320, 320))

        # ByteTrack keeps IDs alive across short gaps (handles YOLO flicker)
        results = self.model.track(
            small, verbose=False, persist=True, conf=self.conf_thresh
        )

        current_ids = set()
        if results[0].boxes.id is not None:
            y_lo, y_hi = self.roi_y_band
            x_lo, x_hi = self.roi_x_center_band

            for box, tid in zip(results[0].boxes, results[0].boxes.id.int().tolist()):
                x1, y1, x2, y2 = box.xyxy[0].numpy()
                cx = ((x1 + x2) / 2) / 320
                cy = ((y1 + y2) / 2) / 320
                if x_lo <= cx <= x_hi and y_lo <= cy <= y_hi:
                    current_ids.add(tid)

        # When a track ID departs, clear it from fired_ids.
        # This means the same product type returning later gets a fresh
        # ByteTrack ID and will fire again — correct warehouse behaviour.
        departed = self._active_ids - current_ids
        for tid in departed:
            self._fired_ids.discard(tid)

        self._active_ids = current_ids

        # New IDs we haven't fired for yet
        new_ids = current_ids - self._fired_ids
        if new_ids:
            tid = next(iter(new_ids))
            self._fired_ids.add(tid)
            return True

        return False

    def reset(self):
        self._fired_ids.clear()
        self._active_ids.clear()
