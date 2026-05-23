import cv2


class TriggerDetector:
    def __init__(
        self,
        roi_y_band=(0.25, 0.75),
        roi_x_center_band=(0.2, 0.8),
        min_contour_area=3000,
        min_gap_frames=20,
        check_every_n_frames=2,
        mog2_history=300,
        mog2_var_threshold=40,
    ):
        self.roi_y_band = roi_y_band
        self.roi_x_center_band = roi_x_center_band
        self.min_area = min_contour_area
        self.min_gap = min_gap_frames
        self.check_every_n = check_every_n_frames

        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=mog2_history,
            varThreshold=mog2_var_threshold,
            detectShadows=False,
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self._triggered = False   # True = already fired this product
        self._clear_count = 0     # consecutive frames with no object (for re-arm)

        print("[Trigger] MOG2 background subtractor ready.")

    def process_frame(self, frame, frame_count):
        """
        Returns True exactly once per product entering the zone.
        Re-arms only after the zone is clear for min_gap_frames consecutive checks.
        """
        if frame_count % self.check_every_n != 0:
            return False

        has_object = self._detect_object(frame)

        if not self._triggered:
            if has_object:
                self._triggered = True
                self._clear_count = 0
                return True          # Fire — product detected
        else:
            if has_object:
                self._clear_count = 0   # still present, keep waiting
            else:
                self._clear_count += 1
                if self._clear_count >= self.min_gap:
                    self._triggered = False  # zone clear long enough — re-arm
                    self._clear_count = 0

        return False

    def _detect_object(self, frame):
        h, w = frame.shape[:2]
        y1 = int(self.roi_y_band[0] * h)
        y2 = int(self.roi_y_band[1] * h)
        x1 = int(self.roi_x_center_band[0] * w)
        x2 = int(self.roi_x_center_band[1] * w)

        fg = self._bg.apply(frame)
        roi = fg[y1:y2, x1:x2]

        # Remove noise
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, self._kernel)
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = sum(cv2.contourArea(c) for c in contours)
        return total_area > self.min_area

    def get_foreground_area(self, frame):
        """For debug/preview — returns foreground pixel area in trigger zone."""
        h, w = frame.shape[:2]
        y1 = int(self.roi_y_band[0] * h)
        y2 = int(self.roi_y_band[1] * h)
        x1 = int(self.roi_x_center_band[0] * w)
        x2 = int(self.roi_x_center_band[1] * w)
        fg = self._bg.apply(frame, learningRate=0)  # don't update model
        roi = fg[y1:y2, x1:x2]
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, self._kernel)
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sum(cv2.contourArea(c) for c in contours)

    def reset(self):
        self._triggered = False
        self._clear_count = 0
