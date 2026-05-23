import datetime
import queue
import time

import cv2

from live import conveyor_config as config
from live.camera_thread import CameraThread
from live.frame_sync import FrameAssembler, FrameSyncBuffer
from live.inspection_worker import InspectionTask, InspectionWorker
from live.result_writer import ResultWriter
from live.trigger_detector import TriggerDetector


class ConveyorSystem:
    def __init__(self):
        self._buffers = [
            FrameSyncBuffer(maxlen=config.FRAME_BUFFER_SIZE)
            for _ in config.CAMERA_INDICES
        ]
        self._cameras = [
            CameraThread(idx, config.CAMERA_RESOLUTION, config.CAMERA_FPS, buf)
            for idx, buf in zip(config.CAMERA_INDICES, self._buffers)
        ]
        self._assembler = FrameAssembler(
            self._buffers,
            window_ms=config.FRAME_SYNC_WINDOW_MS,
            min_variance=config.SHARPNESS_MIN_VARIANCE,
        )
        self._trigger = TriggerDetector(
            yolo_model_path=config.YOLO_TRIGGER_MODEL,
            roi_y_band=config.TRIGGER_ROI_Y_BAND,
            roi_x_center_band=config.TRIGGER_ROI_X_CENTER_BAND,
            confidence_threshold=config.TRIGGER_CONFIDENCE_THRESHOLD,
            check_every_n_frames=config.TRIGGER_CHECK_EVERY_N_FRAMES,
            min_gap_frames=config.TRIGGER_MIN_GAP_FRAMES,
        )
        self._inspection_queue = queue.Queue(maxsize=config.INSPECTION_QUEUE_MAX)
        self._writer = ResultWriter(config.DB_PATH, config.JSON_LOG_PATH)
        self._worker = InspectionWorker(self._inspection_queue, self._writer, config)

    def start(self, session_id):
        self._session_id = session_id
        self._writer.start_session(session_id, datetime.datetime.now().isoformat())
        self._worker.start()
        for cam in self._cameras:
            cam.start()
        print(f"\n[Conveyor] Session: {session_id}")
        print(f"[Conveyor] Cameras: {config.CAMERA_INDICES}")
        print(f"[Conveyor] Max products: {config.MAX_PRODUCTS}")
        print("[Conveyor] Waiting for models to load...\n")
        # Give the worker time to load models before products start arriving
        time.sleep(2)

    def run_session(self, max_products=None):
        if max_products is None:
            max_products = config.MAX_PRODUCTS

        trigger_cam_buf = self._buffers[config.TRIGGER_CAMERA_INDEX]
        trigger_cam = self._cameras[config.TRIGGER_CAMERA_INDEX]

        seq = 0
        frame_count = 0
        print(f"[Conveyor] Running — pass products under Camera-{config.TRIGGER_CAMERA_INDEX} to inspect.")
        print("[Conveyor] Press Ctrl+C to stop early.\n")

        h_res, w_res = config.CAMERA_RESOLUTION[1], config.CAMERA_RESOLUTION[0]
        x1_zone = int(config.TRIGGER_ROI_X_CENTER_BAND[0] * w_res)
        x2_zone = int(config.TRIGGER_ROI_X_CENTER_BAND[1] * w_res)
        y1_zone = int(config.TRIGGER_ROI_Y_BAND[0] * h_res)
        y2_zone = int(config.TRIGGER_ROI_Y_BAND[1] * h_res)

        last_fired_flash = 0  # timestamp of last trigger for green flash

        try:
            while seq < max_products:
                frame = trigger_cam_buf.get_closest(time.monotonic(), window_ms=100)
                if frame is None:
                    time.sleep(0.01)
                    continue

                fired = self._trigger.process_frame(frame, frame_count)
                frame_count += 1

                if fired:
                    trigger_ts = time.monotonic()
                    last_fired_flash = trigger_ts
                    seq += 1
                    frames = self._assembler.collect_snapshot(trigger_ts)
                    task = InspectionTask(
                        frames=frames,
                        trigger_ts=trigger_ts,
                        session_id=self._session_id,
                        seq_number=seq,
                    )
                    try:
                        self._inspection_queue.put_nowait(task)
                        print(f"[Conveyor] Product #{seq} queued (queue size: {self._inspection_queue.qsize()})")
                    except queue.Full:
                        print(f"[Conveyor] WARNING: Queue full — product #{seq} dropped. Slow down the belt.")

                # ── Preview window ─────────────────────────────────────────
                try:
                    preview = frame.copy()
                    flashing = (time.monotonic() - last_fired_flash) < 0.4
                    zone_color = (0, 255, 0) if flashing else (0, 200, 255)
                    cv2.rectangle(preview, (x1_zone, y1_zone), (x2_zone, y2_zone), zone_color, 2)
                    label = f"Products: {seq}  Queue: {self._inspection_queue.qsize()}"
                    cv2.putText(preview, label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(preview, "Move product into box to scan | Q to quit",
                                (10, h_res - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                    cv2.imshow("Conveyor Inspection", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[Conveyor] Quit by user.")
                        break
                except cv2.error:
                    # GUI not available — run headless, use Ctrl+C to stop
                    pass

                time.sleep(0.005)

        except KeyboardInterrupt:
            print("\n[Conveyor] Interrupted by user.")

        cv2.destroyAllWindows()

        print(f"\n[Conveyor] Session complete — {seq} products captured.")
        self._print_summary()

    def stop(self):
        for cam in self._cameras:
            cam.stop()
        # Wait for remaining inspections to finish
        print("[Conveyor] Waiting for inspection queue to drain...")
        self._inspection_queue.join()
        self._worker.stop()
        end_time = datetime.datetime.now().isoformat()
        self._writer.end_session(self._session_id, end_time)
        self._writer.close()
        print("[Conveyor] Shutdown complete.")

    def _print_summary(self):
        summary = self._writer.get_session_summary(self._session_id)
        if not summary:
            return
        print("\n" + "=" * 54)
        print("  SESSION SUMMARY")
        print("=" * 54)
        print(f"  Session ID       : {summary.get('session_id')}")
        print(f"  Total Products   : {summary.get('total_products', 0)}")
        print(f"  Barcode Found    : {summary.get('barcode_count', 0)}")
        print(f"  VLM Inspected    : {summary.get('vlm_count', 0)}")
        print(f"  Incomplete       : {summary.get('incomplete_count', 0)}")
        print("=" * 54)
