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
            CameraThread(idx, config.CAMERA_RESOLUTION, config.CAMERA_FPS, buf,
                         rotate=getattr(config, 'CAMERA_ROTATE', None))
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

        # Start worker thread — models load in background
        self._worker.start()

        # Block until ALL models are loaded before opening cameras
        print("[System] Loading AI models, please wait...")
        if not self._worker.wait_ready(timeout=300):
            raise RuntimeError("[System] Models failed to load within 5 minutes.")

        # Now start cameras
        for cam in self._cameras:
            cam.start()

        # Give cameras a moment to produce their first frames
        time.sleep(1.0)
        print(f"\n[Conveyor] Session : {session_id}")
        print(f"[Conveyor] Cameras : {config.CAMERA_INDICES}")
        print(f"[Conveyor] Max     : {config.MAX_PRODUCTS} products\n")

    def run_session(self, max_products=None):
        if max_products is None:
            max_products = config.MAX_PRODUCTS

        trigger_buf = self._buffers[config.TRIGGER_CAMERA_INDEX]

        seq = 0
        frame_count = 0
        last_fired_ts = 0

        pw = getattr(config, 'PREVIEW_WIDTH', 360)
        ph = getattr(config, 'PREVIEW_HEIGHT', 640)
        cv2.namedWindow("Conveyor Inspection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Conveyor Inspection", pw, ph)

        print("[Conveyor] Running — place products in the camera view.")
        print("[Conveyor] Press Q in preview window or Ctrl+C to stop.\n")

        try:
            while seq < max_products:
                frame = trigger_buf.get_closest(time.monotonic(), window_ms=100)
                if frame is None:
                    time.sleep(0.005)
                    continue

                fired = self._trigger.process_frame(frame, frame_count)
                frame_count += 1

                if fired:
                    trigger_ts = time.monotonic()
                    last_fired_ts = trigger_ts
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
                        print(f"[Conveyor] ► Product #{seq} detected — queued for inspection")
                    except queue.Full:
                        print(f"[Conveyor] WARNING: Queue full — product #{seq} skipped.")

                # ── Preview window ──────────────────────────────────────────
                try:
                    preview = cv2.resize(frame, (pw, ph))
                    flashing = (time.monotonic() - last_fired_ts) < 0.5
                    zone_color = (0, 255, 0) if flashing else (0, 200, 255)

                    # Flash border on trigger
                    if flashing:
                        cv2.rectangle(preview, (0, 0), (pw - 1, ph - 1), (0, 255, 0), 4)

                    status = "SCANNING" if flashing else ("ARMED" if not self._trigger._triggered else "REMOVE PRODUCT")
                    cv2.putText(preview, f"#{seq}  Q:{self._inspection_queue.qsize()}",
                                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(preview, status,
                                (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)
                    cv2.putText(preview, "Q = quit",
                                (8, ph - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

                    cv2.imshow("Conveyor Inspection", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[Conveyor] Quit by user.")
                        break
                except cv2.error:
                    pass  # headless fallback

                time.sleep(0.005)

        except KeyboardInterrupt:
            print("\n[Conveyor] Interrupted.")

        cv2.destroyAllWindows()
        print(f"\n[Conveyor] Session complete — {seq} products captured.")
        self._print_summary()

    def stop(self):
        for cam in self._cameras:
            cam.stop()
        print("[Conveyor] Draining inspection queue...")
        self._inspection_queue.join()
        self._worker.stop()
        self._writer.end_session(self._session_id, datetime.datetime.now().isoformat())
        self._writer.close()
        print("[Conveyor] Shutdown complete.")

    def _print_summary(self):
        summary = self._writer.get_session_summary(self._session_id)
        if not summary:
            return
        print("\n" + "=" * 54)
        print("  SESSION SUMMARY")
        print("=" * 54)
        print(f"  Session          : {summary.get('session_id')}")
        print(f"  Total Products   : {summary.get('total_products', 0)}")
        print(f"  Barcode Found    : {summary.get('barcode_count', 0)}")
        print(f"  VLM Inspected    : {summary.get('vlm_count', 0)}")
        print(f"  Incomplete       : {summary.get('incomplete_count', 0)}")
        print("=" * 54)
