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
    def __init__(self, camera_indices=None, on_trigger=None, on_result=None, on_progress=None):
        self._on_trigger = on_trigger  # callable(cam_idx) — fired when product detected
        self._on_result  = on_result   # callable(result_dict) — fired after inspection
        sources = camera_indices if camera_indices is not None else config.CAMERA_INDICES
        # One buffer per UI slot — None-source slots just stay empty.
        # This preserves slot→buffer index mapping so cam 4 stays in widget 4.
        self._buffers = [
            FrameSyncBuffer(maxlen=config.FRAME_BUFFER_SIZE)
            for _ in sources
        ]
        self._cameras = [
            CameraThread(src, config.CAMERA_RESOLUTION, config.CAMERA_FPS, buf,
                         rotate=getattr(config, 'CAMERA_ROTATE', None))
            for src, buf in zip(sources, self._buffers)
            if src is not None
        ]
        # Trigger buffer: use the configured slot if it's active, else the first active slot
        _active = [i for i, s in enumerate(sources) if s is not None]
        _trig_pref = config.TRIGGER_CAMERA_INDEX
        self._trigger_slot = _trig_pref if _trig_pref in _active else (_active[0] if _active else 0)
        self._assembler = FrameAssembler(
            self._buffers,
            window_ms=config.FRAME_SYNC_WINDOW_MS,
            min_variance=config.SHARPNESS_MIN_VARIANCE,
        )
        self._trigger = TriggerDetector(
            yolo_model_path=config.YOLO_TRIGGER_MODEL,
            confidence_threshold=config.TRIGGER_CONFIDENCE_THRESHOLD,
            min_box_area=config.TRIGGER_MIN_BOX_AREA,
            enter_frames=config.TRIGGER_ENTER_FRAMES,
            leave_frames=config.TRIGGER_LEAVE_FRAMES,
            classes=getattr(config, 'TRIGGER_CLASSES', None),
            trigger_line_y=getattr(config, 'TRIGGER_LINE_Y', None),
        )
        self._inspection_queue = queue.Queue(maxsize=config.INSPECTION_QUEUE_MAX)
        self._writer = ResultWriter(config.DB_PATH, config.JSON_LOG_PATH)
        self._worker = InspectionWorker(self._inspection_queue, self._writer, config,
                                        on_result=self._on_result,
                                        on_progress=on_progress)
        self.auto_trigger = getattr(config, 'TRIGGER_AUTO', True)
        self._seq = 0
        self._stop_requested = False

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

        trigger_buf = self._buffers[self._trigger_slot]

        frame_count = 0
        last_fired_ts = 0

        pw = getattr(config, 'PREVIEW_WIDTH', 360)
        ph = getattr(config, 'PREVIEW_HEIGHT', 640)
        if self._on_trigger is None:
            cv2.namedWindow("Conveyor Inspection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Conveyor Inspection", pw, ph)
            print("[Conveyor] Running — place products in the camera view.")
            print("[Conveyor] Press Q in preview window or Ctrl+C to stop.\n")
        else:
            print("[Conveyor] Running — UI mode active.\n")

        _YOLO_INTERVAL = 1.0 / 15   # cap trigger YOLO at 15 fps — saves ~60% CPU
        _last_yolo = 0.0

        try:
            while self._seq < max_products and not self._stop_requested:
                frame = trigger_buf.get_closest(time.monotonic(), window_ms=100)
                if frame is None:
                    time.sleep(0.005)
                    continue

                now = time.monotonic()
                if now - _last_yolo < _YOLO_INTERVAL:
                    time.sleep(0.005)
                    continue
                _last_yolo = now

                if self.auto_trigger:
                    fired = self._trigger.process_frame(frame)
                else:
                    self._trigger.last_boxes = []
                    fired = False
                frame_count += 1

                if fired:
                    trigger_ts = time.monotonic()
                    last_fired_ts = trigger_ts
                    self._seq += 1
                    seq = self._seq
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
                    if self._on_trigger:
                        self._on_trigger(config.TRIGGER_CAMERA_INDEX)

                # ── Preview window (skipped when UI callbacks are registered) ──
                if self._on_trigger is None:
                    try:
                        preview = cv2.resize(frame, (pw, ph))
                        flashing = (time.monotonic() - last_fired_ts) < 0.5

                        # Draw trigger line when in line-crossing mode
                        if self._trigger.last_line_y is not None:
                            ly = int(self._trigger.last_line_y * ph)
                            line_color = (0, 255, 0) if flashing else (60, 60, 60)
                            cv2.line(preview, (0, ly), (pw, ly), line_color, 1)

                        for (x1n, y1n, x2n, y2n) in self._trigger.last_boxes:
                            bx1 = int(x1n * pw)
                            by1 = int(y1n * ph)
                            bx2 = int(x2n * pw)
                            by2 = int(y2n * ph)
                            box_color = (0, 255, 0) if flashing else (0, 200, 255)
                            cv2.rectangle(preview, (bx1, by1), (bx2, by2), box_color, 2)
                            label = "SCANNING..." if flashing else "PRODUCT"
                            cv2.putText(preview, label, (bx1, max(by1 - 8, 14)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

                        if flashing:
                            cv2.rectangle(preview, (0, 0), (pw - 1, ph - 1), (0, 255, 0), 4)

                        if flashing:
                            status, s_color = "SCANNING", (0, 255, 0)
                        elif self._trigger.product_in_frame:
                            status, s_color = "IN FRAME", (0, 200, 255)
                        else:
                            status, s_color = "READY", (200, 200, 200)

                        cv2.putText(preview, f"#{self._seq}  Q:{self._inspection_queue.qsize()}",
                                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(preview, status,
                                    (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_color, 2)
                        cv2.putText(preview, "Q = quit",
                                    (8, ph - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

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

        if self._on_trigger is None:
            cv2.destroyAllWindows()
        print(f"\n[Conveyor] Session complete — {self._seq} products captured.")
        self._print_summary()

    def request_stop(self):
        self._stop_requested = True

    def manual_snap(self):
        """Capture frames from all cameras right now and queue for inspection. Thread-safe."""
        self._seq += 1
        seq = self._seq
        trigger_ts = time.monotonic()
        frames = self._assembler.collect_snapshot(trigger_ts)
        task = InspectionTask(
            frames=frames,
            trigger_ts=trigger_ts,
            session_id=self._session_id,
            seq_number=seq,
        )
        try:
            self._inspection_queue.put_nowait(task)
            print(f"[Conveyor] ► Manual snap #{seq} — queued for inspection")
            return True
        except queue.Full:
            print(f"[Conveyor] WARNING: Queue full — manual snap #{seq} skipped.")
            self._seq -= 1
            return False

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
