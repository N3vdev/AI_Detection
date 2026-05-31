import os
import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional
import cv2
import torch


@dataclass
class InspectionTask:
    frames: list           # list of np.ndarray | None, one per camera
    trigger_ts: float
    session_id: str
    seq_number: int
    product_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


class InspectionWorker(threading.Thread):
    def __init__(self, queue, result_writer, config, on_result=None, on_progress=None):
        super().__init__(daemon=True, name="InspectionWorker")
        self.queue = queue
        self.result_writer = result_writer
        self.config = config
        self._on_result   = on_result
        self._on_progress = on_progress
        self._stop_event  = threading.Event()
        self._ready_event = threading.Event()
        self._ai = None

    def wait_ready(self, timeout=300):
        """Block until models are fully loaded. Returns False if timeout."""
        return self._ready_event.wait(timeout=timeout)

    def _load_models(self):
        from src.detect import AIInspectionSystem
        if self._on_progress:
            self._on_progress("Loading AI models...")
        self._ai = AIInspectionSystem(
            barcode_model_path=self.config.BARCODE_DETECTOR_MODEL,
            qwen_model_id=self.config.QWEN_MODEL_ID,
            world_model_id=self.config.YOLO_WORLD_MODEL,
            florence2_model_id=self.config.FLORENCE2_MODEL_ID,
            debug=self.config.SAVE_DEBUG_SNAPSHOTS,
            on_progress=self._on_progress,
        )

    def run(self):
        self._load_models()
        self._ready_event.set()   # unblocks ConveyorSystem.start()
        print("[System] All models loaded. Ready to inspect.\n")

        while not self._stop_event.is_set():
            try:
                task = self.queue.get(timeout=1.0)
            except Exception:
                continue

            try:
                self._process(task)
            except Exception as e:
                print(f"[Worker] ERROR on product {task.product_id}: {e}")
                self.result_writer.write({
                    "product_id": task.product_id,
                    "session_id": task.session_id,
                    "seq_number": task.seq_number,
                    "trigger_time": _iso(task.trigger_ts),
                    "status": f"Error: {e}",
                    "barcode": None,
                })
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.queue.task_done()

    def _process(self, task: InspectionTask):
        t0 = time.time()
        print(f"\n[Worker] Product #{task.seq_number} ({task.product_id})")

        valid_frames = [(i, f) for i, f in enumerate(task.frames) if f is not None]
        if not valid_frames:
            print(f"[Worker] No frames for product {task.product_id} — skipping.")
            return

        # ── Fast-path: try barcode on all cameras in parallel before any disk I/O ──
        # pyzbar on 4 frames simultaneously ~15-30ms. If found, skip OCR+Qwen entirely.
        barcode = self._ai.quick_barcode_scan([f for _, f in valid_frames])
        if barcode:
            processing_ms = int((time.time() - t0) * 1000)
            print(f"[Worker] #{task.seq_number} BARCODE (fast): {barcode} ({processing_ms}ms)")
            snapshot_dir = os.path.join(self.config.SNAPSHOT_DIR, f"product_{task.product_id}")
            os.makedirs(snapshot_dir, exist_ok=True)
            for i, frame in valid_frames:
                cv2.imwrite(os.path.join(snapshot_dir, f"cam{i}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, self.config.SNAPSHOT_JPEG_QUALITY])
            fast_result = {
                "product_id":    task.product_id,
                "session_id":    task.session_id,
                "seq_number":    task.seq_number,
                "trigger_time":  _iso(task.trigger_ts),
                "processing_ms": processing_ms,
                "barcode":       barcode,
                "status":        "Complete (Barcode)",
                "snapshot_cam0": _snap(snapshot_dir, 0, task.frames),
                "snapshot_cam1": _snap(snapshot_dir, 1, task.frames),
                "snapshot_cam2": _snap(snapshot_dir, 2, task.frames),
                "snapshot_cam3": _snap(snapshot_dir, 3, task.frames),
            }
            self.result_writer.write(fast_result)
            if self._on_result:
                self._on_result(fast_result)
            return

        # ── Full pipeline: save frames to disk then run OCR + Qwen ────────────────
        snapshot_dir = os.path.join(self.config.SNAPSHOT_DIR, f"product_{task.product_id}")
        os.makedirs(snapshot_dir, exist_ok=True)

        saved_paths = []
        for i, frame in valid_frames:
            path = os.path.join(snapshot_dir, f"cam{i}.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.SNAPSHOT_JPEG_QUALITY])
            saved_paths.append(path)

        result = self._ai.inspect_product(saved_paths)

        processing_ms = int((time.time() - t0) * 1000)
        result.update({
            "product_id":   task.product_id,
            "session_id":   task.session_id,
            "seq_number":   task.seq_number,
            "trigger_time": _iso(task.trigger_ts),
            "processing_ms": processing_ms,
            "snapshot_cam0": _snap(snapshot_dir, 0, task.frames),
            "snapshot_cam1": _snap(snapshot_dir, 1, task.frames),
            "snapshot_cam2": _snap(snapshot_dir, 2, task.frames),
            "snapshot_cam3": _snap(snapshot_dir, 3, task.frames),
        })

        self.result_writer.write(result)
        if self._on_result:
            self._on_result(result)

        if result.get("barcode"):
            print(f"[Worker] #{task.seq_number} BARCODE: {result['barcode']} ({processing_ms}ms)")
        else:
            print(f"[Worker] #{task.seq_number} VLM: {result.get('brand','?')} / "
                  f"{result.get('product_name','?')} / "
                  f"exp={result.get('expiry_date','—')} ({processing_ms}ms)")

    def stop(self):
        self._stop_event.set()
        # Explicitly release model references so VRAM is freed before the next session loads
        if self._ai is not None:
            del self._ai
            self._ai = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _iso(monotonic_ts):
    import datetime
    wall = datetime.datetime.now() - datetime.timedelta(
        seconds=time.monotonic() - monotonic_ts
    )
    return wall.isoformat(timespec="milliseconds")


def _snap(snapshot_dir, cam_idx, frames):
    if cam_idx < len(frames) and frames[cam_idx] is not None:
        return os.path.join(snapshot_dir, f"cam{cam_idx}.jpg")
    return None
