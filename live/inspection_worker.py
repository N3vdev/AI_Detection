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
    def __init__(self, queue, result_writer, config):
        super().__init__(daemon=True, name="InspectionWorker")
        self.queue = queue
        self.result_writer = result_writer
        self.config = config
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._ai = None

    def wait_ready(self, timeout=300):
        """Block until models are fully loaded. Returns False if timeout."""
        return self._ready_event.wait(timeout=timeout)

    def _load_models(self):
        from src.detect import AIInspectionSystem
        print("[System] Loading AI models — this may take a minute on first run...")
        self._ai = AIInspectionSystem(
            barcode_model_path=self.config.BARCODE_DETECTOR_MODEL,
            ocr_model_path=self.config.DOTTED_OCR_MODEL,
            qwen_model_id=self.config.QWEN_MODEL_ID,
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

        snapshot_dir = os.path.join(self.config.SNAPSHOT_DIR, f"product_{task.product_id}")
        os.makedirs(snapshot_dir, exist_ok=True)

        saved_paths = []
        for i, frame in enumerate(task.frames):
            if frame is not None:
                path = os.path.join(snapshot_dir, f"cam{i}.jpg")
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.SNAPSHOT_JPEG_QUALITY])
                saved_paths.append(path)

        if not saved_paths:
            print(f"[Worker] No frames for product {task.product_id} — skipping.")
            return

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
        })

        self.result_writer.write(result)

        if result.get("barcode"):
            print(f"[Worker] #{task.seq_number} BARCODE: {result['barcode']} ({processing_ms}ms)")
        else:
            print(f"[Worker] #{task.seq_number} VLM: {result.get('brand','?')} / "
                  f"{result.get('product_name','?')} / "
                  f"exp={result.get('expiry_date','—')} ({processing_ms}ms)")

    def stop(self):
        self._stop_event.set()


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
