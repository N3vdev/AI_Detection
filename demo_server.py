"""
AI Product Inspector — Web Demo Server
=======================================
Install deps (one-time, separate from main requirements):
    pip install fastapi "uvicorn[standard]" python-multipart psutil pynvml

Run server:
    cd d:\\Projects\\AI_Detection
    python demo_server.py

Expose to internet (no WiFi required) — run in a second terminal:
    cloudflared tunnel --url http://localhost:8000
    Download cloudflared: https://github.com/cloudflare/cloudflared/releases/latest

Then share the https://xxxx.trycloudflare.com URL with anyone.
"""

import asyncio
import os
import sys
import time
import tempfile
from contextlib import asynccontextmanager
from typing import List

import psutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

# Seed CPU percent so first /stats call returns a real value not 0
psutil.cpu_percent(interval=None)

# GPU monitoring via pynvml (comes with CUDA drivers — no extra install needed)
_nvml_handle = None
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live import conveyor_config as config
from src.detect import AIInspectionSystem

_ai: AIInspectionSystem | None = None
_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_static")
_ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ai
    print("[Demo] Loading AI models — please wait...")
    _ai = AIInspectionSystem(
        barcode_model_path=config.BARCODE_DETECTOR_MODEL,
        qwen_model_id=config.QWEN_MODEL_ID,
        world_model_id=config.YOLO_WORLD_MODEL,
        debug=True,  # always save debug snapshots in demo mode
    )
    print("[Demo] Models loaded. Server ready at http://localhost:8000\n")
    yield


app = FastAPI(title="AI Product Inspector", lifespan=lifespan)


@app.get("/stats")
async def stats():
    cpu = round(psutil.cpu_percent(interval=None))
    gpu = 0
    if _nvml_handle is not None:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
            gpu = util.gpu
        except Exception:
            pass
    return {"cpu": cpu, "gpu": gpu}


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(_STATIC, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.post("/inspect")
async def inspect(images: List[UploadFile] = File(...)):
    if _ai is None:
        raise HTTPException(503, "Models not loaded yet — try again in a moment")
    if not images:
        raise HTTPException(400, "No images provided")

    paths = []
    try:
        for upload in images:
            ext = os.path.splitext(upload.filename or "")[1].lower()
            if ext not in _ALLOWED_EXT:
                ext = ".jpg"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(await upload.read())
                paths.append(tmp.name)

        t0 = time.time()
        result = await asyncio.to_thread(_ai.inspect_product, paths)
        result["processing_ms"] = int((time.time() - t0) * 1000)
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    result.pop("images", None)
    result.pop("dotted_label_text", None)

    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
