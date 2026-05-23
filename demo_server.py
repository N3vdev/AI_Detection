"""
AI Product Inspector — Web Demo Server
=======================================
Install deps (one-time, separate from main requirements):
    pip install fastapi "uvicorn[standard]" python-multipart

Run server:
    cd d:\\Projects\\AI_Detection
    python demo_server.py

Expose to internet (no WiFi required) — run in a second terminal:
    cloudflared tunnel --url http://localhost:8000
    Download cloudflared: https://github.com/cloudflare/cloudflared/releases/latest

Then share the https://xxxx.trycloudflare.com URL with anyone.
"""

import os
import sys
import time
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

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
        ocr_model_path=config.DOTTED_OCR_MODEL,
        qwen_model_id=config.QWEN_MODEL_ID,
    )
    print("[Demo] Models loaded. Server ready at http://localhost:8000\n")
    yield


app = FastAPI(title="AI Product Inspector", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(_STATIC, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.post("/inspect")
async def inspect(image: UploadFile = File(...)):
    if _ai is None:
        raise HTTPException(503, "Models not loaded yet — try again in a moment")

    ext = os.path.splitext(image.filename or "")[1].lower()
    if ext not in _ALLOWED_EXT:
        ext = ".jpg"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await image.read())
        path = tmp.name

    try:
        t0 = time.time()
        result = _ai.inspect_product([path])
        result["processing_ms"] = int((time.time() - t0) * 1000)
    finally:
        os.unlink(path)

    # Strip internal fields not useful to the client
    result.pop("images", None)
    result.pop("dotted_label_text", None)

    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
