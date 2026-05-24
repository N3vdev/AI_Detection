# AI Product Inspection System — Overview

## What It Does

An automated real-time inspection system for conveyor belt production lines. Products pass under cameras and the system instantly extracts:

- **Barcode value** (EAN-13, CODE-128, QR code, and others)
- **Brand name** — e.g. Parachute, Amul, Coca-Cola
- **Category** — Food / Drink / Snack / Skincare / Haircare / Medicine / Household
- **Expiry date** — any printed format
- **Manufacture / Packed date**
- **Batch number / Lot number**

---

## How It will work on the Conveyor Belt

### 1 — Product Detection (Trigger)

A lightweight YOLO model (**YOLO11n**) runs on the CPU and watches the trigger camera continuously at full belt speed. Every frame is scanned. When a product enters frame and is confirmed across three consecutive frames, a trigger fires.

### 2 — Camera Snapshot

The moment the trigger fires, the system captures a snapshot from all cameras simultaneously. 

### 3 — Barcode Fast Path

The system scans all captured frames in parallel for a barcode. This runs simultaneously across all cameras using pyzbar (ZBar library). Typical time: 15–30ms.

If any camera has a readable barcode, the result is written to the database immediately and the full AI pipeline never runs. Since most products on a production line have barcodes, this path handles the majority of products almost instantly.

### 4 — Full AI Pipeline (no barcode found)

For products where barcode scanning fails — damaged barcodes, unlabelled items, angle obstruction — the system runs a five-stage vision pipeline:

**Stage A — Barcode retry with preprocessing**
A custom barcode detection model (**YOLOv8s**, trained on barcode images) finds the barcode region and crops it. The crop is retried through six preprocessing variants (upscale, sharpen, adaptive threshold, Otsu threshold) to recover partially damaged or low-contrast barcodes.

**Stage B — Region detection**
**YOLO-World v2 (medium)** scans the image for label stickers, nutrition panels, brand logos, and ingredient blocks. Isolating these regions means later steps work on focused crops rather than the full frame — critical for reading small inkjet-printed dates at the edge of a jar or bottle.

**Stage C — Text reading (OCR)**
**EasyOCR** (CRAFT text detector + recognition network) locates and reads every text region on the label. After OCR runs, the bounding boxes of all detected text are used to compute a tight crop around the text-dense area. This crop is passed to the VLM in the next step — effectively zooming in on the label.

**Stage D — Vision Language Model**
**Qwen2.5-VL-3B-Instruct** (3 billion parameters, 4-bit quantized) receives up to four inputs: region crops from Stage B, the tight text-region crop from Stage C, and the sharpest full image. It extracts brand, product name, category, expiry date, manufacture date, and batch number in a single inference call.

**Stage E — Regex fallback**
A pattern-matching pass over the raw OCR text catches any date or batch fields the VLM missed. It recognises a comprehensive list of keywords (EXP, BBD, BEST BEFORE, MFD, MFG, PKD, DOM, and many more) across all common date formats. Only fills empty fields — never overrides the VLM.


## Inspection Queue (Under Development)

Products with barcodes are handled by the fast path and never queue — they complete in under 30ms. Products without barcodes are queued for the full pipeline (the VLM takes 5–15 seconds per product on GPU). The queue holds up to 10 slots; if the belt outruns the AI during a burst, the excess product is skipped with a log warning rather than crashing or consuming unbounded memory.

---

## Result Format if no barcode is found

```json
{
  "barcode": null,
  "brand": "Parachute",
  "product_name": "Men Advanced Aftershower Hair Cream",
  "product_category": "Skincare",
  "expiry_date": null,
  "manufacture_date": "02/26",
  "batch_number": "KK039-R",
  "status": "Complete (Vision)",
  "processing_ms": 8420
}
```

- **Complete (Barcode)** — barcode decoded in fast path
- **Complete (Vision)** — no barcode, brand extracted by VLM
- **Incomplete** — pipeline ran but could not extract brand or barcode

---

## Demo Web App *(for testing accuracy only)*

A browser-based UI for testing detection accuracy without the physical conveyor setup. Upload photos of a product from any phone or laptop and get the full inspection result. Runs the exact same AI pipeline as the conveyor system. Useful for validating accuracy on new product types before deploying on the line.

Can be shared over the internet using Cloudflare Tunnel — gives a public HTTPS URL with no port forwarding or account required.

---

## Recommended Hardware Requirements (Tested On)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8GB VRAM | 12GB |
| RAM | 16GB | 32GB |
| CPU | 6-core | 8-core+ for multi-camera |
| Storage | 20GB free | 50GB+ (snapshots grow) |
| OS | Windows 10/11 | Windows 11 |
| CUDA | 12.x | 12.6 |
| Python Version| 3.13 | 3.13 |

---

## Installation

### Prerequisites

Before running anything, make sure the following are installed on the machine:

1. **Python 3.13** — download from python.org. During install, tick **"Add Python to PATH"**. `pip` is included with Python — no separate install needed.
2. **CUDA 12.x drivers** — install the latest NVIDIA driver for the GPU. The CUDA toolkit is not needed separately; PyTorch bundles it.

---

### Step 1 — Extract the Files

Unzip the provided archive into any folder, for example `C:\AI_Inspector\`. The folder should contain the Python source files, the batch files, and `cloudflared.exe`.

---

### Step 2 — Install Python Dependencies

This happens automatically. On the first run, `run_demo.bat` detects that dependencies have not been installed yet and runs `pip install -r requirements.txt` before starting anything. Subsequent runs skip this step entirely.

If you prefer to install manually beforehand, open a terminal in the extracted folder and run:

```
pip install -r requirements.txt
```

PyTorch is pulled with CUDA 12.6 support automatically. First-time install takes a few minutes depending on internet speed.

On first run, the following model weights are downloaded automatically and cached locally:
- **Qwen2.5-VL-3B-Instruct** (~7 GB) — vision language model
- **EasyOCR CRAFT + recognition models** (~200 MB) — text detection and reading
- **YOLO-World v2 medium** (~200 MB) — region detection
- **YOLO11n** (~6 MB) — conveyor trigger model
- **YOLOv8s barcode detector** — custom model included in the zip, no download needed

After the first run everything is cached and no internet connection is needed.

---

### Step 3 — Run the Demo (Test Mode)

The zip includes a batch file called **`run_demo.bat`**. Double-click it or run it from a terminal.

It does two things:
1. Starts the AI server on your machine at `http://localhost:8000`
2. Starts a Cloudflare Tunnel using the included `cloudflared.exe`, which prints a public URL like `https://abc123.trycloudflare.com`

Share that URL with anyone on any device. They open it in a browser, upload photos of a product, and get the inspection result. The AI runs on your machine — they only see the web UI. No source code is shared.

Both the server and the tunnel stay running until you close the terminal window.

> **Note:** The public URL changes every time you restart the tunnel. To keep a fixed URL, a free Cloudflare account with a named tunnel can be configured separately.

---

### Conveyor Belt Live Feed

The live conveyor mode — real-time camera feed, automatic trigger, and database logging — will be released later once the physical camera setup and belt configuration have been finalised and fully tested. Camera sources, resolution, and trigger thresholds will be set in the configuration file included with that release.

---

## Python Dependencies

All dependencies are listed in `requirements.txt` included in the zip. Run `pip install -r requirements.txt` to install everything in one step.

| Library | Purpose |
|---------|---------|
| `torch`, `torchvision` | Deep learning framework (CUDA 12.6 build) |
| `transformers` | Loads and runs Qwen2.5-VL vision language model |
| `accelerate`, `bitsandbytes` | 4-bit quantization to reduce VRAM usage |
| `ultralytics` | Runs YOLO models (barcode detector, YOLO-World, YOLO11n) |
| `easyocr` | Text detection and reading (CRAFT + recognition network) |
| `opencv-contrib-python` | Image processing and fallback barcode decoder |
| `pyzbar` | Primary barcode decoder (ZBar library) |
| `Pillow`, `numpy` | Image loading and array operations |
| `fastapi`, `uvicorn` | Web server for the demo app |
| `python-multipart` | Handles image file uploads in the demo |
| `huggingface-hub` | Downloads and caches model weights from HuggingFace |
| `pyyaml` | Configuration file parsing |


