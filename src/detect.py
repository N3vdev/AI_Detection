import cv2
import re
import os
import time
import json
import datetime
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

import easyocr

try:
    from pyzbar.pyzbar import decode as pyzbar_decode, ZBarSymbol
    # Only scan formats found on retail products — excludes PDF417 (boarding passes/IDs)
    # which causes noisy assertion warnings on images with vaguely PDF417-like patterns.
    # Build defensively: some ZBar Windows builds omit DATAMATRIX / CODE93 / I25.
    _PYZBAR_SYMBOLS = [
        getattr(ZBarSymbol, s) for s in
        ('EAN13', 'EAN8', 'UPCA', 'UPCE', 'CODE128', 'CODE39', 'CODE93', 'QRCODE', 'I25')
        if hasattr(ZBarSymbol, s)
    ]
except ImportError:
    pyzbar_decode = None
    _PYZBAR_SYMBOLS = None

# Qwen2.5-VL (newer) with fallback to Qwen2-VL
try:
    from transformers import Qwen2_5_VLProcessor as _VLProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration as _VLModel
except ImportError:
    from transformers import Qwen2VLProcessor as _VLProcessor
    from transformers import Qwen2VLForConditionalGeneration as _VLModel


# Month name — full and abbreviated, all caps (text is uppercased before matching)
_MON = (
    r'(?:JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY'
    r'|JUN(?:E)?|JUL(?:Y)?|AUG(?:UST)?|SEP(?:T(?:EMBER)?)?'
    r'|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)'
)

# Expiry keywords
_EXP_KW = (
    r'(?:'
    r'EXP(?:IRY|IRES?|IRE|IRATION(?:\s*DATE)?)?'   # EXP, EXPIRY, EXPIRES, EXPIRATION DATE
    r'|BB\.?[DE]?'                                   # BB, BBD, BBE (Best Before End)
    r'|B\.B\.?[DE]?'                                 # B.B, B.B.D, B.B.E
    r'|BEST\s*BEFORE\s*(?:END|DATE)?'               # BEST BEFORE, BEST BEFORE END, BEST BEFORE DATE
    r'|BEST\s*(?:BY|USED?\s*BY)'                    # BEST BY, BEST USED BY
    r'|USE\s*(?:BY|BEFORE)(?:\s*DATE)?'              # USE BY, USE BEFORE, USE BY DATE
    r'|USED?\s*BY'                                   # USED BY
    r'|CONSUME\s*(?:BY|BEFORE)'                      # CONSUME BY, CONSUME BEFORE
    r'|SELL\s*(?:BY|BEFORE)(?:\s*DATE)?'             # SELL BY, SELL BEFORE, SELL BY DATE
    r')'
)

# Manufacturing / packed keywords
_MFG_KW = (
    r'(?:'
    r'MF[DG](?:\.?\s*(?:DATE|ON))?'                 # MFD, MFG, MFD DATE, MFG ON
    r'|MANUFACTURED?\s*(?:ON|DATE)?'                 # MANUFACTURE, MANUFACTURED, MANUFACTURED ON
    r'|MANUFACTURING\s*DATE'                         # MANUFACTURING DATE
    r'|DATE\s*OF\s*MANU(?:FACTURE)?'                 # DATE OF MANUFACTURE
    r'|DOM'                                          # DOM (date of manufacture)
    r'|D\.O\.M\.?'                                   # D.O.M
    r'|PROD(?:UCTION)?\s*(?:DATE|ON)?'               # PROD DATE, PRODUCTION DATE, PROD ON
    r'|PACKED?\s*(?:ON|DATE)?'                       # PACK, PACKED, PACK ON, PACK DATE
    r'|PACKING\s*DATE'                               # PACKING DATE
    r'|PKD\.?\s*(?:ON)?'                             # PKD, PKD ON
    r'|PACKAGED\s*ON'                                # PACKAGED ON
    r'|PACKAGING\s*DATE'                             # PACKAGING DATE
    r'|MADE\s*ON'                                    # MADE ON
    r')'
)

_KW_SEP = r'[\s:\.]*'   # gap between keyword and date value

_DATE_PATTERNS = [
    # ── Manufacturing ──────────────────────────────────────────────────────────
    # dd/mm/yy  dd/mm/yyyy  dd-mm-yy  dd.mm.yyyy
    (_MFG_KW + _KW_SEP + r'(\d{1,2}[\s\/\-\.]\d{1,2}[\s\/\-\.]\d{2,4})',           'manufacture_date'),
    # dd/mmm/yy  dd-JAN-2025  dd mmm yyyy
    (_MFG_KW + _KW_SEP + r'(\d{1,2}[\s\/\-\.]' + _MON + r'[\s\/\-\.]\d{2,4})',     'manufacture_date'),
    # mmm/yyyy  mmm-yy  JAN 2025
    (_MFG_KW + _KW_SEP + r'(' + _MON + r'[\s\/\-\.]\d{2,4})',                       'manufacture_date'),
    # mm/yyyy  mm-yyyy  (month-year only)
    (_MFG_KW + _KW_SEP + r'(\d{1,2}[\s\/\-\.]\d{4})',                               'manufacture_date'),
    # yyyy-mm-dd  yyyy/mm/dd
    (_MFG_KW + _KW_SEP + r'(\d{4}[\s\/\-\.]\d{1,2}[\s\/\-\.]\d{1,2})',             'manufacture_date'),
    # compact ddmmyyyy  (inkjet / dot-matrix common)
    (_MFG_KW + _KW_SEP + r'(\d{2})(\d{2})(\d{4})',                                  'manufacture_date'),
    # compact yyyymmdd
    (_MFG_KW + _KW_SEP + r'(\d{4})(\d{2})(\d{2})',                                  'manufacture_date'),

    # ── Expiry ─────────────────────────────────────────────────────────────────
    # dd/mm/yy  dd/mm/yyyy
    (_EXP_KW + _KW_SEP + r'(\d{1,2}[\s\/\-\.]\d{1,2}[\s\/\-\.]\d{2,4})',           'expiry_date'),
    # dd/mmm/yy  dd-JAN-2025
    (_EXP_KW + _KW_SEP + r'(\d{1,2}[\s\/\-\.]' + _MON + r'[\s\/\-\.]\d{2,4})',     'expiry_date'),
    # mmm/yyyy  JAN 2025
    (_EXP_KW + _KW_SEP + r'(' + _MON + r'[\s\/\-\.]\d{2,4})',                       'expiry_date'),
    # mm/yyyy
    (_EXP_KW + _KW_SEP + r'(\d{1,2}[\s\/\-\.]\d{4})',                               'expiry_date'),
    # yyyy-mm-dd
    (_EXP_KW + _KW_SEP + r'(\d{4}[\s\/\-\.]\d{1,2}[\s\/\-\.]\d{1,2})',             'expiry_date'),

    # ── Generic fallback (no label — most unlabeled dates are expiry) ──────────
    # dd/mmm/yyyy or dd-JAN-25 standalone
    (r'\b(\d{1,2}[\s\/\-\.]' + _MON + r'[\s\/\-\.]\d{2,4})\b',                     'expiry_date'),
    # dd/mm/yyyy or dd-mm-yy standalone
    (r'\b(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{2,4})\b',                                   'expiry_date'),
]

# YOLO-World classes — concrete VISUAL objects YOLO-World can reliably detect.
# Avoid semantic/abstract concepts ("expiry date") — YOLO can't see meaning,
# only visual patterns. Detect the label/sticker region, then let Qwen read it.
_WORLD_CLASSES = [
    "product label",          # main printed label on packaging
    "label sticker",          # adhesive sticker label
    "nutrition facts panel",  # nutrition information table
    "barcode",                # 1D linear barcode
    "QR code",                # 2D QR / DataMatrix code
    "brand logo",             # company logo or brand name graphic
    "ingredient list",        # block of small ingredient text
]

_BATCH_PATTERN = re.compile(
    r'(?:BATCH\s*(?:NO\.?|NUMBER|CODE)?|LOT\s*(?:NO\.?|NUMBER)?)[\s:\.]*([A-Z0-9]{5,20})',
    re.IGNORECASE,
)

_PROMPT = (
    "You are a product inspection AI on a factory conveyor belt. "
    "You are given one or more images of the SAME physical product from different angles. "
    "Carefully examine ALL images and extract the following. "
    "Pay special attention to small inkjet-printed or dot-matrix text near edges — "
    "these are typically the manufacture/packed/expiry dates and may appear faint or dotted.\n\n"
    "Brand: [exact brand or company name printed on the product]\n"
    "Product: [full product name including variant, flavour, or size]\n"
    "Category: [Food / Drink / Snack / Skincare / Haircare / Medicine / Household]\n"
    "Expiry: [date after EXP / Expiry / Best Before / Use By / BB / BBD / Consume By — any format including dd/mm/yy, dd/mmm/yyyy, mm/yyyy — else NONE]\n"
    "MFG: [date after MFG / MFD / Mfg Date / Manufactured / Packed On / PKD / Packing Date / Production Date / Made On / DOM — else NONE]\n"
    "Batch: [alphanumeric code after Batch No / Lot No — else NONE]\n\n"
    "Rules:\n"
    "- Output ONLY the six fields above, one per line, no extra text.\n"
    "- If a date appears in multiple formats, output the most complete one.\n"
    "- Never confuse MFG and Expiry — Expiry is always the later date.\n"
    "- If unsure, output NONE rather than guessing."
)


class AIInspectionSystem:
    def __init__(
        self,
        barcode_model_path='models/barcode_detector.pt',
        qwen_model_id='Qwen/Qwen2.5-VL-3B-Instruct',
        world_model_id='yolov8m-worldv2.pt',
        debug=True,
    ):
        """
        debug=True  → save per-step snapshots to debug_snapshots/ (demo / dev use)
        debug=False → skip all debug writes (production conveyor — saves disk + minor speedup)
        """
        self._debug = debug
        print("[System] Initializing AI Inspection Pipeline...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[System] Device: {self.device.upper()}")

        # ── Barcode YOLO ───────────────────────────────────────────────────────
        self.barcode_detector = None
        if os.path.exists(barcode_model_path):
            self.barcode_detector = YOLO(barcode_model_path)
            print("[System] Barcode detector loaded.")
        else:
            print(f"[Warning] Barcode model not found: {barcode_model_path}")

        # ── Qwen2.5-VL ────────────────────────────────────────────────────────
        print(f"[System] Loading {qwen_model_id}...")
        self.qwen_processor = _VLProcessor.from_pretrained(qwen_model_id)

        load_kwargs = dict(device_map="auto")

        if self.device == "cuda":
            free_vram, _ = torch.cuda.mem_get_info(0)
            free_gb = free_vram / 1024 ** 3
            print(f"[System] Free VRAM: {free_gb:.1f} GB")
            if free_gb < 12.0:
                # 4-bit quantization — fits in ~5 GB, minor accuracy loss
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    print("[System] 4-bit quantization enabled (low VRAM mode).")
                except ImportError:
                    print("[Warning] bitsandbytes not installed — loading in float16 (may OOM).")
                    load_kwargs["torch_dtype"] = torch.float16
            else:
                load_kwargs["torch_dtype"] = torch.float16
                print("[System] Loading in float16.")
        else:
            load_kwargs["torch_dtype"] = torch.float32

        try:
            import flash_attn  # noqa
            load_kwargs["attn_implementation"] = "flash_attention_2"
            print("[System] Flash Attention 2 enabled.")
        except ImportError:
            pass

        self.qwen_model = _VLModel.from_pretrained(qwen_model_id, **load_kwargs).eval()
        print(f"[System] {qwen_model_id} loaded.")

        # 640 patches per image keeps small text (dates, batch numbers) legible.
        self.qwen_processor.image_processor.max_pixels = 640 * 28 * 28
        self.qwen_processor.image_processor.min_pixels = 4 * 28 * 28

        # ── RapidOCR — PP-OCR models via ONNX Runtime (no PaddlePaddle executor)
        print("[System] Loading EasyOCR...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'), verbose=False)
        self.ocr_ready = True
        print("[System] EasyOCR loaded.")

        torch.set_num_threads(os.cpu_count() or 4)

        # ── YOLO-World open-vocabulary region detector ─────────────────────────
        self.world_detector = None
        if world_model_id:
            try:
                from ultralytics import YOLOWorld
                print(f"[System] Loading YOLO-World ({world_model_id})...")
                self.world_detector = YOLOWorld(world_model_id)
                self.world_detector.set_classes(_WORLD_CLASSES)
                print("[System] YOLO-World loaded.")
            except Exception as e:
                print(f"[Warning] YOLO-World failed to load: {e}")

        # ── Debug snapshot directory ───────────────────────────────────────────
        self._debug_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'debug_snapshots'
        )
        os.makedirs(self._debug_dir, exist_ok=True)

        print("[System] Ready.\n")

    # ── Qwen2.5-VL — single call with ALL images ───────────────────────────────

    @staticmethod
    def _shrink_pil(pil_img, max_side=640):
        """Downscale PIL image so its longest side ≤ max_side before tokenisation."""
        w, h = pil_img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return pil_img

    def _qwen_extract(self, pil_images):
        content = []
        for img in pil_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": _PROMPT})

        messages = [{"role": "user", "content": content}]
        text = self.qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        t0 = time.time()
        print(f"[Qwen] Tokenizing {len(pil_images)} image(s)...")
        inputs = self.qwen_processor(
            text=[text], images=pil_images, return_tensors="pt"
        ).to(self.device)
        n_tokens = inputs.input_ids.shape[1]
        print(f"[Qwen] Input tokens: {n_tokens} — running inference...")

        with torch.inference_mode():
            output_ids = self.qwen_model.generate(
                **inputs, max_new_tokens=64, do_sample=False
            )

        elapsed = time.time() - t0
        response = self.qwen_processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0].strip()

        print(f"[Qwen] Done in {elapsed:.1f}s\nResponse:\n{response}\n")
        return self._parse_response(response)

    def _parse_response(self, response):
        result = {}
        for line in response.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            value = value.strip()
            key = key.strip().lower()
            if not value or value.upper() == "NONE":
                continue
            if key == "brand":      result["brand"] = value
            elif key == "product":  result["product_name"] = value
            elif key == "category": result["product_category"] = value
            elif key == "expiry":   result["expiry_date"] = value
            elif key == "mfg":      result["manufacture_date"] = value
            elif key == "batch":    result["batch_number"] = value
        return result

    # ── EasyOCR text reader ────────────────────────────────────────────────────

    def _read_paddleocr(self, img_bgr, min_conf=0.25):
        try:
            self._last_ocr_raw = self.ocr_reader.readtext(
                img_bgr, detail=1, paragraph=False, min_size=5
            )
        except Exception as e:
            print(f"[OCR] EasyOCR failed: {e}")
            self._last_ocr_raw = []
            return ''
        return ' '.join(text for (_, text, conf) in self._last_ocr_raw if conf > min_conf)

    def _dbg_ocr_overlay(self, img_bgr):
        """Draw EasyOCR detections from last _read_paddleocr call onto a copy."""
        vis = img_bgr.copy()
        for (bbox, text, conf) in getattr(self, '_last_ocr_raw', []):
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            cv2.putText(vis, f"{text} {conf:.2f}", tuple(pts[0].tolist()),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return vis

    # ── Barcode helpers ────────────────────────────────────────────────────────

    def _decode_crop(self, crop):
        if crop is None or crop.size == 0:
            return None

        def _try(img):
            if pyzbar_decode:
                try:
                    for r in pyzbar_decode(img, symbols=_PYZBAR_SYMBOLS):
                        v = r.data.decode("utf-8")
                        if v:
                            return v
                except Exception:
                    pass
            try:
                det = cv2.barcode.BarcodeDetector()
                ok, decoded, _, _ = det.detectAndDecodeMulti(img)
                if ok:
                    for v in decoded:
                        if v:
                            return v
            except AttributeError:
                pass
            return None

        # 1. Color crop as-is
        v = _try(crop)
        if v: return v

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

        # 2. 2× upscale — helps when barcode is small in the frame
        v = _try(cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR))
        if v: return v

        # 3. 3× upscale
        v = _try(cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LINEAR))
        if v: return v

        # 4. Sharpened — recovers slightly blurry barcodes
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharp = cv2.filter2D(gray, -1, kernel)
        v = _try(cv2.resize(sharp, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR))
        if v: return v

        # 5. Adaptive threshold — handles curved surfaces / uneven lighting
        thresh_a = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        v = _try(cv2.resize(thresh_a, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR))
        if v: return v

        # 6. Otsu threshold — handles low-contrast prints
        _, thresh_o = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return _try(cv2.resize(thresh_o, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR))

    def _padded_crop(self, img, xyxy, pad=0.15):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = xyxy
        pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
        return img[max(0, y1 - ph):min(h, y2 + ph), max(0, x1 - pw):min(w, x2 + pw)]

    # ── Date / batch extraction ────────────────────────────────────────────────

    def _extract_dates(self, text):
        found = {}
        upper = text.upper()
        for pattern, key in _DATE_PATTERNS:
            if key not in found:
                m = re.search(pattern, upper)
                if m:
                    if m.lastindex and m.lastindex >= 3:
                        found[key] = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
                    else:
                        found[key] = m.group(1).strip()
        if "batch_number" not in found:
            m = _BATCH_PATTERN.search(upper)
            if m:
                found["batch_number"] = m.group(1)
        return found

    # ── YOLO-World region detection ────────────────────────────────────────────

    def _detect_regions(self, img_bgr, conf=0.08):
        """
        Run YOLO-World on img_bgr and return a dict of class_name → crop (numpy BGR).
        Only keeps the highest-confidence detection per class.
        Returns {} if world_detector not loaded or nothing found.
        """
        if not self.world_detector:
            return {}
        results = self.world_detector(img_bgr, verbose=False, conf=conf)
        best = {}  # class_name → (score, crop)
        for det in results:
            for box in det.boxes:
                cls_id = int(box.cls[0])
                name = _WORLD_CLASSES[cls_id]
                score = float(box.conf[0])
                if name in best and score <= best[name][0]:
                    continue
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                crop = self._padded_crop(img_bgr, xyxy, pad=0.05)
                if crop is not None and crop.size > 0:
                    best[name] = (score, crop)
        return {name: crop for name, (_, crop) in best.items()}

    # ── Main pipeline ──────────────────────────────────────────────────────────

    def inspect_product(self, image_paths):
        """
        Inspect one or more images of the same product.
        All images are passed to Qwen2.5-VL in a single call — it sees every
        angle at once and extracts brand, product, dates from whichever image
        has them most clearly.

        Usage:
            system.inspect_product("front.jpg")
            system.inspect_product(["front.jpg", "back.jpg", "side.jpg"])
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        result = {
            "images":            [os.path.basename(p) for p in image_paths],
            "barcode":           None,
            "brand":             None,
            "product_name":      None,
            "product_category":  None,
            "expiry_date":       None,
            "manufacture_date":  None,
            "batch_number":      None,
            "dotted_label_text": None,
            "status":            "Incomplete",
        }

        loaded = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"[Warning] Cannot load: {path}")
            else:
                loaded.append((path, img))

        if not loaded:
            result["status"] = "Error: no images loaded"
            return result

        # ── Debug run folder ───────────────────────────────────────────────────
        _ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        if self._debug:
            _rdir = os.path.join(self._debug_dir, _ts)
            os.makedirs(_rdir, exist_ok=True)
            def _dbg(name, img):
                path = os.path.join(_rdir, name)
                if isinstance(img, np.ndarray):
                    cv2.imwrite(path, img)
                else:
                    img.save(path)
        else:
            _rdir = None
            def _dbg(name, img):
                pass  # no-op in production

        # ── Phase 1: Barcode ───────────────────────────────────────────────────
        print(f"[Phase 1] Scanning {len(loaded)} image(s) for barcode...")

        # Pass 1 — direct decode on full image (no YOLO needed).
        # Cap to 1920px so pyzbar stays fast on high-res phone photos;
        # _decode_crop also retries on 2× grayscale internally.
        for path, img in loaded:
            h, w = img.shape[:2]
            long_side = max(h, w)
            if long_side > 1920:
                s = 1920 / long_side
                img_scan = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            else:
                img_scan = img
            value = self._decode_crop(img_scan)
            if value:
                result["barcode"] = value
                result["status"] = "Complete (Barcode)"
                print(f"[Phase 1] Barcode (direct): {value}")
                return result

        # Pass 2 — YOLO region proposal → padded crop → decode.
        # Also saves a debug image with all detected barcode boxes drawn.
        for i, (path, img) in enumerate(loaded):
            if self.barcode_detector:
                vis = img.copy()
                any_box = False
                for det in self.barcode_detector(img, verbose=False):
                    for box in det.boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0])
                        cv2.rectangle(vis, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 3)
                        cv2.putText(vis, f"barcode {conf:.2f}", (xyxy[0], max(0, xyxy[1] - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        any_box = True
                        value = self._decode_crop(self._padded_crop(img, xyxy))
                        if value:
                            cv2.putText(vis, value, (xyxy[0], xyxy[3] + 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                            _dbg(f'2_barcode_yolo_{i}_DECODED.jpg', vis)
                            result["barcode"] = value
                            result["status"] = "Complete (Barcode)"
                            print(f"[Phase 1] Barcode (YOLO crop): {value}")
                            return result
                label = 'boxes' if any_box else 'no_detection'
                _dbg(f'2_barcode_yolo_{i}_{label}.jpg', vis)

        print("[Phase 1] No barcode — moving to Phase 2.")

        # ── YOLO-World: detect label/date/logo regions ─────────────────────────
        # Runs on all images, keeps best-confidence crop per class.
        # Saves a debug image with all boxes drawn + individual crops.
        regions = {}  # class_name → numpy BGR crop
        if self.world_detector:
            print("[World] Scanning for label/date/logo regions...")
            for i, (_, img) in enumerate(loaded):
                vis = img.copy()
                r = self._detect_regions(img)
                for name, crop in r.items():
                    if name not in regions:
                        regions[name] = crop
                        _dbg(f'2b_world_crop_{name.replace(" ", "_")}.jpg', crop)
                        print(f"[World] Detected: {name}")
                # Draw all boxes on vis for the debug overview image
                raw = self.world_detector(img, verbose=False, conf=0.08)
                for det in raw:
                    for box in det.boxes:
                        cls_id = int(box.cls[0])
                        name = _WORLD_CLASSES[cls_id]
                        conf_w = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(vis, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 165, 255), 3)
                        cv2.putText(vis, f"{name} {conf_w:.2f}",
                                    (xyxy[0], max(0, xyxy[1] - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                _dbg(f'2b_world_overview_{i}.jpg', vis)
            if not regions:
                print("[World] No regions detected — Qwen will read full images")

        # ── Phase 2: EasyOCR ──────────────────────────────────────────────────
        # Prefer YOLO-World label crops (focused → less noise).
        # If YOLO-World found nothing, run on full images with a stricter
        # confidence threshold (0.40 vs 0.25) to suppress background garbage.
        _OCR_CLASSES = ["product label", "label sticker", "ingredient list", "nutrition facts panel"]
        ocr_targets = [regions[c] for c in _OCR_CLASSES if c in regions]
        if ocr_targets:
            _ocr_conf = 0.25   # relaxed — focused crop, mostly real text
        else:
            ocr_targets = [img for _, img in loaded]
            _ocr_conf = 0.40   # strict — full image, filter background noise

        _OCR_TARGET = 1920
        _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ocr_text_parts = []
        text_region_crops = []  # tight crops around text clusters → sent to Qwen at higher detail
        _ocr_idx = 0
        if self.ocr_ready:
            for img in ocr_targets:
                h, w = img.shape[:2]
                long_side = max(h, w)
                if long_side > _OCR_TARGET:
                    s = _OCR_TARGET / long_side
                    img_up = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                elif long_side < _OCR_TARGET // 2:
                    img_up = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
                else:
                    img_up = img
                lab = cv2.cvtColor(img_up, cv2.COLOR_BGR2LAB)
                lab[..., 0] = _clahe.apply(lab[..., 0])
                img_up = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                _dbg(f'3_ocr_input_{_ocr_idx}.jpg', img_up)
                t = self._read_paddleocr(img_up, min_conf=_ocr_conf)
                _dbg(f'3_ocr_output_{_ocr_idx}.jpg', self._dbg_ocr_overlay(img_up))

                # Compute a tight crop around all detected text boxes.
                # Uses a low position threshold (0.20) — we only care about WHERE
                # text is, not whether EasyOCR read it correctly.
                # This crop is sent to Qwen so small sticker text (batch, MFD)
                # occupies a much larger fraction of Qwen's token budget.
                _raw = getattr(self, '_last_ocr_raw', [])
                _pos_boxes = [bbox for (bbox, _, c) in _raw if c > 0.20]
                if len(_pos_boxes) >= 5:
                    _pts = [pt for bbox in _pos_boxes for pt in bbox]
                    _xs = [p[0] for p in _pts]
                    _ys = [p[1] for p in _pts]
                    _hh, _ww = img_up.shape[:2]
                    _px, _py = int(_ww * 0.04), int(_hh * 0.04)
                    _cx1 = max(0, int(min(_xs)) - _px)
                    _cy1 = max(0, int(min(_ys)) - _py)
                    _cx2 = min(_ww, int(max(_xs)) + _px)
                    _cy2 = min(_hh, int(max(_ys)) + _py)
                    _frac = (_cx2 - _cx1) * (_cy2 - _cy1) / (_hh * _ww)
                    if 0 < _frac < 0.75:
                        _tc = img_up[_cy1:_cy2, _cx1:_cx2]
                        if _tc.size > 0:
                            text_region_crops.append(_tc)
                            _dbg(f'3_text_region_{_ocr_idx}.jpg', _tc)
                            print(f"[OCR] Text-region crop: {_cx2-_cx1}×{_cy2-_cy1} "
                                  f"({_frac*100:.0f}% of frame) → Qwen")

                _ocr_idx += 1
                if t.strip():
                    print(f"[OCR] Text (conf≥{_ocr_conf}): {t}")
                    if not result["dotted_label_text"]:
                        result["dotted_label_text"] = t
                    ocr_text_parts.append(t)

        # ── Phase 3: Qwen2.5-VL ────────────────────────────────────────────────
        # Build image list for Qwen:
        #   1. YOLO-World label/logo crops (if any)
        #   2. EasyOCR text-region crops (tight crop of detected text — higher detail
        #      than full image, lets Qwen read small stickers / inkjet dates properly)
        #   3. Sharpest full image(s) as fallback / brand-name context
        _LABEL_PRIORITY = ["product label", "label sticker", "brand logo",
                           "nutrition facts panel", "ingredient list"]
        vlm_imgs = [regions[c] for c in _LABEL_PRIORITY if c in regions]

        # Always add text-region crops — they give Qwen focused detail on the label area
        vlm_imgs.extend(text_region_crops)

        if not vlm_imgs:
            # Nothing found at all — give Qwen the sharpest full images
            _VLM_MAX = 2
            if len(loaded) > _VLM_MAX:
                ranked = sorted(loaded, reverse=True,
                                key=lambda x: cv2.Laplacian(
                                    cv2.cvtColor(x[1], cv2.COLOR_BGR2GRAY), cv2.CV_64F
                                ).var())
                vlm_imgs = [img for _, img in ranked[:_VLM_MAX]]
            else:
                vlm_imgs = [img for _, img in loaded]
        else:
            # Add the sharpest full image for brand/product-name context
            sharpest = max(loaded, key=lambda x: cv2.Laplacian(
                cv2.cvtColor(x[1], cv2.COLOR_BGR2GRAY), cv2.CV_64F
            ).var())[1]
            vlm_imgs.append(sharpest)

        vlm_imgs = vlm_imgs[:4]  # cap at 4 — text-region crop adds one slot

        print(f"[Phase 3] Qwen2.5-VL on {len(vlm_imgs)} image(s)...")
        pil_images = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for img in vlm_imgs
        ]
        for i, pil in enumerate(pil_images):
            _dbg(f'4_qwen_input_{i}.jpg', pil)
        qwen_data = self._qwen_extract(pil_images)
        result.update({k: v for k, v in qwen_data.items() if v})

        # ── Phase 4: Date regex on EasyOCR text (fills inkjet dates Qwen missed) ──
        if ocr_text_parts:
            combined = " ".join(ocr_text_parts)
            for k, v in self._extract_dates(combined).items():
                if not result.get(k):
                    result[k] = v

        result["status"] = "Complete (Vision)" if result.get("brand") else "Incomplete"

        if self._debug:
            with open(os.path.join(_rdir, 'result.json'), 'w') as f:
                json.dump({k: v for k, v in result.items() if k != 'images'}, f, indent=2)
            print(f"[Debug] Snapshots saved → debug_snapshots/{_ts}/")

        return result

    def inspect_image(self, image_path):
        """Single-image convenience wrapper."""
        r = self.inspect_product([image_path])
        r["image"] = os.path.basename(image_path)
        return r
