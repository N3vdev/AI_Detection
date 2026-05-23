import cv2
import re
import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from src.ocr_model import CRNN
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
except ImportError:
    pyzbar_decode = None


_DATE_PATTERNS = [
    (r'MF[DG][\s:]*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{2,4})',                                    'manufacture_date'),
    (r'(?:EXP|EXPIRY|BB|BEST\s*BEFORE|USE\s*BY)[\s:]*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{2,4})', 'expiry_date'),
    (r'\b(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{2,4})\b',                                             'expiry_date'),
]

_QWEN_PROMPT = (
    "You are a product label scanner on a factory conveyor belt. "
    "Look at this product image carefully and extract:\n"
    "Brand: [company/brand name]\n"
    "Product: [full product name and type]\n"
    "Category: [Food / Drink / Snack / Skincare / Haircare / Medicine / Household]\n"
    "Expiry: [expiry date if visible, else NONE]\n"
    "MFG: [manufacturing date if visible, else NONE]\n"
    "Batch: [batch or lot number if visible, else NONE]\n"
    "Text: [all other visible text on the label, comma separated, else NONE]\n"
    "Reply using exactly those field names."
)


class AIInspectionSystem:
    def __init__(
        self,
        barcode_model_path='models/barcode_detector.pt',
        ocr_model_path='models/dotted_ocr_retrained.pth',
        qwen_model_id='Qwen/Qwen2-VL-2B-Instruct',
    ):
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

        # ── VLM or OCR depending on device ────────────────────────────────────
        self.qwen_processor = None
        self.qwen_model = None
        self.ocr_reader = None

        if self.device == "cuda":
            # GPU: Qwen2-VL — reads stylized logos accurately, ~2-3s
            print(f"[System] Loading Qwen2-VL ({qwen_model_id})...")
            self.qwen_processor = Qwen2VLProcessor.from_pretrained(qwen_model_id)
            self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                qwen_model_id, torch_dtype=torch.float16,
            ).to(self.device).eval()
            print("[System] Qwen2-VL loaded.")
        else:
            # CPU: EasyOCR — fast (~2s), limited on stylized fonts
            print("[System] CPU mode — loading EasyOCR (fast). GPU required for full accuracy.")
            import easyocr
            self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("[System] EasyOCR loaded.")

        # ── CRNN dotted/inkjet label reader ────────────────────────────────────
        self.alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/-.: "
        self.crnn = CRNN(32, 1, len(self.alphabet) + 1, 256)
        self.crnn_ready = False
        try:
            self.crnn.load_state_dict(torch.load(ocr_model_path, map_location='cpu'))
            self.crnn.eval()
            self.crnn_ready = True
            print("[System] Dotted label CRNN loaded.")
        except Exception as e:
            print(f"[System] Dotted CRNN not loaded ({e}).")

        torch.set_num_threads(os.cpu_count() or 4)
        print("[System] Ready.\n")

    # ── Qwen2-VL inference (GPU) ───────────────────────────────────────────────

    def _qwen_extract(self, img_bgr):
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_pil},
                {"type": "text",  "text": _QWEN_PROMPT},
            ],
        }]
        text = self.qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.qwen_processor(
            text=[text], images=[img_pil], return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            output_ids = self.qwen_model.generate(
                **inputs, max_new_tokens=150, do_sample=False
            )
        response = self.qwen_processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0].strip()
        print(f"[Qwen2-VL] Response:\n{response}\n")
        return self._parse_qwen(response)

    def _parse_qwen(self, response):
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
            elif key == "text":     result["raw_ocr_text"] = value
        return result

    # ── EasyOCR inference (CPU) ────────────────────────────────────────────────

    def _easyocr_extract(self, img_bgr):
        results = self.ocr_reader.readtext(img_bgr)
        if not results:
            return {}

        def bbox_area(r):
            pts = r[0]
            return abs(pts[1][0] - pts[0][0]) * abs(pts[2][1] - pts[0][1])

        sorted_by_size = sorted(results, key=bbox_area, reverse=True)
        all_text = " ".join(r[1] for r in results if r[2] > 0.3)

        out = {"raw_ocr_text": all_text}
        # Largest high-confidence text → brand candidate
        for r in sorted_by_size:
            if r[2] > 0.5 and len(r[1].strip()) >= 3:
                out["brand"] = r[1].strip()
                break
        # Second largest → product name candidate
        skipped_brand = False
        for r in sorted_by_size:
            if not skipped_brand:
                skipped_brand = True
                continue
            if r[2] > 0.4 and len(r[1].strip()) >= 3:
                out["product_name"] = r[1].strip()
                break

        print(f"[EasyOCR] Extracted: {all_text[:120]}")
        return out

    # ── Dotted label helpers ───────────────────────────────────────────────────

    def _preprocess_dotted(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    def _read_crnn(self, crop_bgr):
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 32))
        t = torch.from_numpy(
            (gray.astype(np.float32) / 255.0 - 0.5) / 0.5
        ).unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            preds = self.crnn(t)
            _, idx = torch.max(preds, 2)
            idx = idx.permute(1, 0).cpu().numpy()[0]
        chars = [
            self.alphabet[i - 1]
            for j, i in enumerate(idx)
            if i != 0 and (j == 0 or i != idx[j - 1])
        ]
        return "".join(chars)

    # ── Barcode helpers ────────────────────────────────────────────────────────

    def _decode_crop(self, crop):
        if crop is None or crop.size == 0:
            return None

        def _try(img):
            if pyzbar_decode:
                try:
                    for r in pyzbar_decode(img):
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

        result = _try(crop)
        if result:
            return result
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        return _try(cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR))

    def _padded_crop(self, img, xyxy, pad=0.15):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = xyxy
        pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
        return img[max(0, y1 - ph):min(h, y2 + ph), max(0, x1 - pw):min(w, x2 + pw)]

    def _extract_dates(self, text):
        found = {}
        upper = text.upper()
        for pattern, key in _DATE_PATTERNS:
            if key not in found:
                m = re.search(pattern, upper)
                if m:
                    found[key] = m.group(1)
        return found

    # ── Main pipeline ──────────────────────────────────────────────────────────

    def inspect_product(self, image_paths):
        """
        Inspect one or more images of the same product (front, back, side).
        Results from all views are merged — the best data from each image wins.

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
            "raw_ocr_text":      None,
            "status":            "Incomplete",
        }

        # Load all images upfront
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

        # ── Phase 1: Barcode — check every view ────────────────────────────────
        print(f"[Phase 1] Scanning {len(loaded)} image(s) for barcode...")
        for path, img in loaded:
            if self.barcode_detector:
                for det in self.barcode_detector(path, verbose=False):
                    for box in det.boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        value = self._decode_crop(self._padded_crop(img, xyxy))
                        if value:
                            result["barcode"] = value
                            result["status"] = "Complete (Barcode)"
                            print(f"[Phase 1] Barcode: {value}")
                            return result
        print("[Phase 1] No barcode — moving to Phase 2.")

        # ── Phase 2: CRNN dotted labels on every view ──────────────────────────
        all_text_parts = []
        if self.crnn_ready:
            for _, img in loaded:
                crnn_text = self._read_crnn(self._preprocess_dotted(img))
                if crnn_text.strip():
                    if not result["dotted_label_text"]:
                        result["dotted_label_text"] = crnn_text
                    all_text_parts.append(crnn_text)

        # ── Phase 3: Brand / product / category ────────────────────────────────
        if self.device == "cuda":
            # GPU — Qwen2-VL on first image (most informative view)
            print(f"[Phase 3] Qwen2-VL on {os.path.basename(loaded[0][0])}...")
            qwen_data = self._qwen_extract(loaded[0][1])
            for k, v in qwen_data.items():
                if v and not result.get(k):
                    result[k] = v
            if qwen_data.get("raw_ocr_text"):
                all_text_parts.append(qwen_data["raw_ocr_text"])

            # If dates or batch still missing, run Qwen2-VL on remaining views
            needs_more = not result.get("expiry_date") or not result.get("manufacture_date")
            if needs_more and len(loaded) > 1:
                for path, img in loaded[1:]:
                    print(f"[Phase 3] Qwen2-VL on {os.path.basename(path)} (checking remaining fields)...")
                    extra = self._qwen_extract(img)
                    for k, v in extra.items():
                        if v and not result.get(k):
                            result[k] = v
        else:
            # CPU — EasyOCR on every view, merge best result
            print(f"[Phase 3] EasyOCR on {len(loaded)} image(s)...")
            for _, img in loaded:
                ocr_data = self._easyocr_extract(img)
                for k in ("brand", "product_name", "raw_ocr_text"):
                    if ocr_data.get(k) and not result.get(k):
                        result[k] = ocr_data[k]
                if ocr_data.get("raw_ocr_text"):
                    all_text_parts.append(ocr_data["raw_ocr_text"])

        # ── Date regex over everything collected ───────────────────────────────
        combined = " ".join(all_text_parts)
        result["raw_ocr_text"] = combined or result.get("raw_ocr_text")
        for k, v in self._extract_dates(combined).items():
            if not result.get(k):
                result[k] = v

        result["status"] = "Complete (Vision)" if result.get("brand") else "Incomplete"
        return result

    def inspect_image(self, image_path):
        """Single-image convenience wrapper (backward compatible)."""
        r = self.inspect_product([image_path])
        r["image"] = os.path.basename(image_path)
        return r
