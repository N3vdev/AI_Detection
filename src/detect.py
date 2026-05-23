import cv2
import re
import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from src.ocr_model import CRNN

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
except ImportError:
    pyzbar_decode = None

# Qwen2.5-VL (newer) with fallback to Qwen2-VL
try:
    from transformers import Qwen2_5_VLProcessor as _VLProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration as _VLModel
except ImportError:
    from transformers import Qwen2VLProcessor as _VLProcessor
    from transformers import Qwen2VLForConditionalGeneration as _VLModel


_DATE_PATTERNS = [
    (r'(?:MF[DG]\.?\s*DATE\s*[:.]?|MF[DG]|MFG|MANUFACTURED?(?:\s*DATE)?|DOM)[\s:\.]*'
     r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',                                           'manufacture_date'),
    (r'(?:MF[DG]\.?\s*DATE\s*[:.]?|MF[DG]|MFG|MANUFACTURED?)[\s:\.]*'
     r'(\d{1,2}[\/\-\.](?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\/\-\.]\d{2,4})',
                                                                                              'manufacture_date'),
    (r'(?:MF[DG]|MFG)[\s:\.]*(\d{2})(\d{2})(\d{4})',                                       'manufacture_date'),
    (r'(?:EXP(?:IRY|IRE|\.)?|BB\.?|BEST\s*BEFORE|USE\s*BY|USE\s*BEFORE|BBD)[\s:\.]*'
     r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',                                           'expiry_date'),
    (r'(?:EXP(?:IRY|IRE)?|BB|BEST\s*BEFORE|USE\s*BY)[\s:\.]*'
     r'(\d{1,2}[\/\-\.](?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\/\-\.]\d{2,4})',
                                                                                              'expiry_date'),
    (r'\b(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{2,4})\b',                                           'expiry_date'),
]

_BATCH_PATTERN = re.compile(
    r'(?:BATCH\s*(?:NO\.?|NUMBER|CODE)?|LOT\s*(?:NO\.?|NUMBER)?)[\s:\.]*([A-Z0-9]{5,20})',
    re.IGNORECASE,
)

_PROMPT = (
    "You are a product inspection AI on a factory conveyor belt. "
    "You are given one or more images of the SAME product from different angles "
    "(front, back, side). Look at ALL images together and extract:\n"
    "Brand: [exact brand or company name on the product]\n"
    "Product: [full product name and variant]\n"
    "Category: [Food / Drink / Snack / Skincare / Haircare / Medicine / Household]\n"
    "Expiry: [expiry or best-before date, else NONE]\n"
    "MFG: [manufacturing or packed date, else NONE]\n"
    "Batch: [batch or lot number, else NONE]\n"
    "Reply using exactly those field names, one per line."
)


class AIInspectionSystem:
    def __init__(
        self,
        barcode_model_path='models/barcode_detector.pt',
        ocr_model_path='models/dotted_ocr_retrained.pth',
        qwen_model_id='Qwen/Qwen2.5-VL-7B-Instruct',
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

        # ── Qwen2.5-VL ────────────────────────────────────────────────────────
        print(f"[System] Loading {qwen_model_id}...")
        self.qwen_processor = _VLProcessor.from_pretrained(qwen_model_id)

        load_kwargs = dict(device_map="auto")
        load_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
        try:
            import flash_attn  # noqa
            load_kwargs["attn_implementation"] = "flash_attention_2"
            print("[System] Flash Attention 2 enabled.")
        except ImportError:
            pass

        self.qwen_model = _VLModel.from_pretrained(qwen_model_id, **load_kwargs).eval()
        print(f"[System] {qwen_model_id} loaded.")

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

    # ── Qwen2.5-VL — single call with ALL images ───────────────────────────────

    def _qwen_extract(self, pil_images):
        """
        Pass all product images in one call.
        Qwen2.5-VL sees every angle simultaneously and extracts the best data.
        """
        content = []
        for img in pil_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": _PROMPT})

        messages = [{"role": "user", "content": content}]
        text = self.qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.qwen_processor(
            text=[text], images=pil_images, return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            output_ids = self.qwen_model.generate(
                **inputs, max_new_tokens=100, do_sample=False
            )
        response = self.qwen_processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0].strip()

        print(f"[Qwen2.5-VL] Response:\n{response}\n")
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

        # ── Phase 1: Barcode ───────────────────────────────────────────────────
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

        # ── Phase 2: CRNN dotted labels ────────────────────────────────────────
        crnn_text_parts = []
        if self.crnn_ready:
            for _, img in loaded:
                t = self._read_crnn(self._preprocess_dotted(img))
                if t.strip():
                    if not result["dotted_label_text"]:
                        result["dotted_label_text"] = t
                    crnn_text_parts.append(t)

        # ── Phase 3: Qwen2.5-VL — all images in ONE call ──────────────────────
        print(f"[Phase 3] Qwen2.5-VL on {len(loaded)} image(s) (single call)...")
        pil_images = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for _, img in loaded
        ]
        qwen_data = self._qwen_extract(pil_images)
        result.update({k: v for k, v in qwen_data.items() if v})

        # ── Phase 4: Date regex on CRNN text (fills inkjet dates Qwen missed) ──
        if crnn_text_parts:
            combined = " ".join(crnn_text_parts)
            for k, v in self._extract_dates(combined).items():
                if not result.get(k):
                    result[k] = v

        result["status"] = "Complete (Vision)" if result.get("brand") else "Incomplete"
        return result

    def inspect_image(self, image_path):
        """Single-image convenience wrapper."""
        r = self.inspect_product([image_path])
        r["image"] = os.path.basename(image_path)
        return r
