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

# Support Qwen2.5-VL (newer, more accurate) with fallback to Qwen2-VL
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

_QWEN_FRONT_PROMPT = (
    "You are a product label scanner on a factory conveyor belt. "
    "Look carefully at this product image and extract:\n"
    "Brand: [exact brand or company name visible on the product]\n"
    "Product: [full product name and variant]\n"
    "Category: [Food / Drink / Snack / Skincare / Haircare / Medicine / Household]\n"
    "Expiry: [expiry or best-before date if visible, else NONE]\n"
    "MFG: [manufacturing or packed date if visible, else NONE]\n"
    "Batch: [batch or lot number if visible, else NONE]\n"
    "Reply using exactly those field names, one per line."
)

_QWEN_BACK_PROMPT = (
    "You are reading the back or bottom label of a product. "
    "Look carefully and extract ONLY:\n"
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

        # ── Vision-Language Model ──────────────────────────────────────────────
        # Qwen2.5-VL-7B: state-of-the-art for reading product labels, logos,
        # and dates. Uses device_map="auto" to fill all available VRAM.
        # VRAM needed:  float16 → ~15 GB  |  4-bit → ~5 GB
        print(f"[System] Loading {qwen_model_id}...")
        self.qwen_processor = _VLProcessor.from_pretrained(qwen_model_id)

        load_kwargs = dict(device_map="auto")
        if self.device == "cuda":
            load_kwargs["torch_dtype"] = torch.float16
            # Enable Flash Attention 2 if available (2-4× faster on Ampere/Ada GPUs)
            try:
                load_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                pass
        else:
            load_kwargs["torch_dtype"] = torch.float32

        self.qwen_model = _VLModel.from_pretrained(
            qwen_model_id, **load_kwargs
        ).eval()
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

        # EasyOCR loaded lazily — only for back-label date extraction
        self._ocr_reader = None

        torch.set_num_threads(os.cpu_count() or 4)
        print("[System] Ready.\n")

    # ── Front / back classifier ────────────────────────────────────────────────

    def _classify_label(self, img_bgr):
        """
        Classify image as 'front' or 'back' using Canny edge density.
        Back labels have dense text (many edges); front labels are mostly graphics.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        density = np.sum(edges > 0) / edges.size
        label = "back" if density > 0.07 else "front"
        print(f"[Classify] {label.upper()} (edge density={density:.3f})")
        return label

    # ── Qwen VLM inference ─────────────────────────────────────────────────────

    def _qwen_extract(self, img_bgr, prompt, max_tokens=100):
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_pil},
                {"type": "text",  "text": prompt},
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
                **inputs, max_new_tokens=max_tokens, do_sample=False
            )
        response = self.qwen_processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0].strip()
        print(f"[Qwen] Response:\n{response}\n")
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
        return result

    # ── EasyOCR date scan (lazy-loaded, back labels only) ─────────────────────

    def _get_ocr_reader(self):
        if self._ocr_reader is None:
            import easyocr
            print("[System] Loading EasyOCR for date extraction...")
            self._ocr_reader = easyocr.Reader(['en'], gpu=self.device == "cuda",
                                               verbose=False)
        return self._ocr_reader

    def _easyocr_date_scan(self, img_bgr):
        """Targeted date scan on back label: full image + strips at 3× zoom + CLAHE."""
        reader = self._get_ocr_reader()
        h, w = img_bgr.shape[:2]

        def ocr_region(crop, scale=1.0):
            if crop.size == 0:
                return ""
            if scale != 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(cv2.fastNlMeansDenoising(gray, h=15))
            results = reader.readtext(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))
            text = " ".join(r[1] for r in results if r[2] > 0.2)
            if text.strip():
                print(f"[EasyOCR] {text[:100]}")
            return text

        parts = [
            ocr_region(img_bgr),
            ocr_region(img_bgr[int(h * 0.65):, :], scale=3),
            ocr_region(img_bgr[:int(h * 0.25), :], scale=3),
            ocr_region(img_bgr[:, :int(w * 0.3)],  scale=3),
            ocr_region(img_bgr[:, int(w * 0.7):],  scale=3),
        ]
        return " ".join(p for p in parts if p)

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
        Auto-detects front (logo/brand) vs back (dates/info) and routes each
        image to the right model — Qwen2.5-VL-7B for the front, EasyOCR +
        Qwen fallback for the back.

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

        # ── Phase 2: Classify front / back ────────────────────────────────────
        print("[Phase 2] Classifying images...")
        fronts, backs = [], []
        for path, img in loaded:
            if self._classify_label(img) == "front":
                fronts.append((path, img))
            else:
                backs.append((path, img))

        # Guarantee at least one front — pick the least text-dense image
        if not fronts:
            densities = sorted(
                ((np.sum(cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150) > 0)
                  / img.shape[0] / img.shape[1], path, img)
                 for path, img in loaded)
            )
            fronts = [(densities[0][1], densities[0][2])]
            backs  = [(p, i) for _, p, i in densities[1:]]
            print(f"[Phase 2] Fallback: using {os.path.basename(fronts[0][0])} as front")

        print(f"[Phase 2] Front: {[os.path.basename(p) for p,_ in fronts]}")
        if backs:
            print(f"[Phase 2] Back:  {[os.path.basename(p) for p,_ in backs]}")

        # ── Phase 3: CRNN dotted labels on all images ──────────────────────────
        all_text_parts = []
        if self.crnn_ready:
            for _, img in loaded:
                crnn_text = self._read_crnn(self._preprocess_dotted(img))
                if crnn_text.strip():
                    if not result["dotted_label_text"]:
                        result["dotted_label_text"] = crnn_text
                    all_text_parts.append(crnn_text)

        # ── Phase 4: Qwen2.5-VL on front image → brand, product, category ─────
        print(f"[Phase 4] Qwen2.5-VL-7B on front: {os.path.basename(fronts[0][0])}")
        qwen_data = self._qwen_extract(fronts[0][1], _QWEN_FRONT_PROMPT, max_tokens=100)
        result.update({k: v for k, v in qwen_data.items() if v})

        # ── Phase 5: Back label — EasyOCR first, Qwen fallback ────────────────
        if backs:
            dates_needed = not result.get("expiry_date") or not result.get("manufacture_date")
            if dates_needed:
                print(f"[Phase 5] Date scan on back: {os.path.basename(backs[0][0])}")
                back_text = self._easyocr_date_scan(backs[0][1])
                all_text_parts.append(back_text)

                # Run regex on EasyOCR output
                for k, v in self._extract_dates(back_text).items():
                    if not result.get(k):
                        result[k] = v

                # Still missing? Qwen2.5-VL with date-only prompt (fast — 50 tokens)
                still_missing = not result.get("expiry_date") and not result.get("manufacture_date")
                if still_missing:
                    print("[Phase 5] EasyOCR missed dates — Qwen2.5-VL on back...")
                    qwen_back = self._qwen_extract(backs[0][1], _QWEN_BACK_PROMPT, max_tokens=50)
                    for k, v in qwen_back.items():
                        if v and not result.get(k):
                            result[k] = v

        # ── Final: date regex over all collected text ──────────────────────────
        combined = " ".join(all_text_parts)
        result["raw_ocr_text"] = combined[:500] if combined else None
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
