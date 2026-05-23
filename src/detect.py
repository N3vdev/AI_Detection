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
    (r'MF[DG][\s:]*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{2,4})',                                'manufacture_date'),
    (r'(?:EXP|EXPIRY|BB|BEST\s*BEFORE|USE\s*BY)[\s:]*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{2,4})', 'expiry_date'),
    (r'\b(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{2,4})\b',                                         'expiry_date'),
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

        # ── Piero2411 barcode YOLO ─────────────────────────────────────────────
        self.barcode_detector = None
        if os.path.exists(barcode_model_path):
            self.barcode_detector = YOLO(barcode_model_path)
            print("[System] Barcode detector loaded.")
        else:
            print(f"[Warning] Barcode model not found: {barcode_model_path}")

        # ── Qwen2-VL-2B — visual product understanding ─────────────────────────
        # max_pixels caps visual tokens to ~512 patches (≈634×634 px equivalent).
        # Default is 16384 patches (~12 MP) which is 32× slower for no accuracy gain on labels.
        print(f"[System] Loading Qwen2-VL ({qwen_model_id})...")
        self.qwen_processor = Qwen2VLProcessor.from_pretrained(
            qwen_model_id,
            min_pixels=4 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            qwen_model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()
        torch.set_num_threads(os.cpu_count() or 4)
        print("[System] Qwen2-VL loaded.")

        # ── CRNN — custom dotted/inkjet label reader ───────────────────────────
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

        print("[System] Ready.\n")

    # ── Qwen2-VL inference ─────────────────────────────────────────────────────

    def _qwen_extract(self, img_bgr):
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        # Downscale to max 640px — Qwen2-VL reads labels fine at this resolution
        if max(img_pil.size) > 640:
            ratio = 640 / max(img_pil.size)
            img_pil = img_pil.resize(
                (int(img_pil.width * ratio), int(img_pil.height * ratio)),
                Image.LANCZOS,
            )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_pil},
                    {"type": "text",  "text": _QWEN_PROMPT},
                ],
            }
        ]
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
            if key == "brand":
                result["brand"] = value
            elif key == "product":
                result["product_name"] = value
            elif key == "category":
                result["product_category"] = value
            elif key == "expiry":
                result["expiry_date"] = value
            elif key == "mfg":
                result["manufacture_date"] = value
            elif key == "batch":
                result["batch_number"] = value
            elif key == "text":
                result["raw_ocr_text"] = value
        return result

    # ── Dotted label preprocessing ─────────────────────────────────────────────

    def _preprocess_dotted(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # ── CRNN dotted label reader ───────────────────────────────────────────────

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

    # ── Date extraction ────────────────────────────────────────────────────────

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

    def inspect_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"Cannot load image: {image_path}"}

        result = {
            "image":             os.path.basename(image_path),
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

        # ── Phase 1: Barcode ───────────────────────────────────────────────────
        print("[Phase 1] Scanning for barcode...")
        if self.barcode_detector:
            for det in self.barcode_detector(image_path, verbose=False):
                for box in det.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    value = self._decode_crop(self._padded_crop(img, xyxy))
                    if value:
                        result["barcode"] = value
                        result["status"] = "Complete (Barcode)"
                        print(f"[Phase 1] Barcode: {value}")
                        return result
        print("[Phase 1] No barcode — moving to Phase 2.")

        # ── Phase 2: Qwen2-VL — all fields in one pass ────────────────────────
        print("[Phase 2] Running Qwen2-VL product analysis...")
        qwen_data = self._qwen_extract(img)
        result.update(qwen_data)

        # ── Phase 3: CRNN on dotted-preprocessed image ─────────────────────────
        if self.crnn_ready:
            dotted_img = self._preprocess_dotted(img)
            crnn_text = self._read_crnn(dotted_img)
            if crnn_text.strip():
                result["dotted_label_text"] = crnn_text
                # Fill in dates from dotted label if Qwen2-VL missed them
                ocr_dates = self._extract_dates(crnn_text)
                for k, v in ocr_dates.items():
                    if not result.get(k):
                        result[k] = v

        result["status"] = "Complete (Vision)" if result.get("brand") else "Incomplete"
        return result
