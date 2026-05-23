import cv2
import re
import os
import time
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
    r'|BB\.?D?'                                      # BB, BBD
    r'|B\.B\.?D?'                                    # B.B, B.B.D
    r'|BEST\s*(?:BEFORE|BY|USED?\s*BY)'              # BEST BEFORE, BEST BY, BEST USED BY
    r'|USE\s*(?:BY|BEFORE)(?:\s*DATE)?'              # USE BY, USE BEFORE, USE BY DATE
    r'|USED?\s*BY'                                   # USED BY
    r'|CONSUME\s*(?:BY|BEFORE)'                      # CONSUME BY, CONSUME BEFORE
    r'|SELL\s*(?:BY|BEFORE)'                         # SELL BY, SELL BEFORE
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
    "Expiry: [any expiry/best-before/use-by/sell-by/consume-by/BB/BBD date, else NONE]\n"
    "MFG: [any manufacturing/manufactured-on/packed-on/packing-date/production-date/made-on/DOM date, else NONE]\n"
    "Batch: [batch or lot number, else NONE]\n"
    "Reply using exactly those field names, one per line."
)


class AIInspectionSystem:
    def __init__(
        self,
        barcode_model_path='models/barcode_detector.pt',
        ocr_model_path='models/dotted_ocr_retrained.pth',
        qwen_model_id='Qwen/Qwen2.5-VL-3B-Instruct',
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

    @staticmethod
    def _shrink_pil(pil_img, max_side=640):
        """Downscale PIL image so its longest side ≤ max_side before tokenisation."""
        w, h = pil_img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return pil_img

    def _qwen_extract(self, pil_images):
        # Pre-shrink so the processor receives small images — saves tokenisation time
        pil_images = [self._shrink_pil(img) for img in pil_images]

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

    # ── Auto-orientation ───────────────────────────────────────────────────────

    def _best_rotation(self, img_bgr):
        """Return img rotated to the best upright orientation (0° or 180°).

        Strategy:
        1. If barcode detector loaded: run it on both orientations, pick the one
           with higher max confidence. Fast signal — ~30 ms per call on GPU.
        2. Fallback (no barcode / low confidence): Sobel-Y edge-density heuristic.
           Product labels are top-heavy (logo/name at top, date at bottom), so the
           top third has more horizontal edge energy when upright. ~3 ms.
        """
        img_180 = cv2.rotate(img_bgr, cv2.ROTATE_180)

        if self.barcode_detector:
            def _max_conf(img):
                res = self.barcode_detector(img, verbose=False)
                return max((b.conf[0].item() for r in res for b in r.boxes), default=0.0)

            c0, c180 = _max_conf(img_bgr), _max_conf(img_180)
            if c180 > c0 and c180 > 0.15:
                print("[Orient] Rotated 180° (barcode signal)")
                return img_180
            if c0 > 0.15:
                return img_bgr  # already correct, no message needed

        # Sobel fallback
        def _edge_top_score(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (256, 256))
            gy = cv2.Sobel(small, cv2.CV_32F, 0, 1)
            return float(np.sum(np.abs(gy[:85]))) - float(np.sum(np.abs(gy[171:])))

        if _edge_top_score(img_180) > _edge_top_score(img_bgr):
            print("[Orient] Rotated 180° (edge heuristic)")
            return img_180

        return img_bgr

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

        # ── Auto-orient (fix upside-down products) ─────────────────────────────
        print(f"[Orient] Checking orientation on {len(loaded)} image(s)...")
        loaded = [(path, self._best_rotation(img)) for path, img in loaded]

        # ── Phase 1: Barcode ───────────────────────────────────────────────────
        print(f"[Phase 1] Scanning {len(loaded)} image(s) for barcode...")
        for path, img in loaded:
            if self.barcode_detector:
                for det in self.barcode_detector(img, verbose=False):  # use oriented array
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

        # ── Phase 3: Qwen2.5-VL ────────────────────────────────────────────────
        # Pick the 2 sharpest frames — more images = more tokens = slower.
        # Two angles give enough context; sending 3+ rarely improves accuracy.
        _VLM_MAX = 2
        if len(loaded) > _VLM_MAX:
            ranked = sorted(loaded, reverse=True,
                            key=lambda x: cv2.Laplacian(
                                cv2.cvtColor(x[1], cv2.COLOR_BGR2GRAY), cv2.CV_64F
                            ).var())
            vlm_imgs = [img for _, img in ranked[:_VLM_MAX]]
        else:
            vlm_imgs = [img for _, img in loaded]

        print(f"[Phase 3] Qwen2.5-VL on {len(vlm_imgs)} image(s)...")
        pil_images = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for img in vlm_imgs
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
