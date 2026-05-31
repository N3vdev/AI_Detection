import cv2
import re
import os
import time
import json
import datetime
import threading
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

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

# ── Qwen response parsing helpers ─────────────────────────────────────────────
_NON_ANSWERS = frozenset({
    'none', 'n/a', 'not visible', 'not found', 'not available',
    'not legible', 'cannot read', 'unknown', 'not present',
    'not specified', 'not printed', 'not readable', 'not clear',
    'not detected', 'not applicable', 'unclear',
})

_KEY_MAP = {
    'brand': 'brand',
    'product': 'product_name',
    'product name': 'product_name',
    'category': 'product_category',
    'expiry': 'expiry_date',
    'expiry date': 'expiry_date',
    'best before': 'expiry_date',
    'exp': 'expiry_date',
    'mfg': 'manufacture_date',
    'mfg date': 'manufacture_date',
    'manufacture date': 'manufacture_date',
    'manufactured': 'manufacture_date',
    'manufacturing date': 'manufacture_date',
    'batch': 'batch_number',
    'batch no': 'batch_number',
    'batch number': 'batch_number',
    'lot no': 'batch_number',
    'lot number': 'batch_number',
}

# Qwen prompt template — OCR text injected at runtime so Qwen uses both image and text context
_PROMPT_TMPL = (
    "You are a product inspection AI on a factory conveyor belt. "
    "You are given one or more images of the SAME physical product from different angles.\n\n"
    "OCR text already extracted from these images:\n{ocr_text}\n\n"
    "Using both the images AND the OCR text above, extract the following fields. "
    "Pay special attention to small inkjet-printed or dot-matrix text near edges — "
    "these are typically the manufacture/packed/expiry dates and may appear faint or dotted.\n\n"
    "Brand: [exact brand or company name printed on the product]\n"
    "Product: [full product name including variant, flavour, or size]\n"
    "Category: [Food / Drink / Snack / Skincare / Haircare / Medicine / Household]\n"
    "Expiry: [date after EXP/Expiry/Best Before/Use By/BB/BBD/Consume By — any format — else NONE]\n"
    "MFG: [date after MFG/MFD/Mfg Date/Manufactured/Packed On/PKD/Production Date/Made On/DOM — else NONE]\n"
    "Batch: [alphanumeric code after Batch No/Lot No — else NONE]\n\n"
    "Rules:\n"
    "- Output ONLY the six fields above, one per line, no extra text.\n"
    "- If a date appears in multiple formats, output the most complete one.\n"
    "- Never confuse MFG and Expiry — Expiry is always the later date.\n"
    "- If unsure, output NONE rather than guessing.\n"
    "- Do NOT add explanations or extra sentences — output only the six lines above."
)


class AIInspectionSystem:
    def __init__(
        self,
        barcode_model_path='models/barcode_detector.pt',
        qwen_model_id='Qwen/Qwen2.5-VL-3B-Instruct',
        world_model_id='yolov8m-worldv2.pt',
        florence2_model_id='microsoft/Florence-2-large',
        debug=True,
        on_progress=None,
    ):
        """
        debug=True  → save per-step snapshots to debug_snapshots/ (demo / dev use)
        debug=False → skip all debug writes (production conveyor — saves disk + minor speedup)
        on_progress → callable(str) emitted at each loading step for UI feedback
        """
        def _prog(msg):
            print(f"[System] {msg}")
            if on_progress:
                on_progress(msg)

        self._debug = debug
        _prog("Initializing AI pipeline...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _prog(f"Device: {self.device.upper()}")

        # ── Barcode YOLO ───────────────────────────────────────────────────────
        _prog("Loading barcode detector...")
        self.barcode_detector = None
        if os.path.exists(barcode_model_path):
            self.barcode_detector = YOLO(barcode_model_path)
            _prog("Barcode detector ready.")
        else:
            print(f"[Warning] Barcode model not found: {barcode_model_path}")

        # ── WeChat barcode detector (deep-learning, includes super-resolution) ─
        self._wechat_bc = None
        _wechat_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'wechat_barcode')
        if (os.path.isdir(_wechat_dir)
                and hasattr(cv2, 'wechat_qrcode')
                and hasattr(cv2.wechat_qrcode, 'WeChatQRCode')):
            try:
                self._wechat_bc = cv2.wechat_qrcode.WeChatQRCode(
                    os.path.join(_wechat_dir, 'detect.prototxt'),
                    os.path.join(_wechat_dir, 'detect.caffemodel'),
                    os.path.join(_wechat_dir, 'sr.prototxt'),
                    os.path.join(_wechat_dir, 'sr.caffemodel'),
                )
                _prog("WeChat barcode detector ready.")
            except Exception as e:
                print(f"[Warning] WeChat barcode init failed: {e}")

        # ── Qwen2.5-VL ────────────────────────────────────────────────────────
        _prog("Loading Qwen VLM processor...")
        self.qwen_processor = _VLProcessor.from_pretrained(qwen_model_id)

        load_kwargs = dict(device_map="auto")

        if self.device == "cuda":
            free_vram, _ = torch.cuda.mem_get_info(0)
            free_gb = free_vram / 1024 ** 3
            _prog(f"Free VRAM: {free_gb:.1f} GB")
            if free_gb < 12.0:
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    _prog("4-bit quantization enabled (low VRAM).")
                except ImportError:
                    print("[Warning] bitsandbytes not installed — loading in float16 (may OOM).")
                    load_kwargs["torch_dtype"] = torch.float16
            else:
                load_kwargs["torch_dtype"] = torch.float16
                _prog("Loading Qwen in float16.")
        else:
            load_kwargs["torch_dtype"] = torch.float32

        try:
            import flash_attn  # noqa
            load_kwargs["attn_implementation"] = "flash_attention_2"
            _prog("Flash Attention 2 enabled.")
        except ImportError:
            pass

        _prog("Loading Qwen VLM model  (largest step)...")
        self.qwen_model = _VLModel.from_pretrained(qwen_model_id, **load_kwargs).eval()
        _prog("Qwen VLM ready.")

        _max_side = 960 if self.device == "cuda" else 640
        self.qwen_processor.image_processor.max_pixels = _max_side * 28 * 28
        self.qwen_processor.image_processor.min_pixels = 4 * 28 * 28

        torch.set_num_threads(os.cpu_count() or 4)

        # ── Florence-2 whole-image OCR ─────────────────────────────────────────
        _prog("Loading Florence-2 OCR...")
        from transformers import AutoProcessor, AutoModelForCausalLM as _F2Model
        self.florence2_processor = AutoProcessor.from_pretrained(
            florence2_model_id, trust_remote_code=True
        )
        self.florence2_model = _F2Model.from_pretrained(
            florence2_model_id,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            trust_remote_code=True,
        ).to(self.device).eval()
        _prog("Florence-2 loaded.")

        # ── YOLO-World open-vocabulary region detector ─────────────────────────
        self.world_detector = None
        if world_model_id:
            try:
                _prog(f"Loading YOLO-World detector...")
                from ultralytics import YOLOWorld
                self.world_detector = YOLOWorld(world_model_id)
                self.world_detector.set_classes(_WORLD_CLASSES)
                _prog("YOLO-World ready.")
            except Exception as e:
                print(f"[Warning] YOLO-World failed to load: {e}")

        _prog("All models loaded — ready to inspect.")

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

    def _qwen_extract(self, pil_images, ocr_context=""):
        prompt = _PROMPT_TMPL.format(ocr_text=ocr_context or "None available")
        content = []
        for img in pil_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

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
                **inputs, max_new_tokens=200, do_sample=False, temperature=None,
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
            if ':' not in line:
                continue
            key, _, value = line.partition(':')
            key = key.strip().lower()
            value = value.strip().strip('"\'')
            if not value or value.lower() in _NON_ANSWERS or value.upper() == 'NONE':
                continue
            field = _KEY_MAP.get(key)
            if field and field not in result:
                result[field] = value
        return result

    # ── Florence-2 whole-image OCR ─────────────────────────────────────────────

    def _read_florence2(self, imgs):
        """Whole-image OCR via Florence-2. Returns single string with all detected text."""
        texts = []
        for img in imgs:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            inputs = self.florence2_processor(
                text="<OCR>", images=pil, return_tensors="pt"
            ).to(self.device)
            if self.device == 'cuda':
                inputs = {k: v.half() if v.dtype == torch.float32 else v
                          for k, v in inputs.items()}
            with torch.no_grad():
                ids = self.florence2_model.generate(
                    **inputs, max_new_tokens=256, do_sample=False, num_beams=1,
                )
            raw = self.florence2_processor.decode(ids[0], skip_special_tokens=False)
            parsed = self.florence2_processor.post_process_generation(
                raw, task="<OCR>", image_size=pil.size
            )
            t = parsed.get("<OCR>", "").strip()
            if t:
                texts.append(t)
        return " ".join(texts)

    # ── Barcode helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _validate_barcode(value: str) -> bool:
        """Return False if value is obviously wrong (garbage, wrong checksum, too short)."""
        if not value or len(value) < 4:
            return False
        # Must be ASCII printable only
        if not all(32 <= ord(c) < 127 for c in value):
            return False

        def _gcd_checksum(digits, weights):
            total = sum(int(d) * w for d, w in zip(digits[:-1], weights))
            return (10 - total % 10) % 10 == int(digits[-1])

        # EAN-13: 13 digits, weights alternating 1,3 starting at pos 0
        if len(value) == 13 and value.isdigit():
            return _gcd_checksum(value, [1 if i % 2 == 0 else 3 for i in range(12)])

        # EAN-8: 8 digits, weights alternating 3,1 starting at pos 0
        if len(value) == 8 and value.isdigit():
            return _gcd_checksum(value, [3 if i % 2 == 0 else 1 for i in range(7)])

        # UPC-A: 12 digits, weights alternating 3,1 starting at pos 0
        if len(value) == 12 and value.isdigit():
            return _gcd_checksum(value, [3 if i % 2 == 0 else 1 for i in range(11)])

        # Other formats (CODE128, CODE39, QR, etc.): just sanity length
        return len(value) >= 4

    def _decode_crop(self, crop):
        if crop is None or crop.size == 0:
            return None

        def _try(img):
            # pyzbar: collect all candidates, pick the largest (most prominent in frame)
            if pyzbar_decode:
                try:
                    candidates = []
                    for r in pyzbar_decode(img, symbols=_PYZBAR_SYMBOLS):
                        v = r.data.decode("utf-8", errors="ignore").strip()
                        if v and self._validate_barcode(v):
                            area = r.rect.width * r.rect.height
                            candidates.append((area, v))
                    if candidates:
                        return max(candidates)[1]   # largest bounding box wins
                except Exception:
                    pass
            # WeChat barcode detector — deep-learning with super-resolution module
            if self._wechat_bc is not None:
                try:
                    texts, _ = self._wechat_bc.detectAndDecode(img)
                    for t in (texts or []):
                        t = (t or "").strip()
                        if t and self._validate_barcode(t):
                            return t
                except Exception:
                    pass
            # ZXing C++ — most robust for distant/blurry barcodes
            try:
                import zxingcpp
                for b in zxingcpp.read_barcodes(img):
                    t = (b.text or "").strip()
                    if t and self._validate_barcode(t):
                        return t
            except Exception:
                pass
            # OpenCV barcode detector last-resort fallback
            try:
                det = cv2.barcode.BarcodeDetector()
                ok, decoded, _, _ = det.detectAndDecodeMulti(img)
                if ok:
                    for v in decoded:
                        v = (v or "").strip()
                        if v and self._validate_barcode(v):
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

        # 4. 4× upscale with cubic interpolation — for distant/very small barcodes
        v = _try(cv2.resize(gray, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC))
        if v: return v

        # 5. Mild sharpening — recovers slightly blurry barcodes without creating halos
        sharp = cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        v = _try(cv2.resize(sharp, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR))
        if v: return v

        # 6. Adaptive threshold — handles curved surfaces / uneven lighting
        # Use INTER_NEAREST: binary images must NOT be interpolated with gray values
        thresh_a = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        v = _try(cv2.resize(thresh_a, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST))
        if v: return v

        # 7. Otsu threshold — handles low-contrast prints (INTER_NEAREST for binary)
        _, thresh_o = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        v = _try(cv2.resize(thresh_o, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST))
        if v: return v

        # 8. Rotated attempts — for 1D barcodes oriented perpendicular to scan direction
        for angle in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            rotated = cv2.rotate(gray, angle)
            v = _try(cv2.resize(rotated, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR))
            if v: return v

        return None

    def _padded_crop(self, img, xyxy, pad=0.15):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = xyxy
        pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
        return img[max(0, y1 - ph):min(h, y2 + ph), max(0, x1 - pw):min(w, x2 + pw)]

    def quick_barcode_scan(self, frames):
        """Scan all camera frames for a barcode IN PARALLEL — no disk I/O, no YOLO.
        Call this on raw numpy frames before saving to disk or running the full pipeline.
        Returns the decoded barcode string, or None if not found.

        With 4 cameras all threads run simultaneously; typical wall time ~15-30ms.
        """
        _SCAN_MAX = 1920
        found = threading.Event()
        result = [None]

        def _scan(frame):
            if frame is None or found.is_set():
                return
            h, w = frame.shape[:2]
            if max(h, w) > _SCAN_MAX:
                s = _SCAN_MAX / max(h, w)
                frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            v = self._decode_crop(frame)
            if v and not found.is_set():
                result[0] = v
                found.set()

        threads = [threading.Thread(target=_scan, args=(f,), daemon=True)
                   for f in frames if f is not None]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return result[0]

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

    def _detect_regions(self, img_bgr, conf=0.15):
        """
        Run YOLO-World on img_bgr and return:
          - crops dict: class_name → crop (numpy BGR), best-confidence per class
          - raw_results: raw YOLO results list (for debug overlay reuse — avoids double inference)
        Returns ({}, []) if world_detector not loaded or nothing found.
        """
        if not self.world_detector:
            return {}, []
        raw_results = self.world_detector(img_bgr, verbose=False, conf=conf)
        best = {}  # class_name → (score, crop)
        for det in raw_results:
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
        return {name: crop for name, (_, crop) in best.items()}, raw_results

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
        # Single inference pass per image; raw_results reused for debug overlay.
        regions = {}  # class_name → numpy BGR crop
        if self.world_detector:
            print("[World] Scanning for label/date/logo regions...")
            for i, (_, img) in enumerate(loaded):
                r, raw_results = self._detect_regions(img)
                for name, crop in r.items():
                    if name not in regions:
                        regions[name] = crop
                        _dbg(f'2b_world_crop_{name.replace(" ", "_")}.jpg', crop)
                        print(f"[World] Detected: {name}")
                # Draw debug overlay using already-computed raw_results — no second inference
                vis = img.copy()
                for det in raw_results:
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
                print("[World] No regions detected — Florence-2 will read full images")

        # ── Phase 2 — Florence-2 whole-image OCR ──────────────────────────────
        print("[Phase 2] Running Florence-2 OCR...")
        _OCR_CLASSES = ["product label", "label sticker", "ingredient list", "nutrition facts panel"]
        ocr_imgs = [regions[c] for c in _OCR_CLASSES if c in regions]
        if not ocr_imgs:
            ocr_imgs = [img for _, img in loaded]
        ocr_imgs = ocr_imgs[:2]   # 2 best images — sufficient accuracy, A1000-friendly

        ocr_text_parts = []
        florence_text = self._read_florence2(ocr_imgs)
        if florence_text:
            print(f"[Florence-2] {florence_text[:200]}")
            ocr_text_parts.append(florence_text)
            result["dotted_label_text"] = florence_text

        # ── Phase 2.5: Regex on all OCR text — authoritative for dates/batch ───
        if ocr_text_parts:
            combined = " ".join(ocr_text_parts)
            for k, v in self._extract_dates(combined).items():
                result[k] = v

        # ── Phase 3: Qwen2.5-VL ────────────────────────────────────────────────
        # Build image list for Qwen — prioritise labelled regions, then sharpest frames.
        _LABEL_PRIORITY = ["product label", "label sticker", "brand logo",
                           "nutrition facts panel", "ingredient list"]
        vlm_imgs = [regions[c] for c in _LABEL_PRIORITY if c in regions]

        if not vlm_imgs:
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
            sharpest = max(loaded, key=lambda x: cv2.Laplacian(
                cv2.cvtColor(x[1], cv2.COLOR_BGR2GRAY), cv2.CV_64F
            ).var())[1]
            vlm_imgs.append(sharpest)

        vlm_imgs = vlm_imgs[:2]   # cap at 2 — halves Qwen inference time

        print(f"[Phase 3] Qwen2.5-VL on {len(vlm_imgs)} image(s)...")

        # Convert to PIL directly — no CLAHE (preprocessing distorts text recognition)
        pil_images = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for img in vlm_imgs
        ]

        for i, pil in enumerate(pil_images):
            _dbg(f'4_qwen_input_{i}.jpg', pil)

        ocr_context = " ".join(ocr_text_parts)
        qwen_data = self._qwen_extract(pil_images, ocr_context=ocr_context)

        # Regex results are authoritative for dates/batch — Qwen only fills gaps.
        for k, v in qwen_data.items():
            if v and not result.get(k):
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
