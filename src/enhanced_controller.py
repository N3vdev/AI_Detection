import cv2
import torch
import numpy as np
import time
import os
import re
from ultralytics import YOLO
from pyzbar.pyzbar import decode as decode_barcodes
from src.ocr_model import CRNN
from paddleocr import PaddleOCR
import threading
from queue import Queue
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor

class EnhancedConveyorInspector:
    """
    High-performance conveyor belt inspection system optimized for live feed extraction
    Designed for low-latency, high-throughput product identification
    """

    def __init__(self,
                 yolo_path='yolov8n.pt',
                 ocr_path='models/dotted_ocr.pth',
                 barcode_model_path='models/barcode_detector.pt',
                 wechat_model_dir='models/wechat_barcode',
                 max_workers=4,
                 frame_buffer_size=2):

        print("[Enhanced Inspector] Initializing High-Performance Conveyor System...")

        # === CORE MODELS ===
        self.detector = YOLO(yolo_path)  # General object detection
        self.detector_lock = threading.Lock() # Ensure thread-safety for shared model
        self.barcode_detector = None
        if os.path.exists(barcode_model_path):
            self.barcode_detector = YOLO(barcode_model_path)
        
        # Mapping for standard COCO models to our system's labels
        self.label_map = {
            'bottle': 'product',
            'cup': 'product',
            'wine glass': 'product',
            'bowl': 'product',
            'hair drier': 'product',
            'toothbrush': 'product'
        }

        # === OCR SYSTEMS ===
        self.alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/-.: "
        self.dotted_ocr = CRNN(32, 1, len(self.alphabet) + 1, 256)
        try:
            self.dotted_ocr.load_state_dict(torch.load(ocr_path, map_location='cpu'))
            self.dotted_ocr.eval()
            print("[Enhanced Inspector] Dotted OCR loaded successfully")
        except Exception as e:
            print(f"[Warning] Dotted OCR not loaded: {e}")
            self.dotted_ocr = None

        # === BARCODE ENGINES ===
        self.wechat_detector = None
        try:
            # Check if WeChat detector is available in cv2
            if hasattr(cv2, 'wechat_qrcode_WeChatQRCode') and os.path.exists(os.path.join(wechat_model_dir, "detect.prototxt")):
                self.wechat_detector = cv2.wechat_qrcode_WeChatQRCode(
                    os.path.join(wechat_model_dir, "detect.prototxt"),
                    os.path.join(wechat_model_dir, "detect.caffemodel"),
                    os.path.join(wechat_model_dir, "sr.prototxt"),
                    os.path.join(wechat_model_dir, "sr.caffemodel")
                )
        except Exception as e:
            pass # Silently fail if not available

        # === GENERIC OCR (For Autonomous Identification) ===
        try:
            # use_textline_orientation=True helps with rotated products
            # enable_mkldnn=False avoids NotImplementedError on Windows
            self.generic_ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)
            print("[Enhanced Inspector] Generic OCR (PaddleOCR) initialized")
        except Exception as e:
            print(f"[Warning] Generic OCR not available: {e}")
            self.generic_ocr = None

        try:
            from pyzbar.pyzbar import decode as decode_barcodes
            self.pyzoobar_available = True
        except ImportError:
            self.pyzoobar_available = False
            self.pyzoobar_decode = None

        # === PERFORMANCE OPTIMIZATIONS ===
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.frame_buffer = deque(maxlen=frame_buffer_size)
        self.processing_times = {
            'barcode': deque(maxlen=100),
            'dotted_ocr': deque(maxlen=100),
            'visual': deque(maxlen=100),
            'total': deque(maxlen=100)
        }

        # === PRODUCT DATABASE (EXTENDABLE) ===
        self.product_db = {
            "8901088026864": {"brand": "MARICO", "product": "PARACHUTE HAIR OIL"},
            "8906087779292": {"brand": "MARICO", "product": "HAIR CREAM"},
            "5449000131805": {"brand": "COCA-COLA", "product": "CLASSIC SODA"},
            "8901088133333": {"brand": "PARACHUTE", "product": "ADVANCED ALOE VERA"}
        }

        # === CACHING FOR LIVE FEED ===
        self.recent_detections = {}  # Cache recent results for tracking
        self.cache_timeout = 0.5  # seconds

        # === PREPROCESSING KERNELS (PRE-COMPUTED) ===
        self.sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        print("[Enhanced Inspector] System initialized and ready for live feed")

    def preprocess_variants_fast(self, frame):
        """
        Ultra-fast preprocessing variants for barcode detection
        Returns generator of preprocessed frames ordered by speed/effectiveness
        """
        # Variant 0: Original (fastest)
        yield frame

        # Variant 1: Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yield gray

        # Variant 2: CLAHE enhanced (for uneven lighting)
        yield self.clahe.apply(gray)

        # Variant 3: Sharpened (for motion blur)
        yield cv2.filter2D(gray, -1, self.sharpen_kernel)

        # Variant 4: Otsu threshold (high contrast)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield otsu

        # Variant 5: Adaptive threshold (glare/spots)
        yield cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Variant 6: Denoised + CLAHE (noisy feeds)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        yield self.clahe.apply(denoised)

    def decode_barcode_ultra_fast(self, img):
        """
        Ultra-fast barcode decoding with early exit strategy
        Returns decoded string or None within milliseconds
        """
        start_time = time.perf_counter()

        if img is None or img.size == 0:
            return None

        # Try each preprocessing variant - exit early on success
        for variant in self.preprocess_variants_fast(img):
            # 1. Try pyzbar (usually fastest and most reliable)
            if self.pyzoobar_available:
                try:
                    results = decode_barcodes(variant)
                    for r in results:
                        data = r.data.decode("utf-8")
                        if self.validate_checksum_fast(data):
                            self.processing_times['barcode'].append(time.perf_counter() - start_time)
                            return data
                except:
                    pass  # Continue to next method

            # 2. Try WeChat detector (good for distorted/curved surfaces)
            if self.wechat_detector:
                try:
                    res, _ = self.wechat_detector.detectAndDecode(variant)
                    for r in res:
                        if r and self.validate_checksum_fast(r):
                            self.processing_times['barcode'].append(time.perf_counter() - start_time)
                            return r
                except:
                    pass

            # 3. Try OpenCV barcode detector (fallback)
            try:
                detector = cv2.barcode.BarcodeDetector()
                ok, decoded, _, _ = detector.detectAndDecodeMulti(variant)
                if ok:
                    for val in decoded:
                        if val and self.validate_checksum_fast(val):
                            self.processing_times['barcode'].append(time.perf_counter() - start_time)
                            return val
            except:
                pass  # OpenCV contrib might not be installed

        self.processing_times['barcode'].append(time.perf_counter() - start_time)
        return None

    def validate_checksum_fast(self, code):
        """Ultra-fast checksum validation for UPC/EAN"""
        if not code or not code.isdigit() or len(code) not in [8, 12, 13]:
            return False

        # Optimized checksum calculation
        if len(code) == 12:  # UPC-A
            odd_sum = sum(int(code[i]) for i in range(0, 11, 2))
            even_sum = sum(int(code[i]) for i in range(1, 11, 2))
            checksum = (odd_sum * 3 + even_sum) % 10
            return (10 - checksum) % 10 == int(code[11])
        elif len(code) == 13:  # EAN-13
            odd_sum = sum(int(code[i]) for i in range(0, 12, 2))
            even_sum = sum(int(code[i]) for i in range(1, 12, 2))
            checksum = (odd_sum + even_sum * 3) % 10
            return (10 - checksum) % 10 == int(code[12])
        elif len(code) == 8:   # UPC-E
            # Simplified validation for UPC-E
            return code[-3:] == "000" or code[-3:] == "100"  # Basic check
        return False

    def extract_dotted_label_roi(self, frame, detections):
        """
        intelligently extract regions likely to contain dotted labels
        based on common packaging patterns
        """
        h, w = frame.shape[:2]
        dotted_regions = []

        # Common dotted label locations:
        # 1. Bottom edges (expiry/batch codes often here)
        # 2. Top edges near corners
        # 3. Side seams/flaps

        # Add bottom region (20% height from bottom)
        bottom_region = frame[int(h*0.8):h, :]
        if bottom_region.size > 0:
            dotted_regions.append(("bottom", bottom_region))

        # Add top region (15% height from top)
        top_region = frame[0:int(h*0.15), :]
        if top_region.size > 0:
            dotted_regions.append(("top", top_region))

        # Add left/right edge regions (15% width from edges)
        left_region = frame[:, 0:int(w*0.15)]
        right_region = frame[:, int(w*0.85):w]
        if left_region.size > 0:
            dotted_regions.append(("left", left_region))
        if right_region.size > 0:
            dotted_regions.append(("right", right_region))

        # If we have object detections, focus on their vicinity
        if detections is not None and len(detections.boxes) > 0:
            for box in detections.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                # Expand box slightly to catch label areas
                exp_x1 = max(0, x1 - 10)
                exp_y1 = max(0, y1 - 10)
                exp_x2 = min(w, x2 + 10)
                exp_y2 = min(h, y2 + 10)

                expanded_crop = frame[exp_y1:exp_y2, exp_x1:exp_x2]
                if expanded_crop.size > 0:
                    dotted_regions.append(("object_vicinity", expanded_crop))

        return dotted_regions

    def read_generic_text_fast(self, crop):
        """
        Extract all text from a crop using PaddleOCR
        Returns list of (text, confidence, box) tuples
        """
        if self.generic_ocr is None or crop is None or crop.size == 0:
            return []

        try:
            # PaddleOCR inference
            result = self.generic_ocr.ocr(crop)
            if not result:
                return []

            extracted = []
            
            # Handle new dictionary-based format (PaddleX/OCRv5 wrapper)
            if isinstance(result[0], dict):
                data = result[0]
                texts = data.get('rec_texts', [])
                scores = data.get('rec_scores', [])
                polys = data.get('rec_polys', [])
                
                for i in range(len(texts)):
                    text = texts[i]
                    conf = scores[i]
                    box = polys[i] if i < len(polys) else None
                    if conf > 0.4:
                        extracted.append((text, conf, box))
            
            # Handle standard list-based format
            elif isinstance(result[0], list):
                for line in result[0]:
                    box = line[0]
                    text = line[1][0]
                    conf = line[1][1]
                    if conf > 0.4:
                        extracted.append((text, conf, box))
                        
            return extracted
        except Exception as e:
            return []

    def extract_brand_product_autonomous(self, ocr_results):
        """
        Heuristically identify Brand and Product from a list of OCR strings
        Uses font size (box height) and filters common non-brand words
        """
        if not ocr_results:
            return None, None

        # Calculate heights for each box to estimate font size
        scored_results = []
        stop_words = {
            'THE', 'AND', 'FOR', 'WITH', 'NEW', 'BEST', 'OF', 'NET', 'WT', 'G', 'ML', 'OZ',
            'ORIGINAL', 'TASTE', 'DELICIOUS', 'REFRESHING', 'BOTTLES', 'FL', 'CLASSIC',
            'SERVE', 'CHILLED', 'INGREDIENTS', 'WATER', 'SUGAR', 'CARBONATED', 'MEN', 'LEMON',
            'RIOINAL', 'ORIGIN', 'NATURAL', 'EXTRACT'
        }

        for text, conf, box in ocr_results:
            text_upper = text.strip().upper()
            if len(text_upper) < 2 or text_upper.isdigit(): continue
            if any(char in text_upper for char in '/-.'): continue
            
            # Filter out pure stop words
            if text_upper in stop_words: continue
            
            # Height of the bounding box
            if box is not None:
                try:
                    # Handle different box formats (list of 4 points or [x1, y1, x2, y2])
                    if len(box) == 4 and isinstance(box[0], (list, np.ndarray)):
                        height = abs(box[2][1] - box[0][1])
                    else:
                        height = abs(box[3] - box[1])
                except:
                    height = 10
            else:
                height = 10 # Default
                
            scored_results.append({
                'text': text_upper,
                'conf': conf,
                'height': height
            })

        if not scored_results:
            return None, None

        # Sort by height (largest text first)
        scored_results.sort(key=lambda x: x['height'], reverse=True)
        
        # The largest text is very likely the brand
        brand = scored_results[0]['text']
        
        # The next few large/confident texts are likely the product type
        product_parts = []
        for item in scored_results[1:5]:
            if item['text'] != brand and item['text'] not in stop_words:
                # Avoid repeats
                if item['text'] not in product_parts:
                    product_parts.append(item['text'])

        product = " ".join(product_parts[:2]) if product_parts else None
        return brand, product

    def read_dotted_label_optimized(self, crop):
        """
        Optimized dotted label reading with preprocessing pipeline
        Returns (text, confidence)
        """
        if self.dotted_ocr is None or crop is None or crop.size == 0:
            return None, 0

        start_time = time.perf_counter()

        try:
            # Standard preprocessing for dotted OCR (matching training)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 32))  # Standard size from training

            # Normalize to [-1, 1] range as expected by model
            img = gray.astype(np.float32) / 255.0
            img = (img - 0.5) / 0.5  # [-1, 1] normalization
            img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

            # Inference
            with torch.no_grad():
                preds = self.dotted_ocr(img)
                probs = torch.softmax(preds, 2)
                max_probs, max_indices = torch.max(probs, 2)
                
                # CTC Greedy Decoder
                indices = max_indices.permute(1, 0).cpu().numpy()[0]
                confidences = max_probs.permute(1, 0).cpu().numpy()[0]

                char_list = []
                conf_list = []
                for i in range(len(indices)):
                    if indices[i] != 0 and (not (i > 0 and indices[i] == indices[i - 1])):
                        char_list.append(self.alphabet[indices[i] - 1])
                        conf_list.append(confidences[i])
                
                result = "".join(char_list)
                avg_conf = np.mean(conf_list) if conf_list else 0

                self.processing_times['dotted_ocr'].append(time.perf_counter() - start_time)
                return (result if result.strip() else None), avg_conf

        except Exception as e:
            print(f"[Dotted OCR Error] {e}")
            self.processing_times['dotted_ocr'].append(time.perf_counter() - start_time)
            return None, 0

    def parse_dotted_info(self, dotted_text):
        """
        Parse dotted label text to extract useful information
        Handles formats like: EXP 12/02/25, BATCH A1B2C3, MFD 01/24, etc.
        """
        if not dotted_text:
            return {"expiry": None, "batch": None, "mfd": None, "raw": dotted_text}

        # Pre-clean: common OCR errors
        raw_text = dotted_text.strip().upper()
        # Remove common hallucinations or noise characters at start/end
        raw_text = re.sub(r'^[^A-Z0-9]+|[^A-Z0-9]+$', '', raw_text)
        
        result = {"expiry": None, "batch": None, "mfd": None, "raw": dotted_text}

        def clean_digits(t):
            # OCR correction specifically for digit-heavy parts
            return t.replace('I', '1').replace('L', '1').replace('O', '0').replace('S', '5').replace('Z', '2').replace('B', '8').replace('G', '6')

        # 1. Look for date patterns (Expiry or MFD)
        # Patterns for DD/MM/YYYY, DD-MM-YY, MM.YY, etc.
        date_pattern = r'([A-Z0-9]{1,2})[/\-\.\s]([A-Z0-9]{1,2})(?:[/\-\.\s]([A-Z0-9]{2,4}))?'
        
        # Find all potential dates
        for match in re.finditer(date_pattern, raw_text):
            groups = match.groups()
            g1, g2, g3 = [clean_digits(g) if g else None for g in groups]
            
            date_str = None
            if g1 and g2 and g3:
                if g1.isdigit() and g2.isdigit() and g3.isdigit():
                    if len(g3) == 2: g3 = f"20{g3}"
                    # Basic validation: month <= 12, day <= 31
                    if 1 <= int(g2) <= 12 and 1 <= int(g1) <= 31:
                        date_str = f"{g1.zfill(2)}/{g2.zfill(2)}/{g3}"
            elif g1 and g2 and not g3:
                if g1.isdigit() and g2.isdigit():
                    if len(g2) == 2: g2 = f"20{g2}"
                    if 1 <= int(g1) <= 12:
                        date_str = f"{g1.zfill(2)}/{g2}"

            if date_str:
                # Determine if it's MFD or EXP based on context
                context = raw_text[max(0, match.start()-10):match.start()]
                if 'MFD' in context or 'MFG' in context or 'PKD' in context:
                    result["mfd"] = date_str
                else:
                    # Default to expiry if not sure, or if 'EXP' is present
                    result["expiry"] = date_str

        # 2. Look for batch/lot patterns
        batch_patterns = [
            r'(?:BATCH|LOT|LT|BN|B\.?N\.?|L\.?N\.?)[\s:]?([A-Z0-9\-]+)',
            r'#[\s:]?([A-Z0-9\-]+)'
        ]
        
        for pattern in batch_patterns:
            match = re.search(pattern, raw_text)
            if match:
                batch = match.group(1)
                if len(batch) >= 3:
                    result["batch"] = batch
                    break
        
        if not result["batch"]:
            # Fallback: identify parts that aren't dates and look like codes
            parts = re.split(r'[\s,;]+', raw_text)
            for p in parts:
                p = p.strip()
                # Skip known words
                if p in {'EXP', 'EXPIRY', 'MFD', 'MFG', 'BATCH', 'LOT', 'NET', 'WT', 'DATE'}: continue
                # Look for alphanumeric codes (e.g. A123B, 14ZZ0Z)
                if 5 <= len(p) <= 15 and any(c.isalpha() for c in p) and any(c.isdigit() for c in p):
                    # Avoid picking up dates we already found
                    if result["expiry"] and p in result["expiry"]: continue
                    if result["mfd"] and p in result["mfd"]: continue
                    result["batch"] = p
                    break

        return result

    def process_conveyor_frame_optimized(self, camera_frames, frame_id=None):
        """
        Main processing function optimized for live conveyor belt feed
        Processes 4 camera views: [Front, Back, Left, Right]
        Returns structured product information with timing metrics
        """
        overall_start = time.perf_counter()

        if frame_id is None:
            frame_id = int(time.time() * 1000)  # Use timestamp as ID

        # Check cache first for very recent frames (temporal coherence)
        cache_key = str(hash(str([f.shape for f in camera_frames])))[:16]
        if cache_key in self.recent_detections:
            cached_time, cached_result = self.recent_detections[cache_key]
            if time.time() - cached_time < self.cache_timeout:
                # Return cached result with updated frame ID
                result = cached_result.copy()
                result["frame_id"] = frame_id
                result["from_cache"] = True
                self.processing_times['total'].append(time.perf_counter() - overall_start)
                return result

        final_result = {
            "frame_id": frame_id,
            "timestamp": time.time(),
            "product_id": "Unknown",
            "barcode": None,
            "brand": None,
            "product": None,
            "color": None,
            "expiry_date": None,
            "mfd_date": None,
            "batch_code": None,
            "status": "Processing",
            "processing_time_ms": 0,
            "from_cache": False,
            "sources": []  # Track what methods contributed
        }

        # === STAGE 1: PARALLEL BARCODE SCANNING (HIGHEST PRIORITY) ===
        # Submit barcode tasks for all 4 frames in parallel
        barcode_futures = []
        for i, frame in enumerate(camera_frames):
            future = self.executor.submit(self.decode_barcode_ultra_fast, frame)
            barcode_futures.append((i, future, frame))

        # Check results as they complete (early exit on first success)
        barcode_found = False
        for cam_idx, future, frame in barcode_futures:
            try:
                # Increased timeout for CPU - 200ms
                barcode_result = future.result(timeout=0.2)
                if barcode_result:
                    final_result["barcode"] = barcode_result
                    final_result["status"] = "Complete (Barcode Path)"
                    final_result["sources"].append(f"barcode_cam{cam_idx}")
                    barcode_found = True
                    
                    # Process barcode-specific info from database
                    if barcode_result in self.product_db:
                        info = self.product_db[barcode_result]
                        final_result["brand"] = info["brand"]
                        final_result["product"] = info["product"]
                    break  # Early exit - we got what we need
            except Exception as e:
                continue  # Try next camera

        # === STAGE 2: VISION PATH (if no barcode or need more info) ===
        if not barcode_found or final_result["brand"] is None:
            # Run YOLO detection on all frames in parallel
            # We use a lock because shared YOLO instances aren't strictly thread-safe
            def locked_detect(f):
                with self.detector_lock:
                    return self.detector(f, verbose=False)[0]

            detection_futures = []
            for i, frame in enumerate(camera_frames):
                future = self.executor.submit(locked_detect, frame)
                detection_futures.append((i, future))

            # Process detection results
            brand_candidates = []
            product_candidates = []
            color_candidates = []
            dotted_regions_to_process = []

            for cam_idx, future in detection_futures:
                try:
                    # Increased timeout for CPU - 5.0s
                    # Note: Since they are locked, they will run sequentially.
                    results = future.result(timeout=5.0)

                    # Also extract ROIs based on heuristics for this frame
                    heuristics_regions = self.extract_dotted_label_roi(camera_frames[cam_idx], results)
                    for region_type, crop in heuristics_regions:
                        dotted_regions_to_process.append((region_type, cam_idx, 0.4, crop))

                    # Process detections for this camera
                    for box in results.boxes:
                        label = results.names[int(box.cls[0])]
                        confidence = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)

                        # Map standard COCO labels to our system if needed
                        if label in self.label_map:
                            label = self.label_map[label]

                        # Extract crop for further processing
                        crop = camera_frames[cam_idx][xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]

                        if label == 'logo' and confidence > 0.3:
                            brand_candidates.append(("logo", cam_idx, confidence, crop))
                            dotted_regions_to_process.append(("logo", crop))

                        elif label == 'product' and confidence > 0.3:
                            product_candidates.append(("product", cam_idx, confidence, crop))

                        elif label == 'color_indicator' and confidence > 0.3:
                            color_candidates.append(("color", cam_idx, confidence, crop))

                        elif label == 'dotted_label' or label == 'text_region' or label == 'flavor_text':
                            dotted_regions_to_process.append(("detection", cam_idx, confidence, crop))

                except Exception as e:
                    error_type = type(e).__name__
                    print(f"[Detection Error Cam{cam_idx}] {error_type}: {e}")
                    # Even if detection fails, try heuristics on this frame
                    heuristics_regions = self.extract_dotted_label_roi(camera_frames[cam_idx], None)
                    for region_type, crop in heuristics_regions:
                        dotted_regions_to_process.append((region_type, cam_idx, 0.2, crop))
                    continue

            # === STAGE 3: OCR LABEL PROCESSING ===
            # Process all candidate dotted label and text regions
            best_dotted_result = None
            best_dotted_confidence = 0
            
            # Hallucination detection: track identical results from different regions
            ocr_counts = {}
            valid_results = []

            # First pass: gather results
            for region_info in dotted_regions_to_process:
                if len(region_info) == 3:  # From detection
                    region_type, cam_idx, confidence, crop = (*region_info, None)
                elif len(region_info) == 4:  # With explicit confidence
                    region_type, cam_idx, confidence, crop = region_info
                else:  # From ROI extraction
                    region_type, crop = region_info
                    cam_idx, confidence = 0, 0.5  # defaults

                # A. Try Dotted OCR
                dotted_text, dotted_conf = self.read_dotted_label_optimized(crop)
                if dotted_text and dotted_conf > 0.3: # Threshold for dotted OCR
                    ocr_counts[dotted_text] = ocr_counts.get(dotted_text, 0) + 1
                    valid_results.append({
                        "text": dotted_text,
                        "conf": dotted_conf,
                        "type": f"dotted_{region_type}",
                        "cam": cam_idx,
                        "source": "dotted_ocr"
                    })

                # B. Try Generic OCR on the same region if it's a "dotted_label" or "text_region"
                if region_type in ["dotted_label", "text_region", "detection"]:
                    generic_results = self.read_generic_text_fast(crop)
                    for text, conf, _ in generic_results:
                        if conf > 0.4:
                            valid_results.append({
                                "text": text,
                                "type": f"generic_{region_type}",
                                "cam": cam_idx,
                                "source": "generic_ocr"
                            })

            # Second pass: Filter hallucinations and score results
            for res in valid_results:
                text = res["text"]
                
                # A. Hallucination filter for Dotted OCR
                if res["source"] == "dotted_ocr":
                    # 1. Identical results across many regions
                    if ocr_counts[text] > len(dotted_regions_to_process) // 2 and len(text) > 4:
                        continue
                    
                    # 2. Suspicious patterns (common hallucinations)
                    if any(p in text for p in ["ZZ", "00Z", "DWO", "14Z", "0Z0"]):
                        if res["conf"] < 0.7: continue # Require very high confidence for these
                
                parsed = self.parse_dotted_info(text)
                
                # Score based on having useful information
                score = 0
                if parsed["expiry"]: score += 3
                if parsed["mfd"]: score += 2
                if parsed["batch"]: score += 1
                
                if score > best_dotted_confidence:
                    best_dotted_confidence = score
                    best_dotted_result = parsed
                    best_res_meta = res

                # Early exit if we got a high-confidence expiry/mfd
                if score >= 3:
                    final_result["expiry_date"] = parsed["expiry"]
                    final_result["mfd_date"] = parsed["mfd"]
                    final_result["batch_code"] = parsed["batch"]
                    final_result["sources"].append(f"{res['source']}_{res['type']}")
                    break

            # Apply best result if no early exit
            if best_dotted_result and not final_result["expiry_date"]:
                final_result["expiry_date"] = best_dotted_result["expiry"]
                final_result["mfd_date"] = best_dotted_result["mfd"]
                final_result["batch_code"] = best_dotted_result["batch"]
                final_result["sources"].append(f"{best_res_meta['source']}_label")

            # === STAGE 4: BRAND/PRODUCT INFERENCE ===
            # If we still don't have brand/product, try to infer autonomously using Generic OCR
            if final_result["brand"] is None:
                all_found_text = []
                
                # Option A: Use YOLO candidates
                if brand_candidates or product_candidates:
                    candidates = brand_candidates + product_candidates
                    candidates.sort(key=lambda x: x[2], reverse=True)
                    for _, cam_idx, conf, crop in candidates[:3]:
                        all_found_text.extend(self.read_generic_text_fast(crop))
                
                # Option B: Global fallback - Try ALL frames to find any readable brand/product
                if not all_found_text and self.generic_ocr:
                    for i, frame in enumerate(camera_frames):
                        # Focus OCR on the central part of the frame where product is most likely
                        h, w = frame.shape[:2]
                        # Use a more generous center crop
                        center_crop = frame[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)]
                        all_found_text.extend(self.read_generic_text_fast(center_crop))
                        if len(all_found_text) > 8: break # Got enough text
                
                if all_found_text:
                    brand, product = self.extract_brand_product_autonomous(all_found_text)
                    if brand:
                        final_result["brand"] = brand
                        final_result["sources"].append("ocr_autonomous_brand")
                    if product:
                        final_result["product"] = product
                        final_result["sources"].append("ocr_autonomous_product")

            # Final fallbacks if OCR failed
            if final_result["brand"] is None and brand_candidates:
                final_result["brand"] = "DETECTED_BRAND"
                final_result["sources"].append("visual_brand")

            if final_result["product"] is None and product_candidates:
                final_result["product"] = "DETECTED_PRODUCT"
                final_result["sources"].append("visual_product")

            # Set status based on what we found
            found_something = (final_result["barcode"] or final_result["brand"] or 
                              final_result["product"] or final_result["expiry_date"])
            
            if found_something:
                if final_result["status"] == "Processing":
                    final_result["status"] = "Complete (Autonomous Vision)"
            elif not barcode_found:
                final_result["status"] = "Incomplete - No recognizable features"

        # === PERFORMANCE TRACKING ===
        total_time = time.perf_counter() - overall_start
        final_result["processing_time_ms"] = total_time * 1000
        self.processing_times['total'].append(total_time)

        # === CACHE RESULT FOR TEMPORAL COHERENCE ===
        self.recent_detections[cache_key] = (time.time(), final_result.copy())
        # Limit cache size
        if len(self.recent_detections) > 50:
            # Remove oldest entries
            oldest_key = min(self.recent_detections.keys(),
                           key=lambda k: self.recent_detections[k][0])
            del self.recent_detections[oldest_key]

        # Log performance periodically
        if frame_id % 30 == 0:  # Every 30 frames
            avg_barcode = np.mean(self.processing_times['barcode']) * 1000 if self.processing_times['barcode'] else 0
            avg_dotted = np.mean(self.processing_times['dotted_ocr']) * 1000 if self.processing_times['dotted_ocr'] else 0
            avg_total = np.mean(self.processing_times['total']) * 1000 if self.processing_times['total'] else 0
            print(f"[Performance] Frame {frame_id}: "
                  f"Barcode:{avg_barcode:.1f}ms Dotted:{avg_dotted:.1f}ms Total:{avg_total:.1f}ms "
                  f"Status:{final_result['status']}")

        return final_result

    def process_conveyor_batch(self, frame_batch):
        """
        Process a batch of frames for maximum throughput
        Useful when you have bursts of frames from high-speed cameras
        """
        futures = []
        for frame_data in frame_batch:
            if isinstance(frame_data, tuple) and len(frame_data) == 2:
                frame_id, camera_frames = frame_data
                future = self.executor.submit(self.process_conveyor_frame_optimized, camera_frames, frame_id)
            else:
                # Assume just camera frames
                future = self.executor.submit(self.process_conveyor_frame_optimized, frame_data)
            futures.append(future)

        # Collect results in order
        results = []
        for future in futures:
            try:
                result = future.result(timeout=1.0)  # 1 second timeout per frame
                results.append(result)
            except Exception as e:
                print(f"[Batch Processing Error] {e}")
                results.append({
                    "frame_id": len(results),
                    "status": "Error",
                    "error": str(e),
                    "processing_time_ms": 0
                })

        return results

    def get_performance_stats(self):
        """Get current performance statistics"""
        stats = {}
        for key, times in self.processing_times.items():
            if times:
                times_ms = [t * 1000 for t in times]  # Convert to milliseconds
                stats[key] = {
                    "avg_ms": np.mean(times_ms),
                    "min_ms": np.min(times_ms),
                    "max_ms": np.max(times_ms),
                    "p95_ms": np.percentile(times_ms, 95) if len(times_ms) >= 20 else np.max(times_ms),
                    "samples": len(times_ms)
                }
            else:
                stats[key] = {"avg_ms": 0, "min_ms": 0, "max_ms": 0, "p95_ms": 0, "samples": 0}
        return stats

    def shutdown(self):
        """Clean shutdown of the inspection system"""
        print("[Enhanced Inspector] Shutting down...")
        self.executor.shutdown(wait=True)
        print("[Enhanced Inspector] Shutdown complete")

# Factory function for easy instantiation
def create_enhanced_inspector(**kwargs):
    """Factory function to create an EnhancedConveyorInspector with defaults"""
    return EnhancedConveyorInspector(**kwargs)

# Example usage and testing functions
if __name__ == "__main__":
    import os

    # Create inspector
    inspector = create_enhanced_inspector(
        yolo_path='yolov8n.pt',
        ocr_path='models/dotted_ocr.pth',
        max_workers=4
    )

    print("Enhanced Conveyor Inspector ready for live feed processing")
    print("Use inspector.process_conveyor_frame_optimized([front, back, left, right])")
    print("For batch processing: inspector.process_conveyor_batch([(id1, frames1), (id2, frames2), ...])")

    # Example of how to use with mock data
    # mock_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(4)]
    # result = inspector.process_conveyor_frame_optimized(mock_frames, frame_id=1)
    # print(f"Result: {result}")

    # Clean shutdown example
    # inspector.shutdown()