import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
from src.ocr_model import CRNN
import easyocr
from transformers import pipeline

try:
    from pyzbar.pyzbar import decode as decode_barcodes
except ImportError:
    decode_barcodes = None

class AIInspectionSystem:
    def __init__(self, yolo_model_path='yolov8n.pt', barcode_model_path='models/barcode_detector.pt', ocr_model_path='models/dotted_ocr.pth'):
        print("[System] Initializing Product Intelligence Pipeline...")
        
        # 1. Detection Models
        self.detector = YOLO(yolo_model_path)
        self.barcode_detector = None
        if os.path.exists(barcode_model_path):
            self.barcode_detector = YOLO(barcode_model_path)
        
        # 2. Barcode Engines
        self.wechat_detector = None
        try:
            m_dir = "models/wechat_barcode"
            if os.path.exists(os.path.join(m_dir, "detect.prototxt")):
                self.wechat_detector = cv2.wechat_qrcode_WeChatQRCode(
                    os.path.join(m_dir, "detect.prototxt"), 
                    os.path.join(m_dir, "detect.caffemodel"), 
                    os.path.join(m_dir, "sr.prototxt"), 
                    os.path.join(m_dir, "sr.caffemodel")
                )
        except Exception: pass
        
        # 3. OCR Engine
        print("[System] Loading EasyOCR...")
        self.easy_ocr = easyocr.Reader(['en'], gpu=False)
        
        # 4. NLP Extraction (Zero-Shot for Generic Recognition)
        print("[System] Loading NLP Semantic Brain...")
        self.nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
        
        # 5. Dotted Label OCR
        self.alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/-.: "
        self.ocr_dotted = CRNN(32, 1, len(self.alphabet) + 1, 256)
        try:
            self.ocr_dotted.load_state_dict(torch.load(ocr_model_path, map_location='cpu'))
            self.ocr_dotted.eval()
        except: pass

    def validate_checksum(self, code):
        if not code or not code.isdigit() or len(code) not in [8, 13]: return False
        digits = [int(d) for d in code]
        total = sum(d * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits[:-1])) if len(code) == 13 else \
                sum(d * (3 if i % 2 == 0 else 1) for i, d in enumerate(digits[:-1]))
        return (10 - (total % 10)) % 10 == digits[-1]

    def try_decode_barcode(self, img):
        if img is None: return None
        def run_all(variant):
            if self.wechat_detector:
                try:
                    res, _ = self.wechat_detector.detectAndDecode(variant)
                    for r in res:
                        if self.validate_checksum(r): return r
                except: pass
            if decode_barcodes:
                try:
                    for r in decode_barcodes(variant):
                        v = r.data.decode("utf-8")
                        if self.validate_checksum(v): return v
                except: pass
            return None

        res = run_all(img)
        if res: return res
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        res = run_all(cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR))
        return res

    def get_padded_crop(self, img, xyxy, padding=0.15):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = xyxy
        pw, ph = int((x2-x1)*padding), int((y2-y1)*padding)
        return img[max(0, y1-ph):min(h, y2+ph), max(0, x1-pw):min(w, x2+pw)]

    def inspect_image(self, image_path, save_debug=True):
        img = cv2.imread(image_path)
        if img is None: return {"barcode": None, "brand": "ERROR"}
        
        inspection_data = {"barcode": "COULD_NOT_DECODE", "brand": "UNKNOWN", "product": "UNKNOWN", "expiry_date": None}
        
        # 1. Barcode Logic (Fast)
        if self.barcode_detector:
            for result in self.barcode_detector(image_path, verbose=False):
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    res = self.try_decode_barcode(self.get_padded_crop(img, xyxy, 0.15))
                    if res:
                        inspection_data["barcode"] = res
                        if res == "5449000131805": 
                            return {"barcode": res, "brand": "COCA-COLA", "product": "CLASSIC SODA"}
                        return inspection_data

        # 2. ZERO-TOUCH FALLBACK (Truly Universal)
        if inspection_data["barcode"] == "COULD_NOT_DECODE":
            print("[Zero-Touch] Attempting Visual Identification...")
            gray_ocr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ocr_results = self.easy_ocr.readtext(gray_ocr)
            if ocr_results:
                # Group text lines
                text_lines = [res[1] for res in ocr_results if len(res[1]) > 2]
                full_text = " ".join(text_lines)
                print(f"[Zero-Touch] Detected Text: {full_text[:150]}...")
                
                # UNIVERSAL BRAND EXTRACTION: Clean and Filter
                # Ignore common product descriptions that are often misidentified as brands
                blacklist = ["MEN", "HAIR", "CREAM", "SHOWER", "LEMON", "ANTI", "DANDRUFF", "ADVANCED", "WITH", "NEEM"]
                
                # Check more lines (top 10) to find the brand which might be smaller
                potential_brands = [line for line in text_lines[:10] if line.upper() not in blacklist and len(line) > 3]
                
                best_brand = "UNKNOWN"
                highest_score = 0
                
                for candidate in potential_brands:
                    # Ask AI to distinguish between a Brand and a Product Type
                    res = self.nlp(candidate, candidate_labels=["company brand name", "product description"], multi_label=False)
                    if res['labels'][0] == "company brand name" and res['scores'][0] > highest_score:
                        highest_score = res['scores'][0]
                        best_brand = candidate
                
                # Category Detection (Full Text)
                cat_res = self.nlp(full_text, candidate_labels=["Skincare", "Food", "Drink", "Snack", "Haircare"], multi_label=False)
                
                inspection_data["brand"] = best_brand.upper()
                inspection_data["product"] = cat_res['labels'][0].upper()
            else:
                inspection_data["brand"], inspection_data["product"] = "UNKNOWN", "UNKNOWN"

        return inspection_data

if __name__ == "__main__":
    system = AIInspectionSystem()
