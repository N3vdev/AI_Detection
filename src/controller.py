import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from pyzbar.pyzbar import decode as decode_barcodes
from src.ocr_model import CRNN

class IndustrialAIController:
    def __init__(self, yolo_path='yolov8n.pt', ocr_path='models/dotted_ocr.pth'):
        # Initialize YOLOv8
        self.detector = YOLO(yolo_path)
        
        # Initialize Custom OCR
        self.alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/-.: "
        self.ocr = CRNN(32, 1, len(self.alphabet) + 1, 256)
        try:
            self.ocr.load_state_dict(torch.load(ocr_path, map_location='cpu'))
            self.ocr.eval()
            print("[System] Custom OCR Brain loaded.")
        except:
            print("[System] OCR weights not found, using uninitialized model.")

    def process_conveyor_event(self, camera_frames):
        """
        camera_frames: List of 4 images [Front, Back, Left, Right]
        """
        start_time = time.time()
        print(f"\n[Event] Processing new product event (4 cameras)...")
        
        final_result = {
            "product_id": "Unknown",
            "barcode": None,
            "brand": None,
            "flavor": None,
            "expiry": None,
            "status": "Incomplete"
        }

        # Step 1: Rapid Barcode Scan on all 4 frames
        for i, frame in enumerate(camera_frames):
            barcode_value = self._scan_barcode(frame)
            if barcode_value:
                final_result["barcode"] = barcode_value
                final_result["status"] = "Complete (Barcode Path)"
                print(f"[Barcode] Found on camera {i}: {barcode_value}")
                return final_result

        # Step 2: Full Vision Path (if no barcode)
        for i, frame in enumerate(camera_frames):
            results = self.detector(frame, verbose=False)[0]
            
            for box in results.boxes:
                label = results.names[int(box.cls[0])]
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                crop = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]

                if label == 'logo':
                    # Extract Brand Name (could be a classifier or OCR)
                    final_result["brand"] = "LAYS" 
                elif label == 'flavor':
                    final_result["flavor"] = "Classic Salted"
                elif label == 'dotted_label':
                    # Our custom OCR logic
                    final_result["expiry"] = self.read_dotted_label(crop)

        # Logic: If we found brand and flavor, we mark as complete
        if final_result["brand"] and final_result["expiry"]:
            final_result["status"] = "Complete (Vision Path)"
        
        elapsed = (time.time() - start_time) * 1000
        print(f"[System] Cycle Time: {elapsed:.2f}ms")
        return final_result

    def _try_decode(self, img):
        # pyzbar
        results = decode_barcodes(img)
        if results:
            return results[0].data.decode("utf-8")

        # OpenCV barcode detector fallback — handles cylindrical/curved distortion better
        try:
            detector = cv2.barcode.BarcodeDetector()
            ok, decoded, _, _ = detector.detectAndDecodeMulti(img)
            if ok:
                for val in decoded:
                    if val:
                        return val
        except AttributeError:
            pass  # opencv-contrib not installed

        return None

    def _preprocess_variants(self, frame):
        """
        Yields preprocessed versions of a frame, cheapest first.
        Each variant targets a different failure mode.
        """
        # Raw — fastest, works when image is already clean
        yield frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Grayscale — removes color noise
        yield gray

        # CLAHE — recovers contrast lost to uneven conveyor lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yield clahe.apply(gray)

        # Sharpen — helps with motion blur from belt movement
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        yield cv2.filter2D(gray, -1, kernel)

        # Otsu threshold — handles high-contrast packaging with heavy shadows
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield otsu

        # Adaptive threshold — handles glare spots and uneven illumination
        yield cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Denoise + CLAHE — for noisy/low-light camera feeds
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        yield clahe.apply(denoised)

    def _scan_barcode(self, frame):
        """
        Tries pyzbar against multiple preprocessed versions of the frame.
        Falls back to rotated variants for products sitting at an angle on the belt.
        Returns decoded string or None.
        """
        for variant in self._preprocess_variants(frame):
            result = self._try_decode(variant)
            if result:
                return result

        # Rotation fallback — for products not sitting square on the belt
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        center = (w // 2, h // 2)
        for angle in [-15, -10, -5, 5, 10, 15]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(gray, M, (w, h))
            result = self._try_decode(rotated)
            if result:
                return result

        return None

    def read_dotted_label(self, crop):
        # Preprocess
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 32))
        img = gray.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            preds = self.ocr(img)
            # CTC Greedy Decoder
            _, max_indices = torch.max(preds, 2)
            max_indices = max_indices.permute(1, 0).cpu().numpy()[0]
            
            char_list = []
            for i in range(len(max_indices)):
                if max_indices[i] != 0 and (not (i > 0 and max_indices[i] == max_indices[i - 1])):
                    char_list.append(self.alphabet[max_indices[i] - 1])
            return "".join(char_list)

if __name__ == "__main__":
    controller = IndustrialAIController()
    # Mock data: 4 blank images
    mock_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(4)]
    res = controller.process_conveyor_event(mock_frames)
    print(f"[Output] Result: {res}")
