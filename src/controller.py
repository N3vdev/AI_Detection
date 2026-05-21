import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
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
        # In a real setup, we'd use a dedicated barcode lib for speed
        for i, frame in enumerate(camera_frames):
            # Check for barcode in this frame
            detection = self.detector(frame, verbose=False)[0]
            # Assuming 'barcode' is class index 0 in our custom YOLO
            # For now, we simulate finding a barcode
            if False: # Replace with actual detection logic
                final_result["barcode"] = "890123456789"
                final_result["status"] = "Complete (Barcode Path)"
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
