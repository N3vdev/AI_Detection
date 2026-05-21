import cv2
import torch
import numpy as np
from ultralytics import YOLO
from src.ocr_model import CRNN

class AIInspectionSystem:
    def __init__(self, yolo_model_path='yolov8n.pt', ocr_model_path='models/dotted_ocr.pth'):
        # 1. Load YOLOv8 for detection
        self.detector = YOLO(yolo_model_path)
        
        # 2. Load our custom Dotted OCR
        self.alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/-.: "
        self.n_classes = len(self.alphabet) + 1
        self.ocr = CRNN(32, 1, self.n_classes, 256)
        
        if torch.cuda.is_available():
            self.ocr = self.ocr.cuda()
            
        try:
            self.ocr.load_state_dict(torch.load(ocr_model_path, map_location='cpu'))
            self.ocr.eval()
            print("OCR Model loaded successfully.")
        except:
            print("Warning: OCR model weights not found. Running with uninitialized weights.")

    def decode_prediction(self, output):
        """CTC Decoding"""
        # Get max index
        _, max_indices = torch.max(output, 2)
        max_indices = max_indices.permute(1, 0).cpu().numpy()[0] # [B, W] -> [W]
        
        # Remove duplicates and blanks
        char_list = []
        for i in range(len(max_indices)):
            if max_indices[i] != 0 and (not (i > 0 and max_indices[i] == max_indices[i - 1])):
                char_list.append(self.alphabet[max_indices[i] - 1])
        return "".join(char_list)

    def process_dotted_label(self, crop):
        # Preprocess for OCR
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 32))
        image = gray.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0) # [1, 1, 32, 128]
        
        if torch.cuda.is_available():
            image = image.cuda()
            
        with torch.no_grad():
            preds = self.ocr(image)
            text = self.decode_prediction(preds)
        return text

    def inspect_image(self, image_path):
        results = self.detector(image_path)
        
        inspection_data = {
            "barcode": None,
            "brand": None,
            "expiry_date": None
        }
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                label = result.names[cls]
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                
                # Crop the detected region
                crop = result.orig_img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                
                if label == 'barcode':
                    # Use a decoder library here
                    inspection_data["barcode"] = "DECODED_VALUE" 
                elif label == 'logo':
                    inspection_data["brand"] = "LAYS" # Placeholder
                elif label == 'dotted_label':
                    # Use our custom OCR
                    inspection_data["expiry_date"] = self.process_dotted_label(crop)
                    
        return inspection_data

if __name__ == "__main__":
    system = AIInspectionSystem()
    # In a real scenario, you would pass the camera frame here
    # result = system.inspect_image("test_product.jpg")
    # print(result)
