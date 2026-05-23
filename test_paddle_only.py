from paddleocr import PaddleOCR
import cv2
import sys
from PIL import Image
import numpy as np

def test_paddle(image_path):
    ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)
    
    try:
        pil_img = Image.open(image_path).convert('RGB')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print(f"Image shape: {img.shape}")
    print(f"Methods: {[m for m in dir(ocr) if not m.startswith('_')]}")
    result = ocr.ocr(img)
    print(f"Raw Result: {result}")
    
    if result and result[0]:
        for line in result[0]:
            print(line)

if __name__ == "__main__":
    test_paddle(sys.argv[1] if len(sys.argv) > 1 else "cola.jpeg")
