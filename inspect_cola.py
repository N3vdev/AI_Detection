import cv2
from src.enhanced_controller import create_enhanced_inspector
import numpy as np

from PIL import Image

def inspect_image(image_path):
    inspector = create_enhanced_inspector(
        yolo_path='yolov8n.pt',
        ocr_path='models/dotted_ocr.pth',
        max_workers=2
    )
    
    try:
        pil_img = Image.open(image_path).convert('RGB')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Could not load {image_path}: {e}")
        return

    print(f"Analyzing {image_path}...")
    
    # Try generic OCR on the whole image (or large crops)
    h, w = img.shape[:2]
    # Center crop
    center = img[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
    
    print("\n--- Generic OCR Results ---")
    generic_results = inspector.read_generic_text_fast(img)
    for text, conf, box in generic_results:
        print(f"[{conf:.2f}] {text}")
        parsed = inspector.parse_dotted_info(text)
        if parsed["expiry"] or parsed["batch"]:
            print(f"  -> Parsed: {parsed}")

    print("\n--- Dotted OCR Results on heuristic regions ---")
    heuristics = inspector.extract_dotted_label_roi(img, None)
    for region_type, crop in heuristics:
        dotted_text, dotted_conf = inspector.read_dotted_label_optimized(crop)
        if dotted_text:
            print(f"[{region_type}] {dotted_text} (conf: {dotted_conf:.2f})")
            parsed = inspector.parse_dotted_info(dotted_text)
            print(f"  -> Parsed: {parsed}")

    inspector.shutdown()

if __name__ == "__main__":
    import sys
    image = sys.argv[1] if len(sys.argv) > 1 else "cola.jpeg"
    inspect_image(image)
