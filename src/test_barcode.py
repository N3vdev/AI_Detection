import sys
import cv2
from src.controller import IndustrialAIController

def test_barcode(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[Error] Could not load image: {image_path}")
        return

    controller = IndustrialAIController.__new__(IndustrialAIController)

    print(f"[Test] Scanning: {image_path}")
    for i, variant in enumerate(controller._preprocess_variants(frame)):
        result = controller._try_decode(variant)
        if result:
            print(f"[Found] Decoded on variant {i+1}: {result}")
            return

    # Try rotations
    import numpy as np
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    center = (w // 2, h // 2)
    for angle in [-15, -10, -5, 5, 10, 15]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h))
        result = controller._try_decode(rotated)
        if result:
            print(f"[Found] Decoded at rotation {angle}°: {result}")
            return

    print("[Failed] No barcode found in any variant.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.test_barcode <image_path>")
    else:
        test_barcode(sys.argv[1])
