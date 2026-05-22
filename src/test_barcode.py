import sys
import cv2
import numpy as np
from pyzbar.pyzbar import decode as decode_barcodes

VARIANT_NAMES = [
    "Raw",
    "Grayscale",
    "CLAHE",
    "Sharpen",
    "Otsu threshold",
    "Adaptive threshold",
    "Denoise + CLAHE",
]

def try_decode(img):
    results = decode_barcodes(img)
    if results:
        return results[0].data.decode("utf-8")
    try:
        detector = cv2.barcode.BarcodeDetector()
        ok, decoded, _, _ = detector.detectAndDecodeMulti(img)
        if ok:
            for val in decoded:
                if val:
                    return val
    except AttributeError:
        pass
    return None

def preprocess_variants(frame):
    yield frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    yield gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yield clahe.apply(gray)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    yield cv2.filter2D(gray, -1, kernel)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield otsu
    yield cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    yield clahe.apply(denoised)

def test_barcode(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[Error] Could not load image: {image_path}")
        return

    print(f"[Test] Image: {image_path}  size: {frame.shape[1]}x{frame.shape[0]}")
    print()

    for i, variant in enumerate(preprocess_variants(frame)):
        result = try_decode(variant)
        name = VARIANT_NAMES[i] if i < len(VARIANT_NAMES) else f"Variant {i}"
        if result:
            print(f"[PASS] {name} → {result}")
            return
        else:
            print(f"[fail] {name}")

    # Rotation fallback
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    center = (w // 2, h // 2)
    for angle in [-15, -10, -5, 5, 10, 15]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h))
        result = try_decode(rotated)
        if result:
            print(f"[PASS] Rotation {angle}° → {result}")
            return
        else:
            print(f"[fail] Rotation {angle}°")

    print()
    print("[FAILED] No barcode found. Possible reasons:")
    print("  - Barcode is curved/distorted (cylindrical packaging)")
    print("  - Image resolution too low")
    print("  - Install opencv-contrib: pip install opencv-contrib-python")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.test_barcode <image_path>")
    else:
        test_barcode(sys.argv[1])
