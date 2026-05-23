#!/usr/bin/env python3
"""
Test the enhanced inspector with a single image
"""
import cv2
import numpy as np
from PIL import Image
try:
    from pi_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass
from src.enhanced_controller import create_enhanced_inspector

def test_with_single_image(image_path):
    """Test enhanced inspector with a single image (simulate 4 camera views)"""
    print(f"Testing with image: {image_path}")
    
    # Load the image with PIL for better compatibility then convert to OpenCV BGR
    try:
        pil_img = Image.open(image_path).convert('RGB')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"✗ Could not load image: {e}")
        return
    
    if img is None:
        print("✗ Could not load image")
        return
    
    print(f"Image shape: {img.shape}")
    
    # Create inspector
    inspector = create_enhanced_inspector(
        yolo_path='yolov8n.pt',
        ocr_path='models/dotted_ocr.pth',
        max_workers=2
    )
    
    # Simulate 4 camera views (all same for single image test)
    # In real usage, these would be different angles from conveyor cameras
    camera_frames = [img, img, img, img]
    
    # Process with enhanced inspector
    print("\nProcessing through enhanced inspector:")
    result = inspector.process_conveyor_frame_optimized(camera_frames, frame_id=1)
    
    # Save a diagnostic image if something was detected
    if result["sources"]:
        with inspector.detector_lock:
            # Run one more time just for the debug plot
            debug_res = inspector.detector(img, verbose=False)[0]
            debug_img = debug_res.plot()
            cv2.imwrite("debug_detection.jpg", debug_img)
            print(f"  [Diagnostic] Saved detection plot to 'debug_detection.jpg'")

    print(f"  Status: {result['status']}")
    print(f"  Barcode: {result['barcode']}")
    print(f"  Brand: {result['brand']}")
    print(f"  Product: {result['product']}")
    print(f"  Expiry Date: {result['expiry_date']}")
    print(f"  Batch Code: {result['batch_code']}")
    print(f"  Sources: {', '.join(result['sources']) if result['sources'] else 'None'}")
    print(f"  Processing Time: {result['processing_time_ms']:.2f}ms")
    
    inspector.shutdown()
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python test_single_image.py <image_path>")
        print("Example: python test_single_image.py coke.jpg")
        sys.exit(1)
    
    test_with_single_image(sys.argv[1])
