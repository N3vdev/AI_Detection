#!/usr/bin/env python3
"""
Test script to validate dotted OCR functionality with actual training data
"""

import cv2
import numpy as np
import torch
from src.enhanced_controller import EnhancedConveyorInspector
from src.ocr_model import CRNN

def test_dotted_ocr_directly():
    """Test the dotted OCR model directly with training data"""
    print("Testing Dotted OCR Model Directly...")

    # Load the model
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/-.: "
    ocr_model = CRNN(32, 1, len(alphabet) + 1, 256)

    try:
        ocr_model.load_state_dict(torch.load('models/dotted_ocr.pth', map_location='cpu'))
        ocr_model.eval()
        print("✓ Dotted OCR model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load OCR model: {e}")
        return

    # Test with actual training images
    import os
    test_images = []
    test_labels = []

    synthetic_path = 'data/synthetic'
    for i in range(5):  # Test first 5 images
        img_path = f'{synthetic_path}/dotted_{i:05d}.png'
        label_path = f'{synthetic_path}/dotted_{i:05d}.txt'

        if os.path.exists(img_path) and os.path.exists(label_path):
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"✗ Could not load image: {img_path}")
                continue

            # Load label
            with open(label_path, 'r') as f:
                label = f.read().strip()

            test_images.append(img)
            test_labels.append(label)
            print(f"  Loaded test image {i}: {label}")

    if not test_images:
        print("✗ No test images found")
        return

    # Preprocess and test each image
    print("\nRunning OCR predictions:")
    for idx, (img, expected) in enumerate(zip(test_images, test_labels)):
        try:
            # Preprocess exactly as in training
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 32))

            # Normalize to [-1, 1] range
            img_tensor = gray.astype(np.float32) / 255.0
            img_tensor = (img_tensor - 0.5) / 0.5
            img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).unsqueeze(0)

            # Inference
            with torch.no_grad():
                preds = ocr_model(img_tensor)
                _, max_indices = torch.max(preds, 2)
                max_indices = max_indices.permute(1, 0).cpu().numpy()[0]

                char_list = []
                for i in range(len(max_indices)):
                    if max_indices[i] != 0 and (not (i > 0 and max_indices[i] == max_indices[i - 1])):
                        char_list.append(alphabet[max_indices[i] - 1])
                result = "".join(char_list)

            print(f"  Image {idx}: Expected='{expected}', Got='{result}' {'✓' if result.strip() == expected else '✗'}")

        except Exception as e:
            print(f"  Image {idx}: Error - {e}")

def test_enhanced_inspector_with_dotted():
    """Test the enhanced inspector's dotted label processing"""
    print("\n" + "="*50)
    print("Testing Enhanced Inspector with Dotted Labels")
    print("="*50)

    # Create inspector
    inspector = EnhancedConveyorInspector(
        yolo_path='yolov8n.pt',
        ocr_path='models/dotted_ocr.pth',
        max_workers=2
    )

    # Test with actual dotted label image
    test_img_path = 'data/synthetic/dotted_00000.png'
    expected_label = 'EXP 12/02/25'

    print(f"Testing with image: {test_img_path}")
    print(f"Expected label: {expected_label}")

    # Load the image
    img = cv2.imread(test_img_path)
    if img is None:
        print("✗ Could not load test image")
        return

    print(f"Image shape: {img.shape}")

    # Test direct dotted OCR processing
    print("\nTesting direct dotted OCR processing:")
    dotted_result = inspector.read_dotted_label_optimized(img)
    print(f"Direct OCR result: '{dotted_result}'")

    if dotted_result:
        parsed = inspector.parse_dotted_info(dotted_result)
        print(f"Parsed info: {parsed}")

        # Check if we got the expiry date
        if parsed["expiry"]:
            print(f"✓ Successfully extracted expiry date: {parsed['expiry']}")
        else:
            print("✗ Could not extract expiry date")

    # Test with frame simulation (put the dotted label in a frame)
    print("\nTesting with frame simulation:")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = [50, 50, 50]  # Dark background

    # Place the dotted label image in the bottom region (where expiry codes usually are)
    h, w = img.shape[:2]
    y_offset = 400  # Near bottom
    x_offset = 220  # Centered

    # Make sure we don't go out of bounds
    if y_offset + h <= frame.shape[0] and x_offset + w <= frame.shape[1]:
        frame[y_offset:y_offset+h, x_offset:x_offset+w] = img
        print(f"  Placed dotted label at position ({x_offset}, {y_offset})")
    else:
        # Resize if too large
        scale = min(400/w, 60/h)  # Fit in reasonable size
        new_w, new_h = int(w*scale), int(h*scale)
        resized_img = cv2.resize(img, (new_w, new_h))
        y_offset = 400 - new_h//2
        x_offset = 320 - new_w//2
        frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
        print(f"  Resized and placed dotted label at ({x_offset}, {y_offset})")

    # Create mock 4-camera frames (all same for simplicity)
    camera_frames = [frame, frame, frame, frame]

    # Process with enhanced inspector
    print("\nProcessing frame through enhanced inspector:")
    result = inspector.process_conveyor_frame_optimized(camera_frames, frame_id=1)

    print(f"  Status: {result['status']}")
    print(f"  Barcode: {result['barcode']}")
    print(f"  Brand: {result['brand']}")
    print(f"  Product: {result['product']}")
    print(f"  Expiry Date: {result['expiry_date']}")
    print(f"  Batch Code: {result['batch_code']}")
    print(f"  Sources: {', '.join(result['sources']) if result['sources'] else 'None'}")
    print(f"  Processing Time: {result['processing_time_ms']:.2f}ms")

    # Check results
    success = False
    if result['expiry_date'] and '12/02/25' in result['expiry_date']:
        print("  ✓ SUCCESS: Correctly extracted expiry date!")
        success = True
    elif result['batch_code'] and ('EXP' in result['batch_code'] or '12/02/25' in result['batch_code']):
        print("  ✓ PARTIAL: Found expiry info in batch code")
        success = True
    else:
        print("  ✗ FAILED: Could not extract expiry date correctly")

    # Show performance stats
    print("\nPerformance Statistics:")
    stats = inspector.get_performance_stats()
    for operation, data in stats.items():
        if data['samples'] > 0:
            print(f"  {operation.capitalize():<12} - Avg: {data['avg_ms']:6.2f}ms "
                  f"[{data['samples']} samples]")

    inspector.shutdown()
    return success

if __name__ == "__main__":
    print("DOTTED OCR VALIDATION TEST")
    print("="*50)

    # Test 1: Direct model testing
    test_dotted_ocr_directly()

    # Test 2: Enhanced inspector integration
    success = test_enhanced_inspector_with_dotted()

    print("\n" + "="*50)
    if success:
        print("OVERALL RESULT: ✓ TESTS PASSED - Dotted OCR functionality verified")
    else:
        print("OVERALL RESULT: ✗ TESTS FAILED - Issues detected with dotted OCR")
    print("="*50)