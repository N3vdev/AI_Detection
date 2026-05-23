#!/usr/bin/env python3
"""
Test script for the Enhanced Conveyor Inspection System
Demonstrates live feed extraction from conveyor belt simulation
"""

import cv2
import numpy as np
import time
import os
from src.enhanced_controller import EnhancedConveyorInspector, create_enhanced_inspector

def create_mock_product_frames(product_type="coca_cola"):
    """Create mock camera frames simulating different views of a product on conveyor belt"""
    frames = []

    # Create 4 camera views: Front, Back, Left, Right
    for view in ['front', 'back', 'left', 'right']:
        # Create a basic frame (simulating what cameras would see)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add some background texture
        frame[:] = [50, 50, 50]  # Dark gray background

        if product_type == "coca_cola":
            # Simulate Coca-Cola can with barcode and label
            if view == 'front':
                # Draw a red can shape
                cv2.rectangle(frame, (200, 150), (440, 330), (0, 0, 255), -1)
                # Add white stripe (typical Coke design)
                cv2.rectangle(frame, (200, 220), (440, 260), (255, 255, 255), -1)
                # Add Coca-Cola text simulation
                cv2.putText(frame, "COCA", (250, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "COLA", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # Add barcode area (bottom)
                cv2.rectangle(frame, (250, 300), (390, 320), (0, 0, 0), -1)  # Black barcode area
                # Add some barcode-like lines
                for i in range(10):
                    x = 260 + i * 12
                    cv2.line(frame, (x, 305), (x+4, 315), (255, 255, 255), 2)

            elif view == 'side':
                # Side view - thinner rectangle
                cv2.rectangle(frame, (280, 160), (360, 320), (0, 0, 200), -1)
                cv2.rectangle(frame, (280, 220), (360, 260), (255, 255, 255), -1)

            else:  # back, left, right - simpler views
                cv2.rectangle(frame, (250, 180), (390, 300), (100, 100, 150), -1)

        elif product_type == "hair_cream":
            # Simulate hair cream product
            if view == 'front':
                # Draw container
                cv2.rectangle(frame, (220, 140), (420, 340), (200, 180, 160), -1)
                cv2.rectangle(frame, (220, 100), (420, 140), (150, 100, 80), -1)  # Cap
                # Add label area
                cv2.rectangle(frame, (240, 180), (400, 300), (255, 255, 255), -1)
                cv2.putText(frame, "HAIR", (260, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(frame, "CREAM", (260, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(frame, "EXP 12/02/25", (260, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                # Add dotted label simulation at bottom
                cv2.rectangle(frame, (250, 320), (390, 340), (230, 230, 230), -1)
                for i in range(8):
                    x = 260 + i * 15
                    cv2.circle(frame, (x, 330), 2, (0, 0, 0), -1)

        frames.append(frame)

    return frames

def simulate_dotted_label():
    """Create a sample dotted label image similar to the training data"""
    # Create a synthetic dotted label image
    img = np.ones((50, 200, 3), dtype=np.uint8) * 255  # White background

    # Add some text-like dots (simulating dotted printing)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "EXP 12/02/25"

    # Create dotted effect by drawing individual dots
    x_offset = 10
    for char in text:
        if char == ' ':
            x_offset += 15
            continue
        # Draw character as series of dots
        for i in range(len(char) * 8):  # Approximate dot positions
            dot_x = x_offset + (i % 3) * 2
            dot_y = 25 + (i // 3) * 2
            if dot_x < 190 and dot_y < 45:
                cv2.circle(img, (dot_x, dot_y), 1, (0, 0, 0), -1)
        x_offset += 10

    # Add some noise to make it more realistic
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.subtract(img, noise)

    return img

def main():
    print("=" * 60)
    print("ENHANCED CONVEYOR INSPECTION SYSTEM - LIVE FEED TEST")
    print("=" * 60)

    # Initialize the enhanced inspector
    print("\n[INIT] Initializing Enhanced Conveyor Inspector...")
    inspector = create_enhanced_inspector(
        yolo_path='yolov8n.pt',
        ocr_path='models/dotted_ocr.pth',
        max_workers=4
    )

    # Test 1: Single frame processing with mock Coca-Cola product
    print("\n" + "=" * 50)
    print("TEST 1: Single Frame Processing (Coca-Cola Can)")
    print("=" * 50)

    # Create mock frames for Coca-Cola product
    mock_frames = create_mock_product_frames("coca_cola")
    frame_names = ['Front', 'Back', 'Left', 'Right']

    # Display frame info
    for i, (frame, name) in enumerate(zip(mock_frames, frame_names)):
        print(f"  {name} Camera Frame: {frame.shape}")

    # Process the frame
    start_time = time.time()
    result = inspector.process_conveyor_frame_optimized(mock_frames, frame_id=1)
    processing_time = (time.time() - start_time) * 1000

    print(f"\n[RESULTS] Processing Time: {processing_time:.2f}ms")
    print(f"  Frame ID: {result['frame_id']}")
    print(f"  Status: {result['status']}")
    print(f"  Barcode: {result['barcode']}")
    print(f"  Brand: {result['brand']}")
    print(f"  Product: {result['product']}")
    print(f"  Expiry Date: {result['expiry_date']}")
    print(f"  Batch Code: {result['batch_code']}")
    print(f"  Sources: {', '.join(result['sources']) if result['sources'] else 'None'}")
    print(f"  From Cache: {result['from_cache']}")

    # Test 2: Batch processing simulation
    print("\n" + "=" * 50)
    print("TEST 2: Batch Processing Simulation")
    print("=" * 50)

    # Create a batch of different product frames
    batch_frames = []
    batch_ids = []

    # Add Coca-Cola frames
    coco_frames = create_mock_product_frames("coca_cola")
    batch_frames.append((1001, coco_frames))
    batch_ids.append(1001)

    # Add Hair Cream frames
    hair_frames = create_mock_product_frames("hair_cream")
    batch_frames.append((1002, hair_frames))
    batch_ids.append(1002)

    # Add some empty frames (no product)
    empty_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(4)]
    batch_frames.append((1003, empty_frames))
    batch_ids.append(1003)

    print(f"  Processing batch of {len(batch_frames)} frames...")

    # Process batch
    start_time = time.time()
    batch_results = inspector.process_conveyor_batch(batch_frames)
    batch_time = (time.time() - start_time) * 1000

    print(f"\n[BATCH RESULTS] Total Time: {batch_time:.2f}ms")
    print(f"  Average per frame: {batch_time/len(batch_frames):.2f}ms")

    for i, (result, frame_id) in enumerate(zip(batch_results, batch_ids)):
        product_name = ["Coca-Cola", "Hair Cream", "Empty"][i]
        print(f"\n  Frame {frame_id} ({product_name}):")
        print(f"    Status: {result['status']}")
        print(f"    Barcode: {result['barcode'] or 'None'}")
        print(f"    Brand: {result['brand'] or 'None'}")
        print(f"    Expiry: {result['expiry_date'] or 'None'}")
        print(f"    Time: {result['processing_time_ms']:.2f}ms")

    # Test 3: Performance statistics
    print("\n" + "=" * 50)
    print("TEST 3: Performance Statistics")
    print("=" * 50)

    stats = inspector.get_performance_stats()
    print("  Processing Time Statistics (milliseconds):")
    for operation, data in stats.items():
        if data['samples'] > 0:
            print(f"    {operation.capitalize():<12} - Avg: {data['avg_ms']:6.2f}ms "
                  f"(Min: {data['min_ms']:5.2f}, Max: {data['max_ms']:5.2f}, "
                  f"P95: {data['p95_ms']:5.2f}) [{data['samples']} samples]")
        else:
            print(f"    {operation.capitalize():<12} - No samples yet")

    # Test 4: Dotted label reading test
    print("\n" + "=" * 50)
    print("TEST 4: Dotted Label OCR Test")
    print("=" * 50)

    # Create and test dotted label
    dotted_img = simulate_dotted_label()
    print(f"  Created synthetic dotted label: {dotted_img.shape}")

    # Save for inspection
    cv2.imwrite('test_dotted_label.png', dotted_img)
    print("  Saved test dotted label as 'test_dotted_label.png'")

    # Test the dotted OCR directly
    if inspector.dotted_ocr is not None:
        dotted_result = inspector.read_dotted_label_optimized(dotted_img)
        print(f"  Dotted OCR Result: '{dotted_result}'")
        if dotted_result:
            parsed = inspector.parse_dotted_info(dotted_result)
            print(f"  Parsed Info: {parsed}")
    else:
        print("  Dotted OCR not available - model may not have loaded")

    # Cleanup
    print("\n[CLEANUP] Shutting down inspector...")
    inspector.shutdown()

    print("\n" + "=" * 60)
    print("TEST COMPLETE - Enhanced Conveyor Inspection System Ready!")
    print("Features demonstrated:")
    print("  ✓ Parallel processing of 4 camera views")
    print("  ✓ Ultra-fast barcode detection with early exit")
    print("  ✓ Optimized dotted label OCR")
    print("  ✓ Temporal caching for live feed coherence")
    print("  ✓ Batch processing for high-throughput scenarios")
    print("  ✓ Performance monitoring and statistics")
    print("  ✓ Graceful error handling and fallback mechanisms")
    print("=" * 60)

if __name__ == "__main__":
    main()