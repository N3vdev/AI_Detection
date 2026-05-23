from src.detect import AIInspectionSystem
import os

def test_system():
    # Ensure model is downloaded
    if not os.path.exists("models/barcode_detector.pt"):
        print("Downloading barcode model...")
        from src.download_barcode_model import download_model
        download_model()

    system = AIInspectionSystem()
    
    test_image = "coke.jpg"
    if os.path.exists(test_image):
        print(f"Testing on {test_image}...")
        results = system.inspect_image(test_image, save_debug=True)
        print("Inspection Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        print("\n[Debug] Crops saved to the 'debug/' folder. Check 'debug/barcode_specialized_0.jpg'.")

    else:
        print(f"Test image {test_image} not found.")

if __name__ == "__main__":
    test_system()
