import sys
import os
from src.detect import AIInspectionSystem

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect.py <image_path>")
        return

    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    print(f"--- Starting Inspection: {image_path} ---")
    
    # Initialize system
    system = AIInspectionSystem()
    
    # Perform inspection
    results = system.inspect_image(image_path, save_debug=True)
    
    print("\n--- RESULTS ---")
    print(f"BARCODE: {results['barcode']}")
    print(f"BRAND  : {results['brand']}")
    print(f"EXPIRY : {results['expiry_date']}")
    print("----------------")
    print(f"\nDebug crops saved to 'debug/' folder.")

if __name__ == "__main__":
    main()
