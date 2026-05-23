"""
AI Product Inspection — single image test
Usage:
    python test_image.py <image_path>
    python test_image.py s-l400.jpg
"""
import sys
from src.detect import AIInspectionSystem

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_image.py <image_path>")
        sys.exit(1)

    system = AIInspectionSystem()
    result = system.inspect_image(sys.argv[1])

    print("\n" + "=" * 50)
    print("  INSPECTION RESULT")
    print("=" * 50)
    print(f"  Image            : {result.get('image')}")
    print(f"  Status           : {result.get('status')}")
    print(f"  Barcode          : {result.get('barcode') or '—'}")
    print(f"  Brand            : {result.get('brand') or '—'}")
    print(f"  Product          : {result.get('product_name') or '—'}")
    print(f"  Category         : {result.get('product_category') or '—'}")
    print(f"  Expiry Date      : {result.get('expiry_date') or '—'}")
    print(f"  Manufacture Date : {result.get('manufacture_date') or '—'}")
    print(f"  Batch Number     : {result.get('batch_number') or '—'}")
    print(f"  Dotted Label     : {result.get('dotted_label_text') or '—'}")
    if result.get('raw_ocr_text'):
        print(f"  Label Text       : {result.get('raw_ocr_text')}")
    print("=" * 50)

if __name__ == "__main__":
    main()
