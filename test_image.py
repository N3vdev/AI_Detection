"""
AI Product Inspection — test one or multiple images of the same product

Usage:
    python test_image.py front.jpg
    python test_image.py front.jpg back.jpg
    python test_image.py front.jpg back.jpg side.jpg
"""
import sys
from src.detect import AIInspectionSystem


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_image.py <image1> [image2] [image3] ...")
        sys.exit(1)

    image_paths = sys.argv[1:]
    system = AIInspectionSystem()
    result = system.inspect_product(image_paths)

    print("\n" + "=" * 54)
    print("  INSPECTION RESULT")
    print("=" * 54)
    images = result.get("images") or [result.get("image", "?")]
    print(f"  Images           : {', '.join(images)}")
    print(f"  Status           : {result.get('status')}")
    print(f"  Barcode          : {result.get('barcode') or '—'}")
    print(f"  Brand            : {result.get('brand') or '—'}")
    print(f"  Product          : {result.get('product_name') or '—'}")
    print(f"  Category         : {result.get('product_category') or '—'}")
    print(f"  Expiry Date      : {result.get('expiry_date') or '—'}")
    print(f"  Manufacture Date : {result.get('manufacture_date') or '—'}")
    print(f"  Batch Number     : {result.get('batch_number') or '—'}")
    print(f"  Dotted Label     : {result.get('dotted_label_text') or '—'}")
    if result.get("raw_ocr_text"):
        preview = result["raw_ocr_text"][:80].replace("\n", " ")
        print(f"  Label Text       : {preview}{'...' if len(result['raw_ocr_text']) > 80 else ''}")
    print("=" * 54)


if __name__ == "__main__":
    main()
