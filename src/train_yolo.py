from ultralytics import YOLO

def train_custom_yolo(yaml_path="data/products/dataset.yaml"):
    # Load a pretrained YOLOv8 Nano model
    model = YOLO("yolov8n.pt")
    
    # Train the model
    print("[Training] Starting YOLOv8 training for product features...")
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        device='cpu', # Change to 0 if you have a GPU
        project='models',
        name='product_detector'
    )
    print("[Training] YOLOv8 training complete. Model saved in models/product_detector/weights/best.pt")

if __name__ == "__main__":
    # Note: This requires real images and labels to be present in data/products
    try:
        train_custom_yolo()
    except Exception as e:
        print(f"[Error] Could not start training: {e}")
        print("[Hint] Make sure you have added images and labels to 'data/products' first.")
