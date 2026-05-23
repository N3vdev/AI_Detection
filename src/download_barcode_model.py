from ultralytics import YOLO
import os
from huggingface_hub import hf_hub_download

def download_model():
    repo_id = "Piero2411/YOLOV8s-Barcode-Detection"
    filename = "YOLOV8s_Barcode_Detection.pt"
    
    print(f"Downloading model {filename} from {repo_id}...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download the model file
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="models")
    
    # Rename it for easier use
    target_path = "models/barcode_detector.pt"
    if os.path.exists(target_path):
        os.remove(target_path)
    os.rename(local_path, target_path)
    
    print(f"Model downloaded and saved to: {target_path}")
    
    # Verify it can be loaded
    model = YOLO(target_path)
    print("Model loaded successfully.")
    return model

if __name__ == "__main__":
    download_model()
