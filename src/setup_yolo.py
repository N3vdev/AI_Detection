import os
import yaml

def setup_yolo_training(data_path="data/products"):
    # 1. Create directory structure
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(data_path, sub), exist_ok=True)
    
    # 2. Create the dataset.yaml file
    dataset_info = {
        "path": os.path.abspath(data_path),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "barcode",
            1: "logo",
            2: "flavor_text",
            3: "dotted_label"
        }
    }
    
    with open(os.path.join(data_path, "dataset.yaml"), "w") as f:
        yaml.dump(dataset_info, f, default_flow_style=False)
    
    print(f"[Setup] YOLO Training directory ready at {data_path}")
    print("[Setup] Next step: Put your 10 product images in images/train and labels (YOLO format) in labels/train.")

if __name__ == "__main__":
    setup_yolo_training()
