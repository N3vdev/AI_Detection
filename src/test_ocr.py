import torch
import cv2
import numpy as np
import os
from src.ocr_model import CRNN

def test_single_image(img_path, model_path='models/dotted_ocr.pth'):
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/-.: "
    n_classes = len(alphabet) + 1
    
    # Load model
    model = CRNN(32, 1, n_classes, 256)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Load image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    original_h, original_w = image.shape
    image_resized = cv2.resize(image, (128, 32))
    
    # Preprocess
    input_data = image_resized.astype(np.float32) / 255.0
    input_data = (input_data - 0.5) / 0.5
    input_tensor = torch.from_numpy(input_data).unsqueeze(0).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        preds = model(input_tensor)
        
    # CTC Decode
    _, max_indices = torch.max(preds, 2)
    max_indices = max_indices.permute(1, 0).cpu().numpy()[0]
    
    char_list = []
    for i in range(len(max_indices)):
        if max_indices[i] != 0 and (not (i > 0 and max_indices[i] == max_indices[i - 1])):
            char_list.append(alphabet[max_indices[i] - 1])
    
    prediction = "".join(char_list)
    
    # Read ground truth
    label_path = img_path.replace('.png', '.txt')
    with open(label_path, 'r') as f:
        ground_truth = f.read().strip()
        
    print(f"Image: {img_path}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Prediction  : {prediction}")
    print(f"Match: {ground_truth == prediction}")

if __name__ == "__main__":
    # Test on the first synthetic image
    test_single_image("data/synthetic/dotted_00000.png")
