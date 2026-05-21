import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class DottedDataset(Dataset):
    def __init__(self, data_dir, alphabet, img_h=32, img_w=128):
        self.data_dir = data_dir
        self.alphabet = alphabet
        self.img_h = img_h
        self.img_w = img_w
        
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        # Read image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.img_w, self.img_h))
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5 # Normalize
        image = torch.from_numpy(image).unsqueeze(0) # [1, H, W]
        
        # Read label
        label_path = os.path.join(self.data_dir, img_name.replace('.png', '.txt'))
        with open(label_path, 'r') as f:
            label_text = f.read().strip()
            
        # Convert label to indices
        label = [self.alphabet.find(c) for c in label_text if self.alphabet.find(c) != -1]
        label = torch.LongTensor(label)
        
        return image, label

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    
    # Pack labels for CTC Loss
    label_lengths = torch.LongTensor([len(l) for l in labels])
    all_labels = torch.cat(labels)
    
    return images, all_labels, label_lengths

if __name__ == "__main__":
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/-.: "
    dataset = DottedDataset("data/synthetic", alphabet)
    img, label = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Label: {label}")
