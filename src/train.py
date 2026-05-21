import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.ocr_model import CRNN
from src.dataset import DottedDataset, collate_fn
import os

def train():
    # Configuration
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/-.: "
    n_classes = len(alphabet) + 1 # +1 for CTC blank
    img_h = 32
    nh = 256
    batch_size = 16
    epochs = 20
    lr = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Data
    dataset = DottedDataset("data/synthetic", alphabet, img_h=img_h)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Model
    model = CRNN(img_h, 1, n_classes, nh).to(device)
    
    # Loss and Optimizer
    criterion = torch.nn.CTCLoss(blank=0, reduction='mean').to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (images, labels, label_lengths) in enumerate(dataloader):
            images = images.to(device)
            
            # Forward
            preds = model(images) # [W, B, C]
            
            batch_size = images.size(0)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            
            loss = criterion(preds.log_softmax(2), labels, preds_size, label_lengths)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/dotted_ocr.pth")
    print("Model saved to models/dotted_ocr.pth")

if __name__ == "__main__":
    train()
