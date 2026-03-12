import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Import our setup from existing scripts
from models_unified import SkinToneClassifier, weights_init
from train_tone_classifier import FitzpatrickDataset

def train_weighted_classifier():
    # 1. Settings
    EPOCHS = 15
    BATCH_SIZE = 32
    LR = 0.0002
    IMG_SIZE = 128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'fitzpatrick17k', 'fitzpatrick17k-main', 'fitzpatrick17k.csv')
    img_dir = os.path.join(base_dir, 'data', 'fitzpatrick17k', 'images')
    save_dir = os.path.join(base_dir, 'checkpoints', 'unified')
    os.makedirs(save_dir, exist_ok=True)

    # 2. Calculate Class Weights based on population
    print("Calculating class weights...")
    df = pd.read_csv(csv_path)
    df = df[df['fitzpatrick_scale'].isin([1, 2, 3, 4, 5, 6])]
    class_counts = df['fitzpatrick_scale'].value_counts().sort_index().values
    
    # Weight = Total_Samples / (Num_Classes * Class_Samples)
    total_samples = sum(class_counts)
    num_classes = len(class_counts)
    weights = total_samples / (num_classes * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    
    print(f"Class counts: {class_counts}")
    print(f"Calculated weights: {weights}")

    # 3. Data Loader
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    dataset = FitzpatrickDataset(csv_filepath=csv_path, img_dir=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    # 4. Initialize Model & Loss with Weights
    model = SkinToneClassifier(num_tones=6, img_size=IMG_SIZE).to(device)
    model.apply(weights_init)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # Applying weights directly to the CrossEntropyLoss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 5. Training Loop
    print("Starting WEIGHTED pre-training for Skin Tone Classifier...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i, (imgs, labels) in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device).long()
            
            # Forward
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 10 == 0:
                pbar.set_postfix({'Loss': f"{(total_loss/(i+1)):.4f}", 'Acc': f"{(100 * correct / total):.2f}%"})

        # Save specifically as a weighted model
        save_path = os.path.join(save_dir, "C_weighted.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved weighted checkpoint to {save_path}")

if __name__ == "__main__":
    train_weighted_classifier()
