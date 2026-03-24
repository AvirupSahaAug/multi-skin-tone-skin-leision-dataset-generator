import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Import our setup from existing scripts
from models_unified import StrongSkinToneClassifier, weights_init
from train_tone_classifier import FitzpatrickDataset

def train_strong_classifier():
    # 1. Settings - ResNet18 needs more epochs and careful LR
    EPOCHS = 30
    BATCH_SIZE = 32
    LR = 0.0001 
    IMG_SIZE = 128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'fitzpatrick17k', 'fitzpatrick17k-main', 'fitzpatrick17k.csv')
    img_dir = os.path.join(base_dir, 'data', 'fitzpatrick17k', 'images')
    save_dir = os.path.join(base_dir, 'checkpoints', 'unified')
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return

    # 2. Oversampling Logic (Duplicates minority classes)
    print("Calculating oversampling weights to handle class imbalance...")
    df = pd.read_csv(csv_path)
    df = df[df['fitzpatrick_scale'].isin([1, 2, 3, 4, 5, 6])]
    labels = df['fitzpatrick_scale'].values - 1
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    print(f"Original counts: {class_counts}")
    print(f"Oversampling to achieve balance across all 6 tones.")

    # 3. Enhanced Augmentations (Crucial for oversampling)
    # Using Normalize(0.5, 0.5) to match GAN range [-1, 1]
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 40.0)),
            A.MotionBlur(blur_limit=3),
        ], p=0.3),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), 
        ToTensorV2()
    ])
    
    dataset = FitzpatrickDataset(csv_filepath=csv_path, img_dir=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, drop_last=True)

    # 4. Initialize Model (Strong ResNet18)
    print("Initializing Strong (ResNet18) Skin Tone Classifier...")
    model = StrongSkinToneClassifier(num_tones=6, pretrained=True).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    print(f"Starting Training...")
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i, (imgs, labels) in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 10 == 0:
                acc = 100 * correct / total
                pbar.set_postfix({'Loss': f"{(total_loss/(i+1)):.4f}", 'Acc': f"{acc:.2f}%"})

        # Save Checkpoint
        final_acc = 100 * correct / total
        save_path = os.path.join(save_dir, "C_strong.pth")
        torch.save(model.state_dict(), save_path)
        
        if final_acc > best_acc:
            best_acc = final_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "C_strong_best.pth"))
            print(f" New Best Epoch Accuracy: {best_acc:.2f}% - Model Saved.")

    print(f"Training Complete. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_strong_classifier()
