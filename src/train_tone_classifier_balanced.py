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
from models_unified import SkinToneClassifier, weights_init
from train_tone_classifier import FitzpatrickDataset

def train_balanced_classifier():
    # 1. Settings
    EPOCHS = 20
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

    # 2. Calculate Oversampling Weights
    print("Calculating oversampling weights for balance...")
    df = pd.read_csv(csv_path)
    df = df[df['fitzpatrick_scale'].isin([1, 2, 3, 4, 5, 6])]
    
    # Get labels (0-5)
    labels = df['fitzpatrick_scale'].values - 1
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    
    # Map each sample in the dataset to its oversampling weight
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    print(f"Class counts: {class_counts}")
    print(f"Targeting perfectly balanced sampling using replacement.")

    # 3. Enhanced Augmentations (to prevent overfitting on duplicates)
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.MotionBlur(blur_limit=3),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    dataset = FitzpatrickDataset(csv_filepath=csv_path, img_dir=img_dir, transform=transform)
    # Use the sampler instead of shuffle=True
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, drop_last=True)

    # 4. Initialize Model
    model = SkinToneClassifier(num_tones=6, img_size=IMG_SIZE).to(device)
    model.apply(weights_init)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss() # Weights are handled by the Sampler now

    # 5. Training Loop
    print("Starting BALANCED OVERSAMPLED pre-training for Skin Tone Classifier...")
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

        # Save specifically as a balanced model
        save_path = os.path.join(save_dir, "C_balanced.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved balanced checkpoint to {save_path}")

if __name__ == "__main__":
    train_balanced_classifier()
