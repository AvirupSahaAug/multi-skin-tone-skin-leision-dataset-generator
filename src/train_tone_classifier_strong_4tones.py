import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Import our setup from existing scripts
from models_unified import StrongSkinToneClassifier, weights_init

# Create a specialized dataset isolated to Tone 1 to 4 to avoid touching existing code
class FitzpatrickDataset4Tones(Dataset):
    def __init__(self, csv_filepath, img_dir, transform=None, target_size=(128, 128)):
        self.img_dir = img_dir
        self.transform = transform
        self.target_size = target_size
        
        # Load CSV and filter to tones 1, 2, 3, 4
        df = pd.read_csv(csv_filepath)
        df = df[df['fitzpatrick_scale'].isin([1, 2, 3, 4])]
        
        # Reset index after dropping rows
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get md5hash for image name
        img_name = str(self.df.iloc[idx]['md5hash']) + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            image = torch.zeros((3, self.target_size[0], self.target_size[1]))
            label = int(self.df.iloc[idx]['fitzpatrick_scale']) - 1
            return image, label
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # Get label (Scale 1-4 -> mapped to 0-3 for CrossEntropyLoss)
        label = int(self.df.iloc[idx]['fitzpatrick_scale']) - 1
        
        return image, label


def train_strong_classifier_4tones():
    # 1. Settings 
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

    # 2. Oversampling Logic (Focuses on 4 tones)
    print("Calculating oversampling weights to handle class imbalance (Tones 1-4)...")
    df = pd.read_csv(csv_path)
    df = df[df['fitzpatrick_scale'].isin([1, 2, 3, 4])]
    labels = df['fitzpatrick_scale'].values - 1
    class_counts = np.bincount(labels) # should be size 4
    class_weights = 1. / (class_counts + 1e-6) # avoid div by zero safely
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    print(f"Original counts: {class_counts}")

    # 3. Enhanced Augmentations
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
    
    dataset = FitzpatrickDataset4Tones(csv_filepath=csv_path, img_dir=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, drop_last=True)

    # 4. Initialize Model (Strong ResNet18) for 4 tones!
    print("Initializing Strong (ResNet18) Skin Tone Classifier for 4 Tones...")
    model = StrongSkinToneClassifier(num_tones=4, pretrained=True).to(device)
    
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
        save_path = os.path.join(save_dir, "C_strong_4tones.pth")
        torch.save(model.state_dict(), save_path)
        
        if final_acc > best_acc:
            best_acc = final_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "C_strong_4tones_best.pth"))
            print(f" New Best Epoch Accuracy: {best_acc:.2f}% - Model Saved.")

    print(f"Training Complete. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_strong_classifier_4tones()
