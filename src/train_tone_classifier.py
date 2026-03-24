import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import the classifier from the unified models we just created
from models_unified import SkinToneClassifier, weights_init

# Dataset definition for Fitzpatrick17k
class FitzpatrickDataset(Dataset):
    def __init__(self, csv_filepath, img_dir, transform=None, target_size=(128, 128)):
        self.img_dir = img_dir
        self.transform = transform
        self.target_size = target_size
        
        # Load CSV and filter to valid Fitzpatrick scales (1 to 6)
        df = pd.read_csv(csv_filepath)
        df = df[df['fitzpatrick_scale'].isin([1, 2, 3, 4, 5, 6])]
        
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
            # Fallback gracefully if image is missing by returning a zero tensor
            # In a strict environment, you'd clean the CSV first.
            image = torch.zeros((3, self.target_size[0], self.target_size[1]))
            # Target scale 1...6 -> Map to 0...5
            label = int(self.df.iloc[idx]['fitzpatrick_scale']) - 1
            return image, label
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform is None:
            self.transform = A.Compose([
                A.Resize(self.target_size[0], self.target_size[1]),
                # Standard normalization for image classifiers
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            
        augmented = self.transform(image=image)
        image = augmented['image']

        # Get label (Scale 1-6 -> mapped to 0-5 for CrossEntropyLoss)
        label = int(self.df.iloc[idx]['fitzpatrick_scale']) - 1
        
        return image, label

def train_classifier():
    # Settings
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

    print("Loading Fitzpatrick17k dataset...")
    # Add simple data augmentation to prevent overfitting
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), # Match GAN normalization
        ToTensorV2()
    ])
    
    dataset = FitzpatrickDataset(csv_filepath=csv_path, img_dir=img_dir, transform=transform, target_size=(IMG_SIZE, IMG_SIZE))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    print(f"Total valid images loaded: {len(dataset)}")

    # Initialize model (6 tones: 1 to 6)
    model = SkinToneClassifier(num_tones=6, img_size=IMG_SIZE).to(device)
    model.apply(weights_init)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("Starting pre-training for Skin Tone Classifier...")
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

        # Save checkpoint after each epoch
        save_path = os.path.join(save_dir, "C_pretrained.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    train_classifier()
