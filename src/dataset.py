import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class SkinLesionDataset(Dataset):
    def __init__(self, csv_filepath, img_dir=None, transform=None, target_size=(128, 128)):
        """
        Args:
            csv_filepath (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images. (Not strictly needed if csv has full paths)
            transform (callable, optional): Optional transform to be applied on a sample.
            target_size (tuple): Target image size (H, W).
        """
        self.df = pd.read_csv(csv_filepath)
        self.transform = transform
        self.target_size = target_size
        
        # Mappings for HAM10000 diagnosis
        # nv: Melanocytic nevi
        # mel: Melanoma
        # bkl: Benign keratosis-like lesions
        # bcc: Basal cell carcinoma
        # akiec: Actinic keratoses
        # vasc: Vascular lesions
        # df: Dermatofibroma
        self.class_to_idx = {
            'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 
            'akiec': 4, 'vasc': 5, 'df': 6
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get image path
        img_path = self.df.iloc[idx]['image_path']
        
        # 2. Load image (OpenCV loads as BGR)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image not found at {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Default transform if none provided
            default_tf = A.Compose([
                A.Resize(self.target_size[0], self.target_size[1]),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), # GAN normalization (-1 to 1)
                ToTensorV2()
            ])
            augmented = default_tf(image=image)
            image = augmented['image']

        # 4. Get label
        label_str = self.df.iloc[idx]['dx']
        label = self.class_to_idx[label_str]
        
        return image, label

def get_transforms(image_size=128):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

if __name__ == "__main__":
    # Test block
    import matplotlib.pyplot as plt
    
    # Path to your processed train csv
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/processed/ham10000_train.csv')
    
    if os.path.exists(csv_path):
        dataset = SkinLesionDataset(csv_filepath=csv_path, target_size=(128, 128))
        print(f"Dataset loaded with {len(dataset)} samples.")
        
        img, label = dataset[0]
        print(f"Sample 0 shape: {img.shape} (Range: {img.min():.2f} to {img.max():.2f})")
        print(f"Sample 0 label: {label} ({dataset.idx_to_class[label]})")
        
        # Verification passed
    else:
        print(f"CSV not found at {csv_path}. Run prepare_data.py first.")
