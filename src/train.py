import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np

# Import our custom modules
from dataset import SkinLesionDataset
from models import Generator, Discriminator, weights_init
from tqdm import tqdm

def train(args):
    # 1. Setup Directories
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 2. device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Load Data
    print("Loading dataset...")
    dataset = SkinLesionDataset(csv_filepath=args.csv_path, target_size=(args.img_size, args.img_size))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"Dataset size: {len(dataset)}")

    # 4. Initialize Models
    G = Generator(z_dim=args.z_dim, num_classes=7, img_size=args.img_size).to(device)
    D = Discriminator(num_classes=7, img_size=args.img_size).to(device)
    
    # Initialize weights
    G.apply(weights_init)
    D.apply(weights_init)

    # 5. Optimizers
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Loss function
    adversarial_loss = nn.BCEWithLogitsLoss()

    # Fixed noise for sampling (to track progress)
    fixed_z = torch.randn(64, args.z_dim).to(device)
    # Generate labels 0-6 repeated
    fixed_labels = torch.cat([torch.full((8,), i) for i in range(7)] + [torch.zeros(8)]).long().to(device) # 7*8=56 + 8 = 64
    fixed_labels = fixed_labels[:64] # Ensure 64

    # 6. Training Loop
    start_time = time.time()
    print("Starting training...")
    
    for epoch in range(args.start_epoch, args.n_epochs):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{args.n_epochs}")
        for i, (imgs, labels) in pbar:
            
            # Identify real images
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            batch_size = real_imgs.size(0)

            # Adversarial ground truths
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)
            
            # Generally for PatchGAN output (B, 1, H, W), we need to match shape
            # D output is (B, 1, 5, 5) -> We can target same shape or mean
            # Let's adjust target tensor shape automatically
            
            # --- Train Generator ---
            optimizer_G.zero_grad()

            # Sample noise and labels
            z = torch.randn(batch_size, args.z_dim).to(device)
            gen_labels = torch.randint(0, 7, (batch_size,)).to(device)

            # Generate images
            gen_imgs = G(z, gen_labels)

            # Loss: D(G(z)) should be valid
            # D output shape: (batch_size, 1, 5, 5)
            d_out_fake = D(gen_imgs, gen_labels)
            target_real = torch.ones_like(d_out_fake).to(device)
            
            g_loss = adversarial_loss(d_out_fake, target_real)

            g_loss.backward()
            optimizer_G.step()

            # --- Train Discriminator ---
            optimizer_D.zero_grad()

            # Real Loss
            d_out_real = D(real_imgs, labels)
            real_loss = adversarial_loss(d_out_real, torch.ones_like(d_out_real).to(device))

            # Fake Loss
            d_out_fake = D(gen_imgs.detach(), gen_labels)
            fake_loss = adversarial_loss(d_out_fake, torch.zeros_like(d_out_fake).to(device))

            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # Logging
            batches_done = epoch * len(dataloader) + i
            if i % args.log_interval == 0:
                pbar.set_postfix({'D loss': f'{d_loss.item():.4f}', 'G loss': f'{g_loss.item():.4f}'})

        # End of Epoch: Save Samples
        if epoch % args.sample_interval == 0:
            with torch.no_grad():
                G.eval()
                sample_imgs = G(fixed_z, fixed_labels)
                G.train()
                save_image(sample_imgs, f"{args.sample_dir}/{epoch}.png", nrow=8, normalize=True)
                print(f"Saved sample to {args.sample_dir}/{epoch}.png")

        # Save Checkpoint
        if epoch % args.checkpoint_interval == 0:
            torch.save(G.state_dict(), f"{args.checkpoint_dir}/G_epoch_{epoch}.pth")
            torch.save(D.state_dict(), f"{args.checkpoint_dir}/D_epoch_{epoch}.pth")
            print(f"Saved checkpoint to {args.checkpoint_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--z_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--sample_interval", type=int, default=1, help="interval between image sampling")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--log_interval", type=int, default=100, help="interval for logging")
    parser.add_argument("--start_epoch", type=int, default=0, help="starting epoch")
    
    # Paths
    parser.add_argument("--csv_path", type=str, default="data/processed/ham10000_train.csv", help="path to train csv")
    parser.add_argument("--sample_dir", type=str, default="samples/phase1", help="directory to save samples")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/phase1", help="directory to save model checkpoints")

    args = parser.parse_args()
    print(args)
    
    train(args)
