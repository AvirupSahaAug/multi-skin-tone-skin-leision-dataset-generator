import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F

# Import our custom modules
from dataset import SkinLesionDataset
from models import MultiHeadGenerator, Discriminator, weights_init
from diversity import ResNetEncoder

def train(args):
    # 1. Setup
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data
    print("Loading dataset...")
    dataset = SkinLesionDataset(csv_filepath=args.csv_path, target_size=(args.img_size, args.img_size))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 3. Models
    # Use MultiHeadGenerator for Phase 2
    G = MultiHeadGenerator(z_dim=args.z_dim, num_classes=7, img_size=args.img_size).to(device)
    D = Discriminator(num_classes=7, img_size=args.img_size).to(device)
    
    # Feature Extractor for Diversity Loss
    FeatureExtractor = ResNetEncoder().to(device)
    FeatureExtractor.eval() # Always eval

    # Initialize weights
    G.apply(weights_init)
    D.apply(weights_init)
    
    # Load Phase 1 checkpoint if provided
    if args.pretrained_G:
        print(f"Loading Phase 1 Generator weights from {args.pretrained_G}")
        # Phase 1 had a different architecture (Single Head). 
        # We can load the backbone weights.
        state_dict = torch.load(args.pretrained_G, map_location=device)
        
        # Filter out incompatible keys (heads)
        model_dict = G.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        # Manually initialize heads with the single head from phase 1 if desired,
        # or just leave them random. Let's load the backbone.
        model_dict.update(pretrained_dict)
        G.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)} keys into MultiHeadGenerator.")

    if args.pretrained_D:
        print(f"Loading Phase 1 Discriminator from {args.pretrained_D}")
        D.load_state_dict(torch.load(args.pretrained_D, map_location=device))

    # 4. Optimizers
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Loss Functions
    adversarial_loss = nn.BCEWithLogitsLoss()
    
    # Helper for diversity loss
    def compute_diversity_loss(imgs):
        # imgs: [x1, x2, x3], each (B, 3, H, W)
        # Extract features: (B, 512)
        # Normalize features to prevent scaling attacks
        feats = [F.normalize(FeatureExtractor(x), p=2, dim=1) for x in imgs]
        
        loss = 0
        count = 0
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                # L1 distance between NORMALIZED features
                # Max distance is 2.0 (opposite vectors)
                dist = torch.mean(torch.abs(feats[i] - feats[j])) 
                loss -= dist
                count += 1
        return loss / count

    # Fixed input for sampling
    fixed_z = torch.randn(32, args.z_dim).to(device)
    fixed_labels = torch.cat([torch.full((4,), i) for i in range(7)] + [torch.zeros(4)]).long().to(device)
    fixed_labels = fixed_labels[:32]

    # 5. Training Loop
    print(f"Starting Phase 2 Training (Multi-Output + Diversity)...")
    for epoch in range(args.start_epoch, args.n_epochs):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{args.n_epochs}")
        for i, (real_imgs, labels) in pbar:
            
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            batch_size = real_imgs.size(0)

            # --- Train Generator ---
            optimizer_G.zero_grad()

            z = torch.randn(batch_size, args.z_dim).to(device)
            gen_labels = torch.randint(0, 7, (batch_size,)).to(device)

            # Generate 3 variations
            x1, x2, x3 = G(z, gen_labels)
            
            # 1. Adversarial Loss (Discriminator should solve ALL of them)
            # We can average the loss for the 3 heads, or sum it.
            # D output: (B, 1, 5, 5)
            d_out1 = D(x1, gen_labels)
            d_out2 = D(x2, gen_labels)
            d_out3 = D(x3, gen_labels)
            
            valid = torch.ones_like(d_out1).to(device)
            
            g_adv_loss = (adversarial_loss(d_out1, valid) + 
                          adversarial_loss(d_out2, valid) + 
                          adversarial_loss(d_out3, valid)) / 3.0

            # 2. Diversity Loss
            div_loss = compute_diversity_loss([x1, x2, x3])
            
            # Total G Loss
            g_loss = g_adv_loss + (args.lambda_div * div_loss)

            g_loss.backward()
            optimizer_G.step()

            # --- Train Discriminator ---
            optimizer_D.zero_grad()

            # Real Loss
            d_real = D(real_imgs, labels)
            real_loss = adversarial_loss(d_real, torch.ones_like(d_real).to(device))

            # Fake Loss (Randomly pick one of the 3 heads to show D, or all? 
            # Showing all might overpower D. Let's pick one random head per batch sample or mix.)
            # Simpler: just concatenate and detach.
            # Let's show D all 3 generated batches (detached).
            
            d_fake1 = D(x1.detach(), gen_labels)
            d_fake2 = D(x2.detach(), gen_labels)
            d_fake3 = D(x3.detach(), gen_labels)
            
            fake_loss = (adversarial_loss(d_fake1, torch.zeros_like(d_fake1).to(device)) +
                         adversarial_loss(d_fake2, torch.zeros_like(d_fake2).to(device)) +
                         adversarial_loss(d_fake3, torch.zeros_like(d_fake3).to(device))) / 3.0

            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # Logging
            if i % args.log_interval == 0:
                pbar.set_postfix({
                    'D': f'{d_loss.item():.3f}', 
                    'G_adv': f'{g_adv_loss.item():.3f}',
                    'Div': f'{div_loss.item():.3f}'
                })

        # Save Samples
        if epoch % args.sample_interval == 0:
            with torch.no_grad():
                G.eval()
                # x1, x2, x3 = G(fixed_z, fixed_labels)
                # Concatenate vertically: Row 1 = x1, Row 2 = x2, Row 3 = x3
                # shape: (B, 3, H, W). 
                # We want to see [x1_0, x2_0, x3_0, x1_1, x2_1, x3_1, ...]
                # Actually, simpler: Save 3 separate grids or stacked.
                
                out1, out2, out3 = G(fixed_z, fixed_labels)
                # Stack them: (3*B, 3, H, W) -> Interleave
                
                # Let's make a combined grid where each column represents 3 variants of same Z
                # fixed_z is size 32. 
                # We have 32 * 3 = 96 images.
                # Arrange in grid of nrow=8.
                # Actually, let's just stack them vertically for checking diversity
                combined = torch.cat([out1, out2, out3], dim=2) # Concatenate Height-wise
                save_image(combined, f"{args.sample_dir}/{epoch}_diversity.png", nrow=8, normalize=True)
                
                G.train()
                print(f"Saved sample to {args.sample_dir}/{epoch}_diversity.png")

        # Save Checkpoint
        if epoch % args.checkpoint_interval == 0:
            torch.save(G.state_dict(), f"{args.checkpoint_dir}/G_phase2_{epoch}.pth")
            torch.save(D.state_dict(), f"{args.checkpoint_dir}/D_phase2_{epoch}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--z_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--sample_interval", type=int, default=1, help="interval between image sampling")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--log_interval", type=int, default=100, help="interval for logging")
    parser.add_argument("--start_epoch", type=int, default=0, help="starting epoch")
    
    # Phase 2 Specifics
    parser.add_argument("--lambda_div", type=float, default=1.0, help="weight for diversity loss")
    parser.add_argument("--pretrained_G", type=str, default="", help="path to phase1 G weights")
    parser.add_argument("--pretrained_D", type=str, default="", help="path to phase1 D weights")
    
    # Paths
    parser.add_argument("--csv_path", type=str, default="data/processed/ham10000_train.csv", help="path to train csv")
    parser.add_argument("--sample_dir", type=str, default="samples/phase2", help="directory to save samples")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/phase2", help="directory to save model checkpoints")

    args = parser.parse_args()
    print(args)
    
    train(args)
