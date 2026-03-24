import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import SkinLesionDataset
from models_unified import DynamicMultiHeadGenerator, ConditionalDiscriminator, StrongSkinToneClassifier, weights_init

def train(args):
    # 1. Setup Directories
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data
    print("Loading lesion dataset (HAM10000)...")
    dataset = SkinLesionDataset(csv_filepath=args.csv_path, target_size=(args.img_size, args.img_size))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 3. Initialize Models
    G = DynamicMultiHeadGenerator(z_dim=args.z_dim, num_classes=args.num_classes, 
                                  num_heads=args.num_tones, img_size=args.img_size).to(device)
    
    D = ConditionalDiscriminator(num_classes=args.num_classes, img_size=args.img_size).to(device)
    
    # Use the STRONG classifier (ResNet18) for 4 Tones
    C = StrongSkinToneClassifier(num_tones=args.num_tones, pretrained=False).to(device)

    G.apply(weights_init)
    D.apply(weights_init)

    if args.pretrained_classifier and os.path.exists(args.pretrained_classifier):
        print(f"Loading pretrained STRONG tone classifier from {args.pretrained_classifier}")
        C.load_state_dict(torch.load(args.pretrained_classifier, map_location=device, weights_only=True))
        C.eval() 
        for param in C.parameters():
            param.requires_grad = False
    else:
        print("Warning: Strong Tone Classifier weights not found or not provided. Training G will be ineffective for tone control.")
        C.apply(weights_init)
        C.train()

    # 4. Optimizers & Losses
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    
    adversarial_loss = nn.BCEWithLogitsLoss()
    tone_loss_fn = nn.CrossEntropyLoss()

    # Fixed noise for sampling
    fixed_z = torch.randn(args.num_classes, args.z_dim).to(device)
    fixed_labels = torch.arange(args.num_classes).long().to(device)

    # 5. Training Loop
    print(f"Starting Unified Training with Strong Classifier ({args.num_tones} Heads)...")
    for epoch in range(args.start_epoch, args.n_epochs):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{args.n_epochs}")
        for i, (real_imgs, lesion_labels) in pbar:
            
            real_imgs = real_imgs.to(device)
            lesion_labels = lesion_labels.to(device)
            batch_size = real_imgs.size(0)

            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            real_pred = D(real_imgs, lesion_labels)
            d_real_loss = adversarial_loss(real_pred, torch.ones_like(real_pred))

            z = torch.randn(batch_size, args.z_dim).to(device)
            gen_labels = torch.randint(0, args.num_classes, (batch_size,)).to(device)
            
            fake_imgs_list = G(z, gen_labels)
            
            d_fake_loss = 0
            for fake_img in fake_imgs_list:
                fake_pred = D(fake_img.detach(), gen_labels)
                d_fake_loss += adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
            d_fake_loss /= args.num_tones

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()
            g_adv_loss = torch.tensor(0.0).to(device)
            g_tone_loss = torch.tensor(0.0).to(device)
            g_cons_loss = torch.tensor(0.0).to(device)

            for t, fake_img in enumerate(fake_imgs_list):
                # 1. Adversarial Loss
                fake_pred = D(fake_img, gen_labels)
                g_adv_loss += adversarial_loss(fake_pred, torch.ones_like(fake_pred))

                # 2. Tone Loss (Using Strong C)
                tone_preds = C(fake_img)
                if args.tone_loss_type == "ce":
                    target_tones = torch.full((batch_size,), t, dtype=torch.long).to(device)
                    g_tone_loss += tone_loss_fn(tone_preds, target_tones)
                else:
                    # Ordinal Distance Penalty
                    probs = torch.softmax(tone_preds, dim=1)
                    indices = torch.arange(args.num_tones).float().to(device)
                    soft_predict = torch.sum(probs * indices, dim=1)
                    dist = torch.abs(soft_predict - t)
                    g_tone_loss += torch.mean(torch.exp(dist) - 1.0)

                # 3. Structural Consistency Loss
                if t > 0:
                    g_cons_loss += torch.mean(torch.abs(fake_img - fake_imgs_list[0]))

            g_adv_loss /= args.num_tones
            g_tone_loss /= args.num_tones
            g_cons_loss /= (args.num_tones - 1) if args.num_tones > 1 else 1

            # --- Dynamic Lambda Scaling ---
            adv_loss_val = g_adv_loss.item()
            scale_factor = 1.0
            if adv_loss_val > 1.5:  
                scale_factor = max(0.5, 1.5 / adv_loss_val)

            curr_lambda_tone = args.lambda_tone * scale_factor
            curr_lambda_cons = args.lambda_consistency * scale_factor

            g_loss = g_adv_loss + (curr_lambda_tone * g_tone_loss) + (curr_lambda_cons * g_cons_loss)
            g_loss.backward()
            optimizer_G.step()

            if i % args.log_interval == 0:
                pbar.set_postfix({
                    'D_loss': f'{d_loss.item():.4f}', 
                    'G_adv': f'{g_adv_loss.item():.4f}',
                    'G_tone': f'{g_tone_loss.item():.4f}'
                })

        # Save Samples 
        if epoch % args.sample_interval == 0:
            with torch.no_grad():
                G.eval()
                sample_imgs_list = G(fixed_z, fixed_labels)
                combined_images = torch.cat(sample_imgs_list, dim=3)
                save_image(combined_images, f"{args.sample_dir}/epoch_{epoch}.png", normalize=True, nrow=1)
                G.train()

        if epoch % args.checkpoint_interval == 0:
            torch.save(G.state_dict(), f"{args.checkpoint_dir}/G_strong_4tones_{epoch}.pth")
            torch.save(D.state_dict(), f"{args.checkpoint_dir}/D_strong_4tones_{epoch}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--sample_interval", type=int, default=1)
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--num_tones", type=int, default=4) # DEFAULT TO 4 TONES
    parser.add_argument("--lambda_tone", type=float, default=1.0)
    parser.add_argument("--tone_loss_type", type=str, default="exp", choices=["ce", "exp"])
    parser.add_argument("--lambda_consistency", type=float, default=10.0)
    # USE 4 TONE CHECKPOINT
    parser.add_argument("--pretrained_classifier", type=str, default="checkpoints/unified/C_strong_4tones_best.pth")
    parser.add_argument("--csv_path", type=str, default="data/processed/ham10000_train.csv")
    parser.add_argument("--sample_dir", type=str, default="samples/strong_unified_4tones")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/strong_unified_4tones")

    args = parser.parse_args()
    train(args)
