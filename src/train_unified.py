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
from models_unified import DynamicMultiHeadGenerator, ConditionalDiscriminator, SkinToneClassifier, weights_init

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

    # Note: If you want to train the Tone Classifier on real Fitzpatrick images concurrently,
    # you would load the Fitzpatrick dataloader here too. For now, the script defines the models
    # and trains G and D, while ensuring G's heads match the Tone Classifier output.

    # 3. Initialize Models
    # N-Heads Generator, 1 head for each of the `args.num_tones` skin tones
    G = DynamicMultiHeadGenerator(z_dim=args.z_dim, num_classes=args.num_classes, 
                                  num_heads=args.num_tones, img_size=args.img_size).to(device)
    
    # Conditional Discriminator (Real vs Fake based on Lesion class)
    D = ConditionalDiscriminator(num_classes=args.num_classes, img_size=args.img_size).to(device)
    
    # Skin Tone Classifier (Classifies Fitzpatrick Tone 0 to N-1)
    # Ideally pre-trained! If pre-trained, we freeze it.
    C = SkinToneClassifier(num_tones=args.num_tones, img_size=args.img_size).to(device)

    G.apply(weights_init)
    D.apply(weights_init)
    C.apply(weights_init)

    if args.pretrained_classifier:
        print(f"Loading pretrained tone classifier from {args.pretrained_classifier}")
        C.load_state_dict(torch.load(args.pretrained_classifier, map_location=device, weights_only=True))
        # Usually, when pushing gradients back to G to enforce skin tone, C doesn't need to be trained simultaneously
        C.eval() 
        for param in C.parameters():
            param.requires_grad = False
    else:
        print("Warning: Tone Classifier is training from scratch on generator outputs only, which won't enforce real-world skin tones without actual Fitzpatrick data.")
        C.train()

    # 4. Optimizers & Losses
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    
    # Only optimize C if we intend to train it (e.g., if we had joint fitzpatrick data)
    optimizer_C = optim.Adam(C.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    adversarial_loss = nn.BCEWithLogitsLoss()
    tone_loss_fn = nn.CrossEntropyLoss()

    # Fixed noise for sampling (e.g. 1 sample per disease class to see variation)
    fixed_z = torch.randn(args.num_classes, args.z_dim).to(device)
    fixed_labels = torch.arange(args.num_classes).long().to(device)

    # 5. Training Loop
    print(f"Starting Unified Training ({args.num_tones} Heads)...")
    for epoch in range(args.start_epoch, args.n_epochs):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{args.n_epochs}")
        for i, (real_imgs, lesion_labels) in pbar:
            
            real_imgs = real_imgs.to(device)
            lesion_labels = lesion_labels.to(device)
            batch_size = real_imgs.size(0)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_pred = D(real_imgs, lesion_labels)
            d_real_loss = adversarial_loss(real_pred, torch.ones_like(real_pred))

            z = torch.randn(batch_size, args.z_dim).to(device)
            gen_labels = torch.randint(0, args.num_classes, (batch_size,)).to(device)
            
            # Generate N images, one for each skin tone head
            fake_imgs_list = G(z, gen_labels)
            
            # Fake Loss (Randomly select one head's output to train D, or average them)
            d_fake_loss = 0
            for fake_img in fake_imgs_list:
                fake_pred = D(fake_img.detach(), gen_labels)
                d_fake_loss += adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
            d_fake_loss /= args.num_tones

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            g_adv_loss = 0
            g_tone_loss = 0
            g_cons_loss = 0

            # For each head 't', it should fool D, AND be classified as tone 't' by C
            for t, fake_img in enumerate(fake_imgs_list):
                # 1. Adversarial Loss
                fake_pred = D(fake_img, gen_labels)
                g_adv_loss += adversarial_loss(fake_pred, torch.ones_like(fake_pred))

                # 2. Tone Loss
                tone_preds = C(fake_img)
                if args.tone_loss_type == "ce":
                    target_tones = torch.full((batch_size,), t, dtype=torch.long).to(device)
                    g_tone_loss += tone_loss_fn(tone_preds, target_tones)
                else:
                    # Exponential Distance Penalty (Ordinal)
                    # We calculate a soft prediction index and penalize the distance to 't'
                    probs = torch.softmax(tone_preds, dim=1)
                    indices = torch.arange(args.num_tones).float().to(device)
                    soft_predict = torch.sum(probs * indices, dim=1)
                    dist = torch.abs(soft_predict - t)
                    # exp(dist)-1 makes the penalty grow much faster the further away we are
                    g_tone_loss += torch.mean(torch.exp(dist) - 1.0)

                # 3. Structural Consistency Loss (L1 distance to the first head)
                # This ensures the lesion shape doesn't drift between heads
                if t > 0:
                    g_cons_loss += torch.mean(torch.abs(fake_img - fake_imgs_list[0]))

            g_adv_loss /= args.num_tones
            g_tone_loss /= args.num_tones
            g_cons_loss /= (args.num_tones - 1) if args.num_tones > 1 else 1

            # Total Generator loss
            g_loss = g_adv_loss + (args.lambda_tone * g_tone_loss) + (args.lambda_consistency * g_cons_loss)
            g_loss.backward()
            optimizer_G.step()
            
            # ----------------------------------------
            # Train Classifier (C) on generated data?
            # ----------------------------------------
            # Without real fitzpatrick data mixed in, C won't learn *real* skin tones.
            # In a full run, you'd insert real fitzpatrick images here and train C.
            # E.g.: optimizer_C.step() after computing tone_loss on real fitzpatrick batch.

            # Logging
            if i % args.log_interval == 0:
                pbar.set_postfix({
                    'D_loss': f'{d_loss.item():.4f}', 
                    'G_adv': f'{g_adv_loss.item():.4f}',
                    'G_tone': f'{g_tone_loss.item():.4f}'
                })

        # Save Samples dynamically depending on numb of heads
        if epoch % args.sample_interval == 0:
            with torch.no_grad():
                G.eval()
                sample_imgs_list = G(fixed_z, fixed_labels)
                # Output shape for each list item: (num_classes, 3, H, W)
                # We want a grid where each Row is a disease class, each Col is a Skin Tone
                # Stack along Width (dim=3) to merge the heads horizontally
                combined_images = torch.cat(sample_imgs_list, dim=3)
                save_image(combined_images, f"{args.sample_dir}/epoch_{epoch}.png", normalize=True, nrow=1)
                G.train()
                print(f"Saved generated tone grid to {args.sample_dir}/epoch_{epoch}.png")

        if epoch % args.checkpoint_interval == 0:
            torch.save(G.state_dict(), f"{args.checkpoint_dir}/G_unified_{epoch}.pth")
            torch.save(D.state_dict(), f"{args.checkpoint_dir}/D_unified_{epoch}.pth")
            torch.save(C.state_dict(), f"{args.checkpoint_dir}/C_unified_{epoch}.pth")

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
    
    # Unified Args
    parser.add_argument("--num_classes", type=int, default=7, help="Number of lesion classes in HAM10000")
    parser.add_argument("--num_tones", type=int, default=6, help="Number of skin tones (i.e., num heads)")
    parser.add_argument("--lambda_tone", type=float, default=1.0, help="Weight for skin tone classification loss")
    parser.add_argument("--tone_loss_type", type=str, default="exp", choices=["ce", "exp"], help="Type of tone loss: 'ce' or 'exp'")
    parser.add_argument("--lambda_consistency", type=float, default=10.0, help="Weight for structural consistency between heads")
    parser.add_argument("--pretrained_classifier", type=str, default="", help="Path to pre-trained tone classifier")
    
    # Paths
    parser.add_argument("--csv_path", type=str, default="data/processed/ham10000_train.csv")
    parser.add_argument("--sample_dir", type=str, default="samples/unified")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/unified")

    args = parser.parse_args()
    train(args)
