import os
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import SkinLesionDataset
from models_unified import DynamicMultiHeadGenerator, ConditionalDiscriminator, weights_init

def get_ita(rgb_images):
    """
    Calculate average ITA (Individual Typology Angle) for a batch of RGB images.
    Expects rgb_images to be in the range [-1.0, 1.0] with shape (B, 3, H, W).
    """
    # 1. Normalize [-1, 1] to [0, 1], rigidly clamp to avoid any negatives triggering NaNs in math
    rgb = (rgb_images + 1.0) / 2.0
    rgb = torch.clamp(rgb, min=1e-8, max=1.0)
    
    # 2. RGB to XYZ conversion
    # Inverse sRGB companding
    mask = rgb > 0.04045
    # Clamp the base to ensure we never evaluate power on negatives backwards through torch.where
    base = torch.clamp((rgb + 0.055) / 1.055, min=1e-8)
    rgb_lin = torch.where(mask, torch.pow(base, 2.4), rgb / 12.92)
    
    r = rgb_lin[:, 0:1, :, :]
    g = rgb_lin[:, 1:2, :, :]
    b_rgb = rgb_lin[:, 2:3, :, :]
    
    # D65 illuminant
    x = r * 0.4124 + g * 0.3576 + b_rgb * 0.1805
    y = r * 0.2126 + g * 0.7152 + b_rgb * 0.0722
    z = r * 0.0193 + g * 0.1192 + b_rgb * 0.9505
    
    # 3. XYZ to LAB conversion
    # Reference white (D65)
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    
    # Clamp bounds strictly above 0 as well to prevent fractional pow(-val) NaNs
    x = torch.clamp(x / xn, min=1e-8)
    y = torch.clamp(y / yn, min=1e-8)
    z = torch.clamp(z / zn, min=1e-8)
    
    def f_t(t):
        mask = t > 0.008856
        return torch.where(mask, torch.pow(t, 1/3), 7.787 * t + 16/116)
    
    fx = f_t(x)
    fy = f_t(y)
    fz = f_t(z)
    
    L = torch.clamp(116.0 * fy - 16.0, min=0.0)
    b_val = 200.0 * (fy - fz)
    
    # 4. Calculate ITA (in degrees)
    # Using torch.atan2(y, x). Standard formula: arctan((L - 50) / b)
    ita = torch.atan2(L - 50.0, b_val + 1e-8) * (180.0 / torch.pi)
    
    # Calculate average ITA per image (spatial average), ignoring dark vignette masks
    valid_mask = (L > 15.0).float()
    valid_count = torch.sum(valid_mask, dim=(1, 2, 3)) + 1e-8
    avg_ita = torch.sum(ita * valid_mask, dim=(1, 2, 3)) / valid_count
    return avg_ita

def get_sobel_edge_mask(img):
    """
    Computes an isotropic Sobel Edge Map Mask from a batch of generated RGB images.
    Used for resilient structural consistency locking across skin-tone heads.
    """
    import torch.nn.functional as F
    
    # Convert image to grayscale for robust structural edge detection
    # img is (B, 3, H, W). Weights for RGB to Grayscale
    gray = 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
    
    # Define standard 3x3 Sobel kernels for X and Y edge detection
    # Registered directly to the image device to avoid Tensor device mismatch
    kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(img.device)
    kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).to(img.device)
    
    # Convolve to get edge map filters
    edge_x = F.conv2d(gray, kernel_x, padding=1)
    edge_y = F.conv2d(gray, kernel_y, padding=1)
    
    # Calculate the combined edge magnitude (The specific mathematical Edge Mask)
    edge_mask = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
    return edge_mask

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

    G.apply(weights_init)
    D.apply(weights_init)

    # 3.5 Auto-Resume Checkpoint Logic
    import glob
    g_ckpts = glob.glob(os.path.join(args.checkpoint_dir, "G_ita_6tones_*.pth"))
    if g_ckpts:
        try:
            # Extract integers by splitting off the '.pth' and the last underscore dynamically
            epochs = [int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in g_ckpts]
            latest_epoch = max(epochs)
            
            g_ckpt_path = os.path.join(args.checkpoint_dir, f"G_ita_6tones_{latest_epoch}.pth")
            d_ckpt_path = os.path.join(args.checkpoint_dir, f"D_ita_6tones_{latest_epoch}.pth")
            
            print(f"🔄 Found existing checkpoint. Resuming model weights from Epoch {latest_epoch}...")
            # We map_location safely to guarantee no GPU/CPU state mismatches occur
            G.load_state_dict(torch.load(g_ckpt_path, map_location=device))
            if os.path.exists(d_ckpt_path):
                D.load_state_dict(torch.load(d_ckpt_path, map_location=device))
                
            args.start_epoch = latest_epoch + 1
        except Exception as e:
            print(f"⚠️ Failed to automatically load checkpoint: {e}")

    # 4. Optimizers & Losses
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    
    adversarial_loss = nn.BCEWithLogitsLoss()

    # Fixed noise for sampling
    fixed_z = torch.randn(args.num_classes, args.z_dim).to(device)
    fixed_labels = torch.arange(args.num_classes).long().to(device)

    # Define target ITAs for the heads (now 6 tones cleanly bounded between [50.0, -10.0])
    # Tones mapped evenly: 50.0 -> 38.0 -> 26.0 -> 14.0 -> 2.0 -> -10.0
    target_itas = torch.tensor([50.0, 38.0, 26.0, 14.0, 2.0, -10.0]).to(device)

    # 5. Training Loop
    print(f"Starting Unified Training w/ ITA Algorithm ({args.num_tones} Heads)...")
    for epoch in range(args.start_epoch, args.n_epochs):
        # -------------------------------------------------------------
        # Live Config Reload (for Streamlit GUI)
        # -------------------------------------------------------------
        config_path = "training_config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                    args.lambda_adv = cfg.get("lambda_adv", args.lambda_adv)
                    args.lambda_head_adv = cfg.get("lambda_head_adv", args.lambda_head_adv)
                    args.lambda_tone = cfg.get("lambda_tone", args.lambda_tone)
                    args.lambda_consistency = cfg.get("lambda_cons", args.lambda_consistency)
            except Exception as e:
                pass # Ignore parsing errors during live file writes
                
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{args.n_epochs} | L_adv={args.lambda_adv:.1f} L_tone={args.lambda_tone:.1f} L_cons={args.lambda_consistency:.1f}")
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
            
            base_img_fake, fake_imgs_list = G(z, gen_labels)
            
            # Form the primary structural adversarial signal from the base image
            fake_pred_base = D(base_img_fake.detach(), gen_labels)
            d_fake_loss = adversarial_loss(fake_pred_base, torch.zeros_like(fake_pred_base))
            
            # Feed the 6 heads into D as well with a deliberately smaller multiplier
            # so D learns to reject poor coloration textures without getting overwhelmed!
            for fake_img in fake_imgs_list:
                fake_pred_head = D(fake_img.detach(), gen_labels)
                d_fake_loss += args.lambda_head_adv * adversarial_loss(fake_pred_head, torch.zeros_like(fake_pred_head))

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()
            g_adv_loss = torch.tensor(0.0).to(device)
            g_tone_loss = torch.tensor(0.0).to(device)
            g_cons_loss = torch.tensor(0.0).to(device)

            # 1. Adversarial Loss (Discriminator primarily targets the base_image layout to guarantee robust, unbroken structures)
            fake_pred_base = D(base_img_fake, gen_labels)
            g_adv_loss = adversarial_loss(fake_pred_base, torch.ones_like(fake_pred_base))

            # The 6 Output heads guarantee exact physical similarity to the base_img_fake
            edge_base = get_sobel_edge_mask(base_img_fake)
            
            g_adv_head_loss = torch.tensor(0.0).to(device)

            for t, fake_img in enumerate(fake_imgs_list):
                # Add the small discriminator loss onto the actual colored output to maintain realistic skin textures!
                fake_pred_head = D(fake_img, gen_labels)
                g_adv_head_loss += adversarial_loss(fake_pred_head, torch.ones_like(fake_pred_head))
                
                # 2. Tone Loss (Using Linear Distance)
                ita_vals = get_ita(fake_img)
                target_ita = target_itas[t]
                
                # Linear distance (L1 Penalty) from generated image's ITA to the target ITA
                # Scaled down dynamically by 100 to keep the gradient scale balanced with adversarial/consistency losses
                normalized_ita_diff = torch.abs(ita_vals - target_ita) / 100.0
                g_tone_loss += torch.mean(normalized_ita_diff)

                # 3. Structural Consistency Loss (Sobel Edge Mask Locking)
                # We enforce consistency purely on the *edges* / physical structure of the lesion,
                # leaving the raw color completely free for the generator to shift safely.
                
                # Extract the true shape structure via an isotropic Sobel convolution map
                edge_fake = get_sobel_edge_mask(fake_img)
                
                # Penalize ONLY if the physical Edge Maps diverge from the clean Base Image
                g_cons_loss += torch.mean(torch.abs(edge_fake - edge_base))

            # We don't divide the main g_adv_loss by num_tones anymore since there's only 1 base layout targeted
            # But we softly map the downscaled head-realism penalties back into the generator's global adv loss!
            g_adv_loss = g_adv_loss + (args.lambda_head_adv * (g_adv_head_loss / args.num_tones))
            
            g_tone_loss /= args.num_tones
            g_cons_loss /= args.num_tones

            # --- Dynamic Lambda Scaling ---
            adv_loss_val = g_adv_loss.item()
            scale_factor = 1.0
            if adv_loss_val > 1.5:  
                scale_factor = max(0.5, 1.5 / adv_loss_val)

            curr_lambda_adv  = args.lambda_adv * scale_factor
            curr_lambda_tone = args.lambda_tone * scale_factor
            curr_lambda_cons = args.lambda_consistency * scale_factor

            g_loss = (curr_lambda_adv * g_adv_loss) + (curr_lambda_tone * g_tone_loss) + (curr_lambda_cons * g_cons_loss)
            g_loss.backward()
            optimizer_G.step()

            if i % args.log_interval == 0:
                pbar.set_postfix({
                    'D_loss': f'{d_loss.item():.4f}', 
                    'G_adv': f'{g_adv_loss.item():.4f}',
                    'G_ita': f'{g_tone_loss.item():.4f}'
                })

        # Save Samples for checking and GUI
        if epoch % args.sample_interval == 0:
            with torch.no_grad():
                G.eval()
                # Run the static seed sample tracking
                _, sample_imgs_list = G(fixed_z, fixed_labels)
                
                # Combine for the checkpoint dir history
                combined_images = torch.cat(sample_imgs_list, dim=3)
                # Resize the combined grid to be strictly smaller by exactly half scale!
                import torch.nn.functional as F
                small_combined = F.interpolate(combined_images, scale_factor=0.5, mode='bilinear', align_corners=False)
                
                # Save out the singular combined grids
                save_image(small_combined, f"{args.sample_dir}/epoch_{epoch}.png", normalize=True, nrow=1)
                save_image(small_combined, "latest_sample.png", normalize=True, nrow=1)
                
                # Calculate metrics for the web dashboard mapping
                head_itas = []
                for fake_imgs in sample_imgs_list:
                    # Store average ITA across the batch for this head
                    avg_score = get_ita(fake_imgs).mean().item()
                    head_itas.append(round(avg_score, 1))
                
                # Expose the ITA metrics file to Streamlit
                with open("latest_itas.json", "w") as f:
                    json.dump(head_itas, f)
                    
                G.train()

        if epoch % args.checkpoint_interval == 0:
            torch.save(G.state_dict(), f"{args.checkpoint_dir}/G_ita_6tones_{epoch}.pth")
            torch.save(D.state_dict(), f"{args.checkpoint_dir}/D_ita_6tones_{epoch}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=70) # 70 EPOCHS as requested
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
    parser.add_argument("--num_tones", type=int, default=6) # DEFAULT TO 6 TONES
    parser.add_argument("--lambda_adv", type=float, default=1.0)
    parser.add_argument("--lambda_head_adv", type=float, default=0.05)
    parser.add_argument("--lambda_tone", type=float, default=5.0) 
    parser.add_argument("--lambda_consistency", type=float, default=10.0)
    parser.add_argument("--csv_path", type=str, default="data/processed/ham10000_train.csv")
    parser.add_argument("--sample_dir", type=str, default="samples/ita_6tones")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ita_6tones")

    args = parser.parse_args()
    train(args)
