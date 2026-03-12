import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=7, img_size=128, channels=3, conv_dim=64):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.init_size = img_size // 16  # 128 // 16 = 8
        
        # Label Embedding
        self.label_emb = nn.Embedding(num_classes, z_dim)
        
        # Initial linear layer
        self.l1 = nn.Sequential(
            nn.Linear(z_dim * 2, conv_dim * 8 * self.init_size ** 2)
        )

        # Shared Backbone (upsampling)
        # 8x8 -> 16x16 -> 32x32 -> 64x64
        self.backbone = nn.Sequential(
            # 8x8 -> 16x16
            nn.BatchNorm2d(conv_dim * 8),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(conv_dim * 8, conv_dim * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim * 4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(conv_dim * 4, conv_dim * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim * 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 64x64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(conv_dim * 2, conv_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Head (Final upsampling to image)
        # 64x64 -> 128x128
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(conv_dim, conv_dim // 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim // 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(conv_dim // 2, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Conditional Input: Concatenate noise and label embedding
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        
        out = self.l1(x)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        
        features = self.backbone(out)
        img = self.head(features)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes=7, img_size=128, channels=3, conv_dim=64):
        super(Discriminator, self).__init__()
        
        # We will append label info as an extra channel
        self.img_size = img_size
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)
        
        self.model = nn.Sequential(
            # Input: (channels + 1) x 128 x 128
            nn.Conv2d(channels + 1, conv_dim, 4, stride=2, padding=1), # -> 64x64
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(conv_dim, conv_dim * 2, 4, stride=2, padding=1), # -> 32x32
            nn.InstanceNorm2d(conv_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(conv_dim * 2, conv_dim * 4, 4, stride=2, padding=1), # -> 16x16
            nn.InstanceNorm2d(conv_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(conv_dim * 4, conv_dim * 8, 4, stride=2, padding=1), # -> 8x8
            nn.InstanceNorm2d(conv_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer (Validity)
            nn.Conv2d(conv_dim * 8, 1, 4, stride=1, padding=0) # -> 5x5 (PatchGAN-like) or similar
        )

    def forward(self, img, labels):
        # Create label channel
        c = self.label_embedding(labels).view(labels.size(0), 1, self.img_size, self.img_size)
        
        # Concatenate: (N, C+1, H, W)
        x = torch.cat([img, c], 1)
        
        out = self.model(x)
        return out # Return logits (BCEWithLogitsLoss will be used)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__":
    # Test Sanity
    z_dim = 100
    batch_size = 4
    num_classes = 7
    
    G = Generator(z_dim=z_dim, num_classes=num_classes)
    D = Discriminator(num_classes=num_classes)
    
    z = torch.randn(batch_size, z_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    fake_imgs = G(z, labels)
    print(f"Generator Output Shape: {fake_imgs.shape}") # Should be [4, 3, 128, 128]
    
    d_out = D(fake_imgs, labels)
    print(f"Discriminator Output Shape: {d_out.shape}") # Should be [4, 1, H, W] or [4, 1] depends on padding
