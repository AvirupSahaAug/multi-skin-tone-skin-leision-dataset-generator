import torch
import torch.nn as nn
from torchvision import models

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_features)
        )
    def forward(self, x):
        return x + self.block(x)

class MiniUNetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiniUNetHead, self).__init__()
        # Depth 1: Down
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), # e.g. 128x128 -> 64x64
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Depth 2: Down
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1), # 64x64 -> 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Bottleneck
        self.bottleneck = ResidualBlock(128)
        
        # Depth 2: Up
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 32x32 -> 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Depth 1: Up
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 64x64 -> 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final Output mapping
        self.final = nn.Sequential(
            nn.Conv2d(64 + in_channels, out_channels, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # x is the output from the backbone
        d1 = self.down1(x)
        d2 = self.down2(d1)
        
        b = self.bottleneck(d2)
        
        u1 = self.up1(b)
        u1_skip = torch.cat([u1, d1], dim=1) # 64 + 64 = 128 channels
        
        u2 = self.up2(u1_skip)
        u2_skip = torch.cat([u2, x], dim=1) # 64 + in_channels channels
        
        return self.final(u2_skip)


class DynamicMultiHeadGenerator(nn.Module):
    def __init__(self, z_dim=100, num_classes=7, num_heads=6, img_size=128, channels=3, conv_dim=64):
        super(DynamicMultiHeadGenerator, self).__init__()
        self.img_size = img_size
        self.init_size = img_size // 16
        self.num_heads = num_heads
        
        # Label embedding for the disease
        self.label_emb = nn.Embedding(num_classes, z_dim)
        
        self.l1 = nn.Sequential(
            nn.Linear(z_dim * 2, conv_dim * 8 * self.init_size ** 2)
        )

        # Shared Backbone (Structure-Locked: All spatial upsampling happens here)
        # Upgraded to be more powerful utilizing ResidualBlocks
        self.backbone = nn.Sequential(
            nn.BatchNorm2d(conv_dim * 8),
            nn.Upsample(scale_factor=2), # 8 -> 16
            nn.Conv2d(conv_dim * 8, conv_dim * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim * 4),
            nn.ReLU(inplace=True),
            ResidualBlock(conv_dim * 4),
            
            nn.Upsample(scale_factor=2), # 16 -> 32
            nn.Conv2d(conv_dim * 4, conv_dim * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim * 2),
            nn.ReLU(inplace=True),
            ResidualBlock(conv_dim * 2),
            
            nn.Upsample(scale_factor=2), # 32 -> 64
            nn.Conv2d(conv_dim * 2, conv_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(inplace=True),
            ResidualBlock(conv_dim),

            nn.Upsample(scale_factor=2), # 64 -> 128
            nn.Conv2d(conv_dim, conv_dim // 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim // 2),
            nn.ReLU(inplace=True),
            ResidualBlock(conv_dim // 2)
        )

        # Create the Base Image converter mapping dense features to a 3-channel pristine image evaluated by D
        self.base_head = nn.Sequential(
            nn.Conv2d(conv_dim // 2, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

        # Create N dynamic Mini-UNet heads for N skin tones
        # Notice in_channels=channels (3). The UNets are strictly Image-to-Image local structural colorizers now!
        self.heads = nn.ModuleList([self._make_head(channels, channels) for _ in range(num_heads)])

    def _make_head(self, in_channels, out_channels):
        # Heads now utilize a Mini UNet to refine local color mappings heavily without losing structural context
        return MiniUNetHead(in_channels, out_channels)

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        
        out = self.l1(x)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        
        features = self.backbone(out)
        base_img = self.base_head(features)
        
        # The 6 Output heads directly recolor the generated Base Image
        outputs = [head(base_img) for head in self.heads]
        return base_img, outputs

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=7, img_size=128, channels=3, conv_dim=64):
        super(ConditionalDiscriminator, self).__init__()
        
        self.img_size = img_size
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)
        
        self.model = nn.Sequential(
            nn.Conv2d(channels + 1, conv_dim, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(conv_dim, conv_dim * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(conv_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(conv_dim * 2, conv_dim * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(conv_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(conv_dim * 4, conv_dim * 8, 4, stride=2, padding=1),
            nn.InstanceNorm2d(conv_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(conv_dim * 8, 1, 4, stride=1, padding=0)
        )

    def forward(self, img, labels):
        c = self.label_embedding(labels).view(labels.size(0), 1, self.img_size, self.img_size)
        x = torch.cat([img, c], 1)
        return self.model(x)

class SkinToneClassifier(nn.Module):
    """
    Classifies the Fitzpatrick skin tone into `num_tones` categories.
    Can be pre-trained or trained jointly.
    """
    def __init__(self, num_tones=6, img_size=128, channels=3, conv_dim=64):
        super(SkinToneClassifier, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(channels, conv_dim, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(conv_dim, conv_dim * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(conv_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(conv_dim * 2, conv_dim * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(conv_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(conv_dim * 4, conv_dim * 8, 4, stride=2, padding=1),
            nn.InstanceNorm2d(conv_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Flatten and Classify
            nn.Flatten(),
            nn.Linear(conv_dim * 8 * (img_size // 16) * (img_size // 16), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_tones)
        )

    def forward(self, img):
        # Outputs unnormalized logits for each skin tone
        return self.model(img)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)

class StrongSkinToneClassifier(nn.Module):
    def __init__(self, num_tones=6, pretrained=True):
        super(StrongSkinToneClassifier, self).__init__()
        # Use ResNet18 as a much stronger feature extractor
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Replace the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_tones)
        )

    def forward(self, x):
        return self.resnet(x)
