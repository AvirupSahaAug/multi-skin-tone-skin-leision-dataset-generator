import torch
import torch.nn as nn

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

        # Shared Backbone
        self.backbone = nn.Sequential(
            nn.BatchNorm2d(conv_dim * 8),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(conv_dim * 8, conv_dim * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim * 4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(conv_dim * 4, conv_dim * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim * 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(conv_dim * 2, conv_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Create N dynamic heads for N skin tones
        self.heads = nn.ModuleList([self._make_head(conv_dim, channels) for _ in range(num_heads)])

    def _make_head(self, conv_dim, channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(conv_dim, conv_dim // 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim // 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(conv_dim // 2, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        
        out = self.l1(x)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        
        features = self.backbone(out)
        
        # Return a list of N images, one for each skin tone
        outputs = [head(features) for head in self.heads]
        return outputs

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
        nn.init.constant_(m.bias.data, 0)
