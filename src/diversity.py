import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        # Load pre-trained ResNet18
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        
        # We only need features, so remove the fully connected layer
        # Output of avgpool is 512
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        
        # Freeze parameters to use as fixed feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # x: (B, 3, 128, 128)
        # Output: (B, 512, 1, 1) -> Flatten to (B, 512)
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return features

def diversity_loss(imgs):
    """
    imgs: List of 3 image tensors [x1, x2, x3]. Each is (B, 3, H, W).
    """
    # Simply Average Pairwise L1 distance in pixel space? 
    # Or Feature space? Feature space is better (per literature).
    pass 
    # We will implement the loss logic in the training loop using the encoder above.
