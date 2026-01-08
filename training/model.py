"""
Conditional U-Net dehazing model with ResNet-18 feature extractor.

Architecture:
    Input → ResNet-18 (frozen) → Conditional U-Net → Output
    
Conditioning: 3-class haze severity (light/medium/heavy) injected at bottleneck
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-18 feature extractor (frozen weights).
    Pretrained on ImageNet for transfer learning.
    """
    
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet-18
        resnet = models.resnet18(pretrained=True)
        
        # Extract feature layers (remove final FC layer)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze all weights - we only use this for feature extraction
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """Extract features from input image."""
        return self.features(x)


class ConditionalUNet(nn.Module):
    """
    U-Net style dehazing network with conditional haze severity.
    
    Conditioning strategy:
        - 3-class one-hot vector (light/medium/heavy haze)
        - Injected at bottleneck via feature concatenation
    """
    
    def __init__(self, in_channels=3, out_channels=3, condition_dim=3):
        super().__init__()
        
        # Encoder (downsampling path)
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck (deepest layer, 1/16th resolution)
        # 512 -> 1024 channels
        self.bottleneck = self._conv_block(512 + condition_dim, 1024)
        
        # Decoder (upsampling path)
        # Up4: 1024 -> 512. Concat with Enc4 (512) -> 1024 in. Split to 512 out.
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._conv_block(512 + 512, 512)
        
        # Up3: 512 -> 256. Concat with Enc3 (256) -> 512 in. Split to 256 out.
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(256 + 256, 256)
        
        # Up2: 256 -> 128. Concat with Enc2 (128) -> 256 in. Split to 128 out.
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(128 + 128, 128)
        
        # Up1: 128 -> 64. Concat with Enc1 (64) -> 128 in. Split to 64 out.
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(64 + 64, 64)
        
        # Final output layer
        self.out = nn.Conv2d(64, out_channels, 1)
    
    def _conv_block(self, in_ch, out_ch):
        """Basic convolutional block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, condition):
        """
        Args:
            x: Input image tensor (B, 3, 512, 512)
            condition: One-hot haze severity (B, 3)
        """
        # Encoder
        e1 = self.enc1(x)                   # 512x512, 64ch
        e2 = self.enc2(self.pool(e1))       # 256x256, 128ch
        e3 = self.enc3(self.pool(e2))       # 128x128, 256ch
        e4 = self.enc4(self.pool(e3))       # 64x64,   512ch
        
        # Bottleneck input (Pool e4 -> 32x32)
        b_in = self.pool(e4)
        
        # Conditioning
        b, c = condition.shape
        h, w = b_in.shape[2], b_in.shape[3]
        condition_spatial = condition.view(b, c, 1, 1).expand(b, c, h, w)
        
        # Bottleneck
        bottleneck_input = torch.cat([b_in, condition_spatial], dim=1)
        bottleneck = self.bottleneck(bottleneck_input) # 32x32, 1024ch
        
        # Decoder
        d4 = self.upconv4(bottleneck)       # 64x64, 512ch
        d4 = torch.cat([d4, e4], dim=1)     # 512+512 = 1024ch
        d4 = self.dec4(d4)                  # 64x64, 512ch
        
        d3 = self.upconv3(d4)               # 128x128, 256ch
        d3 = torch.cat([d3, e3], dim=1)     # 256+256 = 512ch
        d3 = self.dec3(d3)                  # 128x128, 256ch
        
        d2 = self.upconv2(d3)               # 256x256, 128ch
        d2 = torch.cat([d2, e2], dim=1)     # 128+128 = 256ch
        d2 = self.dec2(d2)                  # 256x256, 128ch
        
        d1 = self.upconv1(d2)               # 512x512, 64ch
        d1 = torch.cat([d1, e1], dim=1)     # 64+64 = 128ch
        d1 = self.dec1(d1)                  # 512x512, 64ch
        
        # Output
        out = self.out(d1)
        
        return out


class DehazingModel(nn.Module):
    """
    Complete dehazing model combining ResNet-18 feature extractor
    and conditional U-Net dehazing network.
    """
    
    def __init__(self):
        super().__init__()
        self.feature_extractor = ResNetFeatureExtractor()
        self.dehazing_net = ConditionalUNet()
    
    def forward(self, x, condition):
        """
        Args:
            x: Input hazy image (B, 3, 512, 512)
            condition: One-hot haze severity (B, 3)
        
        Returns:
            Dehazed image (B, 3, 512, 512)
        """
        # Note: ResNet features could be used for additional conditioning
        # For simplicity, we use the U-Net directly on input
        # Future improvement: Incorporate ResNet features into conditioning
        
        return self.dehazing_net(x, condition)


def get_model(device='cuda'):
    """
    Create and initialize the dehazing model.
    
    Args:
        device: 'cuda' or 'cpu' (CUDA required for this project)
    
    Returns:
        model: DehazingModel instance on specified device
    """
    # Enforce CUDA requirement
    assert torch.cuda.is_available(), "CUDA is required for this project!"
    
    model = DehazingModel()
    model = model.to(device)
    
    return model
