import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetAutoencoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 1, H, W] -> [B, 16, H/2, W/2]
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> [B, 32, H/4, W/4]
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> [B, 64, H/8, W/8]
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> [B, 32, H/4, W/4]
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),  # -> [B, 16, H/2, W/2]
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # -> [B, 1, H, W]
            nn.Sigmoid()  # Use sigmoid if spectrograms are normalized to [0, 1]
        )

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)  # [B, 16, H/2, W/2]
        e2 = self.enc2(e1)  # [B, 32, H/4, W/4]
        e3 = self.enc3(e2)  # [B, 64, H/8, W/8]

        # Decode with skip connections
        d3 = self.dec3(e3)  # [B, 32, H/4, W/4]
        d3 = torch.cat((d3, e2), dim=1)  # Skip connection

        d2 = self.dec2(d3)  # [B, 16, H/2, W/2]
        d2 = torch.cat((d2, e1), dim=1)  # Skip connection

        d1 = self.dec1(d2)  # [B, 1, H, W]
        return d1
