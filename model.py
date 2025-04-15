import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetAutoencoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        # # Encoder
        # self.enc1 = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 1, H, W] -> [B, 16, H/2, W/2]
        #     nn.BatchNorm2d(16),
        #     nn.ReLU()
        # )
        # self.enc2 = nn.Sequential(
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> [B, 32, H/4, W/4]
        #     nn.BatchNorm2d(32),
        #     nn.ReLU()
        # )
        # self.enc3 = nn.Sequential(
        #     nn.Conv2d(32, 96, kernel_size=3, stride=2, padding=1),  # -> [B, 96, H/8, W/8]
        #     nn.BatchNorm2d(96),
        #     nn.ReLU()
        # )

        # # Bottleneck
        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(96, 128, kernel_size=3, padding=1),  # [B, 96, H/8, W/8] -> [B, 128, H/8, W/8]
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 96, kernel_size=3, padding=1),  # [B, 128, H/8, W/8] -> [B, 96, H/8, W/8]
        #     nn.BatchNorm2d(96),
        #     nn.ReLU(inplace=True)
        # )

        # # Decoder
        # self.dec3 = nn.Sequential(
        #     nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1),  # -> [B, 64, H/4, W/4]
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        # self.dec2 = nn.Sequential(
        #     nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),  # -> [B, 32, H/2, W/2]
        #     nn.BatchNorm2d(32),
        #     nn.ReLU()
        # )
        # self.dec1 = nn.Sequential(
        #     nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # -> [B, 1, H, W]
        #     nn.Sigmoid()  # Use sigmoid if spectrograms are normalized to [0, 1]
        # )
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
            nn.Conv2d(32, 96, kernel_size=3, stride=2, padding=1),  # -> [B, 96, H/8, W/8]
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),  # [B, 96, H/8, W/8] -> [B, 128, H/8, W/8]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 96, kernel_size=3, padding=1),  # [B, 128, H/8, W/8] -> [B, 96, H/8, W/8]
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=4, stride=2, padding=1),  # 96 (bottleneck) + 96 (e3) -> [B, 64, H/4, W/4]
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(96, 32, kernel_size=4, stride=2, padding=1),  # 64 (d3) + 32 (e2) -> [B, 32, H/2, W/2]
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(48, 1, kernel_size=4, stride=2, padding=1),  # 32 (d2) + 16 (e1) -> [B, 1, H, W]
            nn.Sigmoid()  # Use sigmoid if spectrograms are normalized to [0, 1]
        )

    def center_crop(self, enc_feat, dec_feat):
        _, _, h, w = enc_feat.shape
        _, _, h2, w2 = dec_feat.shape
        dh = (h - h2) // 2
        dw = (w - w2) // 2
        return enc_feat[:, :, dh:dh+h2, dw:dw+w2]

    def forward(self, x):
        # # Encoder
        # e1 = self.enc1(x)      # [B, 16, H/2, W/2]
        # e2 = self.enc2(e1)     # [B, 32, H/4, W/4]
        # e3 = self.enc3(e2)     # [B, 96, H/8, W/8]

        # # Bottleneck
        # b = self.bottleneck(e3)  # [B, 96, H/8, W/8]

        # # Decoder
        # d3 = self.dec3(b)         # [B, 64, H/4, W/4]
        
        # # Resize e3 to match d3 dimensions before concatenation
        # e3_cropped = F.interpolate(e3, size=d3.shape[2:], mode='bilinear', align_corners=False)
        # d3 = torch.cat((d3, e3_cropped), dim=1)  # Skip connection

        # d2 = self.dec2(d3)        # [B, 32, H/2, W/2]
        
        # # Resize e2 to match d2 dimensions before concatenation
        # e2_cropped = F.interpolate(e2, size=d2.shape[2:], mode='bilinear', align_corners=False)
        # d2 = torch.cat((d2, e2_cropped), dim=1)

        # d1 = self.dec1(d2)        # [B, 1, H, W]
        
        # # Resize e1 to match d1 dimensions before concatenation
        # e1_cropped = F.interpolate(e1, size=d1.shape[2:], mode='bilinear', align_corners=False)
        # d1 = torch.cat((d1, e1_cropped), dim=1)

        # return d1
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        e3_cropped = self.center_crop(e3, b)
        b_cat = torch.cat((b, e3_cropped), dim=1)  # [B, 192, H/8, W/8]
        d3 = self.dec3(b_cat)                      # [B, 64, H/4, W/4]

        e2_cropped = self.center_crop(e2, d3)
        d3_cat = torch.cat((d3, e2_cropped), dim=1)  # [B, 96, H/4, W/4]
        d2 = self.dec2(d3_cat)

        e1_cropped = self.center_crop(e1, d2)
        d2_cat = torch.cat((d2, e1_cropped), dim=1)  # [B, 48, H/2, W/2]
        out = self.dec1(d2_cat)  # [B, 1, H, W]

        return out

