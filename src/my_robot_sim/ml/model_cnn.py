import torch
import torch.nn as nn
import torch.nn.functional as F


class DynCellNet(nn.Module):
    def __init__(self, in_channels=2):  # <-- ahora por defecto 2
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # Decoder
        self.deconv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.deconv1 = nn.Conv2d(32, 16, 3, padding=1)

        # Output layer
        self.out_conv = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        # x: (B,K,100,100)

        x1 = F.relu(self.conv1(x))   # (B,16,100,100)
        x2 = F.relu(self.conv2(x1))  # (B,32,100,100)
        x3 = F.relu(self.conv3(x2))  # (B,64,100,100)

        y2 = F.relu(self.deconv2(x3))  # (B,32,100,100)
        y1 = F.relu(self.deconv1(y2))  # (B,16,100,100)

        out = self.out_conv(y1)        # (B,1,100,100)
        out = torch.sigmoid(out)
        return out
