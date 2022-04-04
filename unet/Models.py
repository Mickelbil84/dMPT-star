""" Implementing UNet architecture.
Dervied from: https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py
"""

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from unet.Layers import DoubleConv, Down, OutConv, Up


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_points=None, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.inc = DoubleConv(n_channels, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

        #################################
        # Novel addition: Path predictor
        #################################
        self.n_points = n_points
        self.pathPred = nn.Sequential(
            # Assume we start after class pred, i.e. having shape (b x n x w x h)
            Rearrange("b n w h -> b (n w h)"),
            nn.ReLU(True),
            nn.Linear(n_classes * 480 * 480, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, self.n_points * 2),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        if self.n_points is not None:
            path_prediction = self.pathPred(logits)
            path_prediction = torch.tanh(path_prediction)
            path_prediction = rearrange(path_prediction, "b (n d) -> b n d", d=2)
            return path_prediction

        return logits
