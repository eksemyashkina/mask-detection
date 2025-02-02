from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_batch(in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU()
    )


class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        reduced_channels = in_channels // 2
        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        return x + self.layer2(self.layer1(x))
    

class Darknet53(nn.Module):
    def __init__(self, block: nn.Module = DarkResidualBlock) -> None:
        super().__init__()
        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)

    def make_layer(self, block: nn.Module, in_channels: int, num_blocks: int) -> nn.Sequential:
        layers = []
        for _ in range(num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual_block1(x)
        x = self.conv3(x)
        x = self.residual_block2(x)
        x = self.conv4(x)
        x = self.residual_block3(x)
        c4 = x
        x = self.conv5(x)
        x = self.residual_block4(x)
        c5 = x
        x = self.conv6(x)
        x = self.residual_block5(x)
        c6 = x
        return c4, c5, c6


def conv_leaky(in_ch: int, out_ch: int, k: int = 1, s: int = 1, p: int = 0):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True)
    )


class DetectionHead(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, num_anchors: int = 3, num_classes: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            conv_leaky(in_ch, mid_ch, k=1, s=1, p=0),
            conv_leaky(mid_ch, mid_ch * 2, k=3, s=1, p=1),
            conv_leaky(mid_ch * 2, mid_ch, k=1, s=1, p=0),
            conv_leaky(mid_ch, mid_ch * 2, k=3, s=1, p=1),
            conv_leaky(mid_ch * 2, mid_ch, k=1, s=1, p=0)
        )
        self.out_conv = nn.Conv2d(mid_ch, num_anchors * (5 + num_classes), kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        out = self.out_conv(x)
        return out
    

class YOLOv3(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.backbone = Darknet53()
        self.num_classes = num_classes
        self.num_anchors = 3
        self.head_large = DetectionHead(in_ch=1024, mid_ch=512, num_anchors=3, num_classes=num_classes)
        self.head_medium = DetectionHead(in_ch=1024, mid_ch=256, num_anchors=3, num_classes=num_classes)
        self.head_small = DetectionHead(in_ch=512, mid_ch=128, num_anchors=3, num_classes=num_classes)
        self.conv_upsample_l2 = conv_leaky(1024, 512, k=1, s=1, p=0) 
        self.conv_upsample_l3 = conv_leaky(1024, 256, k=1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        c4, c5, c6 = self.backbone(x)
        out_l = self.head_large(c6)
        x_l2 = self.conv_upsample_l2(c6)
        x_l2_up = F.interpolate(x_l2, scale_factor=2, mode="nearest")
        x_merge_l2 = torch.cat([x_l2_up, c5], dim=1)
        out_m = self.head_medium(x_merge_l2)
        x_l3 = self.conv_upsample_l3(x_merge_l2)
        x_l3_up = F.interpolate(x_l3, scale_factor=2, mode="nearest")
        x_merge_l3 = torch.cat([x_l3_up, c4], dim=1)
        out_s = self.head_small(x_merge_l3)
        return out_l, out_m, out_s