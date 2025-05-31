import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetEncoder(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=False):
        super(ResNetEncoder, self).__init__()
        if backbone == 'resnet18':
            net = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            net = models.resnet34(pretrained=pretrained)
        else:
            raise NotImplementedError("Only resnet18 or resnet34 supported")

        self.initial = nn.Sequential(
            net.conv1,  # (B, 64, H/2, W/2)
            net.bn1,
            net.relu,
        )
        self.maxpool = net.maxpool  # (B, 64, H/4, W/4)
        self.layer1 = net.layer1    # (B, 64, H/4, W/4)
        self.layer2 = net.layer2    # (B, 128, H/8, W/8)
        self.layer3 = net.layer3    # (B, 256, H/16, W/16)
        self.layer4 = net.layer4    # (B, 512, H/32, W/32)

    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.maxpool(x0)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return [x0, x2, x3, x4, x5]

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

class SiameseUNetLab(nn.Module):
    def __init__(self):
        super(SiameseUNetLab, self).__init__()
        self.encoder1 = ResNetEncoder(pretrained=False)
        self.encoder2 = ResNetEncoder(pretrained=False)

        self.decoder4 = DecoderBlock(512, 256*2, 256)
        self.decoder3 = DecoderBlock(256, 128*2, 128)
        self.decoder2 = DecoderBlock(128, 64*2, 64)
        self.decoder1 = DecoderBlock(64, 64*2, 32)
        self.decoder0 = DecoderBlock(32, 64*2, 16)

        self.final_conv = nn.Conv2d(16, 3, kernel_size=1)  # Output deltaL, deltaA, deltaB

    def forward(self, input1, input2):
        feats1 = self.encoder1(input1)
        feats2 = self.encoder2(input2)

        x = torch.abs(feats1[-1] - feats2[-1])
        x = self.decoder4(x, torch.cat([feats1[-2], feats2[-2]], dim=1))
        x = self.decoder3(x, torch.cat([feats1[-3], feats2[-3]], dim=1))
        x = self.decoder2(x, torch.cat([feats1[-4], feats2[-4]], dim=1))
        x = self.decoder1(x, torch.cat([feats1[-5], feats2[-5]], dim=1))
        x = self.decoder0(x, torch.cat([feats1[0], feats2[0]], dim=1))

        out = self.final_conv(x)  # (B, 3, H, W) => deltaL, deltaA, deltaB
        return out