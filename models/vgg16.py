import torch
import torch.nn as nn
import torch.functional as F
from torchsummary import summary

class VGGLayer(nn.Module):
    """Implements convolution and ReLU VGG"""
    def __init__(self, in_chs, out_chs):
        super(VGGLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chs),
            nn.LeakyReLU(True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class LarssonVGG16(nn.Module):
    """Implements VGG16-Gray by Larsson et al. paper"""

    def __init__(self):
        super(LarssonVGG16, self).__init__()
        self.max_pool = nn.MaxPool2d(2)

        self.conv_block_1 = nn.Sequential(
            VGGLayer(1, 64),
            VGGLayer(64, 64)
        )

        self.conv_block_2 = nn.Sequential(
            VGGLayer(64, 128),
            VGGLayer(128, 128)
        )

        self.conv_block_3 = nn.Sequential(
            VGGLayer(128, 256),
            VGGLayer(256, 256),
            VGGLayer(256, 256)
        )

        self.conv_block_4 = nn.Sequential(
            VGGLayer(256, 512),
            VGGLayer(512, 512),
            VGGLayer(512, 512)
        )

        self.conv_block_5 = nn.Sequential(
            VGGLayer(512, 512),
            VGGLayer(512, 512),
            VGGLayer(512, 512)
        )

        self.conv6 = VGGLayer(512, 1024)

        self.conv7 = VGGLayer(1024, 1024)

        self.upsample = nn.Upsample((64, 64))

        self.h_fc1 = nn.Sequential(
            nn.Conv2d(3520, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
        )

    def forward(self, x):
        conv1 = self.conv_block_1(x)
        x = self.max_pool(conv1)
        conv2 = self.conv_block_2(x)
        x = self.max_pool(conv2)
        conv3 = self.conv_block_3(x)
        x = self.max_pool(conv3)
        conv4 = self.conv_block_4(x)
        x = self.max_pool(conv4)
        conv5 = self.conv_block_5(x)
        x = self.max_pool(conv5)
        conv6 = self.conv6(x)
        conv7 = self.conv7(conv6)

        hypercolumn = torch.concat((self.upsample(conv1),
                                     self.upsample(conv2),
                                       self.upsample(conv3),
                                         self.upsample(conv4),
                                           self.upsample(conv5),
                                             self.upsample(conv6),
                                               self.upsample(conv7)
                                               ), dim=1) 

        histograms = self.h_fc1(hypercolumn)
        return histograms
    

if __name__ == '__main__':
    test_vgg = LarssonVGG16()
    out = test_vgg(torch.rand(1, 1, 64, 64))
    assert out.shape == torch.Size([1, 1024, 64, 64])

    summary(test_vgg, (1, 64, 64), device='cpu')