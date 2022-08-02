import torch
import torch.nn as nn
import torchvision.transforms.functional as trfn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, middle_channels=None):
        super().__init__()
        if middle_channels is None:
            middle_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1,
                      bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1,
                      bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        pad = 1 if kernel_size == 3 else 0
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=pad, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv_block(x)


class DownStream(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_stream = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.down_stream(x)


class UpStream(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                         kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)
    def forward(self, x_1, x_2):
        x_1 = self.upconv(x_1)
        x = torch.cat([x_1, x_2], dim=1)
        x = self.double_conv(x)
        return x


class PredictConv(nn.Module):
    def __init__(self, in_channels, num_classes, output_size=512):
        super().__init__()
        self.predict_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.output_size = output_size
    def forward(self, x):
        x = self.predict_conv(x)
        x = trfn.center_crop(x, output_size=self.output_size)
        return x


class WeightedCat(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.weights = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
    def forward(self, x_1, x_2):
        x = torch.cat([x_1, x_2], dim=1)
        x = self.weights(x)
        return x


class _DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_1 = ConvBlock(2 * in_channels, 2 * in_channels)
        self.conv_2 = ConvBlock(4 * in_channels, 4 * in_channels)
        self.conv_3 = ConvBlock(8 * in_channels, 8 * in_channels)
        self.bottle_neck = ConvBlock(
            8 * in_channels, out_channels, kernel_size=1
        )
    def forward(self, x):
        x = torch.cat([x, self.conv_1(x)])
        x = torch.cat([x, self.conv_2(x)])
        x = torch.cat([x, self.conv_3(x)])
        x = self.bottle_neck(x)
        return x


class DenseBlock(_DenseBlock):
    def __init__(self, in_channels, out_channels):
        super(DenseBlock, self).__init__(in_channels, out_channels)
        self.input_conv = ConvBlock(in_channels, in_channels)
    def forward(self, x):
        x = torch.cat([x, self.input_conv(x)])
        x = super(DenseBlock, self).forward(x)
        return x


class CatDenseBlock(_DenseBlock):
    def __init__(self, in_channels, out_channels):
        super(CatDenseBlock, self).__init__(in_channels, out_channels)
    def forward(self, x_1, x_2):
        x = torch.cat([x_1, x_2])
        x = super(CatDenseBlock, self).forward(x)
        return x

