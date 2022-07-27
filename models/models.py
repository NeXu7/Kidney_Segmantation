import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
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

class _DownStream(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = _DoubleConv(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x

class _UpStream(nn.Module):
    def __init__(self, in_cnannels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_cnannels, in_cnannels // 2,
                                         kernel_size=2, stride=2)
        self.double_conv = _DoubleConv(in_cnannels, out_channels)
    def forward(self, x_1, x_2):
        x_1 = self.upconv(x_1)
        x = torch.cat([x_1, x_2], dim=1)
        x = self.double_conv(x)
        return x

class _PredictConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.predict_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    def forward(self, x):
        return self.predict_conv(x)

class _WeightedCat(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.weights = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
    def forward(self, x_1, x_2):
        x = torch.cat([x_1, x_2], dim=1)
        x = self.weights(x)
        return x

class UNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.input_conv = _DoubleConv(self.input_channels, 64)
        self.down_1 = _DownStream(64, 128)
        self.down_2 = _DownStream(128, 256)
        self.down_3 = _DownStream(256, 512)
        self.down_4 = _DownStream(512, 1024)

        self.up_4 = _UpStream(1024, 512)
        self.up_3 = _UpStream(512, 256)
        self.up_2 = _UpStream(256, 128)
        self.up_1 = _UpStream(128, 64)

        self.predict_map = _PredictConv(64, self.num_classes)

    def forward(self, x):
        x_i = self.input_conv(x)
        x_1 = self.down_1(x_i)
        x_2 = self.down_2(x_1)
        x_3 = self.down_3(x_2)
        x_4 = self.down_4(x_3)

        x = self.up_4(x_4, x_3)
        x = self.up_3(x, x_2)
        x = self.up_2(x, x_1)
        x = self.up_1(x, x_i)
        x = self.predict_map(x)
        return x

class LeakyUNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.input_conv = _DoubleConv(self.input_channels, 64)
        self.down_1 = _DownStream(64, 128)
        self.down_2 = _DownStream(128, 256)
        self.leak_1 = _DoubleConv(self.input_channels,
                                  middle_channels=64,
                                  out_channels=256)
        self.waight_cat = _WeightedCat(256)
        self.up_2 = _UpStream(256, 128)
        self.up_1 = _UpStream(128, 64)
        self.predict_map = _PredictConv(64, self.num_classes)
        self.down_scaler = nn.AvgPool2d(kernel_size=4, stride=4)

    def forward(self, x):
        x_i = self.input_conv(x) # 3 => 64
        x_1 = self.down_1(x_i) # 64 => 128
        x_2 = self.down_2(x_1) # 128 => 256
        x = self.down_scaler(x)
        x = self.leak_1(x) # 3 => 64 => 256
        x = self.waight_cat(x, x_2) # 256 + 256
        x = self.up_2(x, x_1) # (512 + 128
        x = self.up_1(x, x_i)
        x = self.predict_map(x)
        return x