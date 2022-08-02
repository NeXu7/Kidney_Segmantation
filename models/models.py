from .bricks import *
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.input_conv = DoubleConv(self.input_channels, 64)
        self.down_1 = DownStream(64, 128)
        self.down_2 = DownStream(128, 256)
        self.down_3 = DownStream(256, 512)
        self.down_4 = DownStream(512, 1024)

        self.up_4 = UpStream(1024, 512)
        self.up_3 = UpStream(512, 256)
        self.up_2 = UpStream(256, 128)
        self.up_1 = UpStream(128, 64)

        self.predict_map = PredictConv(64, self.num_classes)

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

class LeakyUNet3(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.input_conv = DoubleConv(self.input_channels, 64)
        self.down_1 = DownStream(64, 128)
        self.down_2 = DownStream(128, 256)
        self.leak_1 = DoubleConv(self.input_channels,
                                  middle_channels=64,
                                  out_channels=256)
        self.weight_cat = WeightedCat(256)
        self.up_2 = UpStream(256, 128)
        self.up_1 = UpStream(128, 64)
        self.predict_map = PredictConv(64, self.num_classes)
        self.down_scaler = nn.AvgPool2d(kernel_size=4, stride=4)

    def forward(self, x):
        x_i = self.input_conv(x)
        x_1 = self.down_1(x_i)
        x_2 = self.down_2(x_1)
        x_l = self.down_scaler(x)
        x_l = self.leak_1(x)
        x_2 = self.weight_cat(x_l, x_2)
        x = self.up_2(x_2, x_1)
        x = self.up_1(x, x_i)
        x = self.predict_map(x)
        return x


class LeakyUNet5(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.input_conv = DoubleConv(self.input_channels, 64)
        self.down_1 = DownStream(64, 128)
        self.down_2 = DownStream(128, 256)
        self.leak_1 = DoubleConv(self.input_channels,
                                  middle_channels=64,
                                  out_channels=256)
        self.down_3 = DownStream(256, 512)
        self.down_4 = DownStream(512, 1024)
        self.leak_2 = DoubleConv(self.input_channels,
                                 middle_channels=256,
                                 out_channels=1024)

        self.weight_cat_1 = WeightedCat(256)
        self.weight_cat_2 = WeightedCat(1024)

        self.up_4 = UpStream(1024, 512)
        self.up_3 = UpStream(512, 256)
        self.up_2 = UpStream(256, 128)
        self.up_1 = UpStream(128, 64)

        self.predict_map = PredictConv(64, self.num_classes)

        self.down_scaler_4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.down_scaler_16 = nn.AvgPool2d(kernel_size=16, stride=16)

    def forward(self, x):
        x_i = self.input_conv(x)
        x_1 = self.down_1(x_i)
        x_2 = self.down_2(x_1)
        x_l = self.down_scaler(x)
        x_l = self.leak_1(x_l)
        x_2 = self.weight_cat_1(x_l, x_2)
        x_3 = self.down_3(x_2)
        x_4 = self.down_4(x_3)
        x_l = self.down_scaler_16(x)
        x_l = self.leak_2(x_l)
        x_4 = self.weight_cat_2(x_l, x_4)

        x = self.up_4(x_4, x_3)
        x = self.up_3(x, x_2)
        x = self.up_2(x, x_1)
        x = self.up_1(x, x_i)
        x = self.predict_map(x)
        return x
