import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU6(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out
    
class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU6(inplace=True)

        self.dsconv1 = DepthwiseSeparableConv(32, 64, stride=1)
        self.dsconv2 = DepthwiseSeparableConv(64, 128, stride=2)
        self.dsconv3 = DepthwiseSeparableConv(128, 128, stride=1)
        self.dsconv4 = DepthwiseSeparableConv(128, 256, stride=2)
        self.dsconv5 = DepthwiseSeparableConv(256, 256, stride=1)
        self.dsconv6 = DepthwiseSeparableConv(256, 512, stride=2)

        self.dsconv7 = DepthwiseSeparableConv(512, 512, stride=1)
        self.dsconv8 = DepthwiseSeparableConv(512, 512, stride=1)
        self.dsconv9 = DepthwiseSeparableConv(512, 512, stride=1)
        self.dsconv10 = DepthwiseSeparableConv(512, 512, stride=1)
        self.dsconv11 = DepthwiseSeparableConv(512, 512, stride=1)

        self.dsconv12 = DepthwiseSeparableConv(512, 1024, stride=2)
        self.dsconv13 = DepthwiseSeparableConv(1024, 1024, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dsconv1(out)
        out = self.dsconv2(out)
        out = self.dsconv3(out)
        out = self.dsconv4(out)
        out = self.dsconv5(out)
        out = self.dsconv6(out)

        out = self.dsconv7(out)
        out = self.dsconv8(out)
        out = self.dsconv9(out)
        out = self.dsconv10(out)
        out = self.dsconv11(out)

        out = self.dsconv12(out)
        out = self.dsconv13(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out