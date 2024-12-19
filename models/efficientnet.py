import torch
import torch.nn as nn
import torch.nn.functional as F


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, padding, se_ratio=0.25):
        super(MBConv, self).__init__()
        self.stride = stride
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        # 扩展通道数（如果需要）
        expanded_channels = in_channels * expand_ratio
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels)

        # 深度可分离卷积
        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, groups=expanded_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(expanded_channels)

        # 压缩和激励（SE）模块（可选，根据se_ratio判断是否添加）
        if se_ratio > 0:
            se_channels = int(in_channels * se_ratio)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded_channels, se_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(se_channels, expanded_channels, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.se = None

        # 投影卷积，恢复到输出通道数
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        # 扩展通道数
        out = F.relu(self.bn1(self.expand_conv(x)))

        # 深度可分离卷积
        out = F.relu(self.bn2(self.depthwise_conv(out)))

        # SE模块操作（如果存在）
        if self.se is not None:
            se_out = self.se(out)
            out = out * se_out

        # 投影卷积
        out = self.bn3(self.project_conv(out))

        # 残差连接（如果满足条件）
        if self.use_res_connect:
            out += identity
        return out


class EfficientNet(nn.Module):
    def __init__(self, num_classes, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2):
        super(EfficientNet, self).__init__()
        self.stem_conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        mbconv_config = [
            # (in_channels, out_channels, expand_ratio, kernel_size, stride, padding)
            (32, 16, 1, 3, 1, 1),
            (16, 24, 6, 3, 2, 1),
            (24, 40, 6, 5, 2, 2),
            (40, 80, 6, 3, 2, 1),
            (80, 112, 6, 5, 1, 2),
            (112, 192, 6, 5, 2, 2),
            (192, 320, 6, 3, 1, 1)
        ]

        # 根据深度系数调整每个MBConv模块的重复次数，这里简单地向下取整，你也可以根据实际情况采用更合理的方式
        repeat_counts = [max(1, int(depth_coefficient * 1)) for _ in mbconv_config]
        layers = []
        for i, config in enumerate(mbconv_config):
            in_channels, out_channels, expand_ratio, kernel_size, stride, padding = config
            for _ in range(repeat_counts[i]):
                layers.append(MBConv(int(in_channels * width_coefficient),
                                     int(out_channels * width_coefficient),
                                     expand_ratio, kernel_size, stride, padding))

        self.mbconv_layers = nn.Sequential(*layers)

        self.last_conv = nn.Conv2d(int(320 * width_coefficient), 1280, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.stem_conv(x)))
        out = self.mbconv_layers(out)
        out = F.relu(self.bn2(self.last_conv(out)))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out