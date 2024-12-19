import torch
import torch.nn as nn
import torch.nn.functional as F

class SSE(nn.Module):
    def __init__(self, in_channels):
        super(SSE, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        out = self.global_avgpool(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = torch.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        
        return x * out
    

class ParNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ParNetBlock, self).__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.sse = SSE(out_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        out = branch1x1 + branch3x3
        out = self.sse(out)
        out = F.silu(out)

        return out
    

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SSE(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.se(out)

        return out
    

class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SSE(out_channels)
        self.concat = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x1, x2):
        x1, x2 = self.conv1x1(x1), self.conv1x1(x2)
        x1, x2 = self.bn(x1), self.bn(x2)
        x1, x2 = self.relu(x1), self.relu(x2)
        x1, x2 = self.se(x1), self.se(x2)
        out = torch.cat([x1, x2], dim=1)
        out = self.concat(out)

        return out
    
class ParNet(nn.Module):
    def __init__(self, num_classes):
        super(ParNet, self).__init__()
        self.downsampling_blocks = nn.ModuleList([
            DownsamplingBlock(3, 64),
            DownsamplingBlock(64, 128),
            DownsamplingBlock(128, 256),
        ])

        self.streams = nn.ModuleList([
            nn.Sequential(
                ParNetBlock(64, 64),
                ParNetBlock(64, 64),
                ParNetBlock(64, 64),
                DownsamplingBlock(64, 128)
            ),
            nn.Sequential(
                ParNetBlock(128, 128),
                ParNetBlock(128, 128),
                ParNetBlock(128, 128),
                ParNetBlock(128, 128)
            ),
            nn.Sequential(
                ParNetBlock(256, 256),
                ParNetBlock(256, 256),
                ParNetBlock(256, 256),
                ParNetBlock(256, 256)
            )
        ])

        self.fusion_blocks = nn.ModuleList([
            FusionBlock(128, 256),
            FusionBlock(256, 256)
        ])

        self.final_downsampling = DownsamplingBlock(256, 1024)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        downsampled_features = []
        for i, downsampling_block in enumerate(self.downsampling_blocks):
            x = downsampling_block(x)
            downsampled_features.append(x)

        stream_features = []
        for i, stream in enumerate(self.streams):
            stream_feature = stream(downsampled_features[i])
            stream_features.append(stream_feature)

        fused_features = stream_features[0]
        for i in range(1, len(stream_features)):
            fused_features = self.fusion_blocks[i - 1](fused_features, stream_features[i])

        x = self.final_downsampling(fused_features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x