import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # input size: (B, 3, 32, 32)   (Batch_size, Channel, Height, Width)
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), # (B, 64, 16, 16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),    # (B, 64, 8, 8)
            nn.Conv2d(64, 192, kernel_size=3, padding=1),   # (B, 192, 8, 8)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),    # (B, 192, 4, 4)
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # (B, 384, 4, 4)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # (B, 256, 4, 4)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # (B, 256, 4, 4)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),    # (B, 256, 2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 *2)
        x = self.classifier(x)
        return x