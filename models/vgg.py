import torch
import torch.nn as nn


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'A-LRN' : [64, 'LRN', 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 'conv1-256', 'M', 512, 512, 'conv1-512', 'M', 512, 512, 'conv1-512', 'M'], 
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class Vgg(nn.Module):
    def __init__(self, cfg_vgg, num_classes):
        super(Vgg, self).__init__()
        self.features = self._make_layers(cfg_vgg)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)

        return x

    def _make_layers(self, cfg_vgg):
        layers = []
        in_channels = 3
        for i in cfg[cfg_vgg]:
            if i == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif i == 'LRN':
                layers += [nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.7, k=2)]
            elif i == 'conv1-256':
                conv2d = nn.Conv2d(in_channels, 256, kernel_size=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = 256
            else:
                conv2d = nn.Conv2d(in_channels, i, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = i
        
        return nn.Sequential(*layers)