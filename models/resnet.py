import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()