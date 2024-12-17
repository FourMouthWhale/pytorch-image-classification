import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models import *

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# 定义数据预处理操作
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))])

# 加载CIFAR10测试集
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

# 定义设备（GPU优先，若可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化模型
# 实例化模型
model_name = 'Vgg_A'
if model_name == 'AlexNet':
    model = AlexNet(num_classes=10).to(device)
elif model_name == 'Vgg_A':
    model = Vgg(cfg_vgg='A', num_classes=10).to(device)
elif model_name == 'Vgg_A-LRN':
    model = Vgg(cfg_vgg='A-LRN', num_classes=10).to(device)
elif model_name == 'Vgg_B':
    model = Vgg(cfg_vgg='B', num_classes=10).to(device)
elif model_name == 'Vgg_C':
    model = Vgg(cfg_vgg='C', num_classes=10).to(device)
elif model_name == 'Vgg_D':
    model = Vgg(cfg_vgg='D', num_classes=10).to(device)
elif model_name == 'Vgg_E':
    model = Vgg(cfg_vgg='E', num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()

# 加载模型权重
weights_path = f"weights/{model_name}_epoch_15.pth"  
model.load_state_dict(torch.load(weights_path, map_location=device))

def test(model, testloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(testloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

if __name__ == "__main__":
    test_loss, test_acc = test(model, testloader, criterion, device)
    print(f"================{model_name} Test================")
    print(f"Load Model Weights From: {weights_path}")
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')