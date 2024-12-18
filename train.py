import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models import *
import matplotlib.pyplot as plt

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 定义数据预处理操作
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))])

# 加载CIFAR10训练集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

# 定义设备（GPU优先，若可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化模型
model_name = 'ResNet18'
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
elif model_name == 'GoogleNet':
    model = GoogleNet(num_classes=10).to(device)
elif model_name == 'ResNet18':
    model = ResNet18(num_classes=10).to(device)
elif model_name == 'ResNet34':
    model = ResNet34(num_classes=10).to(device)
elif model_name == 'ResNet50':
    model = ResNet50(num_classes=10).to(device)
elif model_name == 'ResNet101':
    model = ResNet101(num_classes=10).to(device)
elif model_name == 'ResNet152':
    model = ResNet152(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练轮次
epochs = 15

def train(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

if __name__ == "__main__":
    loss_history, acc_history = [], []
    for epoch in range(epochs):
        train_loss, train_acc = train(model, trainloader, criterion, optimizer, device)
        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        loss_history.append(train_loss)
        acc_history.append(train_acc)
        # 保存模型权重，每5轮次保存到weights文件夹下
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'weights/{model_name}_epoch_{epoch + 1}.pth')
    
    # 绘制损失曲线
    plt.plot(range(1, epochs+1), loss_history, label='Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(f'results\\{model_name}_train_loss_curve.png')
    plt.close()

    # 绘制准确率曲线
    plt.plot(range(1, epochs+1), acc_history, label='Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Curve')
    plt.legend()
    plt.savefig(f'results\\{model_name}_train_acc_curve.png')
    plt.close()