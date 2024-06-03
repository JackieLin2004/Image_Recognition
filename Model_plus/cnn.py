import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
# from tqdm import tqdm

from pre_processing import load_data
import torch.nn.functional as F

# 检测是否有可用的GPU，如果没有则用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 设置随机种子，保证结果的可重复性
def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.pooling = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(38 * 28 * 64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 11)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = F.relu(self.pooling(self.conv3(x)))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# 定义模型


def train(epoch, x, y):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        # inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 10))
            running_loss = 0.0

    x.append(epoch + 1)
    y.append(running_loss / len(train_loader))


def test(z):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    z.append(100 * correct / total)
    print('accuracy on test set: %d %% ' % (100 * correct / total))


def smooth(a):
    bat = 10
    size = len(a)
    i = 0
    while i < size:
        sum = 0
        for j in range(i, i + bat):
            if j < size:
                sum += a[j]
            else:
                bat = j - i
                break
        sum /= bat
        for j in range(i, i + bat):
            if j < size:
                a[j] = sum
            else:
                break
        i += bat
    return a


if __name__ == '__main__':
    model = CNN()
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
    sample = 0.5
    train_loader, test_loader = load_data(sample)
    x1 = []
    y1 = []
    z1 = []
    for epoch in range(200):
        train(epoch, x1, y1)
        test(z1)
    y1 = smooth(y1)
    z1 = smooth(z1)
    print('5:5 finished-----------------------')

    model = CNN()
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
    sample = 0.6
    train_loader, test_loader = load_data(sample)
    x2 = []
    y2 = []
    z2 = []
    for epoch in range(200):
        train(epoch, x2, y2)
        test(z2)
    y2 = smooth(y2)
    z2 = smooth(z2)
    print('6:4 finished-----------------------')

    model = CNN()
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
    sample = 0.7
    train_loader, test_loader = load_data(sample)
    x3 = []
    y3 = []
    z3 = []
    for epoch in range(200):
        train(epoch, x3, y3)
        test(z3)
    y3 = smooth(y3)
    z3 = smooth(z3)
    print('7:3 finished-----------------------')

    model = CNN()
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
    sample = 0.8
    train_loader, test_loader = load_data(sample)
    x4 = []
    y4 = []
    z4 = []
    for epoch in range(200):
        train(epoch, x4, y4)
        test(z4)
    y4 = smooth(y4)
    z4 = smooth(z4)
    print('8:2 finished-----------------------')

    model = CNN()
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
    sample = 0.9
    train_loader, test_loader = load_data(sample)
    x5 = []
    y5 = []
    z5 = []
    for epoch in range(200):
        train(epoch, x5, y5)
        test(z5)
    y5 = smooth(y5)
    z5 = smooth(z5)
    print('9:1 finished-----------------------')

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(x1, y1, color='green', label='5:5')
    plt.plot(x2, y2, color='red', label='6:4')
    plt.plot(x3, y3, color='yellow', label='7:3')
    plt.plot(x4, y4, color='blue', label='8:2')
    plt.plot(x5, y5, color='orange', label='9:1')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Image')
    plt.legend()
    plt.savefig(f"./loss_test.png")
    plt.show()

    # 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(x1, z1, color='green', label='5:5')
    plt.plot(x2, z2, color='red', label='6:4')
    plt.plot(x3, z3, color='yellow', label='7:3')
    plt.plot(x4, z4, color='blue', label='8:2')
    plt.plot(x5, z5, color='orange', label='9:1')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Image')
    plt.legend()
    plt.savefig(f"./accuracy_test.png")
    plt.show()
