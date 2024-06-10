import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from pre_processing import load_data
import torch.nn.functional as F

from util import simple_moving_average, exponential_moving_average, apply_gaussian_filter, apply_lowess, \
    apply_median_filter, plot_result

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


if __name__ == '__main__':
    times = 1000
    model = CNN()
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    sample = 0.5
    train_loader, test_loader = load_data(sample)
    x1 = []
    y1 = []
    z1 = []
    for epoch in range(times):
        train(epoch, x1, y1)
        test(z1)
    sma11 = simple_moving_average(y1, 10)
    sma12 = simple_moving_average(z1, 10)
    ema11 = exponential_moving_average(y1, 0.1)
    ema12 = exponential_moving_average(z1, 0.1)
    gaussian11 = apply_gaussian_filter(y1, sigma=5)
    gaussian12 = apply_gaussian_filter(z1, sigma=5)
    median11 = apply_median_filter(y1, kernel_size=9)
    median12 = apply_median_filter(z1, kernel_size=9)
    lowess11 = apply_lowess(y1, x1, frac=0.1)
    lowess12 = apply_lowess(z1, x1, frac=0.1)
    print('5:5 finished-----------------------')

    model = CNN()
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    sample = 0.6
    train_loader, test_loader = load_data(sample)
    x2 = []
    y2 = []
    z2 = []
    for epoch in range(times):
        train(epoch, x2, y2)
        test(z2)
    sma21 = simple_moving_average(y2, 10)
    sma22 = simple_moving_average(z2, 10)
    ema21 = exponential_moving_average(y2, 0.1)
    ema22 = exponential_moving_average(z2, 0.1)
    gaussian21 = apply_gaussian_filter(y2, sigma=5)
    gaussian22 = apply_gaussian_filter(z2, sigma=5)
    median21 = apply_median_filter(y2, kernel_size=9)
    median22 = apply_median_filter(z2, kernel_size=9)
    lowess21 = apply_lowess(y2, x2, frac=0.1)
    lowess22 = apply_lowess(z2, x2, frac=0.1)
    print('6:4 finished-----------------------')

    model = CNN()
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    sample = 0.7
    train_loader, test_loader = load_data(sample)
    x3 = []
    y3 = []
    z3 = []
    for epoch in range(times):
        train(epoch, x3, y3)
        test(z3)
    sma31 = simple_moving_average(y3, 10)
    sma32 = simple_moving_average(z3, 10)
    ema31 = exponential_moving_average(y3, 0.1)
    ema32 = exponential_moving_average(z3, 0.1)
    gaussian31 = apply_gaussian_filter(y3, sigma=5)
    gaussian32 = apply_gaussian_filter(z3, sigma=5)
    median31 = apply_median_filter(y3, kernel_size=9)
    median32 = apply_median_filter(z3, kernel_size=9)
    lowess31 = apply_lowess(y3, x3, frac=0.1)
    lowess32 = apply_lowess(z3, x3, frac=0.1)
    print('7:3 finished-----------------------')

    model = CNN()
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    sample = 0.8
    train_loader, test_loader = load_data(sample)
    x4 = []
    y4 = []
    z4 = []
    for epoch in range(times):
        train(epoch, x4, y4)
        test(z4)
    sma41 = simple_moving_average(y4, 10)
    sma42 = simple_moving_average(z4, 10)
    ema41 = exponential_moving_average(y4, 0.1)
    ema42 = exponential_moving_average(z4, 0.1)
    gaussian41 = apply_gaussian_filter(y4, sigma=5)
    gaussian42 = apply_gaussian_filter(z4, sigma=5)
    median41 = apply_median_filter(y4, kernel_size=9)
    median42 = apply_median_filter(z4, kernel_size=9)
    lowess41 = apply_lowess(y4, x4, frac=0.1)
    lowess42 = apply_lowess(z4, x4, frac=0.1)
    print('8:2 finished-----------------------')

    model = CNN()
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    sample = 0.9
    train_loader, test_loader = load_data(sample)
    x5 = []
    y5 = []
    z5 = []
    for epoch in range(times):
        train(epoch, x5, y5)
        test(z5)
    sma51 = simple_moving_average(y5, 10)
    sma52 = simple_moving_average(z5, 10)
    ema51 = exponential_moving_average(y5, 0.1)
    ema52 = exponential_moving_average(z5, 0.1)
    gaussian51 = apply_gaussian_filter(y5, sigma=5)
    gaussian52 = apply_gaussian_filter(z5, sigma=5)
    median51 = apply_median_filter(y5, kernel_size=9)
    median52 = apply_median_filter(z5, kernel_size=9)
    lowess51 = apply_lowess(y5, x5, frac=0.1)
    lowess52 = apply_lowess(z5, x5, frac=0.1)
    print('9:1 finished-----------------------')

    # 绘制损失曲线
    plot_result(x1, sma11, sma21, sma31, sma41, sma51,
                'SMA Loss Image', 'Loss', './SMA_Loss.png')
    plot_result(x1, ema11, ema21, ema31, ema41, ema51,
                'EMA Loss Image', 'Loss', './EMA_Loss.png')
    plot_result(x1, gaussian11, gaussian21, gaussian31, gaussian41, gaussian51,
                'GAU Loss Image', 'Loss', './GAU_Loss.png')
    plot_result(x1, median11, median21, median31, median41, median51,
                'MED Loss Image', 'Loss', './MED_Loss.png')
    plot_result(x1, lowess11, lowess21, lowess31, lowess41, lowess51,
                'LOW Loss Image', 'Loss', './LOW_Loss.png')

    # 绘制准确率曲线
    plot_result(x1, sma12, sma22, sma32, sma42, sma52,
                'SMA Accuracy Image', 'Accuracy', './SMA_Accuracy.png')
    plot_result(x1, ema12, ema22, ema32, ema42, ema52,
                'EMA Accuracy Image', 'Accuracy', './EMA_Accuracy.png')
    plot_result(x1, gaussian12, gaussian22, gaussian32, gaussian42, gaussian52,
                'GAU Accuracy Image', 'Accuracy', './GAU_Accuracy.png')
    plot_result(x1, median12, median22, median32, median42, median52,
                'MED Accuracy Image', 'Accuracy', './MED_Accuracy.png')
    plot_result(x1, lowess12, lowess22, lowess32, lowess42, lowess52,
                'LOW Accuracy Image', 'Accuracy', './LOW_Accuracy.png')
