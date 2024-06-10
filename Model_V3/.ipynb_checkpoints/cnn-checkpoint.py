import os
import random
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim

from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

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
            print('[%d, %5d] Loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 10))
            running_loss = 0.0

    x.append(epoch + 1)
    y.append(running_loss / len(train_loader))


def test(z, precisions, recalls):
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    z.append(100 * correct / total)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro')
    precisions.append(precision)
    recalls.append(recall)
    print('Accuracy on test set: %d %% ' % (100 * correct / total))
    print(f'Precision on test set: {precision:.4f}, Recall on test set: {recall:.4f}')


def test_for_pr_curve(y_true, y_scores):
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())


if __name__ == '__main__':
    times = 1000
    model = CNN()
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    sample = 0.5
    train_loader, test_loader = load_data(sample)
    x1 = []  # 轮次
    y1 = []  # 损失
    z1 = []  # 准确率
    p1 = []  # 精确率
    r1 = []  # 召回率
    for epoch in range(times):
        train(epoch, x1, y1)
        test(z1, p1, r1)
    # ==================================================================================
    y_true1 = []
    y_scores1 = []
    test_for_pr_curve(y_true1, y_scores1)
    y_true1 = label_binarize(y_true1, classes=np.arange(11))
    precision1 = dict()
    recall1 = dict()
    average_precision1 = dict()
    y_ture1 = np.array(y_true1)
    y_scores1 = np.array(y_scores1)
    for i in range(11):
        precision1[i], recall1[i], _ = precision_recall_curve(y_true1[:, i], y_scores1[:, i])
        average_precision1[i] = average_precision_score(y_true1[:, i], y_scores1[:, i])
    plt.figure(figsize=(10, 6))
    for i in range(11):
        plt.plot(recall1[i], precision1[i], lw=2, label=f'class {i} (area = {average_precision1[i]:0.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('5:5 Precision-Recall Curve')
    plt.legend(loc="best")
    plt.savefig(f"./5:5-Precision-Recall-Curve.png")
    plt.show()
    plt.close()
    # ==================================================================================
    sma11 = simple_moving_average(y1, 10)
    sma12 = simple_moving_average(z1, 10)
    sma13 = simple_moving_average(p1, 10)
    sma14 = simple_moving_average(r1, 10)
    ema11 = exponential_moving_average(y1, 0.1)
    ema12 = exponential_moving_average(z1, 0.1)
    ema13 = exponential_moving_average(p1, 0.1)
    ema14 = exponential_moving_average(r1, 0.1)
    gaussian11 = apply_gaussian_filter(y1, sigma=5)
    gaussian12 = apply_gaussian_filter(z1, sigma=5)
    gaussian13 = apply_gaussian_filter(p1, sigma=5)
    gaussian14 = apply_gaussian_filter(r1, sigma=5)
    median11 = apply_median_filter(y1, kernel_size=9)
    median12 = apply_median_filter(z1, kernel_size=9)
    median13 = apply_median_filter(p1, kernel_size=9)
    median14 = apply_median_filter(r1, kernel_size=9)
    lowess11 = apply_lowess(y1, x1, frac=0.1)
    lowess12 = apply_lowess(z1, x1, frac=0.1)
    lowess13 = apply_lowess(p1, x1, frac=0.1)
    lowess14 = apply_lowess(r1, x1, frac=0.1)
    print('5:5 finished-----------------------')
    # ------------------------------------------------------------------------------------------

    model = CNN()
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    sample = 0.6
    train_loader, test_loader = load_data(sample)
    x2 = []
    y2 = []
    z2 = []
    p2 = []
    r2 = []
    for epoch in range(times):
        train(epoch, x2, y2)
        test(z2, p2, r2)
    # ==================================================================================
    y_true2 = []
    y_scores2 = []
    test_for_pr_curve(y_true2, y_scores2)
    y_true2 = label_binarize(y_true2, classes=np.arange(11))
    precision2 = dict()
    recall2 = dict()
    average_precision2 = dict()
    y_ture2 = np.array(y_true2)
    y_scores2 = np.array(y_scores2)
    for i in range(11):
        precision2[i], recall2[i], _ = precision_recall_curve(y_true2[:, i], y_scores2[:, i])
        average_precision2[i] = average_precision_score(y_true2[:, i], y_scores2[:, i])
    plt.figure(figsize=(10, 6))
    for i in range(11):
        plt.plot(recall2[i], precision2[i], lw=2, label=f'class {i} (area = {average_precision2[i]:0.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('6:4 Precision-Recall Curve')
    plt.legend(loc="best")
    plt.savefig(f"./6:4-Precision-Recall-Curve.png")
    plt.show()
    plt.close()
    # ==================================================================================
    sma21 = simple_moving_average(y2, 10)
    sma22 = simple_moving_average(z2, 10)
    sma23 = simple_moving_average(p2, 10)
    sma24 = simple_moving_average(r2, 10)
    ema21 = exponential_moving_average(y2, 0.1)
    ema22 = exponential_moving_average(z2, 0.1)
    ema23 = exponential_moving_average(p2, 0.1)
    ema24 = exponential_moving_average(r2, 0.1)
    gaussian21 = apply_gaussian_filter(y2, sigma=5)
    gaussian22 = apply_gaussian_filter(z2, sigma=5)
    gaussian23 = apply_gaussian_filter(p2, sigma=5)
    gaussian24 = apply_gaussian_filter(r2, sigma=5)
    median21 = apply_median_filter(y2, kernel_size=9)
    median22 = apply_median_filter(z2, kernel_size=9)
    median23 = apply_median_filter(p2, kernel_size=9)
    median24 = apply_median_filter(r2, kernel_size=9)
    lowess21 = apply_lowess(y2, x2, frac=0.1)
    lowess22 = apply_lowess(z2, x2, frac=0.1)
    lowess23 = apply_lowess(p2, x2, frac=0.1)
    lowess24 = apply_lowess(r2, x2, frac=0.1)
    print('6:4 finished-----------------------')
    # ------------------------------------------------------------------------------------------

    model = CNN()
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    sample = 0.7
    train_loader, test_loader = load_data(sample)
    x3 = []
    y3 = []
    z3 = []
    p3 = []
    r3 = []
    for epoch in range(times):
        train(epoch, x3, y3)
        test(z3, p3, r3)
    # ==================================================================================
    y_true3 = []
    y_scores3 = []
    test_for_pr_curve(y_true3, y_scores3)
    y_true3 = label_binarize(y_true3, classes=np.arange(11))
    precision3 = dict()
    recall3 = dict()
    average_precision3 = dict()
    y_ture3 = np.array(y_true3)
    y_scores3 = np.array(y_scores3)
    for i in range(11):
        precision3[i], recall3[i], _ = precision_recall_curve(y_true3[:, i], y_scores3[:, i])
        average_precision3[i] = average_precision_score(y_true3[:, i], y_scores3[:, i])
    plt.figure(figsize=(10, 6))
    for i in range(11):
        plt.plot(recall3[i], precision3[i], lw=2, label=f'class {i} (area = {average_precision3[i]:0.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('7:3 Precision-Recall Curve')
    plt.legend(loc="best")
    plt.savefig(f"./7:3-Precision-Recall-Curve.png")
    plt.show()
    plt.close()
    # ==================================================================================
    sma31 = simple_moving_average(y3, 10)
    sma32 = simple_moving_average(z3, 10)
    sma33 = simple_moving_average(p3, 10)
    sma34 = simple_moving_average(r3, 10)
    ema31 = exponential_moving_average(y3, 0.1)
    ema32 = exponential_moving_average(z3, 0.1)
    ema33 = exponential_moving_average(p3, 0.1)
    ema34 = exponential_moving_average(r3, 0.1)
    gaussian31 = apply_gaussian_filter(y3, sigma=5)
    gaussian32 = apply_gaussian_filter(z3, sigma=5)
    gaussian33 = apply_gaussian_filter(p3, sigma=5)
    gaussian34 = apply_gaussian_filter(r3, sigma=5)
    median31 = apply_median_filter(y3, kernel_size=9)
    median32 = apply_median_filter(z3, kernel_size=9)
    median33 = apply_median_filter(p3, kernel_size=9)
    median34 = apply_median_filter(r3, kernel_size=9)
    lowess31 = apply_lowess(y3, x3, frac=0.1)
    lowess32 = apply_lowess(z3, x3, frac=0.1)
    lowess33 = apply_lowess(p3, x3, frac=0.1)
    lowess34 = apply_lowess(r3, x3, frac=0.1)
    print('7:3 finished-----------------------')
    # ------------------------------------------------------------------------------------------

    model = CNN()
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    sample = 0.8
    train_loader, test_loader = load_data(sample)
    x4 = []
    y4 = []
    z4 = []
    p4 = []
    r4 = []
    for epoch in range(times):
        train(epoch, x4, y4)
        test(z4, p4, r4)
    # ==================================================================================
    y_true4 = []
    y_scores4 = []
    test_for_pr_curve(y_true4, y_scores4)
    y_true4 = label_binarize(y_true4, classes=np.arange(11))
    precision4 = dict()
    recall4 = dict()
    average_precision4 = dict()
    y_ture4 = np.array(y_true4)
    y_scores4 = np.array(y_scores4)
    for i in range(11):
        precision4[i], recall4[i], _ = precision_recall_curve(y_true4[:, i], y_scores4[:, i])
        average_precision4[i] = average_precision_score(y_true4[:, i], y_scores4[:, i])
    plt.figure(figsize=(10, 6))
    for i in range(11):
        plt.plot(recall4[i], precision4[i], lw=2, label=f'class {i} (area = {average_precision4[i]:0.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('8:2 Precision-Recall Curve')
    plt.legend(loc="best")
    plt.savefig(f"./8:2-Precision-Recall-Curve.png")
    plt.show()
    plt.close()
    # ==================================================================================
    sma41 = simple_moving_average(y4, 10)
    sma42 = simple_moving_average(z4, 10)
    sma43 = simple_moving_average(p4, 10)
    sma44 = simple_moving_average(r4, 10)
    ema41 = exponential_moving_average(y4, 0.1)
    ema42 = exponential_moving_average(z4, 0.1)
    ema43 = exponential_moving_average(p4, 0.1)
    ema44 = exponential_moving_average(r4, 0.1)
    gaussian41 = apply_gaussian_filter(y4, sigma=5)
    gaussian42 = apply_gaussian_filter(z4, sigma=5)
    gaussian43 = apply_gaussian_filter(p4, sigma=5)
    gaussian44 = apply_gaussian_filter(r4, sigma=5)
    median41 = apply_median_filter(y4, kernel_size=9)
    median42 = apply_median_filter(z4, kernel_size=9)
    median43 = apply_median_filter(p4, kernel_size=9)
    median44 = apply_median_filter(r4, kernel_size=9)
    lowess41 = apply_lowess(y4, x4, frac=0.1)
    lowess42 = apply_lowess(z4, x4, frac=0.1)
    lowess43 = apply_lowess(p4, x4, frac=0.1)
    lowess44 = apply_lowess(r4, x4, frac=0.1)
    print('8:2 finished-----------------------')
    # ------------------------------------------------------------------------------------------

    model = CNN()
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    sample = 0.9
    train_loader, test_loader = load_data(sample)
    x5 = []
    y5 = []
    z5 = []
    p5 = []
    r5 = []
    for epoch in range(times):
        train(epoch, x5, y5)
        test(z5, p5, r5)
    # ==================================================================================
    y_true5 = []
    y_scores5 = []
    test_for_pr_curve(y_true5, y_scores5)
    y_true5 = label_binarize(y_true5, classes=np.arange(11))
    precision5 = dict()
    recall5 = dict()
    average_precision5 = dict()
    y_ture5 = np.array(y_true5)
    y_scores5 = np.array(y_scores5)
    for i in range(11):
        precision5[i], recall5[i], _ = precision_recall_curve(y_true5[:, i], y_scores5[:, i])
        average_precision5[i] = average_precision_score(y_true5[:, i], y_scores5[:, i])
    plt.figure(figsize=(10, 6))
    for i in range(11):
        plt.plot(recall5[i], precision5[i], lw=2, label=f'class {i} (area = {average_precision5[i]:0.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('9:1 Precision-Recall Curve')
    plt.legend(loc="best")
    plt.savefig(f"./9:1-Precision-Recall-Curve.png")
    plt.show()
    plt.close()
    # ==================================================================================
    sma51 = simple_moving_average(y5, 10)
    sma52 = simple_moving_average(z5, 10)
    sma53 = simple_moving_average(p5, 10)
    sma54 = simple_moving_average(r5, 10)
    ema51 = exponential_moving_average(y5, 0.1)
    ema52 = exponential_moving_average(z5, 0.1)
    ema53 = exponential_moving_average(p5, 0.1)
    ema54 = exponential_moving_average(r5, 0.1)
    gaussian51 = apply_gaussian_filter(y5, sigma=5)
    gaussian52 = apply_gaussian_filter(z5, sigma=5)
    gaussian53 = apply_gaussian_filter(p5, sigma=5)
    gaussian54 = apply_gaussian_filter(r5, sigma=5)
    median51 = apply_median_filter(y5, kernel_size=9)
    median52 = apply_median_filter(z5, kernel_size=9)
    median53 = apply_median_filter(p5, kernel_size=9)
    median54 = apply_median_filter(r5, kernel_size=9)
    lowess51 = apply_lowess(y5, x5, frac=0.1)
    lowess52 = apply_lowess(z5, x5, frac=0.1)
    lowess53 = apply_lowess(p5, x5, frac=0.1)
    lowess54 = apply_lowess(r5, x5, frac=0.1)
    print('9:1 finished-----------------------')
    # ------------------------------------------------------------------------------------------

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

    # 绘制精确率图像
    plot_result(x1, sma13, sma23, sma33, sma43, sma53,
                'SMA Precision Image', 'Precision', './SMA_Precision.png')
    plot_result(x1, ema13, ema23, ema33, ema43, ema53,
                'EMA Precision Image', 'Precision', './EMA_Precision.png')
    plot_result(x1, gaussian13, gaussian23, gaussian33, gaussian43, gaussian53,
                'GAU Precision Image', 'Precision', './GAU_Precision.png')
    plot_result(x1, median13, median23, median33, median43, median53,
                'MED Precision Image', 'Precision', './MED_Precision.png')
    plot_result(x1, lowess13, lowess23, lowess33, lowess43, lowess53,
                'LOW Precision Image', 'Precision', './LOW_Precision.png')

    # 绘制召回率图像
    plot_result(x1, sma14, sma24, sma34, sma44, sma54,
                'SMA Recall Image', 'Recall', './SMA_Recall.png')
    plot_result(x1, ema14, ema24, ema34, ema44, ema54,
                'EMA Recall Image', 'Recall', './EMA_Recall.png')
    plot_result(x1, gaussian14, gaussian24, gaussian34, gaussian44, gaussian54,
                'GAU Recall Image', 'Recall', './GAU_Recall.png')
    plot_result(x1, median14, median24, median34, median44, median54,
                'MED Recall Image', 'Recall', './MED_Recall.png')
    plot_result(x1, lowess14, lowess24, lowess34, lowess44, lowess54,
                'LOW Recall Image', 'Recall', './LOW_Recall.png')
