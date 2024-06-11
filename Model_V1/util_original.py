import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
import statsmodels.api as sm


def generate_data():
    """生成模拟的损失值数据和对应的epoch列表"""
    np.random.seed(0)
    x = np.arange(1, 1001)  # Epochs from 1 to 1000
    y = np.random.normal(0.5, 0.1, size=1000)  # 随机生成的损失值，均值为0.5，标准差为0.1
    return x, y


def simple_moving_average(data, window_size):
    """计算简单移动平均（SMA）"""
    """window_size: 移动窗口的大小，值越大，平滑效果越好。"""
    weights = np.ones(window_size) / window_size
    sma = np.convolve(data, weights, mode='valid')
    return sma


def exponential_moving_average(data, alpha):
    """计算指数移动平均（EMA）"""
    """ alpha: 平滑因子，值越小平滑效果越强，反应速度越慢。"""
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return ema


def apply_gaussian_filter(data, sigma):
    """应用高斯滤波平滑处理"""
    """高斯核的标准差，值越大，平滑效果越好。"""
    return gaussian_filter1d(data, sigma=sigma)


def apply_median_filter(data, kernel_size):
    """应用中位数滤波平滑处理"""
    """kernel_size: 滤波器的大小，必须为奇数，值越大，平滑效果越好。"""
    return medfilt(data, kernel_size=kernel_size)


def apply_lowess(data, x, frac):
    """应用局部加权回归平滑（LOWESS）处理"""
    """frac: 用于估计每个点回归权重的数据点比例。值越大，平滑效果越好。"""
    return sm.nonparametric.lowess(data, x, frac=frac)[:, 1]


def plot_result(x, y, smooth_y, title):
    """绘制原始数据与平滑数据的对比图"""
    plt.figure(figsize=(10, 6))
    # 原始图
    plt.plot(x, y, label='Original', alpha=0.5)
    # 平滑处理后的图
    plt.plot(x[len(x) - len(smooth_y):] if len(smooth_y) < len(x) else x, smooth_y, label='Smoothed', linewidth=2)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    x, y = generate_data()
    sma = simple_moving_average(y, 10)
    ema = exponential_moving_average(y, 0.1)
    gaussian = apply_gaussian_filter(y, sigma=5)
    median = apply_median_filter(y, kernel_size=9)
    lowess = apply_lowess(y, x, frac=0.1)

    # 单独绘制每个平滑方法的结果
    plot_result(x, y, sma, '简单移动平均 (SMA)')
    plot_result(x, y, ema, '指数移动平均 (EMA)')
    plot_result(x, y, gaussian, '高斯平滑')
    plot_result(x, y, median, '中位数滤波')
    plot_result(x, y, lowess, '局部加权回归平滑 (LOWESS)')
