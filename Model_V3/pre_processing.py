from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import sys


# 使用PIL包来处理图片，加载图片并转为RGB格式
def my_loader(path):
    return Image.open(path).convert('RGB')


# 在data列表中存入图片路径和标签
test = []


def init_process(path, lens, group, sample):
    data = []
    label = find_label(path)

    num = int(lens[1] * sample)
    my_list = list(range(lens[0], lens[1]))
    random_nums = random.sample(my_list, num)
    # print(my_list)
    # print("==================================================================================")
    # print(random_nums)
    # print("==================================================================================")

    for i in range(lens[0], lens[1]):
        # 表示没被用来训练的这部分之后会被用来测试
        if i not in random_nums:
            test[group].append(i)
        else:
            data.append([path % i, label])
    # print(data)
    # print("==================================================================================")

    return data


def init_test(path, group):
    # 这里的lens没用了其实
    data = []
    label = find_label(path)
    for i in test[group]:
        data.append([path % i, label])
    # print(data)
    # print("==================================================================================")
    return data


# 重写pytorch中的Dataset
class MyDataset(Dataset):
    def __init__(self, data, transform, loader):
        self.data = data
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


# 数据集需要的标签函数，通过标签来辨别图片类别
def find_label(path_name):
    index = 0
    for i in range(len(path_name)):
        if path_name[i] == ' ':
            index = i - 1
            break

    if path_name[index] == 'A':
        return 0
    elif path_name[index] == 'B':
        return 1
    elif path_name[index] == 'C':
        return 2
    elif path_name[index] == 'D':
        return 3
    elif path_name[index] == 'E':
        return 4
    elif path_name[index] == 'F':
        return 5
    elif path_name[index] == 'G':
        return 6
    elif path_name[index] == 'H':
        return 7
    elif path_name[index] == 'I':
        return 8
    elif path_name[index] == 'J':
        return 9
    else:
        return 10


def load_data(sample):
    global test
    test = [[] for i in range(12)]

    print('Data processing...')
    with open('cnn_process.txt', 'a', encoding='utf-8') as f:
        # 将标准输出重定向到文件
        sys.stdout = f
        # 在此处写入你想打印的内容
        print('Data processing...')
        # 重置标准输出到控制台
        sys.stdout = sys.__stdout__

    # 图像变换操作
    transform = transforms.Compose([
        # 随机水平和垂直翻转，进行数据增强
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        # 转为（C， H， W）的张量
        transforms.ToTensor(),
        # 归一化处理，对RGB三个通道进行相同的标准化处理
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    print(sample)
    with open('cnn_process.txt', 'a', encoding='utf-8') as f:
        # 将标准输出重定向到文件
        sys.stdout = f
        # 在此处写入你想打印的内容
        print(sample)
        # 重置标准输出到控制台
        sys.stdout = sys.__stdout__

    # 训练集的数据（路径+标签）的列表
    train_data = (init_process('../Dataset/datas/A (%d).bmp', [1, 39], 1, sample)
                  + init_process('../Dataset/datas/B (%d).bmp', [1, 32], 2, sample)
                  + init_process('../Dataset/datas/C (%d).bmp', [1, 43], 3, sample)
                  + init_process('../Dataset/datas/D (%d).bmp', [1, 32], 4, sample)
                  + init_process('../Dataset/datas/E (%d).bmp', [1, 39], 5, sample)
                  + init_process('../Dataset/datas/F (%d).bmp', [1, 35], 6, sample)
                  + init_process('../Dataset/datas/G (%d).bmp', [1, 38], 7, sample)
                  + init_process('../Dataset/datas/H (%d).bmp', [1, 36], 8, sample)
                  + init_process('../Dataset/datas/I (%d).bmp', [1, 29], 9, sample)
                  + init_process('../Dataset/datas/J (%d).bmp', [1, 26], 10, sample)
                  + init_process('../Dataset/datas/K (%d).bmp', [1, 31], 11, sample)
                  )
    print('得到的训练集共有 ' + len(train_data).__str__() + ' 条')
    with open('cnn_process.txt', 'a', encoding='utf-8') as f:
        # 将标准输出重定向到文件
        sys.stdout = f
        # 在此处写入你想打印的内容
        print('得到的训练集共有 ' + len(train_data).__str__() + ' 条')
        # 重置标准输出到控制台
        sys.stdout = sys.__stdout__
    # print(train_data)

    # 测试集的数据（路径+标签）的列表
    test_data = (init_test('../Dataset/datas/A (%d).bmp', 1)
                 + init_test('../Dataset/datas/B (%d).bmp', 2)
                 + init_test('../Dataset/datas/C (%d).bmp', 3)
                 + init_test('../Dataset/datas/D (%d).bmp', 4)
                 + init_test('../Dataset/datas/E (%d).bmp', 5)
                 + init_test('../Dataset/datas/F (%d).bmp', 6)
                 + init_test('../Dataset/datas/G (%d).bmp', 7)
                 + init_test('../Dataset/datas/H (%d).bmp', 8)
                 + init_test('../Dataset/datas/I (%d).bmp', 9)
                 + init_test('../Dataset/datas/J (%d).bmp', 10)
                 + init_test('../Dataset/datas/K (%d).bmp', 11)
                 )
    print('得到的测试集共有 ' + len(test_data).__str__() + ' 条')
    with open('cnn_process.txt', 'a', encoding='utf-8') as f:
        # 将标准输出重定向到文件
        sys.stdout = f
        # 在此处写入你想打印的内容
        print('得到的测试集共有 ' + len(test_data).__str__() + ' 条')
        # 重置标准输出到控制台
        sys.stdout = sys.__stdout__

    # print(test_data)

    train_dataset = MyDataset(train_data, transform=transform, loader=my_loader)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=8, num_workers=0)
    test_dataset = MyDataset(test_data, transform=transform, loader=my_loader)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=8, num_workers=0)

    print('End of data processing...')
    with open('cnn_process.txt', 'a', encoding='utf-8') as f:
        # 将标准输出重定向到文件
        sys.stdout = f
        # 在此处写入你想打印的内容
        print('End of data processing...')
        # 重置标准输出到控制台
        sys.stdout = sys.__stdout__
    return train_loader, test_loader

# 单元测试
# if __name__ == '__main__':
#     a, b = load_data(0.5)
#     c, d = load_data(0.6)
#     e, f = load_data(0.7)
#     g, h = load_data(0.8)
#     i, j = load_data(0.9)
