from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# 使用PIL包来处理图片，加载图片并转为RGB格式
def my_loader(path):
    return Image.open(path).convert('RGB')


# 在data列表中存入图片路径和标签
def init_process(path, lens):
    data = []
    label = find_label(path)
    for i in range(lens[0], lens[1]):
        data.append([path % i, label])

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
        return 'A'
    elif path_name[index] == 'B':
        return 'B'
    elif path_name[index] == 'C':
        return 'C'
    elif path_name[index] == 'D':
        return 'D'
    elif path_name[index] == 'E':
        return 'E'
    elif path_name[index] == 'F':
        return 'F'
    elif path_name[index] == 'G':
        return 'G'
    elif path_name[index] == 'H':
        return 'H'
    elif path_name[index] == 'I':
        return 'I'
    elif path_name[index] == 'J':
        return 'J'
    else:
        return 'K'


def load_data():
    print('Data processing...')

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

    # 训练集的数据（路径+标签）的列表
    train_data = (init_process('../Dataset/train/A (%d).bmp', [1, 31])
                  + init_process('../Dataset/train/B (%d).bmp', [1, 25])
                  + init_process('../Dataset/train/C (%d).bmp', [1, 34])
                  + init_process('../Dataset/train/D (%d).bmp', [1, 25])
                  + init_process('../Dataset/train/E (%d).bmp', [1, 31])
                  + init_process('../Dataset/train/F (%d).bmp', [1, 28])
                  + init_process('../Dataset/train/G (%d).bmp', [1, 30])
                  + init_process('../Dataset/train/H (%d).bmp', [1, 29])
                  + init_process('../Dataset/train/I (%d).bmp', [1, 23])
                  + init_process('../Dataset/train/J (%d).bmp', [1, 21])
                  + init_process('../Dataset/train/K (%d).bmp', [1, 25])
                  )
    print('得到的训练集共有 ' + len(train_data).__str__() + ' 条')

    # 测试集的数据（路径+标签）的列表
    test_data = (init_process('../Dataset/test/A (%d).bmp', [1, 9])
                 + init_process('../Dataset/test/B (%d).bmp', [1, 8])
                 + init_process('../Dataset/test/C (%d).bmp', [1, 10])
                 + init_process('../Dataset/test/D (%d).bmp', [1, 8])
                 + init_process('../Dataset/test/E (%d).bmp', [1, 9])
                 + init_process('../Dataset/test/F (%d).bmp', [1, 8])
                 + init_process('../Dataset/test/G (%d).bmp', [1, 9])
                 + init_process('../Dataset/test/H (%d).bmp', [1, 8])
                 + init_process('../Dataset/test/I (%d).bmp', [1, 7])
                 + init_process('../Dataset/test/J (%d).bmp', [1, 6])
                 + init_process('../Dataset/test/K (%d).bmp', [1, 7])
                 )
    print('得到的测试集共有 ' + len(test_data).__str__() + ' 条')

    train_dataset = MyDataset(train_data, transform=transform, loader=my_loader)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32, num_workers=0)
    test_dataset = MyDataset(test_data, transform=transform, loader=my_loader)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32, num_workers=0)

    print('End of data processing...')

    return train_loader, test_loader


# 单元测试
if __name__ == '__main__':
    a, b = load_data()
    print(a)
    print(b)
