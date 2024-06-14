"""
用HOG提取彩色图像（RGB）特征并使用SVM解决多分类问题
"""

import os
import glob
import numpy as np
from skimage.feature import hog
from PIL import Image
import joblib
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
import shutil
import sys

# 定义路径和参数
# 训练集图片的位置
train_image_path = '../Dataset/train91'
# 测试集图片的位置
test_image_path = '../Dataset/test91'

# 训练集标签的位置
train_label_path = os.path.join('../Dataset/train91', 'train91.txt')
# 测试集标签的位置
test_label_path = os.path.join('../Dataset/test91', 'test91.txt')

# 图片宽高参数，若已预处理好图像宽高可直接设置成处理好的图像尺寸即可
# 如果没有进行预处理直接用你的图像尺寸就可以了
# 由于对图像预处理对最终模型预测的结果增益不算明显，故已将预处理程序删除
# 原预处理程序可在old/change_size.py中查看
image_height = 320
image_width = 240

"""
“55”“64”“73”“82”“91”指的是测试集与数据集数据量（图像数量）的比例
由于本项目的特征文件比较大（平均一个就有15m），每一个比例所用的特征文件不可能全部存下来,也难以上传到github;
同时一方面训练模型只需用到训练集的特征文件（文件较多占用内存较大），而模型预测只需用到测试集的特征文件（文件较少占用内存较小[大概...]）
另一方面SVM训练好的模型完全依赖于支持向量，而不是由数据的维度决定的。所以训练过一次，后续只要不改变数据集，不管怎么训练都会是相同的结果
因此每次提取完一次特征并跑完一次模型之后，可以把训练集的特征文件删除，只保留测试集的特征文件和对应的模型
"""

# 存放训练集图像特征的路径（初始没有对应的文件夹会在程序运行后自动生成）
train_feat_path = './features/train_feat91/'
# 存放测试集图像特征的路径（初始没有对应的文件夹会在程序运行后自动生成）
test_feat_path = './features/test_feat91/'
# 存放训练的模型
model_path = 'models/'


# 获取图像列表
def get_image_list(filePath, nameList):
    # 用于存储加载的图像
    img_list = []
    for name in nameList:
        temp = Image.open(os.path.join(filePath, name))
        # 调用 temp.copy() 创建图像的副本，并将其追加到 img_list 列表中。
        # copy() 方法确保原始图像在关闭后不会受到影响。
        img_list.append(temp.copy())
        # 调用 temp.close() 关闭图像文件。这是为了释放文件资源，因为已经有了图像的副本。
        temp.close()
    return img_list


# 对彩色图像的每个通道分别提取HOG特征
"""
图像被分成8×8的很多cell，提取HOG特征需要计算梯度直方图，而梯度直方图是在这些cell中计算出来的，固定使用8*8的cell来切分图像的原因是：
一个8×8的图片块包含了8×8×3=192个像素值。而这个图像块的每个像素点梯度信息包含梯度幅值和方向两个值，一共是8×8×2=128个值，
这128个值可以通过用包含9个bin的直方图表示成一个一维数组（包含9个值）。
这样做不仅可以使图像表示更紧凑，而且在一个图片块中计算直方图可以让这种表示方法对噪声有更强的稳健性。
单个像素的梯度信息可能包含噪声，而一个8×8的图片块中的直方图让这种表示方法对噪声更不敏感。
HOG特征起初是被用来检测行人的，8×8的cell在一张64×128的行人图片块中的足以捕捉感兴趣的特征（如人脸、头顶等）。
因此梯度直方图本质上是一个包含9个数字的向量（或数组），这9个数字分别对应0°、20°、40°、… 160°，就是将0°~180°进行分割
"""


def get_color_hog_features(image):
    hog_features = []
    # image.shape[2] 表示图像的第三个维度（颜色通道）的数量。例如，对于 RGB 图像，这个值通常为 3。
    for channel in range(image.shape[2]):
        # image[:, :, channel] 提取图像的单个颜色通道（二维数组）。
        fd = hog(image[:, :, channel],
                 # orientations表示设置 HOG 特征计算中的方向数量为 9，
                 # 这决定了每个像素点的梯度方向分桶数量。这个值通常在 9 到 18 之间，9和12较为常用
                 orientations=9,
                 # 这里的pixels_per_cell就是上述的cell，一般不作修改
                 # 设置每个单元（cell）中的像素数量为 8x8，这个参数控制分块时每个单元的大小。
                 pixels_per_cell=(8, 8),
                 # 设置每个块（block）中的单元数量为 2x2。这个参数控制在计算 HOG 特征时用于归一化的单元块大小。
                 cells_per_block=(2, 2),
                 # 设置块归一化方法为 L2-Hys。这是常用归一化方法之一，有助于提高特征的鲁棒性（稳健性）。
                 block_norm='L2-Hys',
                 # transform_sqrt 设置为 True，对输入图像进行平方根变换。这可以增强图像的对比度。
                 transform_sqrt=True,
                 # 设置 visualize 参数为 False，表示不需要返回 HOG 特征图像的可视化。
                 visualize=False)
        # 将计算得到的 HOG 特征向量 fd 追加到 hog_features 列表中。
        hog_features.append(fd)
    # 使用 np.ravel 将 hog_features 列表展平成一个一维数组，并返回该数组。
    return np.ravel(hog_features)


# 提取特征并保存
def get_feat(image_list, name_list, label_list, savePath):
    # 初始化图片计数器
    i = 0
    for image in image_list:
        try:
            # 确保图像是RGB图像
            # 将图像转换为 NumPy 数组，并检查图像是否为 RGB 格式（即第三个维度的大小是否为 3）。
            image = np.array(image)
            if image.shape[2] != 3:
                print('Image is not in RGB format:', name_list[i])
                print('图像不是RGB格式:', name_list[i])
                continue
        except Exception as e:
            print('Error processing image:', name_list[i], e)
            print('处理图像时出错:', name_list[i], e)
            continue

        # 使用 PIL 库将图像重新调整到指定的 image_width 和 image_height 大小，并将其转换回 NumPy 数组。
        image = np.array(Image.fromarray(image).resize((image_width, image_height)))

        # 提取HOG特征
        fd = get_color_hog_features(image)

        # 添加标签
        # 将标签值追加到特征向量的末尾，形成一个包含特征和标签的数组。
        fd = np.concatenate((fd, [label_list[i]]))
        # 生成特征文件的名称，格式为图像名称加上 .feat 后缀。
        fd_name = name_list[i] + '.feat'
        # 生成特征文件的完整路径。
        fd_path = os.path.join(savePath, fd_name)
        # 使用 joblib.dump 将特征数组保存到特征文件中。
        joblib.dump(fd, fd_path)
        # 处理下一张图像。
        i += 1
    print("Features are extracted and saved.")
    print("所有特征已被提取并保存。")


# 获得图片名称与对应的类别
def get_name_label(file_path):
    # name_list 用于存储图像名称，label_list 用于存储图像标签。
    name_list = []
    label_list = []
    with open(file_path) as f:
        for line in f.readlines():
            if len(line.strip()) >= 3:
                parts = line.split(' ')

                # 由于本项目的训练集和测试集命名都是“字母 （编号）”格式（有一个空格），
                # 在导出标签生成txt文件后格式就是“字母 （编号） 类别”格式（有两个空格）
                # 使用其他命名的数据集注意一下空格分隔的索引，可自行按需修改

                # 将行的前两个部分（图像名称）连接成一个字符串并添加到 name_list 中。
                name_list.append(parts[0] + ' ' + parts[1])
                # 将行的第三个部分（图像标签）去除首尾空白后添加到 label_list 中。
                label_list.append(parts[2].strip())

                if not str(label_list[-1]).isdigit():
                    print("Label must be a number. Found:", label_list[-1])
                    print("标签类别必须为数字。发现标签部分为:", label_list[-1])
                    # 终止程序
                    sys.exit(1)
    return name_list, label_list


# 提取数据集特征
def extra_feat():
    # 获取训练集和测试集的图形名称列表和图像标签列表
    train_name, train_label = get_name_label(train_label_path)
    test_name, test_label = get_name_label(test_label_path)

    # 获取训练集和测试集的图像列表
    train_image = get_image_list(train_image_path, train_name)
    test_image = get_image_list(test_image_path, test_name)

    # 提取特征
    print('\n提取训练集特征中...')
    get_feat(train_image, train_name, train_label, train_feat_path)
    print('\n提取测试集特征中...')
    get_feat(test_image, test_name, test_label, test_feat_path)


# 创建存放特征的文件夹
def mkdir():
    if not os.path.exists(train_feat_path):
        os.mkdir(train_feat_path)
    if not os.path.exists(test_feat_path):
        os.mkdir(test_feat_path)


# 训练和测试
def train_and_test():
    # features 用于存储训练特征，labels 用于存储训练标签。
    features = []
    labels = []

    # 使用 glob 模块查找训练特征文件夹中所有扩展名为 .feat 的文件，并遍历每个文件路径。
    for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
        # 使用 joblib.load 加载每个特征文件。
        data = joblib.load(feat_path)
        # 将特征数据（不包括最后一个元素）添加到 features 列表中，将标签数据（最后一个元素）添加到 labels 列表中。
        features.append(data[:-1])
        labels.append(data[-1])

    need_retrain_model = input('Do we need to retrain the model?\n是否需要重新训练模型？\n(y/n):')
    try:
        if need_retrain_model.lower() == 'y':
            # 创建一个核函数为线性的支持向量机分类器（SVM），并启用概率估计。然后使用 features 和 labels 训练分类器。
            # 本项目尝试使用过另一个分类器svm.LinearSVR()，但效果不如使用核函数的写法；其他核函数也可以尝试使用，但线性核依旧最佳
            clf = svm.SVC(kernel='linear', probability=True)
            print('\n模型训练中...')
            clf.fit(features, labels)
            # 如果模型存储路径不存在，则创建该路径。
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            joblib.dump(clf, os.path.join(model_path, 'rgb_model91'))
            print("Model saved.")
            print("训练模型已保存。\n")
        elif need_retrain_model.lower() == 'n':
            # 直接加载模型，不进行训练
            clf = joblib.load(model_path + 'rgb_model91')
    except Exception as e:
        print('Unable to determine whether to train the model:', need_retrain_model.lower(), e)
        print('无法知道是否训练模型:', need_retrain_model.lower(), e)

    # test_features 用于存储测试特征
    # test_labels 用于存储测试标签
    # result_list 用于存储测试结果。
    test_features = []
    test_labels = []
    result_list = []

    for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
        data_test = joblib.load(feat_path)

        # 将特征数据（不包括最后一个元素）添加到 test_features 列表中，将标签数据（最后一个元素）添加到 test_labels 列表中。
        test_features.append(data_test[:-1])
        test_labels.append(data_test[-1])
        # 将文件名（去掉 .feat 后缀）和标签添加到 result_list 中。
        result_list.append(os.path.basename(feat_path).replace('.feat', '') + ' ' + str(int(data_test[-1])) + '\n')

    # 将 test_features 和 test_labels 列表转换为 NumPy 数组。
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    # 使用训练好的分类器对测试特征进行预测，得到预测结果 predictions 和预测概率 prediction_probs。
    predictions = clf.predict(test_features)
    prediction_probs = clf.predict_proba(test_features)

    # 将 result_list 写入 result.txt 文件。
    with open('results/rgb_result91.txt', 'w') as f:
        f.writelines(result_list)

    # accuracy: 准确率，使用 accuracy_score 计算。
    # precision: 精确率，使用 precision_score 计算，采用宏平均（macro）。
    # recall: 召回率，使用 recall_score 计算，采用宏平均（macro）。
    # loss: 损失，使用 log_loss 计算。
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='macro')
    recall = recall_score(test_labels, predictions, average='macro')
    loss = log_loss(test_labels, prediction_probs)

    print(f'Accuracy: {100 * accuracy:.4f} %')
    print(f'Precision: {100 * precision:.4f} %')
    print(f'Recall: {100 * recall:.4f} %')
    print(f'Log Loss: {loss:.4f}')


if __name__ == '__main__':
    mkdir()
    need_extra_feat = input('Do you need to extract features again?\n是否需要重新获取特征？\n(y/n): ')
    if need_extra_feat.lower() == 'y':
        shutil.rmtree(train_feat_path)
        shutil.rmtree(test_feat_path)
        mkdir()
        extra_feat()
    train_and_test()
