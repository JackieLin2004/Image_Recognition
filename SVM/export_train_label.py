"""
此文件用于导出训练集所需的标签txt文件
仅需修改测试集图片所在的文件夹image_folder和文本文件路径output_txt即可
"""

import os


def extract_image_names_and_labels(image_folder, output_txt):
    """
    从文件夹中提取图片名称及类别，并保存到文本文件中。
    :param image_folder: 图片文件夹路径
    :param output_txt: 输出文本文件路径
    """
    # 确保输出文件路径的目录存在
    output_dir = os.path.dirname(output_txt)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开文件（如果文件不存在会自动创建）
    with open(output_txt, 'w') as f:
        # 遍历文件夹中的所有文件
        for file_name in os.listdir(image_folder):
            # 检查文件是否为图像文件（根据文件扩展名）
            if file_name.endswith('.bmp'):
                # 提取类别，即文件名中的第一个字母
                if file_name[0] == 'A':
                    label = 1
                elif file_name[0] == 'B':
                    label = 2
                elif file_name[0] == 'C':
                    label = 3
                elif file_name[0] == 'D':
                    label = 4
                elif file_name[0] == 'E':
                    label = 5
                elif file_name[0] == 'F':
                    label = 6
                elif file_name[0] == 'G':
                    label = 7
                elif file_name[0] == 'H':
                    label = 8
                elif file_name[0] == 'I':
                    label = 9
                elif file_name[0] == 'J':
                    label = 10
                else:
                    label = 11
                # 写入文件，格式为：文件名 类别
                f.write(f"{file_name} {label}\n")
        print(f"文件已保存到 {output_txt}")


if __name__ == '__main__':
    image_folder1 = '../Dataset/train55'  # 修改为你的图像文件夹路径
    output_txt1 = '../Dataset/train55/train55.txt'  # 输出的文本文件路径
    image_folder2 = '../Dataset/train64'  # 修改为你的图像文件夹路径
    output_txt2 = '../Dataset/train64/train64.txt'  # 输出的文本文件路径
    image_folder3 = '../Dataset/train73'  # 修改为你的图像文件夹路径
    output_txt3 = '../Dataset/train73/train73.txt'  # 输出的文本文件路径
    image_folder4 = '../Dataset/train82'  # 修改为你的图像文件夹路径
    output_txt4 = '../Dataset/train82/train82.txt'  # 输出的文本文件路径
    image_folder5 = '../Dataset/train91'  # 修改为你的图像文件夹路径
    output_txt5 = '../Dataset/train91/train91.txt'  # 输出的文本文件路径
    extract_image_names_and_labels(image_folder1, output_txt1)
    extract_image_names_and_labels(image_folder2, output_txt2)
    extract_image_names_and_labels(image_folder3, output_txt3)
    extract_image_names_and_labels(image_folder4, output_txt4)
    extract_image_names_and_labels(image_folder5, output_txt5)
