# Image_Recognition

## Python语言设计与实践 第三次项目

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

### 需求

利用老师发的感应电机红外图像数据集，通过卷积神经网络实现图像的分类，同时运行出效果图

### 环境

`python` 版本:3.11

### 快速开始

```
pip install -r requirements.txt

python cnn.py
```

### 一些说明

- 在机器学习中有两种主要的方法，一种是**监督学习**，一种是**无监督学习**


- 本次实验中我们分别采用这两种方法来实现图像的多分类问题，无监督学习方面采用 `K-Means` 聚类分析，而监督学习采用卷积神经网络 `CNN` 来进行

### K-Means 实现思路

- 将图片转换成张量，放到高纬度坐标系里求笛卡尔距离，不断更新每个簇（`cluster`）的中心，最终分好的类就是给定的 `cluster` 的数量


- 其属于无监督学习，朴素的 `K-Means` 不引入网络，分好的；类效果如图：<br/>

![kms.png](K_Means%2Fkms.png)

- 可以明显地看出分的类并没有理想中的这么好，但是整体是把11个类别都分出来了

### CNN 实现思路

- 本项目主要是基于 `Pytorch` 框架进行卷积神经网络训练模型，版本为2.3.0


- 首先进行数据处理，打上相应的标签。将369张图像进行测试训练二八分，分别存在 `Dataset/train` 和 `Dataset/test` 下，然后根据类别各自命名，通过名字来实现手动分类，以下是分类的一些说明图：<br/>
![classify.png](classify.png)


- 然后由于图像数量较少，这里做了一些数据增强，在 `load_data` 中对图像做了随机的水平和垂直翻转，同时对三个颜色通道都做了归一化处理，然后将处理好的数据导入神经网络中


- 卷积神经网络的核心部分主要分为三个，我们设计了三个卷积层和三个全连接层，同时池化层的规格是2x2，每次实现的时候都用激活函数 `relu` 做激活，最后一层全连接函数不用激活


- 另外，我们也设置了随机种子，来保证结果的可重复性；设置的 `device` 变量也可以自动检测运行机器有没有合适的GPU，如果有则使用GPU运行，否则使用CPU


- 最后实现训练函数和测试函数，将学习率设置为0.001，训练1000轮，同时画出损失函数和准确率函数的曲线，如图：<br/>
  ![loss.png](loss.png)
  ![accuracy.png](accuracy.png)


- 可以看到整体曲线非常符合曲线函数，同时到最后，损失趋近于0，且准确率趋近于100%


- 一些反思：根据我们现有对模型训练的理解，损失接近0和准确率达到100%是非常非常少见的事情，然后我们下来查阅了相关资料<br/>
发现我们所用的数据集数量确实太少，所以是有可能达到这种非常理想的情况的

### 改进

- 由于一般模型训练都是对训练集和测试集八二分，所以在保证学习率和训练轮次的情况下，<br/>
我们做了一个训练集测试集分别是五五分、六四分、七三分、八二分、九一分的实现，进行改进升级


- 同时，由上述的训练和测试结果可以看到，曲线虽然有整体趋势，但是并不光滑，中间有几次比较大的跳动，所以我们也对曲线的绘制<br/>
进行了平滑处理，也给图像上的曲线不同颜色打上标签，进行损失和准确率的对比，平滑处理我们主要采用了五种方法，分别对比一下：


- 我们也增加性能指标来评估模型，主要包括 “**轮次**-**损失值** / **准确率** / **精确率** / **召回率**” 的图像，以及 “**召回率** / **精确率**” 的图像

**简单平均移动** ![SMA_Loss.png](Model_V3%2FSMA_Loss.png)![SMA_Accuracy.png](Model_V3%2FSMA_Accuracy.png)![SMA_Precision.png](Model_V3%2FSMA_Precision.png)![SMA_Recall.png](Model_V3%2FSMA_Recall.png)
<br/>
<br/>

**指数平均移动** ![EMA_Loss.png](Model_V3%2FEMA_Loss.png)![EMA_Accuracy.png](Model_V3%2FEMA_Accuracy.png)![EMA_Precision.png](Model_V3%2FEMA_Precision.png)![EMA_Recall.png](Model_V3%2FEMA_Recall.png)
<br/>
<br/>

**高斯滤波平滑** <br/>![GAU_Loss.png](Model_V3%2FGAU_Loss.png)![GAU_Accuracy.png](Model_V3%2FGAU_Accuracy.png)![GAU_Precision.png](Model_V3%2FGAU_Precision.png)![GAU_Recall.png](Model_V3%2FGAU_Recall.png)
<br/>
<br/>

**中位数滤波平滑** <br/>![MED_Loss.png](Model_V3%2FMED_Loss.png)![MED_Accuracy.png](Model_V3%2FMED_Accuracy.png)![MED_Precision.png](Model_V3%2FMED_Precision.png)![MED_Recall.png](Model_V3%2FMED_Recall.png)
<br/>
<br/>

**局部加权回归平滑** ![LOW_Loss.png](Model_V3%2FLOW_Loss.png)![LOW_Accuracy.png](Model_V3%2FLOW_Accuracy.png)![LOW_Precision.png](Model_V3%2FLOW_Precision.png)![LOW_Recall.png](Model_V3%2FLOW_Recall.png)

### 总结

最后根据图像分析，结合平滑程度和整体趋势，可以看出高斯滤波平滑的效果最好

##### 损失图像

![GAU_Loss.png](Model_V3%2FGAU_Loss.png)

##### 准确率图像

![GAU_Accuracy.png](Model_V3%2FGAU_Accuracy.png)

##### 精确率图像

![GAU_Precision.png](Model_V3%2FGAU_Precision.png)

##### 召回率图像

![GAU_Recall.png](Model_V3%2FGAU_Recall.png)

最后根据数据集的样本数量以及训练集与测试集比例的不同，综合可视化的图像来分析，可以明显看出训练测试比例为7:3的时候效果最好，其余不相上下，但是整体都是呈现出曲线函数的趋势，且最后收敛

另外，我们也给出不同训练集测试集比例下的 `Recall-Precision` 图像：<br/>
![5_5-Precision-Recall-Curve.png](Model_V3%2F5_5-Precision-Recall-Curve.png)
![6_4-Precision-Recall-Curve.png](Model_V3%2F6_4-Precision-Recall-Curve.png)
![7_3-Precision-Recall-Curve.png](Model_V3%2F7_3-Precision-Recall-Curve.png)
![8_2-Precision-Recall-Curve.png](Model_V3%2F8_2-Precision-Recall-Curve.png)
![9_1-Precision-Recall-Curve.png](Model_V3%2F9_1-Precision-Recall-Curve.png)

`Recall-Precision` 图像围成的面积越大，表示这个类别分类效果越好

### 小组成员

Lin Xiaoyi, Tang Jiajun, Wang Zhen, Chen Guanrui, Wang Jing

### 此项目仅供学习交流使用，转载请注明出处！
