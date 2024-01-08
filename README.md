# 				  MaskRcnn实例分割 代码报告

## 1. 数据集选取

​    ![pic1](https://github.com/thelifeangel/MyNetwork/blob/main/display/image-20240107200036985.png) 

从coco数据集选取cat、dog两个类，训练集共1000个实例（每个类500个），验证集共300个实例。得到训练集810张图片，验证集264张图片。数据集标注格式如下图：

![image-20240107200118282](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240107200118282.png)

## 2.数据集可视化

​											                    Mask掩码可视化

![image-20240107200245913](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240107200245913.png)

![image-20240107200751568](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240107200751568.png)

![image-20240107200757224](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240107200757224.png)

​													                 数据裁剪

​									                 ![image-20240107201024163](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240107201024163.png)

## 3 复现 训练阶段

![image-20240107201209040](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240107201209040.png)

在官方预训练权重基础上，以学习率0.001训练30 epochs head层网络，学习率0.0001再训练10轮 ResNet stage 4以上全部网络，最后以学习率0.0001再训练30 epochs全部层网络。

## 4  训练部分超参数设置

BACKBONE = "resnet101"
IMAGES_PER_GPU = 2（batchsize=2)

NUM_CLASSES = 1 + 2 # 

 STEPS_PER_EPOCH = 450

VALIDATION_STEPS = 100

USE_MINI_MASK = True

LEARNING_RATE = 0.001

LEARNING_MOMENTUM = 0.9

DETECTION_MIN_CONFIDENCE = 0.85

WEIGHT_DECAY = 0.0001

## 5  损失曲线

### 5.1 训练集损失

![image-20240107201733140](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240107201733140.png)

### 5.2 验证集损失

![image-20240107201822933](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240107201822933.png)

### 5.3 RP曲线及IOU值计算

![image-20240107201902108](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240107201902108.png)

![image-20240107201925345](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240107201925345.png)

### 5.4 测试结果

​                                           ![image-20240107202035748](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240107202035748.png)![image-20240107202039948](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240107202039948.png)

### 6 总结

技术优势：

1.高精度：Mask R-CNN在多个标准数据集上显示出卓越的精度，尤其是在对象检测和实例分割方面。

2.多任务学习：它能够同时解决对象检测（给出边界框）和实例分割（给出像素级对象掩码）的任务，使得该框架非常实用。

3.灵活性和泛化能力：Mask R-CNN适用于多种不同的对象类别，并且能够适应各种大小和形状的对象。

4.可扩展性：该模型可以与其他神经网络架构集成，例如，可以替换使用不同的骨干网络来提取特征。

技术局限：

1.计算密集和资源消耗：由于其复杂性，Mask R-CNN需要显著的计算资源，包括大量的GPU计算能力，这可能不适合资源有限的应用。2.速度较慢：与一些更轻量级的模型相比，Mask R-CNN在推理时可能比较慢，这对于需要实时处理的应用场景是一个限制。

3.训练挑战：由于其复杂性，Mask R-CNN的训练可能比较困难，需要精细的超参数调整和大量的标注数据。

4.对遮挡的敏感性：在处理高度遮挡的对象时，Mask R-CNN可能无法准确分割或检测全部对象。