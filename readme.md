# NNN
> Numpy Neural Network

## nnn
Check [NNN README](./nnn/readme.md) for documentation.


## 任务描述
手工搭建三层神经网络分类器，在数据集[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)上进行训练以实现图像分类。

## 基本要求
（1） 本次作业要求自主实现反向传播，不允许使用pytorch，tensorflow等现成的支持自动微分的深度学习框架，可以使用numpy；

（2） 最终提交的代码中应至少包含模型、训练、测试和参数查找四个部分，鼓励进行模块化设计；

（3） 其中模型部分应允许自定义隐藏层大小、激活函数类型，支持通过反向传播计算给定损失的梯度；训练部分应实现SGD优化器、学习率下降、交叉熵损失和L2正则化，并能根据验证集指标自动保存最优的模型权重；参数查找环节要求调节学习率、隐藏层大小、正则化强度等超参数，观察并记录模型在不同超参数下的性能；测试部分需支持导入训练好的模型，输出在测试集上的分类准确率（Accuracy）。