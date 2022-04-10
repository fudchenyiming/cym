# cym
代码介绍：

data.py中实现了dataset类和dataloader类，用来读取和装载训练集数据。

functions.py的class layer和 class linear(layer)实现了神经网络最基本的线性层。

relu() sigmoid() tanh()实现基本的激活函数

functions.py中class EntropyLoss和class MseLoss实现loss的计算以及反向传播和梯度下降

selection.py 用来实现学习率、隐藏层个数、学习率衰减的参数查找。

model.pkl为参数查找后最优模型的保存位置。

optimizers.py中实现了带动量的SGD优化方法。里面代码实现了L2正则化（权重衰减）和学习率下降。

sp.py实现Squentialprocess类用来连续处理神经网络。

train.py是神经网络训练的主函数。



训练过程：运行selection.py 里面设置了参数查找的参数变化范围，并把模型保存下来。



测试过程：运行cvtest.py 输出最优模型的accuracy。
