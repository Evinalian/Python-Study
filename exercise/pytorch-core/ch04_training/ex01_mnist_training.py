"""
练习 4-1: MNIST 分类完整训练循环

任务：
用 PyTorch 实现完整的 MNIST 手写数字分类训练流程。

要求：
1. 数据加载：
   - 使用 torchvision.datasets.MNIST
   - transform: ToTensor() + Normalize((0.1307,), (0.3081,))
   - train_loader batch_size=64, shuffle=True
   - test_loader batch_size=1000, shuffle=False

2. 模型：参考第 3 章的 SimpleCNN 或自定义 CNN：
   - Conv(1,32,3,padding=1) -> ReLU -> MaxPool(2)
   - Conv(32,64,3,padding=1) -> ReLU -> MaxPool(2)
   - Flatten -> Linear(64*7*7, 128) -> ReLU -> Dropout(0.25)
   - Linear(128, 10)

3. 训练配置：
   - criterion: nn.CrossEntropyLoss()
   - optimizer: AdamW, lr=0.001, weight_decay=0.01
   - scheduler: CosineAnnealingLR, T_max=10
   - num_epochs=10

4. 记录：每个 epoch 记录 train_loss 和 test_accuracy

5. 输出：
   - 打印最终测试准确率（目标 > 97%）
   - 用 matplotlib 绘制 train_loss 和 test_accuracy 随 epoch 变化的双 y 轴图

提示：
- datasets.MNIST 需要 download=True
- 记录测试准确率时在 torch.no_grad() 下运行
- 双 y 轴图：fig, ax1 = plt.subplots(); ax2 = ax1.twinx()
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# TODO: 1. 设备检测和数据加载


# TODO: 2. 定义 CNN 模型


# TODO: 3. 定义训练函数 train_one_epoch


# TODO: 4. 定义测试函数 test


# TODO: 5. 设置 optimizer 和 scheduler


# TODO: 6. 训练循环，记录 train_loss 和 test_accuracy


# TODO: 7. 绘制双 y 轴图并打印最终准确率


