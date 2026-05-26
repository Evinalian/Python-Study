"""
练习 3-2: 搭建简单 CNN

任务：
1. 构建类 SimpleCNN(nn.Module)，结构如下：
   - Conv2d(1, 16, kernel_size=3, padding=1) -> ReLU -> MaxPool2d(2, 2)
   - Conv2d(16, 32, kernel_size=3, padding=1) -> ReLU -> MaxPool2d(2, 2)
   - Flatten
   - Linear(32 * 7 * 7, 128) -> ReLU
   - Linear(128, 10)

2. 输入：MNIST 风格灰度图 (batch, 1, 28, 28)
   - 28 -> pool: 14 -> pool: 7  故展平后 32*7*7 = 1568

3. 验证各阶段输出 shape：
   - 用 torch.randn(4, 1, 28, 28) 逐步测试
   - 在 forward 中添加临时 print 或在外部验证
   - 确保最终输出 shape 为 (4, 10)

4. 打印每层的参数形状和参数量

提示：
- MaxPool2d(kernel_size=2, stride=2) 将 H, W 各减半
- view(x.size(0), -1) 展平
- 注意验证 32*7*7 = 1568 是否正确（取决于输入尺寸和 padding）
"""

import torch
import torch.nn as nn


# TODO: 1. 定义 SimpleCNN 类


# TODO: 2. 在 __init__ 中定义各层（conv1, conv2, pool, fc1, fc2）


# TODO: 3. 在 forward 中分步传播，并可选打印中间 shape


# TODO: 4. 实例化模型，用 (4, 1, 28, 28) 输入验证各阶段输出 shape


# TODO: 5. 打印每层的参数形状和参数量


