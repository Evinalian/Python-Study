"""
练习 3-1: 搭建 MLP（多层感知机）

任务：
1. 构建类 MLP(nn.Module)，结构如下：
   - input_dim=784 (MNIST 风格: 28x28 展平)
   - hidden_dims=[512, 256, 128]
   - output_dim=10（10 分类）
   - 每层：Linear -> BatchNorm1d -> ReLU -> Dropout(0.2)
   - 最后一层：Linear -> 无激活（输出 logits）

2. 在 __init__ 中使用 nn.Sequential 或逐个定义层

3. 打印模型结构 (print(model))

4. 统计可训练参数总数并打印

5. 用随机输入 torch.randn(32, 784) 测试前向传播，验证输出 shape 为 (32, 10)

提示：
- 用 nn.ModuleList 或 nn.Sequential 管理多个全连接层
- BatchNorm1d 的 num_features 参数为对应 hidden_dim
- 参数统计：sum(p.numel() for p in model.parameters() if p.requires_grad)
"""

import torch
import torch.nn as nn


# TODO: 1. 定义 MLP 类 (继承 nn.Module)


# TODO: 2. 在 __init__ 中构建网络层


# TODO: 3. 在 forward 中定义前向传播


# TODO: 4. 实例化并打印模型结构


# TODO: 5. 统计并打印可训练参数总数


# TODO: 6. 用 torch.randn(32, 784) 测试输出 shape


