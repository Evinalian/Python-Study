"""
练习 3-3: 损失函数实验

任务：
使用随机生成的 logits 和标签，分别计算三种常用损失函数，理解它们的输入格式和输出范围。

要求：
1. MSELoss：
   - pred = torch.randn(8, 1)（8 个样本的预测值）
   - target = torch.randn(8, 1)（8 个样本的真实值）
   - 计算 loss，打印结果

2. CrossEntropyLoss：
   - logits = torch.randn(8, 5)（8 个样本，5 分类）
   - labels = torch.randint(0, 5, (8,))（类别索引）
   - 计算 loss，打印结果
   - 尝试传入 one-hot 标签 torch.eye(5)[labels]，观察报错信息（用 try/except 捕获）

3. BCEWithLogitsLoss：
   - logits = torch.randn(8)（8 个样本的二分类 logits）
   - labels = torch.randint(0, 2, (8,)).float()（0 或 1）
   - 计算 loss，打印结果

4. 打印每种损失函数的输入要求（文档字符串中的说明）

提示：
- CrossEntropyLoss 的标签类型应为 torch.long（int64）
- 用 try/except 捕获 one-hot 导致的 RuntimeError
"""

import torch
import torch.nn as nn


# TODO: 1. MSELoss 演示


# TODO: 2. CrossEntropyLoss 演示 + one-hot 错误演示（try/except）


# TODO: 3. BCEWithLogitsLoss 演示


# TODO: 4. 总结各损失函数的输入要求


