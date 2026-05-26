"""
练习 2-1: 自定义表格 Dataset

任务：
1. 用 NumPy 生成一个模拟的 CSV 数据集并保存为 data.csv：
   - 200 行样本
   - 5 列特征：x1, x2, x3, x4, x5（从标准正态分布采样）
   - 1 列标签 y = 2*x1 - 3*x2 + x3 + 0*x4 + 0.5*x5 + noise（noise ~ N(0, 0.5)）
   - y 用 float32

2. 继承 torch.utils.data.Dataset，实现 CSVDataset：
   - __init__: 读取 CSV 文件，将特征和标签分离
   - __len__: 返回总样本数
   - __getitem__: 返回 (特征 Tensor, 标签 Tensor)

3. 用 DataLoader 加载该 Dataset，设置 batch_size=16, shuffle=True
4. 遍历一个 epoch，打印每批次的 X.shape 和 y.shape
5. 验证所有批次的特征和标签值都在合理范围内

提示：
- 使用 np.savetxt() 配合 header 写入 CSV，或使用 pandas (to_csv)
- 在 __getitem__ 中返回 torch.tensor 的 float32
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# TODO: 1. 生成模拟数据 (200 行, 5 特征 + 1 标签) 并保存为 data.csv


# TODO: 2. 定义 CSVDataset 类（继承 Dataset）


# TODO: 3. 实例化 Dataset 和 DataLoader (batch_size=16, shuffle=True)


# TODO: 4. 遍历一个 epoch，打印每批次的 shape


# TODO: 5. 验证数据的统计量（特征均值和标签范围）


