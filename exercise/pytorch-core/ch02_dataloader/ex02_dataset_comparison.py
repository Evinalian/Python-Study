"""
练习 2-2: TensorDataset 与自定义 Dataset 对比

任务：
1. 生成相同的训练数据（X: 100 个样本，10 维特征，y: 100 个标签 3 分类）
2. 分别用 TensorDataset 和自定义 Dataset 包装同一份数据
3. 设置相同的 DataLoader 参数（batch_size=20, shuffle=False —— 便于对比）
4. 分别遍历两个 DataLoader，逐批次对比数据是否完全一致

验证方法：
- 两个 DataLoader 使用相同的 random seed 或都不 shuffle
- 逐批次用 torch.allclose() 比对 X 和 y
- 如果所有批次都 allclose，说明自定义 Dataset 实现正确

提示：
- 自定义 Dataset 的 __getitem__ 需要对原始 Tensor 做索引切片
- 由于两个 DataLoader 从同一个数据源读取，shuffle=False 时顺序应该一致
"""

import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader


# TODO: 1. 生成数据: X ~ N(0,1) shape (100, 10), y ~ randint(0,3) shape (100,)


# TODO: 2. 定义自定义 CustomDataset，在 __getitem__ 中从 self.X 和 self.y 取索引


# TODO: 3. 实例化两个 Dataset 和两个 DataLoader (batch_size=20, shuffle=False)


# TODO: 4. 逐批次用 torch.allclose() 验证数据一致性


# TODO: 5. 打印比对结果（全部一致 vs 发现差异）


