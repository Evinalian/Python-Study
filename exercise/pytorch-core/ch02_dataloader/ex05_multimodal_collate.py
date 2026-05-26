"""
练习 2-5（进阶）: 自定义 collate_fn 处理多模态数据

任务：
模拟多模态数据加载：每个样本包含 4 种不同形态的数据，需要自定义 collate_fn 处理。

每个样本的结构（由自定义 Dataset 的 __getitem__ 返回）：
{
    "image":   Tensor, shape (3, 64, 64),  float32,
    "text":    Tensor, 变长 1D,             int64 (token IDs),
    "tabular": Tensor, shape (5,),          float32 (5 个数值特征),
    "label":   int (标量),
}

要求：
1. 生成 200 个模拟样本的 Dataset：
   - image: 每个样本 torch.randn(3, 64, 64) 夹在 [0, 1]
   - text: 每个样本长度在 [5, 20] 随机，值为 randint(0, 100)
   - tabular: 每个样本 torch.randn(5)
   - label: randint(0, 10)

2. 实现 multimodal_collate_fn(batch)：
   - batch 是 list of dicts
   - 输出一个 dict：
     {
         "image":   Tensor (batch_size, 3, 64, 64),
         "text":    Tensor (batch_size, max_len),    # padding 到最大长度
         "text_lens": Tensor (batch_size,),          # 原始长度
         "tabular": Tensor (batch_size, 5),
         "label":   Tensor (batch_size,),
     }

3. 用 DataLoader (batch_size=16, shuffle=True) 配合此 collate_fn

4. 遍历一个 batch，打印所有字段的 shape 和 dtype

提示：
- 用 torch.stack 处理 image, tabular
- 用 pad_sequence 处理 text（batch_first=True, padding_value=0）
- 用 torch.tensor 汇总 label 列表
- 用 torch.tensor 记录原始长度列表
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# TODO: 1. 定义 MultiModalDataset，__getitem__ 返回 dict


# TODO: 2. 实现 multimodal_collate_fn


# TODO: 3. 创建 DataLoader (batch_size=16, shuffle=True)


# TODO: 4. 遍历一个 batch，打印所有字段的 shape 和 dtype


