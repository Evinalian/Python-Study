"""
练习 2-3: 变长序列的 collate_fn

任务：
1. 生成 100 个样本，每个样本是一个长度在 [3, 15] 间随机取值的一维 Tensor（值为随机整数 0-9）
2. 每个样本的标签是该序列长度（整数）

3. 定义一个默认会报错的版本：直接用 DataLoader 不加 collate_fn
   - 尝试遍历，观察报错信息

4. 实现 `collate_with_padding(batch)` 函数：
   - 输入：list of (sequence, label) tuples
   - 输出：一个元组 (padded_sequences, lengths, labels)
     - padded_sequences: shape (batch_size, max_len_in_batch)，用 0 填充
     - lengths: shape (batch_size,)，每序列的原始长度
     - labels: shape (batch_size,)
   - 使用 torch.nn.utils.rnn.pad_sequence 实现 padding

5. 用自定义 collate_fn 创建 DataLoader，遍历验证输出 shapes 正确

提示：
- 自定义 Dataset 的 __getitem__ 返回 (sequence_tensor, length_label)
- pad_sequence(sequences, batch_first=True, padding_value=0)
- lengths = torch.tensor([s.size(0) for s in sequences])
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# TODO: 1. 定义 VarLenDataset，__getitem__ 返回 (随机长度 Tensor, 长度标签)


# TODO: 2. 尝试用默认 collate_fn 的 DataLoader，观察报错（用 try/except 捕获）


# TODO: 3. 实现 collate_with_padding 函数


# TODO: 4. 用自定义 collate_fn 创建 DataLoader (batch_size=16, shuffle=False)


# TODO: 5. 遍历一个 batch，打印 shapes，验证 padding 正确


