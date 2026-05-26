"""
练习 4-2: 验证 model.train() 和 model.eval() 对 Dropout 的影响

任务：
构建一个包含 Dropout(p=0.5) 的网络，验证 train 和 eval 模式下的行为差异。

要求：
1. 构建简单网络：
   - Linear(100, 100) -> Dropout(p=0.5) -> Linear(100, 10)

2. 创建固定输入 x = torch.ones(1, 100)（方便观察哪些元素被置零）

3. 在 model.train() 模式下：
   - 前向传播 3 次（x 不变），打印每次的 Linear(100, 100) 输出（即 Dropout 的输入）
   - 记录 Dropout 后有多少元素被置零（比例应接近 0.5）

4. 在 model.eval() 模式下：
   - 前向传播 3 次，打印每次的输出
   - 验证 eval 模式下没有元素被置零（Dropout 被禁用）
   - 验证 eval 模式下的三次输出完全相同（确定性）

5. 打印 train 模式和 eval 模式下元素被置零的比例

提示：
- 每次前向传播结果不同说明 Dropout 在生效（train 模式）
- (output == 0).sum().item() / output.numel() 计算置零比例
"""

import torch
import torch.nn as nn


# TODO: 1. 构建网络: Linear(100,100) -> Dropout(0.5) -> Linear(100,10)


# TODO: 2. 创建固定输入 torch.ones(1, 100)


# TODO: 3. train 模式：3 次前向传播，观察 Dropout 效果


# TODO: 4. eval 模式：3 次前向传播，验证 Dropout 被禁用


# TODO: 5. 打印对比结果


