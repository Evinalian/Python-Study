"""
练习 4-4（进阶）: 实现早停 + 学习率调度 + tqdm 可视化

任务：
在 MNIST 分类任务上整合训练模块，实现完整的高级训练功能。

要求：
1. 实现 EarlyStopping 类：
   - __init__(patience=5, min_delta=0.0)
   - __call__(val_loss): 返回 True（停止）或 False（继续）
   - 属性 best_loss 记录历史最佳验证损失
   - 属性 counter 记录连续无改善的 epoch 数

2. MNIST 数据加载（与练习 4-1 相同）

3. 模型：与练习 4-1 相同的 CNN

4. 训练配置：
   - optimizer: AdamW, lr=0.001
   - scheduler: ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
   - EarlyStopping(patience=5)

5. tqdm 集成：
   - 外层 epoch 循环用 tqdm 显示（含 train_loss, val_loss, val_acc, lr）

6. 当早停触发时：
   - 打印触发时的 epoch、best_val_loss、当前 val_acc

7. 如果达到最大 epoch 也未触发早停：
   - 打印"训练正常结束"和最终指标

8. 用 matplotlib 绘制训练和验证 loss 曲线，标注早停发生的位置

提示：
- ReduceLROnPlateau.step(val_loss) 在每个 epoch 的验证后调用
- 早停判断放在 scheduler.step() 之后
- tqdm 的 set_postfix 可动态显示多个指标
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


# TODO: 1. 实现 EarlyStopping 类


# TODO: 2. 数据加载（MNIST）


# TODO: 3. 定义 CNN 模型


# TODO: 4. 定义 train_one_epoch 函数（含 tqdm）


# TODO: 5. 定义 evaluate 函数（返回 val_loss 和 val_acc）


# TODO: 6. 训练主循环（含早停、调度器、tqdm）


# TODO: 7. 绘制 loss 曲线并标注早停点


