"""
练习 4-5（进阶）: 自定义训练引擎类 (Trainer)

任务：
将训练引擎封装为一个可复用的 Trainer 类。

要求：
1. Trainer 类的 __init__ 方法：
   - 参数: model, optimizer, criterion, device, train_loader, val_loader,
           scheduler=None, early_stopping=None, grad_clip=None
   - 存储以上配置
   - 初始化 self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

2. 方法：
   a) train_one_epoch() -> float:
      - 执行一个 epoch 的训练
      - 返回平均 training loss
      - 内部包含 optimizer.zero_grad(), forward, loss.backward()
      - 如果 self.grad_clip 不为 None，在 backward 后调用 clip_grad_norm_

   b) validate() -> tuple[float, float]:
      - @torch.no_grad() 装饰
      - 返回 (avg_val_loss, val_accuracy)

   c) fit(num_epochs, verbose=True) -> dict:
      - 执行 num_epochs 个 epoch 的训练
      - 每个 epoch 调用 train_one_epoch() 和 validate()
      - 记录 loss 和 acc 到 self.history
      - 如果有 scheduler，调用 scheduler.step()（ReduceLROnPlateau 需 val_loss）
      - 如果有 early_stopping，每次 validate 后检查是否触发
      - verbose=True 时打印 epoch 级别的日志

3. 测试：
   - 用 MNIST 数据 + CNN 模型测试 Trainer 类
   - 测试三种配置：
     a) 基础配置：AdamW + StepLR，无早停、无梯度裁剪
     b) 高级配置：AdamW + ReduceLROnPlateau + EarlyStopping(patience=5) + grad_clip=1.0
     c) 验证 Trainer 对不同的模型（MLP 替代 CNN）也能正常工作

4. 打印 Trainer 的 history 进行对比验证

提示：
- 需要判断 scheduler 是否为 ReduceLROnPlateau（isinstance check），以决定传入 val_loss
- early_stopping(val_loss) 返回 bool，True 时 break
- 梯度裁剪在 loss.backward() 之后、optimizer.step() 之前
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# TODO: 1. 实现 EarlyStopping 类（可从练习 4-4 复用）


# TODO: 2. 实现 Trainer 类


# TODO: 3. MNIST 数据加载


# TODO: 4. 测试配置 a: 基础 AdamW + StepLR


# TODO: 5. 测试配置 b: AdamW + ReduceLROnPlateau + EarlyStopping + grad_clip


# TODO: 6. 测试配置 c: 用简单 MLP 替换 CNN，验证 Trainer 通用性


# TODO: 7. 对比三种配置的 history


