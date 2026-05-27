"""
练习 2：梯度累积实现
====================

在 DDP 训练中加入梯度累积功能，并正确处理梯度同步。

你需要完成:
1. 在训练循环中实现梯度累积
2. 正确处理 DDP 的梯度同步:
   - 前 N-1 个 micro-batch: 设置 model.require_backward_grad_sync = False
   - 第 N 个 micro-batch: 设置 model.require_backward_grad_sync = True
   - 只在累积的最后一步调用 optimizer.step() 和 optimizer.zero_grad()
3. 实验不同的 accumulation_steps (1, 4, 8, 16)
4. 比较:
   - 相同 effective batch size 下的显存占用
   - 训练速度和吞吐量 (tokens/sec)
   - 最终 loss 是否受影响

注意: 梯度累积中的损失缩放:
    如果每个 micro-batch 的 loss.backward() 都是原始 loss，
    那么最终每个参数的梯度将是 N 个 micro-batch 梯度的和。
    为了得到"平均梯度"，应该:
    - 每个 micro-batch: loss.backward()（梯度累加）
    - 不做缩放（因为梯度的和 = N × 平均梯度, 恰好等于大 batch 的梯度）
    或者:
    - 每个 micro-batch: (loss / N).backward()
    - 累加 N 次后梯度就是平均梯度

思考题:
- 为什么梯度累积需要特殊处理 DDP 的梯度同步？
- gradient_accumulation_steps 增大后，为什么显存基本不变？
- 梯度累积和增大 micro_batch_size 有什么本质区别？
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# TODO: 实现带梯度累积的训练循环
def train_with_gradient_accumulation(
    model, dataloader, optimizer, accumulation_steps, num_epochs,
):
    """
    TODO: 带梯度累积的 DDP 训练循环。

    关键点:
    1. 使用 model.require_backward_grad_sync 控制同步时机
    2. 在累积步之间不调用 optimizer.step() 和 optimizer.zero_grad()
    3. 正确处理损失缩放
    """
    pass


# TODO: 对比实验
def compare_accumulation_strategies():
    """
    TODO: 实验不同的 accumulation_steps。

    固定 effective_batch_size = micro_batch_size × accumulation_steps = 64。
    变化:
    - micro_batch_size=64, accumulation_steps=1 (基准)
    - micro_batch_size=16, accumulation_steps=4
    - micro_batch_size=8,  accumulation_steps=8
    - micro_batch_size=4,  accumulation_steps=16

    记录每种配置的:
    - 单步训练时间
    - GPU 显存占用
    - tokens/sec
    - 前 100 步的 loss
    """
    pass


if __name__ == "__main__":
    compare_accumulation_strategies()
