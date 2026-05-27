"""
练习 4（进阶）：DDP → FSDP 迁移
===============================

将 DDP 训练脚本迁移到 FSDP（FULL_SHARD 模式）。

你需要完成:
1. 从 ex01 的 DDP 版本出发，改为 FSDP
2. 主要改动:
   a. 导入 FSDP 相关模块:
      from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
      from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
   b. 定义 auto_wrap_policy（以 TransformerBlock 为单位包装）
   c. 配置 MixedPrecision（BF16）
   d. 设置 sharding_strategy=FULL_SHARD
   e. 使用 model = FSDP(model, ...) 替代 DDP
3. 对比 DDP 和 FSDP 在相同配置下的:
   - 单 GPU 显存占用
   - 每步训练时间
   - 相同 effective batch size 下能训练的最大模型

FSDP vs DDP 的关键差异:
- DDP: 每 GPU 有完整模型副本 + 完整梯度 + 完整优化器状态
- FSDP: 每 GPU 只存 1/N 的参数 + 1/N 的梯度 + 1/N 的优化器状态

思考题:
- FSDP 为什么比 DDP 多了一次通信？（AllGather 参数 + ReduceScatter 梯度）
- FSDP 的显存节省是否和 GPU 数量成正比？
- 在什么条件下 FSDP 和 DDP 速度接近？
"""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools


# TODO: 定义 TransformerBlock（用于 auto_wrap_policy）


# TODO: 实现 FSDP 训练函数
def train_fsdp(local_rank, world_size, args):
    """
    TODO: FSDP 版本的训练函数。

    关键配置:
    - auto_wrap_policy = functools.partial(
          transformer_auto_wrap_policy,
          transformer_layer_cls={TransformerBlock},
      )
    - mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, ...)
    - sharding_strategy = ShardingStrategy.FULL_SHARD
    - backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    """
    pass


# TODO: 对比 DDP vs FSDP
def compare_ddp_vs_fsdp():
    """
    TODO: 量化对比。

    实验:
    1. 尝试递增模型大小（增加 n_layers 和 d_model）
    2. 记录 DDP 在多大时 OOM
    3. 记录 FSDP 在多大时 OOM
    4. 对比两者在"都可以运行"的规模下的速度和显存
    """
    pass


if __name__ == "__main__":
    pass
