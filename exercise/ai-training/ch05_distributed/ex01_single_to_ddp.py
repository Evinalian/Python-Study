"""
练习 1：单卡 → DDP 迁移
========================

将一个已有的单卡训练脚本改造为 DDP 版本。

你需要完成:
1. 准备一个简单的单卡训练脚本（使用教程中的 DemoTransformer）
2. 添加 DDP 所需的改动:
   a. 初始化进程组: dist.init_process_group(backend='nccl')
   b. 模型包装: model = DDP(model, device_ids=[local_rank])
   c. 数据采样: DistributedSampler(dataset)
   d. 只在 rank 0 打印日志和保存模型
   e. 在 epoch 开始时调用 sampler.set_epoch(epoch)
3. 测试 DDP 版本在 1 GPU 和 2+ GPU 下的运行
4. 对比单卡和 DDP 版本的:
   - 代码改动的行数
   - 训练速度（tokens/sec）
   - 相同有效 batch_size 下的 loss 曲线（应一致）

关键改动清单:
- 每个进程开始: setup(rank, world_size)
- 每个进程结束: cleanup()
- 日志/保存: if is_main_process():
- 数据加载: sampler = DistributedSampler
- 模型: model = DDP(model)

思考题:
- 为什么 sampler.set_epoch(epoch) 是必要的？
- 如果忘记 set_epoch，会发生什么？
- DDP 对 batch_size 有什么隐含要求？
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


# TODO: 拷贝你的单卡训练脚本的模型和数据集定义到这里


# TODO: 实现 setup 和 cleanup 函数
def setup(rank, world_size):
    """TODO: 初始化分布式进程组"""
    pass


def cleanup():
    """TODO: 销毁进程组"""
    pass


# TODO: 实现 DDP 训练函数
def train_ddp(rank, world_size, args):
    """
    TODO: DDP 版本的训练主函数。

    流程:
    1. setup(rank, world_size)
    2. 创建模型 → 包装为 DDP
    3. 创建 DistributedSampler → DataLoader
    4. 训练循环（与单卡相同，但 log/save 只在 rank 0）
    5. cleanup()
    """
    pass


if __name__ == "__main__":
    # TODO: 读取环境变量
    # torchrun 启动时会自动设置 RANK, WORLD_SIZE, LOCAL_RANK
    pass
