"""
练习 6-4（进阶）：DDP 多卡训练

目标：
  实现一个完整的 DDP 训练脚本，包含 DistributedSampler、梯度同步、
  只在 rank 0 保存模型和打印日志。

启动方式：
  torchrun --nproc_per_node=2 ex6_4_ddp_training.py

要求：
  1. 定义 CNN 模型（2 层卷积 + 2 层全连接）
  2. 使用 DistributedSampler 分配数据
  3. 在每个 epoch 开始时调用 sampler.set_epoch(epoch)
  4. 只在 rank 0 打印训练进度和保存模型
  5. 使用 dist.all_reduce 汇总各进程的 loss
  6. 正确初始化和销毁进程组

建议步骤：
  1. 导入必要的库（torch, torch.nn, torch.optim, torch.distributed, DDP, DistributedSampler）
  2. 实现 setup() 函数：读取环境变量（LOCAL_RANK, WORLD_SIZE, RANK），初始化进程组
  3. 实现 cleanup() 函数：销毁进程组
  4. 定义 CNN 模型类
  5. 实现 main()：
     a. setup() 获取 local_rank, rank, world_size
     b. 设置 device = torch.device(f"cuda:{local_rank}")
     c. 准备模拟数据 + DistributedSampler + DataLoader
     d. 创建模型 → .to(device) → DDP 包装
     e. 训练循环（5 epoch）：
        - sampler.set_epoch(epoch)
        - train, backward, step
        - dist.all_reduce 汇总 loss
        - rank 0 打印进度
     f. rank 0 保存模型（model.module.state_dict()）
     g. cleanup()
  6. if __name__ == "__main__": main()

提示：
  - 使用 int(os.environ.get("LOCAL_RANK", 0)) 读取环境变量
  - DDP 包装后，原始模型在 model.module 中
  - dist.all_reduce(tensor, op=dist.ReduceOp.SUM) 对所有进程的 tensor 求和
  - 检查 CUDA 可用性
"""

# TODO: 导入 torch, torch.nn, torch.optim, os
# TODO: 导入 torch.distributed as dist, DistributedDataParallel, DistributedSampler


# TODO: 实现 setup() 函数


# TODO: 实现 cleanup() 函数


# TODO: 定义 CNN 模型类


# TODO: 实现 main() 函数：
#   1. setup
#   2. 准备数据和 sampler
#   3. 创建模型 + DDP 包装
#   4. 训练循环
#   5. 保存模型
#   6. cleanup


# TODO: if __name__ == "__main__": main()
