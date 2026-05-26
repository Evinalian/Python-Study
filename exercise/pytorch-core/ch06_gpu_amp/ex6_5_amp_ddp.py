"""
练习 6-5（进阶）：AMP + DDP 组合

目标：
  将 AMP 和 DDP 组合到同一个训练脚本中，实现完整的分布式混合精度训练。

启动方式：
  torchrun --nproc_per_node=2 ex6_5_amp_ddp.py

要求：
  1. 在 DDP 训练的基础上集成 AMP（autocast + GradScaler）
  2. 每个 epoch 打印 rank 0 的 loss 和 GradScaler 的当前缩放因子
  3. 实现梯度裁剪（在 unscale 之后、step 之前）
  4. 只在 rank 0 保存模型
  5. 正确处理进程组的初始化和清理

建议步骤：
  1. 导入必要的库
  2. 实现 setup() 和 cleanup() 函数
  3. 定义模型类
  4. 实现 main()：
     a. setup，获取 local_rank / rank / world_size
     b. 设置 device
     c. 准备数据（DistributedSampler + DataLoader）
     d. 创建模型 → DDP 包装
     e. 创建 criterion, optimizer, GradScaler
     f. 训练循环：
        - sampler.set_epoch(epoch)
        - 每个 batch: zero_grad → autocast forward → scaler.scale(loss).backward()
        - scaler.unscale_(optimizer) → clip_grad_norm_ → scaler.step(optimizer) → scaler.update()
        - 记录并汇总 loss
        - rank 0 打印 loss 和 scale
     g. rank 0 保存模型
     h. cleanup()

提示：
  - DDP 的梯度同步在 backward() 中自动完成，与 GradScaler 兼容
  - 梯度裁剪必须在 scaler.unscale_() 之后、scaler.step() 之前
  - scaler.get_scale() 查看当前缩放因子
  - 如果 scale 持续下降说明频繁出现梯度溢出
"""

# TODO: 导入 torch, torch.nn, torch.optim, os
# TODO: 导入 torch.distributed, DDP, DistributedSampler
# TODO: 导入 torch.cuda.amp (autocast, GradScaler)


# TODO: 实现 setup() 函数


# TODO: 实现 cleanup() 函数


# TODO: 定义模型类（建议含 BatchNorm，测试 AMP 兼容性）


# TODO: 实现 main() 函数：
#   1. setup
#   2. 准备数据 + DistributedSampler
#   3. 创建模型 + DDP 包装
#   4. 创建 criterion, optimizer, GradScaler
#   5. 训练循环（含 autocast, scaler, 梯度裁剪）
#   6. 保存模型
#   7. cleanup


# TODO: if __name__ == "__main__": main()
