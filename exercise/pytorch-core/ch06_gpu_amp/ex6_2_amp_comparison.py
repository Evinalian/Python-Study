"""
练习 6-2：AMP 混合精度训练对比

目标：
  训练同一个模型，分别使用 FP32 和 AMP (FP16) 两种模式，
  对比显存占用、训练速度和最终 loss。

要求：
  1. 定义一个中等规模的 MLP 模型（3-4 层，hidden_dim=512）
  2. 使用模拟数据（10000 样本，特征维度 200，分类 5 类）
  3. 分别用 FP32 和 AMP 模式各训练 5 个 epoch
  4. 记录并打印对比指标：
     - 峰值显存占用（torch.cuda.max_memory_allocated）
     - 总训练耗时
     - 最终 loss
     - 每个 epoch 的平均耗时
  5. 输出格式化对比报告

建议步骤：
  1. 检查 CUDA 可用性（不可用则退出并提示）
  2. 定义模型类
  3. 准备数据（X, y 使用 torch.randn / torch.randint，移到 GPU）
  4. 实现 train_fp32() 函数：标准 FP32 训练，返回耗时、loss 历史、峰值显存
  5. 实现 train_amp() 函数：AMP 训练，使用 autocast + GradScaler，返回相同的指标
  6. 在每次训练前重置显存峰值统计：torch.cuda.reset_peak_memory_stats()
  7. 在主函数中依次运行两种模式，打印对比表格

提示：
  - 使用 torch.cuda.synchronize() 确保计时准确
  - 使用 time.perf_counter() 计时
  - GradScaler 不需要特殊配置，使用默认参数即可
  - 确保两种模式使用相同的随机种子和初始权重
"""

# TODO: 导入 torch, torch.nn, torch.optim, time


# TODO: 定义模型类 BenchMLP (hidden_dim=512, num_layers=4, dropout=0.2)


# TODO: 实现 train_fp32(model, loader, criterion, optimizer, device, epochs) 函数


# TODO: 实现 train_amp(model, loader, criterion, optimizer, device, epochs) 函数


# TODO: 主函数：
#   1. 检查 CUDA
#   2. 准备数据（torch.manual_seed(42)）
#   3. 重置显存统计，训练 FP32，记录指标
#   4. 重置显存统计，训练 AMP（用相同的初始化权重），记录指标
#   5. 打印对比表格
