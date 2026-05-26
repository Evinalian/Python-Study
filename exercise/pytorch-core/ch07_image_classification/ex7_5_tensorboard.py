"""
练习 7-5（进阶）：TensorBoard 可视化

目标：
  在训练循环中集成 TensorBoard 日志记录，可视化训练过程中的各项指标。

要求：
  1. 使用 torch.utils.tensorboard.SummaryWriter
  2. 记录以下内容：
     a. 标量 (scalar)：train_loss, val_loss, train_acc, val_acc, learning_rate（每个 epoch）
     b. 直方图 (histogram)：模型各层权重的分布（每 N 个 epoch，如每 5 个）
     c. 图像 (image)：一批训练图片样本（第一个 epoch 开始时记录一次）
     d. 图 (graph)：模型计算图（使用 add_graph，需要示例输入）
  3. 训练结束后，打印 TensorBoard 启动指令
  4. 可选：使用 tensorboard --logdir=logs 启动后截图

建议步骤：
  1. 导入必要库（torch, torch.nn, torch.optim, torchvision, SummaryWriter, DataLoader）
  2. 定义模型和数据加载器
  3. 创建 SummaryWriter 实例
  4. 在第一个 epoch 开始时：
     a. 获取一批训练图片
     b. 使用 make_grid 创建图片网格
     c. writer.add_image("train_samples", grid)
     d. writer.add_graph(model, example_input)
  5. 每个 epoch 结束时：
     a. writer.add_scalar("Loss/train", train_loss, epoch)
     b. writer.add_scalar("Loss/val", val_loss, epoch)
     c. writer.add_scalar("Accuracy/train", train_acc, epoch)
     d. writer.add_scalar("Accuracy/val", val_acc, epoch)
     e. writer.add_scalar("LR", current_lr, epoch)
  6. 每 5 个 epoch 结束时：
     a. 遍历 model.named_parameters()
     b. writer.add_histogram 记录每层权重
  7. 训练结束后调用 writer.close()
  8. 打印 tensorboard --logdir=... 启动命令

提示：
  - SummaryWriter 默认日志目录为 "./runs"
  - torchvision.utils.make_grid 将多张图片拼成一张大图
  - add_graph 需要传递 (model, example_input) 两个参数
  - add_histogram 可以按层名记录权重分布，观察是否出现梯度消失/爆炸
  - 在浏览器中打开 http://localhost:6006 查看 TensorBoard 面板
"""

# TODO: 导入 torch, torch.nn, torch.optim, torchvision, SummaryWriter, DataLoader, tqdm


# TODO: 定义模型类和训练超参数


# TODO: 实现 build_data 函数（返回 train_loader, val_loader）


# TODO: 在主函数中：
#   1. 创建数据加载器
#   2. 初始化模型、损失函数、优化器、调度器
#   3. 创建 SummaryWriter
#   4. 记录初始训练样本图片（使用 make_grid + writer.add_image）
#   5. 记录模型图（writer.add_graph）
#   6. 训练循环（同教材中的结构，但在每个 epoch 添加 scalar 记录）
#   7. 每 5 个 epoch 记录一次模型权重直方图
#   8. 关闭 writer
#   9. 打印启动 TensorBoard 的命令
