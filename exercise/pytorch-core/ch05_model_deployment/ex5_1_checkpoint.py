"""
练习 5-1：Checkpoint 保存与恢复

目标：
  训练一个简单的 MLP 分类器 10 个 epoch，每个 epoch 保存 checkpoint。
  然后模拟"训练中断"，从第 5 个 epoch 的 checkpoint 恢复并继续训练到第 10 个 epoch。
  验证恢复后的最终 loss 与连续训练一致。

要求：
  1. 定义 MLP 模型：input_dim=20, hidden_dim=64, num_classes=3，含 Dropout
  2. 使用模拟数据（torch.randn 生成特征, torch.randint 生成标签）
  3. 每个 epoch 保存 checkpoint（含 epoch, model_state_dict, optimizer_state_dict, loss）
  4. 实现从任意 checkpoint 恢复训练的函数
  5. 比较"连续训练 10 epoch"和"训练 5 epoch → 中断 → 从 epoch 5 恢复 → 再训练 5 epoch"
     最终 loss 的差异（应非常接近，差异来自 Dropout 的随机性）

建议步骤：
  1. 定义模型类
  2. 准备数据（固定随机种子以保证可复现）
  3. 实现训练函数 train_one_epoch
  4. 实现保存 checkpoint 的函数
  5. 实现从 checkpoint 恢复的函数
  6. 先连续训练 10 epoch，记录 loss 曲线
  7. 再从头训练 5 epoch，保存 checkpoint，然后"恢复"训练剩余 5 epoch，记录 loss 曲线
  8. 对比两条 loss 曲线（考虑 Dropout 导致的随机性差异）

提示：
  - 使用 torch.manual_seed 固定种子
  - 恢复训练时注意 map_location
  - 使用 model.eval() 或 model.train() 控制 Dropout
"""

# TODO: 导入必要的库（torch, torch.nn, torch.optim, DataLoader, TensorDataset, os）


# TODO: 定义 MLP 模型类 SimpleMLP


# TODO: 实现 train_one_epoch 函数（返回平均 loss）


# TODO: 实现 save_checkpoint 函数（参数：filepath, epoch, model, optimizer, loss）


# TODO: 实现 load_checkpoint 函数（参数：filepath, model, optimizer, device，返回 start_epoch）


# TODO: 准备数据（X_train, y_train, 随机种子 42）


# TODO: 连续训练 10 epoch，每 epoch 记录 loss，保存为 loss_continuous 列表


# TODO: 重新初始化模型和优化器（相同种子），训练 5 epoch


# TODO: 保存 epoch 5 的 checkpoint


# TODO: 模拟中断：重新初始化模型和优化器（相同种子），从 epoch 5 的 checkpoint 恢复


# TODO: 继续训练到 epoch 10，记录 loss，保存为 loss_resumed 列表


# TODO: 打印两组 loss 对比（使用 enumerate 遍历 epoch）
