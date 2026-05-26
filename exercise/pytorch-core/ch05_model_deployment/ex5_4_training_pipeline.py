"""
练习 5-4（进阶）：完整的训练管理系统

目标：
  实现一个完整的训练管理系统，集成 checkpoint 保存/恢复、早停、最佳模型管理。

要求：
  1. 定义 CNN 模型用于 MNIST 分类（或模拟数据）：
     - Conv2d -> ReLU -> MaxPool -> Conv2d -> ReLU -> MaxPool -> Linear -> Dropout -> Linear
  2. 实现完整的训练流程：
     a. 初始化模型、优化器（Adam）、学习率调度器（ReduceLROnPlateau）
     b. 每个 epoch 训练并验证
     c. 每个 epoch 保存 checkpoint（含 epoch, model_state_dict, optimizer_state_dict,
        scheduler_state_dict, train_loss, val_loss, val_acc）
     d. 验证集准确率提升时保存最佳模型
     e. 早停机制：patience=5，即连续 5 个 epoch 验证 loss 不下降则停止
     f. 支持 --resume 参数：如果 checkpoint 存在则自动恢复
  3. 训练结束后：
     a. 加载最佳模型
     b. 在测试集上评估
     c. 打印最终结果

建议步骤：
  1. 定义 CNN 模型类
  2. 准备数据（建议使用 MNIST 数据集，如果下载困难则使用模拟数据）
  3. 实现 train_epoch 函数
  4. 实现 validate 函数（返回 loss 和 accuracy）
  5. 实现 save_checkpoint 函数
  6. 实现 load_checkpoint 函数
  7. 实现 EarlyStopping 类
  8. 编写主训练循环：
     - 检查是否存在 checkpoint → 恢复
     - 每个 epoch：训练 → 验证 → 保存 checkpoint → 更新调度器 → 检查早停
     - 如果验证准确率提升 → 保存最佳模型
  9. 加载最佳模型并在测试集评估

提示：
  - EarlyStopping 类需要记录最佳 loss、连续不下降的 epoch 数
  - ReduceLROnPlateau 在验证 loss 不下降时自动降低学习率
  - 使用 tqdm 显示训练进度（可选）
  - 注意 model.train() / model.eval() 的切换
"""

# TODO: 导入 torch, torch.nn, torch.optim, torch.nn.functional as F


# TODO: 导入 DataLoader, TensorDataset（如果使用模拟数据）或 torchvision.datasets.MNIST


# TODO: 定义 CNN 模型类（2 层卷积 + 2 层全连接）


# TODO: 定义早停类 EarlyStopping（属性：patience, best_loss, counter, early_stop）


# TODO: 实现 train_epoch 函数


# TODO: 实现 validate 函数（返回 loss, accuracy）


# TODO: 实现 save_checkpoint 函数


# TODO: 实现 load_checkpoint 函数


# TODO: 主训练逻辑：
#   1. 超参数设置（lr=0.001, epochs=50, patience=5）
#   2. 初始化模型、优化器、调度器、损失函数
#   3. 检查并恢复 checkpoint
#   4. 训练循环（每个 epoch 训练/验证/保存/早停判断）
#   5. 加载最佳模型
#   6. 测试集评估
#   7. 打印最终结果
