"""
练习 7-2：训练循环中的设备管理

目标：
  实现一个健壮的训练函数，自动检测可用设备（CUDA/MPS/CPU），
  确保模型、数据、损失函数都在同一设备上，并处理设备不可用时的降级逻辑。

要求：
  1. 实现 get_device() 函数：
     - 优先级: CUDA > MPS (macOS) > CPU
     - 打印设备名称和显存信息（如果可用）
     - 返回 torch.device 对象
  2. 实现 to_device() 函数：
     - 接受 model, criterion, optimizer
     - 将 model 和 criterion 移到指定 device
     - 提示：optimizer 中的参数已经随 model 移动，不需要单独处理
     - 返回移动后的对象
  3. 实现 check_device_consistency() 函数：
     - 检查 model 参数、输入数据、标签是否都在同一 device
     - 如果不一致，打印警告信息
  4. 实现 train_with_device_management() 函数：
     - 使用 get_device() 获取设备
     - 使用 to_device() 移动模型
     - 训练循环中每次获取 batch 后调用 check_device_consistency()
     - 在 CUDA 可用时启用 AMP 训练
     - CUDA 不可用时使用标准 FP32 训练
     - 打印每轮训练的设备和显存信息

建议步骤：
  1. 导入 torch, torch.nn, torch.optim
  2. 实现 get_device 函数
  3. 实现 to_device 函数
  4. 实现 check_device_consistency 函数
  5. 实现 train_with_device_management 函数（包含完整的训练循环）
  6. 主函数中调用

提示：
  - torch.cuda.is_available() 检查 CUDA
  - torch.backends.mps.is_available() 检查 MPS (macOS Apple Silicon)
  - 使用 next(model.parameters()).device 获取模型所在的 device
  - optimizer 不需要显式移到 GPU（它的参数引用 model 的参数）
"""

# TODO: 导入 torch, torch.nn, torch.optim, torch.utils.data


# TODO: 实现 get_device() 函数


# TODO: 实现 to_device(model, criterion, device) 函数


# TODO: 实现 check_device_consistency(model, inputs, labels) 函数


# TODO: 定义简单模型 SimpleMLP


# TODO: 实现 train_with_device_management(num_epochs=3) 函数
#   1. 获取设备
#   2. 准备数据（模拟数据，注意移到正确的设备）
#   3. 初始化模型、损失函数
#   4. 移动模型和损失函数到设备
#   5. 训练循环：每个 batch 检查设备一致性
#   6. 打印设备和显存信息


# TODO: 主函数入口
