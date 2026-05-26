"""
练习 7-4（进阶）：换一个数据集复现完整流程

目标：
  将教材中 CIFAR-10 的完整流程改为 CIFAR-100（100 类），
  调整模型输出维度，复现完整的训练/验证/测试/推理流程。

CIFAR-100 信息：
  - 100 个类别，每个类别 600 张图片（500 训练 + 100 测试）
  - 图片尺寸与 CIFAR-10 相同（32x32x3）
  - 类别更细粒度（如：aquatic mammals, fish, flowers, trees 等超类）

要求：
  1. 修改模型输出维度 num_classes=100
  2. 由于类别更多，适当增加模型容量（如增加通道数或层数）
  3. 实现与教材中结构等价的训练流程：
     a. 数据加载（torchvision.datasets.CIFAR100）
     b. 模型定义（SimpleResNet，调整 num_classes）
     c. 训练循环（含 tqdm 进度条）
     d. 验证与早停
     e. 测试评估（Top-1 和 Top-5 准确率）
     f. 单张图片推理
  4. 由于 100 类更难，建议训练至少 30 个 epoch
  5. 打印 Top-1 和 Top-5 准确率

建议步骤：
  1. 导入必要的库
  2. 定义配置类（同教材中的 Config，但 DATASET_NAME="CIFAR100", NUM_CLASSES=100）
  3. 实现数据加载函数（使用 torchvision.datasets.CIFAR100）
  4. 定义模型（增加通道数: 64→128→256→512，或增加 ResidualBlock 数量）
  5. 实现训练函数
  6. 实现验证函数
  7. 实现测试函数（计算 Top-1 和 Top-5 准确率）
  8. 实现推理函数（Top-5 预测结果）
  9. 主函数：依次执行训练、测试、推理

提示：
  - CIFAR100 的归一化参数: mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
  - Top-5 准确率：真实标签出现在预测概率最高的 5 个类别中即算正确
  - 可以用 torch.topk(outputs, 5, dim=1) 获取 Top-5 预测
  - 如果训练太慢，可以将 NUM_EPOCHS 设小一些（如 20），batch_size 适当减小
  - 由于 100 类，单纯增大模型会显著增加内存使用，请酌情调整
"""

# TODO: 导入 torch, torch.nn, torch.optim, torch.nn.functional as F
# TODO: 导入 torchvision, torchvision.transforms, DataLoader
# TODO: 导入 tqdm, time, os


# TODO: 定义 Config 类（NUM_CLASSES=100, NUM_EPOCHS=30 等）


# TODO: 实现 build_transforms(is_train)


# TODO: 实现 build_dataloaders(config)（使用 CIFAR100）


# TODO: 定义 ResidualBlock 和 SimpleResNet（num_classes=100, 适当增大容量）


# TODO: 实现 train_one_epoch 函数


# TODO: 实现 validate 函数


# TODO: 实现 compute_topk_accuracy 函数（计算 Top-1 和 Top-5 准确率）


# TODO: 实现 predict_single_image 函数


# TODO: 主函数：
#   1. 加载数据
#   2. 创建模型
#   3. 训练循环
#   4. 测试（Top-1 和 Top-5）
#   5. 推理示例
