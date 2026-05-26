"""
练习 2-4（进阶）: 图像数据增强管线

任务：
构建一套完整的图像分类数据加载管线，包含训练和验证两条独立的 transform。

要求：
1. 用 torch.randn 生成 500 张 dummy 图像（模拟 32x32 RGB 图像）：
   - shape (500, 3, 32, 32)，值在 [0, 1] 范围
   - 500 个标签为二分类（0 或 1，随机生成）

2. 定义训练 transform：
   - transforms.RandomResizedCrop(28, scale=(0.8, 1.0))
   - transforms.RandomHorizontalFlip(p=0.5)
   - transforms.ColorJitter(brightness=0.2, contrast=0.2)
   - transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

3. 定义验证 transform：
   - transforms.Resize(28) 或 transforms.CenterCrop(28)
   - transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

4. 注意：数据已经是 Tensor (C, H, W) 而非 PIL Image
   - transforms.ToTensor() 不需要，但 Normalize 接受 Tensor 输入
   - RandomResizedCrop/RandomHorizontalFlip/ColorJitter 内部要求是 PIL Image
   - 解决方案：用 transforms.ToPILImage() 转换，或直接用 transforms.Compose 中合适的方式

5. 构建两个 DataLoader（train_loader 和 val_loader），batch_size=32
6. 遍历一个 epoch 的训练集，验证：
   - 每批次的 image shape = (32, 3, 28, 28) 或 (32, 3, 32, 32)
   - image 的均值和标准差大致符合 Normalize 后的分布
7. 打印训练和验证 loader 各取一个 batch 的 image 统计量（均值、最大、最小）

提示：
- 由于数据是 Tensor，不是 PIL Image，对训练集需要先用 ToPILImage 转为 PIL 才能用几何增强
- 另一种方案：直接用 torch.randn 生成 PIL Image（使用 PIL.Image.new）作为数据源
- 推荐在 Dataset.__getitem__ 中做 transform（这是标准做法）
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# TODO: 1. 生成 500 张 dummy 图像 (Tensor, [0, 1] 范围) 和随机标签


# TODO: 2. 定义 DummyImageDataset（将 Tensor 转 PIL 再应用 transform）


# TODO: 3. 定义训练 transform 管线（ToPILImage + 增强 + Normalize）


# TODO: 4. 定义验证 transform 管线（ToPILImage + Resize + Normalize）


# TODO: 5. 构建两个 DataLoader (batch_size=32, shuffle 训练 True 验证 False)


# TODO: 6. 遍历训练和验证 loader 各一个 batch，打印 shapes 和统计量


