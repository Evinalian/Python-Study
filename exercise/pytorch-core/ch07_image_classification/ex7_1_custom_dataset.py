"""
练习 7-1：自定义 Dataset 与数据增强实验

目标：
  实现一个自定义 ImageFolderDataset，遍历本地目录加载图片。
  用 matplotlib 展示不同数据增强对同一张图片的效果。

要求：
  1. 实现 CustomImageFolderDataset 类：
     - __init__(root_dir, transform=None): 遍历 root_dir 下的子目录作为类别
     - __len__(): 返回样本总数
     - __getitem__(idx): 返回 (image, label)，并自动打开 PIL Image
     - get_class_names(): 返回类别名列表
     - 使用 try/except 处理损坏的图片文件
  2. 使用 torchvision 的 transforms 定义多种增强方式：
     - 原图（仅 ToTensor）
     - RandomCrop
     - RandomHorizontalFlip
     - ColorJitter
     - RandomRotation
     - RandomAffine
     - 组合增强（Compose 多个）
  3. 选取一张图片，分别应用上述每种增强，使用 matplotlib 在一个 2x4 的
     subplot 中展示所有效果
  4. 如果没有本地图片数据集，可以从网上下载一张示例图片或从 CIFAR-10 中
     提取一张图片作为演示

建议步骤：
  1. 导入必要库（torch, torchvision.transforms, PIL, matplotlib, os）
  2. 实现 CustomImageFolderDataset 类
  3. 准备一张或多张测试图片
  4. 定义各种增强变换
  5. 用 matplotlib subplot 并排展示增强效果
  6. 为每张子图添加标题说明所用的增强方式

提示：
  - 如果没有本地目录，可以先手动创建目录结构和示例图片
  - 使用 image.show() 可以在调试时快速查看
  - transforms.ToPILImage() 可以将 tensor 转回 PIL Image（用于显示）
  - 显示 tensor 图片时需要先 denormalize 并 clamp 到 [0, 1]
"""

# TODO: 导入 torch, torchvision.transforms, PIL.Image, matplotlib.pyplot, os, random


# TODO: 实现 CustomImageFolderDataset 类
#   - __init__(root_dir, transform=None)
#   - __len__()
#   - __getitem__(idx)
#   - get_class_names()


# TODO: 准备测试图片（如果没有本地数据集，用 CIFAR-10 或创建示例图片）


# TODO: 定义一个工具函数：将 tensor 转为可显示的 numpy 数组（denormalize）


# TODO: 定义各种增强变换列表（名称, transform 的元组列表）


# TODO: 使用 matplotlib 在 2x4 的 subplot 中展示所有增强效果


# TODO: 保存图片到文件（可选）
