"""
练习 7-3：混淆矩阵绘制

目标：
  给定模型预测值和真实标签（可以用模拟数据），绘制混淆矩阵，
  并用 seaborn 热力图展示每个类别的分类准确情况。

要求：
  1. 生成模拟数据：
     - 假设 10 个类别，200 个样本
     - 用 np.random 生成真实的标签
     - 根据真实标签生成"带噪声"的预测标签（模拟模型预测）
     - 控制一个合理的准确率（如 70-80%），使混淆矩阵有区分度
  2. 使用 sklearn.metrics.confusion_matrix 计算混淆矩阵
  3. 使用 seaborn.heatmap 绘制混淆矩阵：
     - 每个格子标注数值
     - x 轴为预测类别，y 轴为真实类别
     - 使用有意义的类别名
  4. 在混淆矩阵旁边或下方标注：
     - 总体准确率
     - 每个类别的召回率（对角线除以行和）
  5. 保存图片为 PNG 文件

建议步骤：
  1. 导入 numpy, sklearn.metrics, matplotlib, seaborn
  2. 定义类别名称（使用 CIFAR-10 的 10 个类别名）
  3. 生成模拟的真实标签和预测标签
  4. 计算混淆矩阵
  5. 绘制热力图（带数值标注）
  6. 计算并标注每个类别的召回率
  7. 保存并显示图片

提示：
  - 用 np.random.seed(42) 确保结果可复现
  - sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
  - 用 plt.text 在每个格子中额外标注召回率
  - 对角线单元格可以高亮显示
"""

# TODO: 导入 numpy as np, sklearn.metrics (confusion_matrix), matplotlib.pyplot, seaborn as sns


# TODO: 定义类别名称 CLASS_NAMES = ["airplane", ...]


# TODO: 设置随机种子 np.random.seed(42)


# TODO: 生成模拟数据：
#   - y_true: 0-9 随机整数 200 个
#   - y_pred: 基于 y_true 添加噪声（例如 70% 正确，30% 随机错误）


# TODO: 计算混淆矩阵


# TODO: 使用 matplotlib + seaborn 绘制混淆矩阵热力图


# TODO: 计算并打印总体准确率和各类别召回率


# TODO: 保存图片（plt.savefig）
