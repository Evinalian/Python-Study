"""
章节：第19章 Seaborn 统计可视化
题目：画一个相关系数热力图
类型：基础练习

题目描述：
使用 penguins 数据集的数值列，计算相关系数矩阵并画出带数值标注的热力图。
图片保存为 "basic_03_corr_heatmap.png"。

要求：
- 用 select_dtypes("number") 选择数值列
- 用 .corr() 计算相关系数矩阵
- 热力图显示数值标注（annot=True），保留 2 位小数
- 使用 RdBu_r 配色方案，center=0

提示：
- 热力图用 sns.heatmap()，最常用参数：annot, fmt, cmap, center, vmin, vmax
- 正相关（红色），负相关（蓝色），对角线永远是 1.0
"""

import seaborn as sns
import matplotlib.pyplot as plt
import os

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 用 sns.load_dataset("penguins") 加载数据，用 select_dtypes("number") 选数值列
# 2. 用 .corr() 计算相关系数矩阵
# 3. 用 sns.heatmap() 画热力图：annot=True, fmt=".2f", cmap="RdBu_r", center=0
# 4. 设置标题，调整标签旋转角度
# 5. 分析：找出与 body_mass_g 最相关的特征
# 6. 用 fig.savefig() 保存图片
#
# 提示：
# - RdBu_r: 红色=正相关, 蓝色=负相关, center=0 使 0 为白色
# - 用 plt.savefig() / fig.savefig() 保存图片
# - 参考第19章教程中热力图部分
#
# 完成后运行: python basic_03_相关系数热力图.py
