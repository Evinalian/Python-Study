"""
章节：第19章 Seaborn 统计可视化
题目：画一个箱线图对比不同组的数据分布
类型：基础练习

题目描述：
使用 penguins 数据集（sns.load_dataset("penguins")），按 species 分组
画 flipper_length_mm 的箱线图，观察三种企鹅的鳍肢长度分布差异。
图片保存为 "basic_02_penguin_boxplot.png"。

提示：
- penguins 数据集有 species、flipper_length_mm 等列
- 箱线图显示中位数、四分位数、须线和离群点
- palette 参数可以设置配色方案
"""

import seaborn as sns
import matplotlib.pyplot as plt
import os

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 用 sns.set_theme() 设置主题，用 sns.load_dataset("penguins") 加载数据
# 2. 用 sns.boxplot() 画箱线图，设置 x/y 参数
# 3. 设置标题、轴标签、配色方案
# 4. 扩展：用 hue="sex" 再按性别分组，先 dropna(subset=["sex"]) 去除性别缺失行
# 5. 用 fig.savefig() 保存图片
#
# 提示：
# - 箱线图显示：中位数(线)、四分位数(箱体)、须线、离群点
# - 用 plt.savefig() / fig.savefig() 保存图片
# - 参考第19章教程中箱线图部分
#
# 完成后运行: python basic_02_箱线图对比分组.py
