"""
章节：第19章 Seaborn 统计可视化
题目：用 pairplot 探索鸢尾花数据集
类型：进阶练习

题目描述：
Iris（鸢尾花）是机器学习中最经典的数据集之一。请使用
sns.load_dataset("iris") 画 pairplot，按 species（三种不同的鸢尾花）着色。

观察并回答：
1. 哪些特征对区分三种鸢尾花最有帮助？
2. 哪一种鸢尾花和其他两种最容易区分？

图片保存为 "advanced_01_iris_pairplot.png"。

提示：
- pairplot 生成所有数值列两两配对的散点图矩阵
- hue="species" 实现分组着色
- diag_kind="kde" 在对角线上显示密度曲线
- 对于大数据集，建议先抽样：df.sample(2000)
"""

import seaborn as sns
import matplotlib.pyplot as plt
import os

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 设置中文字体和 Seaborn 主题
# 2. 用 sns.load_dataset("iris") 加载数据，查看数据概况
# 3. 用 sns.pairplot() 画 pairplot：hue="species", diag_kind="kde"
# 4. 设置总标题，用 g.savefig() 保存图片（pairplot 返回 PairGrid 对象）
# 5. 观察分析：哪些特征区分度最高？哪种花最易区分？用注释写下结论
#
# 提示：
# - pairplot 的对角线可显示直方图(hist)或密度曲线(kde)
# - 用 plt.savefig() / g.savefig() 保存图片
# - 参考第19章教程中 pairplot 部分
#
# 完成后运行: python advanced_01_pairplot探索鸢尾花.py
