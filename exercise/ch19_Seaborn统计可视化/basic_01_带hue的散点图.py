"""
章节：第19章 Seaborn 统计可视化
题目：画一个带 hue 的散点图
类型：基础练习

题目描述：
使用 tips 数据集（sns.load_dataset("tips")），画 x=total_bill、y=tip、
hue=smoker 的散点图，观察吸烟者与不吸烟者的小费行为差异。
图片保存为 "basic_01_tips_scatter_hue.png"。

提示：
- sns.set_theme(style="darkgrid") 设置 Seaborn 主题
- sns.scatterplot() 的 hue 参数自动实现分组着色
- 直接传 DataFrame 的列名，无需手动分组
"""

import seaborn as sns
import matplotlib.pyplot as plt
import os

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 用 sns.set_theme() 设置主题，用 sns.load_dataset("tips") 加载数据
# 2. 用 sns.scatterplot() 画散点图，设置 x/y/hue 参数
# 3. 设置标题和轴标签
# 4. 扩展：用 size 和 style 参数编码更多维度
# 5. 用 fig.savefig() 保存图片
#
# 提示：
# - hue 参数自动实现分组着色和图例
# - 用 plt.savefig() / fig.savefig() 保存图片
# - 参考第19章教程中散点图部分
#
# 完成后运行: python basic_01_带hue的散点图.py
