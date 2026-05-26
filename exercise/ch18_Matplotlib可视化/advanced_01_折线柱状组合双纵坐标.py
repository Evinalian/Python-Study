"""
章节：第18章 Matplotlib 可视化
题目：折线图 + 柱状图组合图（双纵坐标）
类型：进阶练习

题目描述：
模拟一份月度销售数据，用柱状图表示每月销量（左轴），用折线图表示累计销量（右轴），
展示在一个图中。

数据：
- months = 1 到 12
- monthly_sales = [120, 135, 148, 162, 155, 178, 190, 185, 200, 210, 195, 220]
- 累计销量需要用 np.cumsum() 计算

要求：
- 柱状图（左轴）、折线图（右轴）
- 双 y 轴标签用不同颜色区分
- 添加标题和图例
- 图片保存为 "advanced_01_dual_axis.png"

提示：
- 使用 ax1.twinx() 创建共享 x 轴的第二个 y 轴
- 合并两个轴的图例：ax1.get_legend_handles_labels()
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 设置中文字体，准备 month 和 monthly_sales 数据，用 np.cumsum() 计算累计销量
# 2. 创建 fig, ax1，用 ax1.bar() 画柱状图（左轴）
# 3. 用 ax1.twinx() 创建 ax2，用 ax2.plot() 画折线图（右轴）
# 4. 给两个轴分别设置不同颜色的标签，合并图例
# 5. 在柱子和数据点上标注数值，用 fig.savefig() 保存图片
#
# 提示：
# - twinx() 共享 x 轴但创建独立的 y 轴
# - 用 plt.savefig() / fig.savefig() 保存图片
# - 参考第18章教程中双纵坐标部分
#
# 完成后运行: python advanced_01_折线柱状组合双纵坐标.py
