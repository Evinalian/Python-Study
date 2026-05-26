"""
章节：第18章 Matplotlib 可视化
题目：画散点图，用不同颜色区分两组数据
类型：基础练习

题目描述：
生成两组随机数（每组 50 个点），A 组以 (2, 2) 为中心，B 组以 (6, 6) 为中心。
用蓝色和红色分别画出两组散点，添加图例。
图片保存为 "basic_03_scatter_groups.png"。

提示：
- A 组：rng.normal(2, 0.8, 50) 生成 x 和 y 坐标
- B 组：rng.normal(6, 0.8, 50)
- s 参数控制点的大小，alpha 参数控制透明度
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 用 rng.normal() 生成 A 组 (2,2) 和 B 组 (6,6) 的随机坐标
# 2. 用 ax.scatter() 分别画两组散点，不同颜色+图例标签
# 3. 设置标题、轴标签、图例、网格
# 4. 用 fig.savefig() 保存为 PNG 图片
#
# 提示：
# - scatter() 参数: color, alpha, s (大小), edgecolors
# - 用 plt.savefig() / fig.savefig() 保存图片
# - 参考第18章教程中散点图部分
#
# 完成后运行: python basic_03_散点图区分两组数据.py
