"""
章节：第18章 Matplotlib 可视化
题目：画一个柱状图并添加标题和轴标签
类型：基础练习

题目描述：
用以下数据画柱状图：

fruits = ["苹果", "香蕉", "橙子", "葡萄", "西瓜"]
sales = [120, 85, 95, 60, 45]

要求：
- 每根柱子不同颜色
- 添加标题"水果销量"和轴标签
- 在每个柱子上方显示数值
- 图片保存为 "basic_02_fruit_sales.png"

提示：
- 需要设置中文字体：plt.rcParams["font.sans-serif"] = ["SimHei"]
- 用 ax.text() 在柱子上方添加文字
- 用 ax.set_ylim() 给数值标签留出空间
"""

import matplotlib.pyplot as plt
import os

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 设置中文字体：plt.rcParams["font.sans-serif"] = ["SimHei"] 并关闭负号方框
# 2. 用 ax.bar() 画柱状图，每根柱子不同颜色
# 3. 用 ax.text() 在每个柱子上方标注数值
# 4. 设置标题、轴标签，调整 ylim 给标签留空间
# 5. 用 fig.savefig() 保存为 PNG 图片
#
# 提示：
# - Windows 推荐 SimHei 或 Microsoft YaHei 中文字体
# - 用 plt.savefig() / fig.savefig() 保存图片
# - 参考第18章教程中柱状图部分
#
# 完成后运行: python basic_02_柱状图加标题轴标签.py
