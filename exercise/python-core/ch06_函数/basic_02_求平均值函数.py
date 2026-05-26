"""
章节：第6章 函数
题目：求任意多个数的平均值
类型：基础练习

题目描述：
写一个函数 average(*args)，接收任意多个数，返回它们的平均值。
如果没有参数，返回 None。

示例输入/输出：
average(1, 2, 3, 4, 5) → 3.0
average(10, 20) → 15.0
average() → None

提示：
- 使用 *args 接收任意数量的位置参数
- 用 if not args 判断是否为空
- sum(args) / len(args) 计算平均值
- 注意：整数除法 / 在 Python 3 中自动返回 float
"""

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 average(*args) 函数
# 2. 无参数时返回 None
# 3. 用 sum(args) / len(args) 计算平均值
#
# 提示：
# - if not args 判断参数为空
# - / 在 Python 3 中自动返回 float
#
# 完成后，运行 python basic_02_求平均值函数.py 测试你的代码。


def average(*args):
    """计算任意多个数的平均值"""
    pass  # TODO: 实现此函数
