"""
章节：第6章 函数
题目：判断素数
类型：基础练习

题目描述：
写一个函数 is_prime(n) 判断一个正整数是否为素数。

示例输入/输出：
is_prime(1) → False
is_prime(2) → True
is_prime(3) → True
is_prime(4) → False
is_prime(17) → True
is_prime(100) → False

提示：
- 小于 2 的数不是素数
- 只需检查 2 到 sqrt(n) 之间的数是否能整除 n
- 使用 int(n ** 0.5) 或 math.isqrt(n) 计算平方根
- 找到任意一个因数即可返回 False，循环结束未找到则为素数
"""

import math

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 is_prime(n) 函数
# 2. n < 2 直接返回 False
# 3. 遍历 2 到 int(n**0.5)+1，找到因数返回 False
#
# 提示：
# - 使用 math.isqrt(n) 可替代 int(n**0.5)
#
# 完成后，运行 python basic_01_判断素数.py 测试你的代码。


def is_prime(n):
    """判断 n 是否为素数"""
    pass  # TODO: 实现此函数
