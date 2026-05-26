"""
章节：第6章 函数
题目：计时装饰器与生成器
类型：进阶练习

题目描述：
本练习包含两个任务：

任务一：计时装饰器
写一个装饰器 timer，它能打印被装饰函数的执行时间（秒）。
要求：
- 使用 functools.wraps 保留原函数的元信息（__name__、__doc__ 等）
- 使用 time.perf_counter() 进行高精度计时

任务二：斐波那契生成器
写一个生成器函数 fibonacci_gen()，每次 yield 下一个斐波那契数，无限生成。

示例用法：
    @timer
    def slow_sum(n):
        return sum(range(n))

    slow_sum(10_000_000)
    # 输出类似：slow_sum 执行时间：0.1234 秒

    gen = fibonacci_gen()
    for i, num in enumerate(gen, start=1):
        print(f"F({i}) = {num}")
        if i >= 10:
            break
    # F(1) = 1, F(2) = 1, F(3) = 2, F(4) = 3, ...

提示：
- 装饰器本质是"接受函数、返回新函数的高阶函数"
- @timer 等价于 func = timer(func)
- 生成器使用 yield 语句暂停并返回值，下次调用 next() 时从暂停处继续
- 斐波那契数列：1, 1, 2, 3, 5, 8, 13, ...（每一项都是前两项之和）
"""

import time
from functools import wraps

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 timer 装饰器，用 @wraps 保留元信息
# 2. 使用 time.perf_counter() 计算函数耗时
# 3. 实现 fibonacci_gen() 生成器，无限 yield 斐波那契数
#
# 提示：
# - 装饰器返回 wrapper(*args, **kwargs) 内部函数
# - 生成器用 a, b = b, a + b 更新斐波那契数列
#
# 完成后，运行 python advanced_01_计时装饰器与生成器.py 测试你的代码。


def timer(func):
    """装饰器：打印函数执行所花费的时间"""
    pass  # TODO: 实现装饰器


def fibonacci_gen():
    """无限生成斐波那契数列：1, 1, 2, 3, 5, 8, 13, ..."""
    pass  # TODO: 实现生成器
