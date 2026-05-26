"""
章节：第7章 异常处理与上下文管理器
题目：用 contextmanager 实现计时上下文
类型：进阶练习

题目描述：
用 `@contextmanager` 实现一个计时的上下文管理器 `timed_block(name)`，
统计代码块的执行时间。要求：
- 使用 time.perf_counter() 获取高精度时间
- 进入 with 块时记录开始时间，退出时打印耗时
- 即使在 with 块中发生异常，也要保证计时信息被打印

示例输入/输出：
    with timed_block("生成 100 万个平方数"):
        squares = [x ** 2 for x in range(1_000_000)]
    # [生成 100 万个平方数] 耗时: 0.123 秒

    with timed_block("出错的代码"):
        1 / 0
    # [出错的代码] 耗时: 0.001 秒
    # ZeroDivisionError 随后抛出

提示：
- 从 contextlib 导入 contextmanager
- yield 之前的代码在进入 with 时执行
- yield 之后的 finally 代码在退出 with 时执行（无论是否出错）
"""

from contextlib import contextmanager
import time


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 用 @contextmanager 装饰 timed_block 生成器函数
# 2. yield 前记录开始时间，finally 中打印耗时
# 3. 使用 time.perf_counter() 获取高精度时间
#
# 提示：参考第7章 contextmanager 和 finally 示例
#
# 完成后运行: python advanced_02_计时上下文管理器.py


@contextmanager
def timed_block(name="操作"):
    """计时上下文管理器"""
    pass  # TODO: 实现上下文管理器


if __name__ == "__main__":
    pass  # TODO: 编写测试代码
