"""
章节：第12章 内置函数与 functools
题目：用 lru_cache 优化递归
类型：进阶练习

题目描述：
写一个函数 `climb_stairs(n)` 计算爬楼梯的方法数。
爬楼梯规则：每次可以爬 1 级或 2 级台阶。
递推公式：f(1)=1, f(2)=2, f(n)=f(n-1)+f(n-2)。

要求：
1. 使用 @lru_cache 装饰器缓存计算结果
2. 打印缓存统计信息（命中次数、未命中次数）
3. 用 n=50 测试并对比有/无缓存的性能差异

示例输入/输出：
    climb_stairs(50) = 20365011074, 耗时: 0.000123s
    缓存信息: CacheInfo(hits=97, misses=50, maxsize=None, currsize=50)

提示：
- 无缓存的递归复杂度是 O(2^n)，n=50 时几乎不可计算
- @lru_cache(maxsize=None) 表示不限制缓存大小
- cache_info() 方法可查看命中/未命中统计
"""

from functools import lru_cache
import time


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现无缓存版本 climb_stairs_slow(n) 供对比
# 2. 用 @lru_cache 装饰 climb_stairs(n)
# 3. 测试 n=50 并打印 cache_info() 统计
#
# 提示：参考第12章 lru_cache 装饰器示例
#
# 完成后运行: python advanced_01_lru_cache优化递归.py


def climb_stairs_slow(n):
    """无缓存的爬楼梯（仅供对比，不要用大 n 测试）"""
    pass  # TODO: 实现递推公式


@lru_cache(maxsize=None)
def climb_stairs(n):
    """有缓存的爬楼梯 —— 每个 n 只计算一次"""
    pass  # TODO: 实现递推公式


if __name__ == "__main__":
    pass  # TODO: 编写测试代码
