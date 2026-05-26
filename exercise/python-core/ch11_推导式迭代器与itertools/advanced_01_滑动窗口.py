"""
章节：第11章 推导式、迭代器与 itertools
题目：用 itertools 实现滑动窗口
类型：进阶练习

题目描述：
写一个生成器函数 `sliding_window(seq, n)`，返回长度为 n 的滑动窗口。
例如 `sliding_window([1,2,3,4,5], 3)` 产生 (1,2,3), (2,3,4), (3,4,5)。

要求：惰性求值，不一次性加载所有窗口到内存。

示例输入/输出：
    list(sliding_window([1, 2, 3, 4, 5], 3))
    # [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    list(sliding_window("ABCDEF", 2))
    # [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F')]

提示：
- 使用 itertools.islice 取前 n 个元素作为初始窗口
- 后续每次弹出第一个元素，追加下一个元素
- 如果数据不足 n 个，生成器直接结束
"""

from itertools import islice


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 sliding_window(seq, n) 生成器
# 2. 用 islice 取前 n 个元素作为初始窗口
# 3. 用 pop(0) + append 滑动，yield 每个窗口
#
# 提示：参考第11章 itertools.islice 示例
#
# 完成后运行: python advanced_01_滑动窗口.py


def sliding_window(seq, n):
    """惰性滑动窗口生成器"""
    pass  # TODO: 实现函数体


if __name__ == "__main__":
    pass  # TODO: 编写测试代码
