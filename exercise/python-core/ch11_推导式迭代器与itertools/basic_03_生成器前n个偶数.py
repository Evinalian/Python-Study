"""
章节：第11章 推导式、迭代器与 itertools
题目：生成器 —— 产生前 n 个偶数
类型：基础练习

题目描述：
写一个生成器函数 `even_numbers(n)`，产生前 n 个偶数（0, 2, 4, ...）。

示例输入/输出：
    for num in even_numbers(5):
        print(num, end=" ")
    # 0 2 4 6 8

    print(list(even_numbers(10)))
    # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

提示：
- 使用 yield 关键字让函数变成生成器
- 生成器每次 yield 一个值后就暂停，等待下一次迭代
"""


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 even_numbers(n) 生成器函数
# 2. 用 yield 依次产出前 n 个偶数
# 3. 用 for 循环和 list() 测试
#
# 提示：参考第11章 yield 生成器示例
#
# 完成后运行: python basic_03_生成器前n个偶数.py


def even_numbers(n):
    """生成前 n 个偶数"""
    pass  # TODO: 实现生成器函数体


if __name__ == "__main__":
    pass  # TODO: 编写测试代码
