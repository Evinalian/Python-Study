"""
章节：第8章 文件读写与序列化
题目：统计文件的行数和字符数
类型：基础练习

题目描述：
写一个函数 `count_file(filepath)`，返回一个文本文件的总行数和总字符数。
注意：使用逐行读取的方式，内存友好。

示例输入/输出：
    假设 test_count.txt 内容为:
        hello
        world
        python

    lines, chars = count_file("test_count.txt")
    print(f"行数: {lines}, 字符数: {chars}")
    # 行数: 3, 字符数: 17

提示：
- 使用 for line in f 逐行读取（内存友好）
- len(line) 计算每行的字符数（包含换行符）
"""

from pathlib import Path


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 count_file(filepath) 函数
# 2. 用 with open + for line in f 逐行读取
# 3. 累计行数和每行的字符数，返回二元组
#
# 提示：参考第8章文件读取基础示例
#
# 完成后运行: python basic_01_统计文件行数.py


def count_file(filepath):
    """统计文本文件的行数和字符数"""
    pass  # TODO: 实现函数体


if __name__ == "__main__":
    pass  # TODO: 编写测试代码
