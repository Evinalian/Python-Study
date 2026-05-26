"""
章节：第11章 推导式、迭代器与 itertools
题目：用生成器管道处理大文件
类型：进阶练习

题目描述：
写一个生成器管道，逐行读取一个大文件，经过以下处理步骤：
1. 逐行读取文件（去掉末尾换行符）
2. 过滤掉空行和注释行（以 # 开头）
3. 将每行转成大写

使用三个生成器函数串联（管道）完成，内存中始终保持恒定的数据量。

示例输入/输出：
    输入文件 sample.txt:
        hello world
        # 这是一行注释

        python is great
        # 另一行注释
        the end

    输出:
        HELLO WORLD
        PYTHON IS GREAT
        THE END

提示：
- 每个生成器只做一件事（单一职责）
- 管道组合方式：transform_lines(filter_lines(read_lines(filepath)))
- 适合处理超大文件，因为内存中永远只存当前行
"""


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 read_lines(filepath) 逐行读取并去换行符
# 2. 实现 filter_lines(lines) 过滤空行和注释
# 3. 实现 transform_lines(lines) 转大写，管道串联
#
# 提示：参考第11章生成器管道示例
#
# 完成后运行: python advanced_02_生成器管道处理大文件.py


def read_lines(filepath):
    """生成器 1：逐行读取文件"""
    pass  # TODO: 实现函数体


def filter_lines(lines):
    """生成器 2：过滤空行和注释行"""
    pass  # TODO: 实现函数体


def transform_lines(lines):
    """生成器 3：转大写"""
    pass  # TODO: 实现函数体


if __name__ == "__main__":
    pass  # TODO: 编写测试代码
