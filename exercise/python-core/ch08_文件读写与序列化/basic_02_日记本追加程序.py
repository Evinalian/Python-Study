"""
章节：第8章 文件读写与序列化
题目：日记本追加程序
类型：基础练习

题目描述：
写一个函数 `write_diary(entry)`，把日记内容追加到一个文件中。
每次调用都在内容前自动加上时间戳，格式为 [YYYY-MM-DD HH:MM:SS]。

示例输入/输出：
    write_diary("今天开始学 Python 文件操作。")
    write_diary("学会了用 with open 读写文件。")

    diary.txt 内容:
        [2024-01-15 14:30:00] 今天开始学 Python 文件操作。
        [2024-01-15 14:30:05] 学会了用 with open 读写文件。

提示：
- 使用追加模式 "a" 打开文件（不覆盖已有内容）
- 使用 datetime.now().strftime() 生成时间戳
"""

from datetime import datetime


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 write_diary(entry) 函数
# 2. 用 datetime.now().strftime() 生成时间戳
# 3. 用追加模式 "a" 打开文件写入
#
# 提示：参考第8章文件追加写入示例
#
# 完成后运行: python basic_02_日记本追加程序.py


def write_diary(entry):
    """追加一条日记到 diary.txt"""
    pass  # TODO: 实现函数体


if __name__ == "__main__":
    pass  # TODO: 编写测试代码
