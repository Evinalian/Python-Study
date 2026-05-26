"""
章节：第12章 内置函数与 functools
题目：用 sorted 对列表按不同规则排序
类型：基础练习

题目描述：
有如下字符串列表，分别按以下三种规则排序并打印结果：
1. 按字符串长度排序（使用 key=len）
2. 按最后一个字母排序（使用 key=lambda w: w[-1]）
3. 按字母倒序排列（使用 reverse=True）

示例输入/输出：
    words = ["banana", "apple", "cherry", "date", "elderberry", "fig"]

    按长度: ['fig', 'date', 'apple', 'banana', 'cherry', 'elderberry']
    按最后一个字母: ['banana', 'apple', 'date', 'elderberry', 'fig', 'cherry']
    按字母倒序: ['fig', 'elderberry', 'date', 'cherry', 'banana', 'apple']

提示：
- sorted() 返回新列表，原列表不变
- key 参数接收一个函数，对每个元素调用该函数后按返回值排序
- lambda 是匿名函数：lambda 参数: 返回值
"""


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 main() 函数
# 2. 分别用 key=len、lambda、reverse 三种方式排序
# 3. 打印每种排序的结果
#
# 提示：参考第12章 sorted() 函数示例
#
# 完成后运行: python basic_01_sorted多规则排序.py


def main():
    """三种排序规则演示"""
    pass  # TODO: 实现函数体


if __name__ == "__main__":
    main()
