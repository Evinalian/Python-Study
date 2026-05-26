"""
章节：第11章 推导式、迭代器与 itertools
题目：用字典推导式把列表转成字典
类型：基础练习

题目描述：
给定一个水果列表，用字典推导式生成 `{水果名: 名称长度}` 的字典。
再进一步，只保留名称长度 >= 5 的水果。

示例输入/输出：
    fruits = ["apple", "banana", "cherry", "date", "elderberry"]

    全部: {'apple': 5, 'banana': 6, 'cherry': 6, 'date': 4, 'elderberry': 10}
    长度 >= 5: {'apple': 5, 'banana': 6, 'cherry': 6, 'elderberry': 10}

提示：
- 字典推导式语法：{key_expr: value_expr for item in iterable}
- 加过滤条件：{k: v for item in iterable if condition}
"""


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 main() 函数
# 2. 用字典推导式生成 {fruit: len(fruit)}
# 3. 添加 if 条件只保留长度 >= 5 的水果
#
# 提示：参考第11章字典推导式示例
#
# 完成后运行: python basic_02_字典推导式.py


def main():
    """字典推导式生成水果名-长度映射"""
    pass  # TODO: 实现函数体


if __name__ == "__main__":
    main()
