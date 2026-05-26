"""
章节：第12章 内置函数与 functools
题目：用 any / all 判断列表条件
类型：基础练习

题目描述：
对分数列表，用 any 和 all 配合生成器表达式判断以下条件：
1. 是否全部及格（分数 >= 60）
2. 是否存在优秀（分数 >= 90）
3. 是否存在不及格（分数 < 60）

示例输入/输出：
    scores = [85, 92, 78, 55, 88, 95, 60]

    全部及格? False
    存在优秀? True
    存在不及格? True

提示：
- all(condition for item in seq)：所有元素都满足条件返回 True
- any(condition for item in seq)：至少一个元素满足条件返回 True
- 配合生成器表达式使用（不需要加额外的括号）
"""


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 main() 函数
# 2. 用 all() 判断是否全部及格
# 3. 用 any() 判断是否存在优秀 / 不及格
#
# 提示：参考第12章 any/all 函数示例
#
# 完成后运行: python basic_04_any_all条件判断.py


def main():
    """用 any/all 判断分数条件"""
    pass  # TODO: 实现函数体


if __name__ == "__main__":
    main()
