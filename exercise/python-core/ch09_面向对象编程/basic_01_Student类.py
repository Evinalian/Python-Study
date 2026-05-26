"""
章节：第9章 面向对象编程
题目：Student 类
类型：基础练习

题目描述：
写一个 `Student` 类，包含：
- 属性：姓名（name）、分数（score）
- 方法：`print_info()` 打印 "姓名: XX, 分数: XX"
- 构造函数 `__init__(self, name, score)` 初始化属性

示例输入/输出：
    s = Student("小明", 85)
    s.print_info()
    # 姓名: 小明, 分数: 85

提示：
- 在 __init__ 中通过 self.name 和 self.score 设置实例属性
- print_info 方法中使用 self.name 和 self.score 访问属性
"""


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 定义 Student 类及 __init__ 构造方法
# 2. 实现 print_info() 方法打印学生信息
# 3. 在 if __name__ == "__main__": 块中创建实例测试
#
# 提示：参考第9章类与对象基础示例
#
# 完成后运行: python basic_01_Student类.py


class Student:
    """学生类"""

    def __init__(self, name, score):
        pass  # TODO: 初始化 name 和 score 属性

    def print_info(self):
        """打印学生信息"""
        pass  # TODO: 实现打印方法


if __name__ == "__main__":
    pass  # TODO: 编写测试代码
