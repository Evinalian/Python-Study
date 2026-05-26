"""
章节：第9章 面向对象编程
题目：Person 基类与 Student 子类继承
类型：基础练习

题目描述：
写一个 `Person` 类（有 name 属性），再写一个 `Student` 类继承 `Person`，添加 `student_id` 属性。
要求：
- Person 类有 __init__(self, name) 和 introduce() 方法打印 "我叫 XXX"
- Student 类继承 Person，添加 student_id 属性
- Student 类的 introduce() 覆盖父类方法，打印 "我叫 XXX，学号是 YYY"
- 在 Student.__init__ 中使用 super().__init__(name) 调用父类构造函数

示例输入/输出：
    p = Person("张三")
    p.introduce()             # 我叫 张三

    s = Student("李四", "2024001")
    s.introduce()             # 我叫 李四，学号是 2024001

提示：
- 继承语法：class Student(Person)
- 子类中必须使用 super().__init__(name) 初始化父类属性
"""


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 定义 Person 类（__init__ + introduce）
# 2. 定义 Student 类继承 Person（含 student_id）
# 3. 用 super() 调用父类构造，覆盖 introduce
#
# 提示：参考第9章继承与多态示例
#
# 完成后运行: python basic_03_Person继承Student.py


class Person:
    """人类"""

    def __init__(self, name):
        pass  # TODO: 初始化 name 属性

    def introduce(self):
        pass  # TODO: 实现介绍方法


class Student(Person):
    """学生类：继承 Person，增加学号"""

    def __init__(self, name, student_id):
        pass  # TODO: 用 super() 调用父类构造，初始化 student_id

    def introduce(self):
        """覆盖父类的 introduce，增加学号信息"""
        pass  # TODO: 实现方法覆盖


if __name__ == "__main__":
    pass  # TODO: 编写测试代码
