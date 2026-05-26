"""
章节：第9章 面向对象编程
题目：Rectangle 类
类型：基础练习

题目描述：
写一个 `Rectangle` 类，包含：
- 属性：宽（width）和高（height），通过构造函数传入
- 方法 `area()`：计算面积（宽 x 高）
- 方法 `perimeter()`：计算周长（2 x (宽 + 高)）

示例输入/输出：
    r = Rectangle(10, 5)
    print(f"面积: {r.area()}")        # 50
    print(f"周长: {r.perimeter()}")   # 30

提示：
- area() 和 perimeter() 都是实例方法，第一个参数是 self
"""


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 定义 Rectangle 类及 __init__ 构造方法
# 2. 实现 area() 和 perimeter() 方法
# 3. 在 if __name__ == "__main__": 块中创建实例测试
#
# 提示：参考第9章实例方法示例
#
# 完成后运行: python basic_02_Rectangle类.py


class Rectangle:
    """矩形类"""

    def __init__(self, width, height):
        pass  # TODO: 初始化 width 和 height 属性

    def area(self):
        """计算面积"""
        pass  # TODO: 实现面积计算

    def perimeter(self):
        """计算周长"""
        pass  # TODO: 实现周长计算


if __name__ == "__main__":
    pass  # TODO: 编写测试代码
