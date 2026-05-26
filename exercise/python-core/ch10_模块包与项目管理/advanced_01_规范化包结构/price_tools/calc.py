"""
price_tools 包：价格计算模块

提供折扣计算、含税计算、商品总价计算等功能。
"""

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 apply_discount(price, discount) 折扣计算
# 2. 实现 apply_tax(price, tax_rate) 含税计算
# 3. 实现 total_price(items) 商品总价计算
#
# 提示：参考第10章包与模块示例
#


def apply_discount(price, discount):
    """计算折扣后价格

    参数:
        price: 原价
        discount: 折扣率（如 0.1 表示打 9 折）
    返回:
        折扣后价格
    """
    pass  # TODO: 实现函数体


def apply_tax(price, tax_rate):
    """计算含税价格

    参数:
        price: 原价
        tax_rate: 税率（如 0.13 表示 13%）
    返回:
        含税价格
    """
    pass  # TODO: 实现函数体


def total_price(items):
    """计算商品总价

    参数:
        items: 商品列表，每项为 {"price": 单价, "quantity": 数量} 的字典
    返回:
        所有商品的总价
    """
    pass  # TODO: 实现函数体
