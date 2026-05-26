"""
章节：第9章 面向对象编程
题目：BankAccount 银行账户类
类型：进阶练习

题目描述：
实现一个 `BankAccount` 银行账户类。功能要求：
- 创建账户时设定户名（owner）和初始余额（balance，默认为 0）
- `deposit(amount)` 存钱，金额必须为正数，打印操作结果并返回当前余额
- `withdraw(amount)` 取钱，余额不足时提示"余额不足"且不扣款，返回当前余额
- `get_balance()` 查看余额
- `add_interest(rate)` 按利率增加余额（如 rate=0.03 表示 3%）

示例输入/输出：
    account = BankAccount("小明", 1000)
    account.deposit(500)          # 存入 500 元，当前余额: 1500 元
    account.withdraw(200)         # 取出 200 元，当前余额: 1300 元
    account.withdraw(5000)        # 余额不足！当前余额 1300 元，无法取出 5000 元
    account.add_interest(0.03)    # 利息 39.00 元（利率 3.0%），当前余额: 1339.00 元
    print(f"最终余额: {account.get_balance():.2f} 元")

提示：
- 余额用 _balance 命名（单下划线前缀暗示"内部使用"）
- 金额校验防止负数存款/取款
"""


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 __init__（owner, balance=0）
# 2. 实现 deposit（金额校验）和 withdraw（余额校验）
# 3. 实现 get_balance 和 add_interest
#
# 提示：参考第9章封装与属性保护示例
#
# 完成后运行: python advanced_01_BankAccount类.py


class BankAccount:
    """银行账户类"""

    def __init__(self, owner, balance=0):
        pass  # TODO: 初始化 owner 和 _balance

    def deposit(self, amount):
        """存钱"""
        pass  # TODO: 实现存款逻辑（金额校验）

    def withdraw(self, amount):
        """取钱"""
        pass  # TODO: 实现取款逻辑（余额校验）

    def get_balance(self):
        """查看余额"""
        pass  # TODO: 返回余额

    def add_interest(self, rate):
        """计算利息：rate 是利率（如 0.03 表示 3%）"""
        pass  # TODO: 实现利息计算


if __name__ == "__main__":
    pass  # TODO: 编写测试代码
