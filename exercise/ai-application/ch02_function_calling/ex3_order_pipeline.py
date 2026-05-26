"""
练习 3: 电商工具链编排

场景:
    实现一个"下单 → 支付 → 发货"的电商工具链。
    每个步骤验证前置条件，失败时回滚或补偿。

三个工具:
    1. create_order: 创建订单（验证商品、地址、库存）
    2. process_payment: 处理支付（验证账户余额）
    3. ship_order: 发货（验证订单状态、地址有效性）

依赖关系（串行）:
    create_order → process_payment → ship_order
    前一步成功才能执行下一步，任何一步失败都需要告知用户。

要求:
    1. 定义三个工具的 Schema，明确依赖关系
       - create_order: product_ids, quantities, shipping_address
       - process_payment: order_id, payment_method, amount
       - ship_order: order_id

    2. 实现模拟后端:
       - 库存系统（部分商品可能缺货）
       - 账户系统（余额不足时支付失败）
       - 地址验证（部分地址不支持配送）

    3. 实现 run_order_pipeline(user_query):
       - 模型按照依赖顺序调用工具
       - 每一步失败时，返回清晰的错误原因
       - 不自动重试（让模型基于错误信息给用户建议）

    4. 测试:
       - 正常下单流程
       - 商品缺货
       - 余额不足
"""

import os
import json
import random
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# ============================================================
# 模拟后端系统
# ============================================================
class ECommerceSystem:
    """模拟电商系统后端状态"""

    def __init__(self):
        # 库存: {product_id: {"name": ..., "price": ..., "stock": ...}}
        self.inventory = {
            "P001": {"name": "机械键盘", "price": 299.0, "stock": 10},
            "P002": {"name": "无线鼠标", "price": 149.0, "stock": 5},
            "P003": {"name": "显示器", "price": 1999.0, "stock": 0},  # 缺货!
            "P004": {"name": "USB-C Hub", "price": 89.0, "stock": 20},
        }

        # 账户余额: {user_id: balance}
        self.accounts = {"user_001": 1000.0, "user_002": 50.0}  # user_002 余额不足

        # 订单存储
        self.orders: dict = {}
        self._next_order_id = 1

        # 支持配送的地区
        self.supported_regions = ["北京市", "上海市", "广东省", "浙江省", "江苏省"]

    def create_order(self, items: list[dict], shipping_address: dict, user_id: str) -> dict:
        """
        创建订单。

        验证:
        1. 每个商品是否存在
        2. 库存是否充足
        3. 地址是否在配送范围

        返回: {"order_id": "ORD-001", "status": "pending_payment", ...}
        或: {"error": "..."}
        """
        # TODO:
        # 1. 遍历 items，检查每个 product_id 是否在 inventory 中
        # 2. 检查 stock >= quantity
        # 3. 检查 shipping_address["province"] 是否在 supported_regions 中
        # 4. 扣除库存，创建订单，返回订单信息
        pass

    def process_payment(self, order_id: str, user_id: str, amount: float) -> dict:
        """
        处理支付。

        验证:
        1. 订单是否存在且状态为 pending_payment
        2. 用户余额是否足够

        返回: {"payment_id": "PAY-001", "status": "paid", ...}
        或: {"error": "..."}
        """
        # TODO:
        # 1. 检查订单是否存在
        # 2. 检查订单状态是否为 pending_payment
        # 3. 检查账户余额是否 >= amount
        # 4. 扣除余额，更新订单状态，返回支付成功
        pass

    def ship_order(self, order_id: str) -> dict:
        """
        发货。

        验证:
        1. 订单是否存在
        2. 订单状态是否为 paid

        返回: {"tracking_number": "SF123...", "status": "shipped", ...}
        或: {"error": "..."}
        """
        # TODO:
        # 1. 检查订单是否存在
        # 2. 检查订单状态是否为 paid
        # 3. 生成运单号，更新订单状态，返回发货信息
        pass


# ============================================================
# TODO 1: 定义三个工具的 Schema
# ============================================================
# 提示: 在 description 中说明依赖关系，
# 如 process_payment 的 description: "处理订单支付。仅在 create_order 成功后才能调用。"

TOOLS = [
    # TODO: create_order Schema
    # TODO: process_payment Schema
    # TODO: ship_order Schema
]

# ============================================================
# TODO 2: 实例化系统
# ============================================================
system = ECommerceSystem()

FUNCTION_MAP = {
    "create_order": system.create_order,
    "process_payment": system.process_payment,
    "ship_order": system.ship_order,
}

# ============================================================
# TODO 3: 实现订单流程处理
# ============================================================
def run_order_pipeline(user_query: str) -> str:
    """
    处理电商订单的完整对话流程。

    关键: 这个流程需要串行执行。
    create_order → (成功) → process_payment → (成功) → ship_order
    任一步失败 → 返回错误 → 模型告知用户

    System Prompt 应该:
    - 说明三步流程的顺序
    - 强调每步成功后才能进行下一步
    - 失败时告知用户具体原因和解决建议
    """
    # TODO:
    # 1. 构建 system prompt（描述三步流程和依赖关系）
    # 2. 初始化 messages
    # 3. 实现多轮工具调用循环:
    #    - 调用 API
    #    - 检查 tool_calls
    #    - 执行工具（try/except）
    #    - 追加结果
    #    - 继续循环直到没有 tool_call 或达到最大轮次
    pass


# ============================================================
# TODO 4: 测试
# ============================================================
if __name__ == "__main__":
    print("=== 电商工具链测试 ===\n")

    # 场景1: 正常下单
    # reply = run_order_pipeline(
    #     "我是 user_001，要买 1 个机械键盘和 2 个无线鼠标，收货地址是北京市朝阳区XX路，帮我下单"
    # )

    # 场景2: 商品缺货
    # reply = run_order_pipeline(
    #     "我是 user_001，要买 1 个显示器，送到上海市"
    # )

    # 场景3: 余额不足
    # reply = run_order_pipeline(
    #     "我是 user_002，要买 1 个机械键盘（299元），送到浙江省杭州市"
    # )

    print("请完成 TODO 后取消注释运行。")
