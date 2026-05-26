"""
练习 5: A/B 测试框架

场景:
    实现一个完整的 A/B 测试框架，对比两个 prompt 版本的效果。

背景:
    在实际项目中，修改 prompt 后不能直接全量上线，需要先做 A/B 测试：
    50% 用户看到旧版本（对照组），50% 用户看到新版本（实验组），
    收集指标后做对比，判断新版本是否显著优于旧版本。

要求:
    1. 实现用户分流逻辑（基于 user_id 哈希保证同一用户总看到同一个版本）
    2. 实现多指标收集（准确率、延迟、用户满意度）
    3. 实现统计摘要生成
    4. 模拟测试过程并输出报告

TODO:
    1. 实现 assign_variant(user_id) 函数:
       - 基于 user_id 哈希将用户分配到 A 或 B 组
       - 保证同一 user_id 总是分到同一组（用户体验一致性）
       - 默认 50/50 分流比

    2. 实现 ABTest 类:
       - __init__: 初始化 prompt_a, prompt_b, 统计数据结构
       - record(variant, metrics): 记录一次测试结果
       - summary(): 生成统计摘要（每组的均值、样本量、置信区间）
       - is_significant(): 判断结果是否统计显著（简化版）

    3. 实现模拟测试:
       - 准备 2 个版本的 prompt（如不同风格的客服回复 prompt）
       - 准备测试用例（包含 user_id, question, expected_answer）
       - 模拟用户请求，分流 → 调用 API → 记录结果

    4. 输出对比报告:
       - 两组样本量
       - 各指标均值对比
       - 提升百分比
       - 是否显著

    5. 思考: 什么情况下 A/B 测试的结论不可靠？
"""

import os
import json
import time
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# ============================================================
# TODO 1: 用户分流函数
# ============================================================
def assign_variant(user_id: str, ratio: float = 0.5) -> str:
    """
    基于用户 ID 哈希分流。

    参数:
        user_id: 用户唯一标识
        ratio: 分配给 B 组的比例（默认 0.5 即 50%）

    返回:
        "A" 或 "B"

    原理:
        hash(user_id) % 100 < ratio * 100 → B组
        同一 user_id 的 hash 不变 → 总是分到同一组
    """
    # TODO: 实现
    pass


# ============================================================
# TODO 2: ABTest 类
# ============================================================
class ABTest:
    """
    A/B 测试管理器。

    管理两个 prompt 版本的对比测试：
    - variant_a: 对照组（现有版本）
    - variant_b: 实验组（候选版本）
    """

    def __init__(self, name: str, prompt_a: str, prompt_b: str):
        """
        初始化 A/B 测试。

        参数:
            name: 测试名称
            prompt_a: 对照组 System Prompt
            prompt_b: 实验组 System Prompt
        """
        self.name = name
        self.prompt_a = prompt_a
        self.prompt_b = prompt_b

        # TODO: 初始化统计数据结构
        # 提示: 使用 defaultdict(list) 存储各组各指标的数值列表
        self.stats_a = defaultdict(list)  # {"accuracy": [1, 0, 1, ...], "latency": [1.2, 1.5, ...]}
        self.stats_b = defaultdict(list)

    def get_prompt(self, variant: str) -> str:
        """获取指定变体的 prompt"""
        # TODO: 返回 self.prompt_a 或 self.prompt_b
        pass

    def record(self, variant: str, metrics: dict) -> None:
        """
        记录一次测试结果。

        参数:
            variant: "A" 或 "B"
            metrics: {"accuracy": 1.0, "latency": 1.5, ...}
        """
        # TODO: 将各指标值追加到对应的 stats 列表中
        pass

    def summary(self) -> dict:
        """
        生成统计摘要。

        返回格式:
        {
            "name": "测试名称",
            "variant_A": {
                "sample_count": 100,
                "metrics": {
                    "accuracy": {"mean": 0.85, "min": 0.0, "max": 1.0},
                    "latency": {"mean": 1.2, "min": 0.5, "max": 3.0}
                }
            },
            "variant_B": { ... },
            "improvements": {
                "accuracy": "+5.2%",
                "latency": "-3.1%"
            }
        }
        """
        # TODO:
        # 1. 遍历 self.stats_a 和 self.stats_b
        # 2. 计算每个指标的 mean, min, max, count
        # 3. 计算 B 相对 A 的提升百分比
        pass

    def is_significant(self, metric: str = "accuracy") -> bool:
        """
        简化版显著性判断。

        判断标准: B 组均值 - A 组均值 > 阈值（如 0.05）
        且每组样本量 >= 10

        注意: 真实项目应使用 scipy.stats.ttest_ind
        """
        # TODO:
        # a_vals = self.stats_a.get(metric, [])
        # b_vals = self.stats_b.get(metric, [])
        # 如果样本量不足，返回 False
        # 如果 B 组均值超过 A 组 5% 以上，返回 True
        pass

    def report(self) -> str:
        """生成人类可读的测试报告"""
        # TODO: 基于 summary() 的结果生成格式化文本报告
        pass


# ============================================================
# TODO 3: 模拟测试
# ============================================================
def run_simulation():
    """
    模拟 A/B 测试过程。

    场景: 对比两种客服回复风格
    - 对照组 A: 正式、专业的回复风格
    - 实验组 B: 亲切、口语化的回复风格

    模拟 20 个用户，每个用户发一条客服咨询。
    """
    # --- 定义两个版本的 prompt ---
    prompt_a = """你是一个客服助手。用专业、正式的语气回复用户问题。
回复简洁准确，使用"您"称呼用户。"""

    prompt_b = """你是一个客服助手。用亲切、口语化的语气回复用户问题。
像朋友聊天一样，可以适当使用语气词和表情符号。称呼用户为"你"。"""

    ab = ABTest("客服回复风格对比", prompt_a, prompt_b)

    # 模拟的用户问题
    user_questions = [
        ("user_001", "我的订单什么时候发货？"),
        ("user_002", "收到的东西是坏的，怎么办？"),
        ("user_003", "可以修改收货地址吗？"),
        ("user_004", "有没有优惠券可以领？"),
    ]

    # TODO:
    # 1. 对每个 (user_id, question) 对：
    #    a. assign_variant(user_id) 决定分到 A 或 B
    #    b. 获取对应版本的 prompt
    #    c. 调用 API（记录延迟）
    #    d. 由"用户满意度模拟函数"评分（可简化为规则判断：回答是否友好/有帮助）
    #    e. ab.record(variant, {"latency": ..., "satisfaction": ...})
    #
    # 2. 用 ab.report() 打印对比结果

    print("模拟中...")
    # TODO
    print("请完成 TODO 后运行。")


# ============================================================
# TODO 4: 输出对比报告（实现在 ABTest.report() 中）
# ============================================================


# ============================================================
# TODO 5: 思考题
# ============================================================
"""
Q1: 什么情况下 A/B 测试的结论不可靠？
A1: TODO (提示: 样本量不足、未随机分配、测试时间不一致、辛普森悖论)

Q2: 为什么用 user_id 哈希分流比纯随机要好？
A2: TODO

Q3: 如果 B 组提升了 2% 但不显著，你会怎么决策？
A3: TODO
"""

if __name__ == "__main__":
    # 测试分流逻辑
    print("分流测试:")
    for uid in ["alice", "bob", "charlie", "alice"]:
        print(f"  {uid} → {assign_variant(uid)}")
    print("  (注意 alice 两次分配结果应相同)\n")

    run_simulation()
