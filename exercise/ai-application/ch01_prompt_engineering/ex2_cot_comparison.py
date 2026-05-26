"""
练习 2: Few-shot CoT 对比实验

场景:
    选 3 道数学文字题或逻辑题，分别用三种方式回答：
    1. 直接回答（标准模式）
    2. Zero-shot CoT（加 "Let's think step by step"）
    3. Few-shot CoT（提供一个带推理链的示例）

要求:
    1. 每种方式记录输出内容、正确性、耗时
    2. 形成对比表格，分析 CoT 的有效性
    3. 思考什么类型的题目 CoT 改善最大

TODO:
    1. 设计 3 道需要多步推理的题目（数学、逻辑、计划类各一道）

    2. 实现三种回答方式:
       - direct_answer(question): 不加任何推理提示，直接问
       - zero_shot_cot(question): 在问题末尾加 CoT 触发语
       - few_shot_cot(question): 给一个带完整推理链的示例

    3. 实现 compare(questions) 函数:
       - 对每道题用三种方式各答一次
       - 记录: 输出文本、是否正确（手动标注答案）、耗时
       - 打印对比表格

    4. 分析结论:
       - CoT 在哪些题目上改善最大？
       - Few-shot CoT 比 Zero-shot CoT 好在哪？
       - CoT 的代价是什么（token 消耗、延迟）？
"""

import os
import time
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# ============================================================
# TODO 1: 设计 3 道测试题目
# ============================================================
# 题目要求:
# - 数学题: 涉及多步计算（如鸡兔同笼、行程问题、比例问题）
# - 逻辑题: 涉及条件推理（如谁说真话、三段论）
# - 计划题: 涉及多步骤规划（如安排日程、资源配置）
#
# 每道题包含: question(题目文本), answer(标准答案/判断标准)

questions = [
    {
        "type": "数学",
        "question": "TODO: 出一到数学文字题，需要至少 2 步计算",
        "expected_answer": "TODO: 标准答案",
    },
    {
        "type": "逻辑",
        "question": "TODO: 出一道逻辑推理题",
        "expected_answer": "TODO: 预期推理结论",
    },
    {
        "type": "计划",
        "question": "TODO: 出一道需要多步规划的题目，如安排活动时间表",
        "expected_answer": "TODO: 预期方案要点",
    },
]


# ============================================================
# TODO 2a: 直接回答
# ============================================================
def direct_answer(question: str) -> dict:
    """
    不给任何推理提示，直接要求模型回答。

    返回: {"output": str, "elapsed": float}
    """
    # TODO: 实现
    # tips: prompt = f"请回答以下问题：\n{question}"
    pass


# ============================================================
# TODO 2b: Zero-shot CoT
# ============================================================
def zero_shot_cot(question: str) -> dict:
    """
    使用 Zero-shot CoT 触发推理。

    触发方式（选一种或都试）:
    - 中文: "请一步步思考，先展示推理过程再给出答案"
    - 英文: "Let's think step by step."

    返回: {"output": str, "elapsed": float}
    """
    # TODO: 实现
    pass


# ============================================================
# TODO 2c: Few-shot CoT
# ============================================================
def few_shot_cot(question: str, example: dict = None) -> dict:
    """
    Few-shot CoT: 给一个带完整推理链的示例。

    示例结构: {"question": "...", "reasoning": "步骤1...\n步骤2...\n...", "answer": "..."}

    提示: 示例应该与测试题目类似但不相同（避免模型直接套答案）

    返回: {"output": str, "elapsed": float}
    """
    # TODO: 设计一个带推理链的示例
    # TODO: 将示例和问题一起发给模型
    pass


# ============================================================
# TODO 3: 对比函数
# ============================================================
def compare() -> None:
    """
    对每道题用三种方式回答，打印对比结果。

    输出格式建议: 表格，包含 题目类型 | 方式 | 输出摘要 | 是否正确 | 耗时
    """
    # TODO: 遍历 questions
    # TODO: 调用三种方式
    # TODO: 人工判断（或规则判断）是否正确
    # TODO: 打印对比表格和统计摘要

    # 打印统计示例格式:
    # 方式          | 正确率 | 平均耗时
    # direct_answer | 1/3    | 1.2s
    # zero_shot_cot | 2/3    | 3.5s
    # few_shot_cot  | 3/3    | 4.1s
    pass


# ============================================================
# TODO 4: 分析结论（注释回答）
# ============================================================
"""
Q1: CoT 在哪些类型的题目上改善最大？为什么？
A1: TODO

Q2: Few-shot CoT 相比 Zero-shot CoT 有什么优势？
A2: TODO

Q3: CoT 的代价是什么？（考虑 token 消耗和延迟）
A3: TODO

Q4: 有没有 CoT 反而降低准确率的情况？为什么？
A4: TODO
"""

if __name__ == "__main__":
    print("=== Few-shot CoT 对比实验 ===\n")
    # compare()
    print("请完成 TODO 后运行。")
