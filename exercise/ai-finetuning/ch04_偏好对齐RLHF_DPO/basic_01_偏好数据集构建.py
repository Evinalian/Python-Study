"""
练习 1: 偏好数据集构建

场景:
    DPO 训练需要"偏好对"数据——同一个问题的两个回答，一个是"更好的(chosen)"，
    一个是"更差的(rejected)"。你需要设计一个系统来生成和管理这类数据。

要求:
    1. 设计标注标准（什么构成"更好的"回复）
    2. 用 GPT-4 生成回复对
    3. 用 GPT-4 作为 Judge 进行自动标注
    4. 格式化为 DPO 训练所需的 JSONL 格式

TODO:
    1. 实现 ANNOTATION_CRITERIA 字典:
       - 定义"有帮助性"、"安全性"、"真实性"、"格式规范"四个维度的判断标准
       - 每个维度给出 1-5 分的评分细则

    2. 实现 generate_response_pair(client, instruction, system_prompt) 函数:
       - 用 GPT-4 生成两个不同质量的回复
       - 一个高质量（指定要求: 详细、准确、有帮助）
       - 一个低质量（指定要求: 简短、含糊、可能有小错误）
       - 返回 (good_response, bad_response)

    3. 实现 auto_label(client, instruction, response_a, response_b, criteria) 函数:
       - 用 GPT-4 作为 Judge，根据 criteria 判断哪个回复更好
       - 需要处理位置偏差（position bias）：GPT-4 可能倾向于选择第一个回复
       - 解决方案：双向评判——先 A vs B，再 B vs A，交叉验证
       - 返回 {"chosen": ..., "rejected": ..., "confidence": 0.X}

    4. 实现 validate_pair(instruction, chosen, rejected) 函数:
       - 验证 chosen 确实比 rejected 好（基本检查）:
         * chosen 长度不应太短（至少 50 字符）
         * chosen 不应是拒绝回答（"我不知道"等）
         * rejected 不应比 chosen 更好（长度、内容丰富度倒挂）
       - 返回 (is_valid, issues_list)

    5. 实现 build_dpo_dataset(instructions, system_prompt) 函数:
       - 完整的 pipeline:
         for each instruction:
           1. 生成回复对
           2. 自动标注 chosen/rejected
           3. 验证配对质量
           4. 如果质量不达标，重新生成
       - 输出 JSONL 格式的 DPO 数据

    6. 思考题（注释回答）:
       - 为什么需要用 GPT-4 来标注而不是只用规则（如长度）？
       - 如何检测和处理"位置偏差"问题？
"""
import os
import json
from typing import Optional
from openai import OpenAI


# ============================================================
# TODO 1: 定义标注标准
# ============================================================
ANNOTATION_CRITERIA = {
    # TODO: 定义四个维度的评分标准（各 1-5 分）
    "helpfulness": {
        "1": "TODO: 完全无帮助",
        "2": "...",
        "3": "...",
        "4": "...",
        "5": "TODO: 详细、准确、全面的回答",
    },
    "safety": {
        # TODO
    },
    "truthfulness": {
        # TODO
    },
    "format": {
        # TODO
    },
}


# ============================================================
# TODO 2: 生成回复对
# ============================================================
def generate_response_pair(
    client: OpenAI,
    instruction: str,
    system_prompt: str = "你是一个有帮助的AI助手。",
    model: str = "gpt-4o",
) -> tuple[str, str]:
    """
    生成一对高质量和低质量的回复。

    返回: (good_response, bad_response)
    """
    # TODO: 生成高质量回复
    #   system prompt: "请给出详细、准确、有帮助的回答。"
    # TODO: 生成低质量回复
    #   system prompt: "请给出简短、敷衍的回答。（模拟一个不认真的回答者）"
    # TODO: 返回两个回复
    pass


# ============================================================
# TODO 3: 自动标注（LLM-as-Judge）
# ============================================================
def auto_label(
    client: OpenAI,
    instruction: str,
    response_a: str,
    response_b: str,
    criteria: dict,
    model: str = "gpt-4o",
) -> dict:
    """
    用 GPT-4 判断 response_a 和 response_b 哪个更好。

    处理位置偏差的两种策略：
    1. 双向评判: 先判断 A vs B，再判断 B vs A（swap 顺序）
    2. 如果两次判断一致，取共识结果；如果不一致，标记为 "tie"

    返回:
        {
            "chosen": 更好的回复,
            "rejected": 更差的回复,
            "confidence": 0.0-1.0,  # 置信度
            "reasoning": "判断理由"
        }
    """
    # TODO: 构建 Judge Prompt
    #   提示: 包含 criteria、instruction、response_a、response_b
    #   要求: 输出 JSON 格式的判断结果
    # TODO: 第一次判断（A 在前，B 在后）
    # TODO: 第二次判断（B 在前，A 在后）
    # TODO: 交叉验证
    #   - 两次都是 A 更好 -> 确认 A 是 chosen
    #   - 两次都是 B 更好 -> 确认 B 是 chosen
    #   - 不一致 -> 降低 confidence，取第一轮判断
    # TODO: 返回结果
    pass


# ============================================================
# TODO 4: 验证配对质量
# ============================================================
def validate_pair(instruction: str, chosen: str, rejected: str) -> tuple[bool, list[str]]:
    """
    验证 chosen/rejected 配对的基本质量。

    返回: (is_valid, issues_list)
    """
    issues = []

    # TODO: 检查 chosen 长度 >= 50 字符
    # TODO: 检查 chosen 是否包含拒绝话术
    #   （"我不知道"、"无法回答"、"作为AI"等）
    # TODO: 检查 rejected 不应该明显比 chosen 好
    #   - rejected 长度不应 > chosen * 1.5（太长的"差回复"不合理）
    # TODO: 检查 chosen 和 rejected 不应完全相同或高度相似
    # TODO: 检查指令不包含敏感/不当内容（如果有，标记但不过滤）

    is_valid = len(issues) == 0
    return is_valid, issues


# ============================================================
# TODO 5: 完整 DPO 数据集构建 Pipeline
# ============================================================
def build_dpo_dataset(
    instructions: list[str],
    system_prompt: str,
    output_path: str,
    max_retries: int = 3,
) -> list[dict]:
    """
    完整的 DPO 偏好数据构建 Pipeline。

    参数:
        instructions: 指令列表
        system_prompt: 全局 system prompt
        output_path: 输出 JSONL 路径
        max_retries: 每个 instruction 的最大重试次数

    返回:
        [{"prompt": "...", "chosen": "...", "rejected": "..."}, ...]
    """
    # TODO: 初始化 OpenAI client
    # TODO: 对每个 instruction:
    #   for attempt in range(max_retries):
    #       1. 生成回复对
    #       2. 自动标注 chosen/rejected
    #       3. 验证配对质量
    #       4. 如果通过验证，保存并处理下一个
    #       5. 如果未通过，重试
    # TODO: 保存为 JSONL
    # TODO: 打印统计信息（成功率、平均 confidence 等）
    pass


# ============================================================
# TODO 6: 思考题
# ============================================================
"""
Q1: 为什么需要用 GPT-4 来标注而不是只用规则（如长度）？
A1: TODO

Q2: 如何检测和处理"位置偏差"问题？
A2: TODO

Q3: 如果 GPT-4 把两个回复都判为平局（tie），说明了什么？
A3: TODO
"""


if __name__ == "__main__":
    print("=" * 50)
    print("  DPO 偏好数据集构建")
    print("=" * 50)

    # 示例指令列表
    test_instructions = [
        "解释一下量子计算的基本原理",
        "Python 中如何处理异常？",
        "写一首关于秋天的短诗",
        "推荐三本适合初学者的编程书籍",
    ]

    # TODO: 需要设置 OpenAI API Key
    # api_key = os.getenv("OPENAI_API_KEY")
    # if not api_key:
    #     print("请设置 OPENAI_API_KEY 环境变量")

    print(f"待处理指令数: {len(test_instructions)}")
    print("\n请完成以上 TODO 后取消注释运行。")
