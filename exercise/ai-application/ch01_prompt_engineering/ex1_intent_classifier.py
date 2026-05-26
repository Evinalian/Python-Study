"""
练习 1: 设计意图分类 Prompt

场景:
    为在线客服系统设计一个意图识别 prompt，将用户消息分为:
    - 投诉 (complaint): 用户表达不满、要求赔偿
    - 咨询 (inquiry): 用户询问产品信息、使用方法
    - 建议 (suggestion): 用户提出改进意见
    - 闲聊 (chitchat): 与产品无关的随意对话

要求:
    1. 使用五要素模型设计 System Prompt（指令 + 上下文 + 输出格式 + 约束）
    2. 提供至少 3 个 few-shot 示例（覆盖不同类别 + 一个模糊边界案例）
    3. 输出必须是稳定的 JSON，格式为 {"intent": "...", "confidence": 0.X, "keywords": [...]}
    4. 实现 evaluate() 函数在测试集上评估准确率

TODO:
    1. 设计 system_prompt 字符串，包含:
       - 角色定义（你是谁）
       - 四个类别的明确判断标准
       - 输出 JSON 格式要求
       - fallback 策略（不确定时怎么处理）

    2. 设计 3-5 个 few-shot 示例（examples 列表），注意:
       - 覆盖所有四个类别
       - 包含一个模糊边界案例（如"你们的APP挺好的就是太贵了" → 是建议还是抱怨？）
       - 示例格式: {"input": "...", "output": {"intent": "...", "confidence": ..., "keywords": [...]}}

    3. 实现 classify(text) 函数:
       - 调用 OpenAI API
       - 使用 response_format={"type": "json_object"}
       - 返回解析后的 dict

    4. 准备至少 8 个测试用例（test_cases 列表，每个类别至少 2 个）

    5. 实现 evaluate() 函数:
       - 遍历测试用例
       - 统计每个类别的准确率
       - 打印错误案例
       - 返回整体 accuracy

    6. 思考题（注释回答即可）:
       - 哪个类别最难判断？为什么？
       - 如果去掉 few-shot 示例，准确率会下降多少？
"""

import os
import json
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# ============================================================
# TODO 1: 设计 system_prompt
# ============================================================
# 提示: 包含角色定义、四个类别的判断标准、输出格式、fallback 策略
system_prompt = """
TODO: 在这里写出你的 system prompt
"""

# ============================================================
# TODO 2: 设计 few-shot 示例
# ============================================================
examples = [
    # TODO: 至少 3 个示例，覆盖不同类别
    # {"input": "...", "output": {"intent": "...", "confidence": ..., "keywords": [...]}},
]

# ============================================================
# TODO 3: 实现分类函数
# ============================================================
def classify(text: str) -> dict:
    """
    对用户消息进行意图分类。

    参数:
        text: 用户输入的文本

    返回:
        {"intent": "complaint/inquiry/suggestion/chitchat", "confidence": 0.X, "keywords": [...]}
    """
    # TODO: 构建 messages（system + few-shot examples + user input）
    # TODO: 调用 client.chat.completions.create()
    # TODO: 使用 response_format={"type": "json_object"}
    # TODO: 解析并返回结果
    pass


# ============================================================
# TODO 4: 准备测试用例（每个类别至少 2 个）
# ============================================================
test_cases = [
    # TODO: 填写测试用例
    # {"text": "...", "expected_intent": "complaint"},
    # {"text": "...", "expected_intent": "inquiry"},
    # {"text": "...", "expected_intent": "suggestion"},
    # {"text": "...", "expected_intent": "chitchat"},
]

# ============================================================
# TODO 5: 实现评估函数
# ============================================================
def evaluate() -> dict:
    """
    在测试集上评估分类效果。

    返回:
        {
            "total": 总数,
            "correct": 正确数,
            "accuracy": 准确率,
            "per_category": {"投诉": {"correct": n, "total": m}, ...},
            "errors": [{"text": ..., "expected": ..., "got": ...}, ...]
        }
    """
    # TODO: 遍历 test_cases，调用 classify()
    # TODO: 统计每个类别的准确率
    # TODO: 收集错误案例
    # TODO: 打印详细的评估报告
    pass


# ============================================================
# TODO 6: 思考题（注释回答）
# ============================================================
"""
Q1: 哪个类别最难判断？为什么？
A1: TODO

Q2: 如果去掉 few-shot 示例，准确率会下降多少？
A2: TODO  (可以实际改代码测试一下)
"""


if __name__ == "__main__":
    # 测试单个分类
    test_text = "你们的退款流程太复杂了，半天退不了"
    print(f"输入: {test_text}")
    # result = classify(test_text)
    # print(f"分类结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

    # 评估整个测试集
    # report = evaluate()
    # print(f"\n评估报告:\n{json.dumps(report, ensure_ascii=False, indent=2)}")

    print("\n请完成以上 TODO 后运行测试。")
