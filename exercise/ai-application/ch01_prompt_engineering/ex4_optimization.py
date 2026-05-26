"""
练习 4: Prompt 迭代优化

场景:
    给定一个初始 prompt（新闻分类任务）和测试集，完成至少 3 轮迭代优化。
    每轮记录准确率和错误案例，最终生成优化报告。

任务说明:
    新闻分类: 将新闻标题分为 科技 / 体育 / 财经 / 娱乐 / 教育

测试集已提供（见下方 test_cases），你的目标是:
1. 从初始简单 prompt 开始
2. 至少经过 3 轮迭代
3. 最终准确率达到 85% 以上

TODO:
    1. 实现 evaluate(system_prompt, test_cases) 函数:
       - 对每个测试用例调用 API
       - 判断预测类别是否匹配预期
       - 返回 accuracy, errors, per_category_stats

    2. 实现 print_report(version_results) 函数:
       - 打印每轮的准确率和错误案例
       - 格式化输出，方便对比

    3. 完成至少 3 轮迭代:
       第1轮: 最简 prompt（一句话指令）
       第2轮: 加入类别定义 + 输出格式要求
       第3轮: 加入 few-shot 示例 + 边界规则

    4. 在每轮之间分析错误案例，针对性地修改 prompt

    5. 输出最终优化报告，包含:
       - 各版本准确率变化趋势
       - 最有效的改进是什么
       - 哪些错误仍然存在，为什么难解决
"""

import os
import json
import time
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# ============================================================
# 测试集（不要修改这些测试用例）
# ============================================================
TEST_CASES = [
    {"text": "苹果发布新款MacBook Pro搭载M4芯片", "expected": "科技"},
    {"text": "NBA季后赛湖人队逆转取胜", "expected": "体育"},
    {"text": "央行宣布下调贷款利率0.25个百分点", "expected": "财经"},
    {"text": "某明星官宣结婚引发粉丝热议", "expected": "娱乐"},
    {"text": "教育部发布双减政策实施细则", "expected": "教育"},
    {"text": "OpenAI发布GPT-5引发行业震动", "expected": "科技"},
    {"text": "世界杯预选赛中国男足战胜对手", "expected": "体育"},
    {"text": "A股三大指数全线上涨成交额破万亿", "expected": "财经"},
    {"text": "某导演新片票房突破50亿", "expected": "娱乐"},
    {"text": "某大学宣布扩招人工智能专业", "expected": "教育"},
    {"text": "某科技公司推出AI教育产品进校园", "expected": "教育"},  # 模糊: 科技 vs 教育
    {"text": "某游戏公司举办电竞比赛奖金百万", "expected": "娱乐"},  # 模糊: 体育 vs 娱乐
]


# ============================================================
# TODO 1: 实现评估函数
# ============================================================
def evaluate(system_prompt: str) -> dict:
    """
    在测试集上评估一个 prompt。

    参数:
        system_prompt: 待评估的 System Prompt

    返回:
        {
            "accuracy": 0.85,
            "total": 12,
            "correct": 10,
            "errors": [
                {"text": "...", "expected": "...", "predicted": "..."},
            ],
            "per_category": {
                "科技": {"correct": 2, "total": 2},
                ...
            },
            "avg_latency": 1.2
        }
    """
    # TODO: 遍历 TEST_CASES
    # 提示:
    # for case in TEST_CASES:
    #     response = client.chat.completions.create(
    #         model="gpt-4o",
    #         messages=[
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": case["text"]}
    #         ],
    #         temperature=0.0,
    #     )
    #     predicted = response.choices[0].message.content.strip()
    #     判断 predicted 是否包含 expected 类别
    pass


# ============================================================
# TODO 2: 实现报告打印函数
# ============================================================
def print_report(version_results: list[dict]) -> None:
    """
    打印优化报告。

    参数:
        version_results: [{"version": 1, "note": "...", "result": evaluate()返回的dict}, ...]

    输出格式示例:
        === Prompt 优化报告 ===
        测试用例数: 12

        v1 (最简prompt):
          准确率: 58.3% (7/12)
          错误案例:
            "央行..." → 预测"科技", 期望"财经"
            ...

        v3 (加few-shot):
          准确率: 91.7% (11/12)
          ...
    """
    # TODO: 格式化打印
    pass


# ============================================================
# TODO 3: 完成 3 轮迭代
# ============================================================
def run_optimization():
    """
    执行 prompt 优化流程：
    第1轮 → 分析错误 → 改进 prompt → 第2轮 → ... → 第3轮 → 生成报告
    """
    version_results = []

    # --- 第 1 轮: 最简 prompt ---
    prompt_v1 = """
TODO: 写一个最简单的 prompt，如"将以下新闻分类为科技/体育/财经/娱乐/教育"
"""
    # result_v1 = evaluate(prompt_v1)
    # version_results.append({"version": 1, "note": "最简prompt", "prompt": prompt_v1, "result": result_v1})

    # --- 分析第1轮错误 ---
    # TODO: 看哪些类别容易混淆，思考如何改进

    # --- 第 2 轮: 加入类别定义 ---
    prompt_v2 = """
TODO: 在 v1 基础上加入:
- 每个类别的明确定义和关键词
- 输出格式要求（只输出类别名）
- 如果有模糊情况，如何处理
"""
    # result_v2 = evaluate(prompt_v2)
    # version_results.append({"version": 2, "note": "加类别定义", "prompt": prompt_v2, "result": result_v2})

    # --- 分析第2轮错误 ---
    # TODO: 哪些错误改善了，哪些仍存在？

    # --- 第 3 轮: 加入 few-shot 示例 ---
    prompt_v3 = """
TODO: 在 v2 基础上加入 few-shot 示例
- 重点覆盖前两轮的常错类别
- 示例格式: "输入: xxx → 输出: xxx"
"""
    # result_v3 = evaluate(prompt_v3)
    # version_results.append({"version": 3, "note": "加few-shot", "prompt": prompt_v3, "result": result_v3})

    # --- 生成最终报告 ---
    # print_report(version_results)

    # TODO 4: 分析结论
    """
    Q1: 最有效的改进是什么？为什么？
    A1: TODO

    Q2: 哪些错误仍然存在？为什么难以解决？
    A2: TODO

    Q3: 如果再迭代一轮，你会做什么改进？
    A3: TODO
    """


if __name__ == "__main__":
    run_optimization()
    print("\n请完成 TODO 后运行。")
