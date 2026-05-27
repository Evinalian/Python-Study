"""
练习 1: Benchmark 自动评测

场景:
    你微调了一个模型，需要定量评估它在标准 benchmark 上的表现。
    使用 lm-evaluation-harness（或自实现简化版）进行自动化评测。

要求:
    1. 加载模型并在 MMLU 等 benchmark 上跑分
    2. 对比微调前后的分数变化
    3. 生成对比报告

TODO:
    1. 实现 load_model_for_eval(model_name, adapter_path) 函数:
       - 加载基座模型（支持 LoRA adapter 挂载）
       - 返回可用于评测的模型对象

    2. 定义评测任务列表:
       - knowledge_tasks: MMLU, C-Eval（知识能力）
       - reasoning_tasks: HellaSwag（常识推理）
       - 为每个任务定义 few-shot 设置和指标

    3. 实现 run_evaluation(model, tasks, limit_per_task) 函数:
       - 使用 lm-eval 或自实现的评测逻辑
       - 如果 lm-eval 不可用，实现简化版:
         对 MMLU: 加载题目 -> 构建 prompt -> 模型回答 -> 比对答案 -> 计算 accuracy
       - 返回 {task_name: {"accuracy": float, "total": int, "correct": int}}

    4. 实现 compare_results(before_results, after_results) 函数:
       - 生成对比表格
       - 标注有显著变化的任务（变化 > 2 个百分点）
       - 分析哪些能力提升/下降

    5. 实现 detect_catastrophic_forgetting(before, after, threshold=0.05):
       - 如果某个任务的分数下降了超过 5 个百分点，标记为"可能灾难性遗忘"
       - 打印警告信息

    6. 思考题（注释回答）:
       - 为什么微调后 MMLU 分数通常下降？这是问题吗？
       - 如何区分"灾难性遗忘"和"正常的任务聚焦"？
"""
import os
import json
import torch
from typing import Optional


# ============================================================
# TODO 1: 加载模型用于评测
# ============================================================
def load_model_for_eval(model_name: str, adapter_path: Optional[str] = None):
    """
    加载模型用于评测。

    返回: (model, tokenizer)
    """
    # TODO: 加载 tokenizer
    # TODO: 加载模型（BF16, device_map="auto"）
    # TODO: 如果有 adapter_path，挂载 LoRA adapter
    # TODO: 设置 model.eval()
    pass


# ============================================================
# TODO 2: 定义评测任务
# ============================================================
EVAL_TASKS = {
    # TODO: 定义至少 3 个评测任务
    # "mmlu": {"type": "multiple_choice", "few_shot": 5, "metric": "accuracy"},
    # "hellaswag": {"type": "multiple_choice", "few_shot": 10, "metric": "accuracy_norm"},
}

# ============================================================
# TODO 3: 运行评测
# ============================================================
def run_mmlu_simplified(model, tokenizer, subject: str, limit: int = None) -> dict:
    """
    简化版 MMLU 评测（不使用 lm-eval 框架）。

    MMLU 题目格式:
    {
        "question": "What is the capital of France?",
        "choices": ["London", "Paris", "Berlin", "Madrid"],
        "answer": 1  // 正确答案的索引
    }

    对于每个题目:
    1. 构建 few-shot prompt（将几个示例的 question + choices + answer 作为上下文）
    2. 追加当前题目（question + choices）
    3. 让模型生成答案
    4. 比对模型输出和正确答案
    """
    # TODO: 加载 MMLU subject 数据（或创建几个示例题目）
    # TODO: 对每个题目:
    #   1) 构建 prompt（含 few-shot 示例）
    #   2) Tokenize
    #   3) 模型生成
    #   4) 从输出中提取答案（A/B/C/D 或直接比对 logits）
    # TODO: 计算 accuracy
    pass


def run_evaluation(model, tokenizer, tasks: dict, limit_per_task: int = None) -> dict:
    """
    运行评测。

    返回: {task_name: {"total": int, "correct": int, "accuracy": float}, ...}
    """
    # TODO: 检查是否可以使用 lm-eval（try import）
    # TODO: 如果可以，使用 lm_eval.simple_evaluate
    # TODO: 如果不可以，使用简化版实现
    #   (至少手动实现 20-30 道题目的评测逻辑)
    pass


# ============================================================
# TODO 4: 对比分析
# ============================================================
def compare_results(
    before_results: dict,
    after_results: dict,
    threshold: float = 0.02,  # 2 个百分点
) -> dict:
    """
    对比微调前后的评测结果。

    返回:
        {
            "comparisons": [
                {"task": "mmlu", "before": 0.45, "after": 0.43, "change": -0.02, "significant": True},
                ...
            ],
            "summary": "...",
            "improvements": ["task1", ...],
            "declines": ["task2", ...],
        }
    """
    # TODO: 对每个 task 计算变化
    # TODO: 标注变化 > threshold 的任务
    # TODO: 打印对比表格
    # TODO: 生成文字摘要
    pass


# ============================================================
# TODO 5: 灾难性遗忘检测
# ============================================================
def detect_catastrophic_forgetting(
    before: dict,
    after: dict,
    threshold: float = 0.05,
) -> list[str]:
    """
    检测可能的灾难性遗忘。

    如果某个任务的 accuracy 下降超过 threshold（5 个百分点），
    标记为可能的灾难性遗忘。

    返回: 可能发生遗忘的任务名列表
    """
    # TODO: 遍历所有任务
    # TODO: 计算 accuracy 变化
    # TODO: 如果下降 > threshold，加入警告列表
    # TODO: 打印警告（但要注明"可能"而非"确定"）
    pass


# ============================================================
# TODO 6: 思考题
# ============================================================
"""
Q1: 为什么微调后 MMLU 分数通常下降？这是问题吗？
A1: TODO
    提示: 微调让模型聚焦于特定任务分布，通用知识的"优先级"在权重中被降低。
    但如果下降太多（>10 个百分点），可能说明微调数据有问题或训练过度。

Q2: 如何区分"灾难性遗忘"和"正常的任务聚焦"？
A2: TODO
    提示: 关键在于"遗忘的程度"和"遗忘的内容"。
    如果模型在下游任务上提升 20%，但在通用知识上下降 3%，这是合理的取舍。
    如果通用知识下降 15% 而下游只提升 2%，这是灾难性遗忘。

Q3: few-shot 数量如何影响评测结果？
A3: TODO
"""


if __name__ == "__main__":
    print("=" * 50)
    print("  Benchmark 自动评测")
    print("=" * 50)

    MODEL_NAME = "Qwen/Qwen2.5-0.5B"

    print(f"\n模型: {MODEL_NAME}")
    print(f"评测任务: {list(EVAL_TASKS.keys())}")

    # TODO: 取消注释完成评测
    # model, tokenizer = load_model_for_eval(MODEL_NAME)
    # results = run_evaluation(model, tokenizer, EVAL_TASKS, limit_per_task=50)
    # print(json.dumps(results, ensure_ascii=False, indent=2))

    print("\n请完成以上 TODO 后取消注释运行。")
