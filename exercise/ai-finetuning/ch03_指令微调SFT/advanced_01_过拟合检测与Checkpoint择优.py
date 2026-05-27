"""
进阶练习 1: 过拟合检测与 Checkpoint 自动择优

场景:
    你完成了 SFT 训练，Trainer 保存了 3 个 checkpoint（每个 epoch 一个）。
    你不确定哪个 checkpoint 最好——直觉告诉你可能不是最后一个。

    你需要编写一个自动化的评估脚本：
    1. 加载所有 checkpoint
    2. 在测试集上评估每个 checkpoint
    3. 综合分析选出最佳 checkpoint

要求:
    1. 解析训练日志，绘制 loss 曲线
    2. 加载每个 checkpoint 进行推理评估
    3. 用多种指标综合评分

TODO:
    1. 实现 parse_training_logs(log_dir) 函数:
       - 解析 Trainer 保存的日志文件（trainer_state.json）
       - 提取每个 step/epoch 的 training loss 和 eval loss
       - 返回结构化数据

    2. 实现 plot_loss_curves(logs) 函数:
       - 用 ASCII art 或纯文本绘制 loss 曲线
       - 标注训练 loss 和验证 loss 的分叉点（过拟合起始点）
       - 打印过拟合检测结论

    3. 实现 generate_from_checkpoint(model, tokenizer, test_cases) 函数:
       - 加载指定 checkpoint 的 LoRA adapter
       - 对测试集中的每个问题生成回复
       - 返回 {"question": ..., "response": ...} 列表

    4. 实现 score_responses(responses, reference_answers, judge_model="gpt-4") 函数:
       - 用 GPT-4 作为 Judge 评分（或使用启发式规则）
       - 评分维度: 准确性(1-5), 完整性(1-5), 流畅性(1-5)
       - 返回每条回复的评分

    5. 实现 compute_diversity_score(responses) 函数:
       - 计算生成回复的多样性（避免模型总是输出相同的回复）
       - 用词汇多样性（unique token ratio）和语义多样性（embedding 距离）
       - 返回 0-1 的多样性分数

    6. 实现 select_best_checkpoint(checkpoint_scores) 函数:
       - 输入: 每个 checkpoint 的多维评分
       - 综合评分公式: 0.3 * eval_loss_inverse + 0.3 * judge_score + 0.2 * diversity + 0.2 * length_appropriateness
       - 返回最佳 checkpoint 名称和分析报告

    7. 思考题（注释回答）:
       - 为什么 eval loss 不是选择 checkpoint 的唯一标准？
       - 如果所有 checkpoint 的 eval loss 都在下降但生成质量在恶化，可能是什么原因？
"""
import os
import json
import math
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ============================================================
# TODO 1: 解析训练日志
# ============================================================
def parse_training_logs(log_dir: str) -> dict:
    """
    解析 trainer_state.json 中的训练日志。

    trainer_state.json 通常包含:
    - log_history: 每个 logging step 的 loss 等指标
    - best_metric, best_model_checkpoint 等

    返回:
        {
            "train_steps": [1, 11, 21, ...],
            "train_losses": [2.3, 1.8, 1.5, ...],
            "eval_steps": [100, 200, 300, ...],
            "eval_losses": [2.1, 1.9, 1.7, ...],
            "best_metric": float or None,
            "total_steps": int,
        }
    """
    # TODO: 读取 trainer_state.json
    # TODO: 遍历 log_history
    # TODO: 分离 train loss 和 eval loss
    # TODO: 提取 step 信息
    pass


# ============================================================
# TODO 2: ASCII Loss 曲线
# ============================================================
def plot_loss_curves(logs: dict, width: int = 60, height: int = 20):
    """
    用 ASCII 字符绘制 loss 曲线。

    输出示例:
    Loss
    2.5 |    * (train)
        |   / *
    2.0 |  /   \
        | /     *
    1.5 |*       \   o (eval)
        |         \ /
    1.0 |          o
        +------------------------- Steps
    """
    # TODO: 归一化 loss 值到绘图范围
    # TODO: 绘制 Y 轴标签和刻度
    # TODO: 用 * 标记 train loss，用 o 标记 eval loss
    # TODO: 用 - | / \ 字符绘制曲线
    # TODO: 标注过拟合起始点（train loss 下降但 eval loss 开始上升的分叉点）
    pass


# ============================================================
# TODO 3: 从 Checkpoint 生成回复
# ============================================================
def generate_from_checkpoint(
    base_model_name: str,
    checkpoint_path: str,
    tokenizer,
    test_cases: list[dict],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> list[dict]:
    """
    加载 checkpoint 并对测试集生成回复。

    参数:
        base_model_name: 基座模型名称
        checkpoint_path: checkpoint 目录路径
        tokenizer: tokenizer
        test_cases: [{"question": "...", "reference_answer": "..."}, ...]
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度

    返回:
        [{"question": "Q", "response": "R", "reference": "Ref"}, ...]
    """
    # TODO: 加载基座模型
    # TODO: 加载 LoRA adapter
    # TODO: 对每个 test case 生成回复
    # TODO: 清理显存
    # TODO: 返回结果
    pass


# ============================================================
# TODO 4: LLM-as-Judge 评分
# ============================================================
def score_responses(
    responses: list[dict],
    judge_model: str = "gpt-4",
) -> list[dict]:
    """
    用 GPT-4 对生成回复进行多维评分。

    评分维度（各 1-5 分）:
    - 准确性 (accuracy): 回答是否包含正确信息
    - 完整性 (completeness): 是否全面回答了问题
    - 流畅性 (fluency): 语言是否自然流畅、无语法错误

    对于无法调用 GPT-4 的场景，使用启发式规则:
    - 准确性: 与参考答案的词汇重叠率（简化版）
    - 完整性: 回复长度 vs 参考长度
    - 流畅性: 无重复内容、标点规范

    返回:
        [{"question": "Q", "response": "R", "accuracy": 4, "completeness": 3, "fluency": 5}, ...]
    """
    # TODO: 实现启发式评分（无 GPT-4 API 时使用）:
    #   准确性: 使用 Jaccard 相似度（与 reference 的 token 重叠率）
    #   完整性: min(1.0, len(response) / len(reference)) * 5 但是过度长也不加分
    #   流畅性: 检查独特 token 占比、中英文标点一致性等
    # TODO: 或者调用 OpenAI API（如果有 key）用 GPT-4 评分
    # TODO: 返回评分结果
    pass


# ============================================================
# TODO 5: 多样性评分
# ============================================================
def compute_diversity_score(responses: list[str]) -> float:
    """
    计算回复的多样性分数（0-1）。

    如果模型对所有问题都给出相似的回复（如总是说"这是一个好问题"开头），
    说明模型可能过拟合了某些模式。多样性高的回复应该在用词和结构上都有变化。
    """
    # TODO: 计算词汇多样性: 所有回复的 unique token / total token
    # TODO: 计算回复长度方差: 如果所有回复长度都差不多，可能缺乏多样性
    # TODO: 计算回复开头多样性: 前 10 个 token 的重复率
    # TODO: 综合返回 0-1 的分数
    pass


# ============================================================
# TODO 6: 综合评分选择最佳 Checkpoint
# ============================================================
def select_best_checkpoint(
    checkpoint_names: list[str],
    eval_losses: dict[str, float],
    judge_scores: dict[str, dict],
    diversity_scores: dict[str, float],
) -> dict:
    """
    综合分析所有指标，选出最佳 checkpoint。

    综合评分 = 0.3 * loss_score + 0.3 * judge_avg + 0.2 * diversity + 0.2 * length_score

    其中:
    - loss_score = exp(-eval_loss) 归一化（loss 越低分越高）
    - judge_avg = (accuracy + completeness + fluency) / 3
    - diversity 和 length_score 都已归一化到 0-1

    返回:
        {
            "best_checkpoint": "checkpoint-XXX",
            "best_score": 0.85,
            "all_scores": [...],
            "analysis": "推荐 checkpoint-200: ..."
        }
    """
    # TODO: 对每个 checkpoint 计算综合得分
    # TODO: 排序
    # TODO: 选出得分最高的
    # TODO: 生成分析报告（为什么选这个）
    # TODO: 返回结果
    pass


# ============================================================
# TODO 7: 思考题
# ============================================================
"""
Q1: 为什么 eval loss 不是选择 checkpoint 的唯一标准？
A1: TODO
    提示: eval loss 的下降可能来自模型学会了"输出高频 safe 回复"而非更好的回答质量

Q2: 如果所有 checkpoint 的 eval loss 都在下降但生成质量在恶化，可能是什么原因？
A2: TODO
    提示: 考虑 loss 和实际生成质量之间的关系——loss 衡量的是什么？

Q3: 多样性评分太高是否一定好？
A3: TODO
    提示: 极端情况下，随机乱码也有很高的多样性
"""


if __name__ == "__main__":
    print("=" * 50)
    print("  Checkpoint 自动评估与择优")
    print("=" * 50)

    # 示例配置
    BASE_MODEL = "Qwen/Qwen2.5-0.5B"
    CHECKPOINT_DIR = "./sft_output"
    TEST_CASES = [
        {"question": "解释一下机器学习", "reference_answer": "机器学习是..."},
        {"question": "Python 中如何安装第三方库？", "reference_answer": "使用 pip install..."},
        # ... 更多测试用例
    ]

    # TODO: 取消注释完成流程
    # tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    #
    # # 1. 解析日志
    # logs = parse_training_logs(CHECKPOINT_DIR)
    # plot_loss_curves(logs)
    #
    # # 2. 获取 checkpoint 列表
    # checkpoint_dirs = [
    #     d for d in os.listdir(CHECKPOINT_DIR)
    #     if d.startswith("checkpoint-") and os.path.isdir(os.path.join(CHECKPOINT_DIR, d))
    # ]
    #
    # # 3. 每个 checkpoint 生成并评分
    # all_scores = {}
    # for ckpt in checkpoint_dirs:
    #     ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt)
    #     responses = generate_from_checkpoint(BASE_MODEL, ckpt_path, tokenizer, TEST_CASES)
    #     scores = score_responses(responses)
    #     diversity = compute_diversity_score([r["response"] for r in responses])
    #     all_scores[ckpt] = {"scores": scores, "diversity": diversity}
    #
    # # 4. 选出最佳
    # best = select_best_checkpoint(checkpoint_dirs, eval_losses, ...)
    # print(f"\n最佳 Checkpoint: {best['best_checkpoint']}")

    print("\n请完成以上 TODO 后取消注释运行。")
