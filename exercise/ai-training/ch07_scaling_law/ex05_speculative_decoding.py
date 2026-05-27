"""
练习 5（进阶）：实现 Speculative Decoding
========================================

实现 Speculative Decoding（投机解码）来加速文本生成。

你需要完成:
1. 训练或加载两个模型:
   - Target Model（大模型，如 50M 参数）: 需要精确但慢
   - Draft Model（小模型，如 5M 参数）: 可以粗糙但快
2. 实现 Speculative Decoding 的核心逻辑:
   a. Draft Model 自回归生成 K 个候选 token
   b. Target Model 一次前向验证所有 K 个 token
   c. 修正采样:
      对每个候选 token x_i:
        接受概率 = min(1, p_target(x_i | ...) / p_draft(x_i | ...))
      如果拒绝:
        从 max(0, p_target - p_draft) 的归一化分布中重新采样
        后续候选全部丢弃
3. 测量加速比:
   - 标记是否接受（比较 target 和 draft 的输出概率）
   - 生成相同数量 token 的时间对比
   - 不同 K（1, 3, 5, 10）下的加速比
4. 验证输出分布一致性:
   - 统计大量生成中的 token 频率分布
   - 确认 Speculative Decoding 的输出分布与纯自回归一致

修正采样的数学:
    对于位置 i，draft 生成了 token x_i:
    - r = p_target(x_i) / p_draft(x_i)
    - 以概率 min(1, r) 接受 x_i
    - 如果接受，继续验证 x_{i+1}
    - 如果拒绝，从分布 max(0, p_target - p_draft) / sum(...) 中采样

思考题:
- Speculative Decoding 的加速比上限是多少？
- Draft Model 和 Target Model 的相似度如何影响加速比？
- 如果 Draft Model 和 Target Model 完全相同，加速比是多少？
"""

import torch
import torch.nn.functional as F
import time
import copy


# TODO: 加载/训练 Draft Model 和 Target Model
def load_models():
    """
    TODO: 加载两个不同规模的模型。

    Draft Model: 约 5M 参数的小模型
    Target Model: 约 50M 参数的大模型

    两者使用相同的分词器。
    """
    pass


# TODO: 修正采样
def rejection_sampling(target_probs, draft_probs, draft_token):
    """
    TODO: 修正采样算法。

    参数:
        target_probs: (vocab_size,) Target 模型的概率分布
        draft_probs: (vocab_size,) Draft 模型的概率分布
        draft_token: int, Draft 模型生成的 token

    返回:
        (accepted_token, is_accepted)

    如果 is_accepted:
        accepted_token = draft_token
    否则:
        accepted_token 从 max(0, target_probs - draft_probs) 归一化后采样
    """
    pass


# TODO: Speculative Decoding
def speculative_generate(target_model, draft_model, input_ids,
                         max_new_tokens=100, K=5, temperature=1.0):
    """
    TODO: 实现 Speculative Decoding。

    算法:
    while len(generated) < max_new_tokens:
        1. Draft 模型自回归生成 K 个候选 token (或直到生成 EOS)
        2. Target 模型对输入 + K 个候选 token 做一次前向传播
        3. 修正采样: 按顺序验证每个候选 token
        4. 接受正确的 token，拒绝后从调整分布重新采样
        5. 更新输入序列

    返回:
        generated_ids: 生成的 token 序列
        stats: {'acceptance_rate': ..., 'tokens_per_step': ...}
    """
    pass


# TODO: 性能对比
def benchmark_speculative_decoding(target_model, draft_model, input_ids,
                                   K_values=[1, 3, 5, 10],
                                   max_new_tokens=100):
    """
    TODO: 对比不同 K 值的 Speculative Decoding 性能。

    对每个 K:
    1. 运行 speculative_generate，测量时间
    2. 计算 acceptance_rate 和 tokens_per_step
    3. 与纯自回归（K=1）比较加速比

    绘制:
    - 加速比 vs K 的图
    - acceptance_rate vs K 的图
    """
    pass


# TODO: 验证输出分布
def verify_output_distribution(target_model, draft_model, input_ids,
                                num_samples=100):
    """
    TODO: 验证 Speculative Decoding 的输出分布是否与纯自回归一致。

    统计大量生成中每个 token 的出现频率，比较两种方法的分布。
    """
    pass


if __name__ == "__main__":
    pass
