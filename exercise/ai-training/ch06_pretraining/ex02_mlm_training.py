"""
练习 2：实现 MLM 预训练
========================

在 MiniGPT 的基础上实现 MLM（Masked Language Model）预训练。

你需要完成:
1. 修改 Transformer 的注意力掩码:
   - CLM（原版）: 因果掩码（上三角为 -inf）
   - MLM（新版）: 双向注意力（无因果掩码）
2. 实现随机掩码函数:
   - 随机选择 15% 的位置
   - 其中 80% 替换为 [MASK]
   - 10% 替换为随机 token
   - 10% 保持不变
3. 修改损失计算: 只在被掩码的位置计算 loss
4. 训练并对比 CLM 和 MLM:
   - 固定步数下的 loss 下降速度
   - 最终验证 perplexity
   - 文本生成的可行性（MLM 不能直接生成）

MLM 的掩码策略（BERT 原始设计）:
    在 15% 被选中的位置中:
    - 80%: → [MASK]
    - 10%: → 随机 token
    - 10%: → 原 token（不做改动）
    这样设计是为了:
    - 训练和微调的一致性（微调时没有 [MASK] token）
    - 防止模型只关注 [MASK] token 而忽略上下文

思考题:
- MLM 为什么不能直接用于文本生成？
- MLM 训练中每个样本只有 15% 的 token 被计算 loss。
  相对于 CLM（100% 的 token），是否意味着 MLM 需要更多训练步数？
- [MASK] token 的 embedding 在微调时会被使用吗？
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: 复制教程中的 MiniGPT（但去掉因果掩码）


# TODO: 实现掩码函数
def create_mlm_masking(input_ids, mask_token_id, vocab_size, mlm_prob=0.15):
    """
    TODO: 创建 MLM 掩码。

    参数:
        input_ids: (batch, seq_len)，原始 token IDs
        mask_token_id: [MASK] token 的 ID
        vocab_size: 词表大小
        mlm_prob: 被掩码的概率（默认 15%）

    返回:
        masked_input_ids: 掩码后的输入
        labels: 只在被掩码的位置有原始 token ID，其他位置为 -100（ignore_index）
    """
    pass


# TODO: 修改 MiniGPT 以支持 MLM（去除因果掩码）
class MiniBERT(nn.Module):
    """
    TODO: MLM 版本的 Transformer。

    关键改动:
    - 注意力不使用因果掩码（双向注意力）
    - forward 方法接受 masked_input_ids 和 labels
    - loss 只在被掩码位置计算
    """
    pass


# TODO: 对比实验
def compare_clm_vs_mlm():
    """
    TODO:
    1. 用相同配置训练 CLM 和 MLM 模型
    2. 固定训练步数（如 2000 steps）
    3. 对比:
       - 每步的 loss
       - 最终验证 loss（注意 MLM 只在 15% 位置评估）
       - 训练速度
    """
    pass


if __name__ == "__main__":
    compare_clm_vs_mlm()
