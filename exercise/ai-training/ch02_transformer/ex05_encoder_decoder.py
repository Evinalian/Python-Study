"""
练习 5（进阶）：扩展为 Encoder-Decoder Transformer
=================================================

在 Decoder-Only 架构基础上添加 Encoder 模块，实现完整的 Encoder-Decoder Transformer。

核心改动:
1. Encoder 使用双向注意力（无因果掩码）
2. Cross-Attention: Decoder 的每一层除了 Self-Attention，还增加 Cross-Attention
   层，其中 Q 来自 Decoder 的隐藏状态，K/V 来自 Encoder 的输出
3. 经典的 Encoder-Decoder 架构: Encoder 编码源序列，Decoder 自回归生成目标序列

你需要完成:
1. 实现 EncoderBlock（含 Self-Attention + FFN，使用双向注意力）
2. 实现 CrossAttention（Q 来自 Decoder, K/V 来自 Encoder）
3. 修改 DecoderBlock，在 Self-Attention 和 FFN 之间插入 Cross-Attention
4. 实现 EncoderDecoderTransformer 完整模型
5. 用一个简单的翻译任务验证你的实现

参考架构:
    Encoder:
        Input → Embedding → [EncoderBlock × N] → Encoder Output

    Decoder:
        Input → Embedding → [
            Self-Attention (因果) → Cross-Attention (Encoder) → FFN
        ] × N → LM Head → Output

思考题:
- 为什么 Encoder 使用双向注意力而 Decoder 使用因果注意力？
- Cross-Attention 在机器翻译中起什么作用？
- Encoder-Decoder 和 Decoder-Only（GPT 式）相比，各自的优缺点是什么？
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# TODO: 实现 EncoderBlock（双向注意力，无因果掩码）
class EncoderBlock(nn.Module):
    """
    Encoder Block: Self-Attention (双向) + FFN
    """
    pass  # TODO: 你的代码


# TODO: 实现 Cross-Attention 层
class CrossAttention(nn.Module):
    """
    Cross-Attention: Q 来自 Decoder, K/V 来自 Encoder 输出。

    无因果掩码，Decoder 的每个位置可以 attend 到所有 Encoder 位置。
    """
    pass  # TODO: 你的代码


# TODO: 修改 DecoderBlock，加入 Cross-Attention
class DecoderBlockWithCrossAttention(nn.Module):
    """
    Decoder Block with Cross-Attention:
        Self-Attention (因果) → Cross-Attention (Encoder) → FFN
    """
    pass  # TODO: 你的代码


# TODO: EncoderDecoderTransformer 完整模型
class EncoderDecoderTransformer(nn.Module):
    """
    完整 Encoder-Decoder Transformer。

    forward(encoder_input, decoder_input) → logits
    """
    pass  # TODO: 你的代码


if __name__ == "__main__":
    # TODO: 创建模型并测试前向传播
    pass
