"""
练习 2：可视化注意力权重
=========================

从第02章的 MiniTransformer 中提取注意力权重矩阵，用热力图可视化。

你需要完成:
1. 修改 MultiHeadCausalAttention，使其返回注意力权重（attn_weights）
2. 对一个输入句子进行前向传播，提取每层每个头的注意力权重
3. 选择几个有代表性的句子（短句、长句、含有代词指代的句子），
   用 matplotlib 的 imshow 绘制注意力热力图
4. 分析不同头、不同层的注意力模式差异:
   - 浅层 vs 深层的注意力模式
   - 不同头是否关注不同的语言现象
   - 因果掩码是否正确（上三角是否为零）

思考题:
- 为什么有些头倾向于关注相邻 token？
- 在有代词指代的句子中，能找到关注"指代对象"的头吗？
- 深层和浅层的注意力模式有什么系统性差异？
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# TODO: 修改 MultiHeadCausalAttention，使其可选择性地返回注意力权重


# TODO: 视觉化函数
def plot_attention_heatmap(attn_weights, tokens, title="Attention Weights"):
    """
    TODO: 绘制单个注意力头的热力图。

    参数:
        attn_weights: (seq_len, seq_len) 的注意力权重矩阵
        tokens: token 标签列表，长度 seq_len
        title: 图表标题
    """
    pass  # TODO: 你的代码


# TODO: 分析函数
def analyze_attention_patterns(model, tokenizer, sentences):
    """
    TODO: 对多个句子分析其注意力模式。

    对每个句子:
    1. 编码为 token IDs
    2. 前向传播，提取所有层所有头的注意力权重
    3. 绘制每层第一个头的热力图（或选择性地绘制特定层）
    4. 观察并记录:
       - 浅层（layers[0]）主要关注什么？
       - 深层（layers[-1]）主要关注什么？
       - 是否出现"对角线"模式（每个 token 主要关注自己）？
    """
    pass


if __name__ == "__main__":
    # TODO: 加载模型和 tokenizer，运行分析
    # 建议句子:
    # - "The cat sat on the mat because it was comfortable."
    # - "Yesterday I went to the store and bought some milk."
    # - "Although it was raining, we decided to go for a walk."
    pass
