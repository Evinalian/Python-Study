"""
练习 4（进阶）：用中文数据训练 GPT
===================================

使用中文语料训练一个中文 GPT 模型。

你需要完成:
1. 准备中文训练数据:
   - 下载中文维基百科 dump（或其他中文语料）
   - 提取纯文本并进行基本清洗
2. 加载或训练中文分词器:
   - 使用预训练的中文 BPE/SentencePiece 分词器
   - 或使用第03章的方法从零训练
3. 调整 MiniGPT 配置以适应中文:
   - 中文每个 token 的信息量更大（常用中文字约 3000-5000 个）
   - 可以适当减小 d_model 或增大 vocab_size
   - 考虑使用更大的 max_seq_len（中文更紧凑）
4. 训练并测试:
   - 训练一个小型中文 GPT（约 30-50M 参数）
   - 测试中文文本生成效果
   - 观察生成文本的流利度和连贯性

中文 vs 英文的关键差异:
- 中文每个字符通常是一个语义单元（字），而英文需要多个字符（字母）
- 中文 BPE 分词器通常产生更短的 token 序列
- 中文语法和英文差异大，对位置编码更敏感

部署:
- 如果你有足够的 GPU，尝试训练 > 100M 参数的中文模型
- 可以使用 HuggingFace 的 Trainer API 简化训练流程

思考题:
- 中文 GPT 的生成质量与同等规模的英文 GPT 相比如何？为什么？
- tokenizer 的 vocab_size 对中文模型的影响是否大于英文？
"""

import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path


# TODO: 准备中文数据
def prepare_chinese_corpus(output_dir, max_samples=100000):
    """
    TODO: 下载并处理中文语料。

    选项:
    1. datasets.load_dataset("wikipedia", "20220301.zh")
    2. 使用中文新闻数据集（如 CLUECorpus）
    3. 使用公开的清洗后的中文语料（如 MNBVC）

    保存为纯文本文件。
    """
    pass


# TODO: 加载或训练中文分词器
def get_chinese_tokenizer(vocab_size=32000):
    """
    TODO: 获取中文分词器。

    选项:
    1. 从 HuggingFace 加载预训练的中文分词器:
       - "bert-base-chinese" (BERT 分词器)
       - "THUDM/chatglm-6b" (ChatGLM 分词器，SentencePiece)
    2. 使用 tokenizers 库从零训练（第03章方法）
    """
    pass


# TODO: 训练中文 GPT
def train_chinese_gpt():
    """
    TODO: 使用中文数据训练 MiniGPT。

    建议配置:
    - d_model=512, n_layers=8, n_heads=8, max_seq_len=1024
    - vocab_size=30000（与分词器一致）
    - 训练至少 5000 步

    监控:
    - 训练 loss
    - 验证 perplexity
    - 定期生成样本检查质量
    """
    pass


if __name__ == "__main__":
    train_chinese_gpt()
