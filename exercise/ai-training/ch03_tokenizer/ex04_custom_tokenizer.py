"""
练习 4（进阶）：训练真正的自定义分词器
======================================

找一份中文语料，训练一个实际的中文 BPE 分词器。

你需要完成:
1. 下载中文维基百科 dump（或使用其他公开中文语料）
2. 提取纯文本
3. 使用 HuggingFace tokenizers 训练 BPE 分词器（vocab_size >= 32000）
4. 分析:
   - Token 频率分布: 前 100 个 token 占总出现次数的比例？
   - 找出完全没被使用过的 token（如存在）
   - 测试中英混合文本的编码效果
5. 比较你的分词器和 GPT-2 分词器（英文 BPE）对中文的编码效率

思考题:
- 中文 BPE 和英文 BPE 产生的 token 序列长度有什么差异？
- 为什么中文通常需要更大的 vocab_size？
- SentencePiece Unigram 和 BPE 哪个更适合中文？为什么？
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from pathlib import Path
from collections import Counter
import json


# TODO: 下载并处理中文语料
def download_and_process_corpus(output_dir):
    """
    TODO: 下载中文维基百科 dump 的简化版本。

    可以使用:
    - datasets 库: load_dataset("wikipedia", "20220301.zh")
    - 或直接下载 zhwiki dump: https://dumps.wikimedia.org/zhwiki/

    提取纯文本，保存到 output_dir/corpus.txt
    """
    pass


# TODO: 训练中文 BPE 分词器
def train_chinese_bpe(corpus_file, vocab_size=32000):
    """
    TODO: 使用 HuggingFace tokenizers 训练 BPE 分词器。

    考虑中文的特点:
    - 不需要小写化
    - 预分词可能需要特殊处理（中文无空格）
    - 可以考虑使用 Unigram 算法（SentencePiece 默认）
    """
    pass


# TODO: 分析词表
def analyze_vocab(tokenizer, test_texts):
    """
    TODO:
    1. 计算前 100 个 token 的累计频率
    2. 找出出现次数为 0 的 token（如果存在）
    3. 测试中英混合文本的编码长度
    4. 与 GPT-2 分词器比较编码效率
    """
    pass


if __name__ == "__main__":
    pass
