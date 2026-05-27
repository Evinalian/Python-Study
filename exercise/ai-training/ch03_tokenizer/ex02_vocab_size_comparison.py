"""
练习 2：比较不同 vocab_size 的效果
===================================

训练不同 vocab_size 的 BPE 分词器，比较编码效果。

你需要完成:
1. 准备同一份中文语料（可以用教程中的 create_demo_corpus）
2. 训练 3 个 BPE 分词器，vocab_size 分别为 1000、5000、20000
3. 对同一段测试文本，比较:
   - 编码后的 token 数量
   - 平均每个 token 的字符数（信息密度）
   - 某些 token 是否具有语义可解释性
4. 分析词表中未被使用的 token 数量（词汇利用率）

思考题:
- 为什么 vocab_size 越大，产生的 token 数越少？
- 在 vocab_size=1000 时，中文字符'我'是否被打散成了更小的单元？
- 在 vocab_size=20000 时，有多少 token 在训练语料中出现次数 <= 3？
  这些低频 token 对训练有什么影响？
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from pathlib import Path


# TODO: 创建演示语料
def create_corpus(output_file, num_lines=2000):
    """TODO: 参考教程中的 create_demo_corpus，创建中文演示语料"""
    pass


# TODO: 训练不同 vocab_size 的分词器
def train_tokenizers(corpus_file, vocab_sizes):
    """
    TODO: 为每个 vocab_size 训练一个 BPE 分词器。

    参数:
        corpus_file: 训练语料文件路径
        vocab_sizes: list[int]，如 [1000, 5000, 20000]

    返回:
        tokenizers: dict，{vocab_size: Tokenizer实例}
    """
    pass


# TODO: 比较编码效果
def compare_encodings(tokenizers, test_texts):
    """
    TODO: 用不同分词器编码相同的测试文本，记录并对比。

    对每个测试文本和每个分词器:
    - token 数量
    - token 列表
    - 解码是否无损

    打印对比表格。
    """
    pass


# TODO: 分析词表利用率和频率分布
def analyze_vocab(tokenizers):
    """
    TODO: 分析每个分词器的词表。

    - 实际使用的 token 有多少？
    - 词表利用率（使用的 token / vocab_size）
    - 频率分布的 Top-10 和 Bottom-10 token
    """
    pass


if __name__ == "__main__":
    pass
