"""
练习 5（进阶）：实现 WordPiece 训练器
====================================

实现 WordPiece 训练器的核心逻辑。WordPiece 和 BPE 的唯一区别在于选择标准:
- BPE: 合并频率最高的 pair
- WordPiece: 合并 score = count(a,b) / (count(a) * count(b)) 最高的 pair

你需要完成:
1. 修改 ex03_simple_bpe_trainer.py 中的 SimpleBPETrainer，
   将合并选择标准从"频率"改为"WordPiece score"
2. 在同一语料上训练 BPE 和 WordPiece，比较两者的:
   - 合并顺序的差异
   - 最终词表的差异
   - 对"常见词"和"罕见词"的拆分差异
3. 分析 WordPiece 的合并标准为什么更合理

WordPiece 选择标准推导:
    最优合并 = argmax_{a,b} P(ab) / (P(a) * P(b))
              = argmax_{a,b} count(ab) / (count(a) * count(b))

思考题:
- 为什么 count(ab) / (count(a) * count(b)) 比 count(ab) 更好？
- 在什么情况下两者的选择会明显不同？
- BERT 的 ## 前缀有什么用？在你的实现中需要吗？
"""

from collections import defaultdict, Counter


# TODO: 实现 WordPiece 训练器
class WordPieceTrainer:
    """
    WordPiece 训练器。

    和 BPE 的区别:
    - 合并标准: count(a,b) / (count(a) * count(b)) 最高
    - BERT 风格的 token 前缀: 首 token 不加 ##，后续 token 加 ##
    """

    def __init__(self, vocab_size=1000, min_frequency=1):
        pass  # TODO: 你的代码

    def _compute_wordpiece_score(self, pair, pair_freq, token_freq):
        """
        TODO: 计算 WordPiece 的合并分数。

        参数:
            pair: (token_a, token_b)
            pair_freq: count(pair) → pair 的出现次数
            token_freq: count(token) → 每个 token 的出现次数

        返回:
            score: count(a,b) / (count(a) * count(b))

        注意: 这里的 count 是出现次数（频率），不是概率。
        """
        pass

    def train(self, word_freqs):
        """
        TODO: WordPiece 训练主循环。

        与 BPE 相同，但每步选择 score 最高的 pair。
        """
        pass


# TODO: 对比 BPE vs WordPiece
def compare_bpe_vs_wordpiece(word_freqs, vocab_size=500):
    """
    TODO: 在同一语料上训练 BPE 和 WordPiece，比较:
    1. 前 10 步合并操作的差异
    2. 最终词表的大小和内容
    3. 找几个词如 "unhappiness", "running", "newest"
       比较两种算法对它们的拆分
    """
    pass


if __name__ == "__main__":
    # 测试语料
    test_word_freqs = {
        "un": 100, "able": 80, "unable": 40,
        "run": 200, "ning": 150, "running": 100,
        "happiness": 50, "unhappiness": 30,
    }
    compare_bpe_vs_wordpiece(test_word_freqs, vocab_size=200)
