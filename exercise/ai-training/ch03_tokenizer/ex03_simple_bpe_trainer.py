"""
练习 3：实现一个最简 BPE 训练器
================================

用纯 Python 实现一个最小化的 BPE 训练器（不考虑 Unicode 规范化）。

你需要完成:
1. 实现 SimpleBPETrainer 类
2. 输入: 词频字典 {word: frequency}
3. 输出: 合并规则列表 [(token_a, token_b), ...] 和最终词表
4. 使用 heap（优先队列）高效地找到最高频的符号对
   - 每一次合并后，不需要重新统计所有 pairs
   - 只需要更新受影响的 pairs

数据结构提示:
- 使用 collections.Counter 或 heapq
- 维护一个 pair_freq 字典，键为 (a, b)，值为频率
- 合并 (a, b) -> ab 后:
  - 原来 ...c, a... 和 a, b... + b, d... 的关系变为 ...c, ab... 和 ...ab, d...
  - 删除涉及 a 作为第二个元素、b 作为第一个元素的 pairs
  - 新增涉及 ab 的 pairs

思考题:
- 为什么使用优先队列很重要？（提示: 考虑大规模词表下的性能）
- 合并规则为什么用列表存储（而非字典）？（提示: 顺序重要）
- 如果语料很大（数十亿词），你的实现能处理吗？瓶颈在哪里？
"""

from collections import defaultdict, Counter
import heapq
import json


# TODO: 实现 SimpleBPETrainer
class SimpleBPETrainer:
    """
    最简 BPE 训练器。

    TODO: 实现以下方法:
    - __init__: 初始化
    - _initialize_vocab: 从词频字典创建初始字符级词表
    - _count_pairs: 统计所有相邻符号对
    - _find_best_pair: 找到当前最高频的符号对
    - _merge: 执行一次合并
    - train: 主训练循环
    """

    def __init__(self, vocab_size=1000, min_frequency=1):
        """
        参数:
            vocab_size: 目标词表大小
            min_frequency: 最低合并频率
        """
        pass  # TODO: 你的代码

    def _initialize_vocab(self, word_freqs):
        """
        TODO: 初始化词表。
        将每个词拆为字符序列，词尾加 </w>。

        参数:
            word_freqs: dict {word: frequency}

        返回:
            splits: dict {word: [字符列表]}
            vocab: set，初始字符集合
        """
        pass

    def _count_pairs(self, splits):
        """
        TODO: 统计所有相邻符号对的频率。

        返回:
            pair_freqs: dict {(a, b): frequency}
        """
        pass

    def _merge(self, splits, best_pair):
        """
        TODO: 合并符号对。将 best_pair[0] + best_pair[1] 合并为 best_pair[0]+best_pair[1]。

        对 splits 中所有的词，将其序列中的连续 best_pair[0] + best_pair[1] 替换为合并后的 token。
        """
        pass

    def train(self, word_freqs):
        """
        TODO: 主训练循环。

        1. 初始化词表和切分
        2. 统计 pairs 频率
        3. 循环直到词表达目标大小:
           a. 找最高频 pair
           b. 如果频率 < min_frequency，停止
           c. 执行合并
           d. 记录合并规则

        返回:
            merges: list[tuple]，合并规则列表（按顺序）
            vocab: set，最终词表
        """
        pass


# TODO: 测试
def test_simple_bpe():
    """TODO: 用教程中的例子测试你的实现"""
    word_freqs = {
        "low": 4,
        "lower": 2,
        "newest": 3,
        "widest": 1,
    }
    pass


if __name__ == "__main__":
    test_simple_bpe()
