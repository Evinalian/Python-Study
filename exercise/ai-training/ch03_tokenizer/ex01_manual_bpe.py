"""
练习 1：手动执行 BPE 合并
==========================

给定一个小语料，在纸上（或代码辅助）手动执行 BPE 的前 5 步合并。

语料:
    "low low low low lower lower newest newest newest"

你需要完成:
1. 将每个词表示为字符序列，词尾加 </w>
2. 编写函数统计所有相邻符号对的频率
3. 找出最高频的符号对
4. 执行合并，更新序列
5. 重复步骤 2-4，记录每一步的合并操作
6. 验证最终合并表的前 5 项与教程中的推导一致

思考题:
- 如果最高频的符号对有多个（平局），应该怎么处理？
- </w> 的作用是什么？没有它会有什么问题？
- BPE 的"贪婪"本质（每次只选最高频对）有什么优缺点？
"""

from collections import Counter


# TODO: 初始化语料为字符序列
def initialize_corpus(corpus):
    """
    TODO: 将语料中的每个词拆分为字符序列，词尾加 </w>。

    参数:
        corpus: list[str]，如 ["low", "lower", "newest"]

    返回:
        tokenized: list[list[str]]，如 [["l","o","w","</w>"], ...]
    """
    pass


# TODO: 统计相邻符号对的频率
def count_pairs(tokenized_corpus):
    """
    TODO: 统计所有相邻符号对的频率。

    参数:
        tokenized_corpus: list[list[str]]

    返回:
        Counter，键为 (token_a, token_b)，值为频率
    """
    pass


# TODO: 执行合并操作
def merge_pair(tokenized_corpus, pair):
    """
    TODO: 将语料中所有相邻的 pair[0] + pair[1] 合并为一个 token。

    参数:
        tokenized_corpus: list[list[str]]
        pair: tuple[str, str]，要合并的符号对

    返回:
        merged_corpus: list[list[str]]，合并后的语料
    """
    pass


# TODO: 主函数 - 模拟 BPE 的前 5 步
def simulate_bpe(corpus, num_merges=5):
    """
    TODO: 执行 BPE 的前 num_merges 步合并。

    每步:
    1. 调用 count_pairs 找到最高频的 pair
    2. 打印 pair 和频率
    3. 调用 merge_pair 执行合并
    4. 打印合并后的某些词（如 "lower"）的序列

    返回:
        merge_table: list[tuple]，记录每一步的合并操作
    """
    pass


if __name__ == "__main__":
    corpus = ["low", "low", "low", "low", "lower", "lower", "newest", "newest", "newest"]
    simulate_bpe(corpus, num_merges=8)
