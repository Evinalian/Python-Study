"""
练习 3：数据打包优化
====================

实现数据打包的贪心算法，对比打包和不打包的计算利用率。

你需要完成:
1. 生成不同长度分布的模拟文档集:
   - 场景 A: 大部分是短文档（平均 100 tokens）
   - 场景 B: 大部分是长文档（平均 2000 tokens）
   - 场景 C: 长短混合（均匀分布）
2. 实现贪心打包算法（按长度降序 + first-fit）
3. 实现最优打包（使用 binpacking 库的 first-fit-decreasing）
4. 对比三种场景下打包和不打包的:
   - 序列利用率 (总 token / (n_packs × max_seq_len))
   - padding 浪费的百分比
5. 讨论: 在什么场景下打包最有效？

思考题:
- 如果文档都是 max_seq_len 的整数倍，打包是否还有意义？
- 打包后，attention mask 需要怎么处理？
- 打包和 padding 在训练 loss 计算上有什么区别？
"""

import random
import math


# TODO: 生成模拟文档
def generate_documents(num_docs, length_distribution, min_len=10, max_len=10000):
    """
    TODO: 生成指定分布的模拟文档长度列表。

    参数:
        num_docs: 文档数量
        length_distribution: 'short', 'long', 'uniform'
        min_len, max_len: 长度范围

    返回:
        doc_lengths: list[int]
    """
    pass


# TODO: 贪心打包
def greedy_packing(doc_lengths, max_seq_len, eos_overhead=1):
    """
    TODO: 贪心打包算法。

    1. 按长度降序排序
    2. 对每个文档:
       a. 尝试放入当前 pack（当前长度 + doc_len + EOS <= max_seq_len）
       b. 如果放不下，关闭当前 pack，开新 pack
    3. 返回 packs 列表

    返回:
        packs: list[list[int]]，每个 pack 是一组文档长度
        utilization: float，利用率
    """
    pass


# TODO: 对比打包 vs 不打包
def compare_packing_vs_nopacking(doc_lengths, max_seq_len):
    """
    TODO:
    1. 不打包: 每个文档独立占一个序列（padding 到 max_seq_len）
       - 利用率 = sum(doc_lengths) / (n_docs * max_seq_len)
    2. 打包: 多个文档拼入一个序列
       - 利用率 = sum(doc_lengths) / (n_packs * max_seq_len)
    3. 打印对比结果
    """
    pass


if __name__ == "__main__":
    max_seq_len = 2048
    for dist in ['short', 'long', 'uniform']:
        doc_lengths = generate_documents(1000, dist)
        compare_packing_vs_nopacking(doc_lengths, max_seq_len)
