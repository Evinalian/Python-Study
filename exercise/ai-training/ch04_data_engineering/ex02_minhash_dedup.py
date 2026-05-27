"""
练习 2：MinHash 去重实验
========================

使用 datasketch 库实现文档集的 MinHash 去重。

你需要完成:
1. 准备测试文档集（含已知的重复文档）
2. 实现 create_minhash 函数（如教程中）
3. 实现 deduplicate_documents 函数（使用 MinHashLSH）
4. 实验不同参数对去重效果的影响:
   - num_perm: 128, 256, 512
   - threshold: 0.5, 0.7, 0.9
   - ngram_size: 3, 5, 7
5. 对比已知重复的 ground truth，计算精确率和召回率

思考题:
- num_perm 增大时，去重精度和速度如何变化？
- threshold 增大时，被标记为"重复"的文档对增多还是减少？
- ngram_size 为 1（单字符）有什么问题？
"""

from datasketch import MinHash, MinHashLSH
from collections import defaultdict


# TODO: 准备测试数据
def create_test_documents():
    """
    TODO: 创建包含已知重复关系的测试文档集。

    返回:
        documents: list[str]
        ground_truth_duplicates: set of tuples (i, j) 表示文档 i 和 j 是重复的
    """
    pass


# TODO: 实现 MinHash 创建函数
def create_minhash(text, num_perm=128, ngram_size=5):
    """TODO: 参考教程实现"""
    pass


# TODO: 实现去重函数
def deduplicate_with_minhash(documents, num_perm=128, threshold=0.8, ngram_size=5):
    """
    TODO: 使用 MinHash + LSH 去重。

    返回:
        kept_indices: 保留的文档索引列表
        duplicate_pairs: 被标记为重复的文档对列表 [(i, j), ...]
    """
    pass


# TODO: 评估函数
def evaluate_dedup(duplicate_pairs, ground_truth):
    """
    TODO: 计算精确率(precision)、召回率(recall)和 F1。

    精确率 = 正确发现的重复对 / 所有标记为重复的对
    召回率 = 正确发现的重复对 / 所有真实的重复对
    """
    pass


# TODO: 参数实验
def experiment_parameters(documents, ground_truth):
    """
    TODO: 实验不同参数组合的去重效果。

    参数网格:
        num_perm: [64, 128, 256]
        threshold: [0.5, 0.7, 0.9]
        ngram_size: [3, 5, 7]

    对每个组合，计算精确率、召回率、F1、运行时间。
    打印结果表格并找出最佳参数组合。
    """
    pass


if __name__ == "__main__":
    docs, gt = create_test_documents()
    experiment_parameters(docs, gt)
