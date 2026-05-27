"""
练习 5（进阶）：实现 SimHash 去重
================================

从零实现 SimHash 算法用于近似去重。

你需要完成:
1. 实现 compute_simhash 函数:
   - 将文档 tokenize 为 token 列表
   - 对每个 token 计算 64 位哈希值
   - 维护 64 维浮点向量: 位为 1 则 +1，位为 0 则 -1
   - 二值化: 正值为 1，负值为 0，得到 64 位指纹
2. 实现海明距离计算函数
3. 实现基于 SimHash 的去重:
   - 计算所有文档的 SimHash
   - 找到海明距离 < threshold（如 3）的文档对
   - 保留每个近似重复组中的第一个文档
4. 与 MinHash 方法对比速度和去重效果

思考题:
- SimHash 的 64 位指纹和 MinHash 的 128 维签名哪个更紧凑？
- 海明距离和 Jaccard 相似度在性质上有什么不同？
- 为什么 Google 用 SimHash 做网页去重？（提示: 增量更新）
"""

import hashlib
import struct


# TODO: 计算 token 的 64 位哈希
def hash_token_to_64bit(token):
    """
    TODO: 计算 token 的 64 位哈希值。

    使用 hashlib.md5，取前 8 字节转为 64 位无符号整数。
    """
    pass


# TODO: 计算 SimHash
def compute_simhash(tokens):
    """
    TODO: 计算文档的 SimHash 指纹。

    算法:
    1. 初始化 64 维浮点向量为 0
    2. 对每个 token:
       a. 计算其 64 位哈希
       b. 对每一位: 如果是 1，向量该位 +1；否则 -1
    3. 最终: 向量的正位为 1，负位或零为 0

    返回:
        simhash: int, 64 位无符号整数
    """
    pass


# TODO: 计算海明距离
def hamming_distance(hash1, hash2):
    """
    TODO: 计算两个 64 位整数的海明距离（不同位的数量）。
    提示: 使用 XOR 然后统计 1 的个数: bin(hash1 ^ hash2).count('1')
    """
    pass


# TODO: SimHash 去重
def simhash_dedup(documents, num_perm=128, max_hamming=3):
    """
    TODO: 使用 SimHash 找到近似重复的文档。

    参数:
        documents: 文档文本列表
        max_hamming: 海明距离阈值（<= 此值的文档对视为近似重复）

    返回:
        kept_indices: 去重后保留的文档索引
        duplicate_pairs: 近似重复文档对列表
    """
    pass


# TODO: 对比实验
def compare_simhash_vs_minhash(documents, ground_truth):
    """
    TODO: 在相同的测试数据上比较 SimHash 和 MinHash 的:
    - 运行时间
    - 精确率、召回率、F1
    - 指纹/签名大小

    分析两种方法各自的优劣势。
    """
    pass


if __name__ == "__main__":
    pass
