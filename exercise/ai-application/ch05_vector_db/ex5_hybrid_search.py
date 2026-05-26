"""
练习 5: 混合搜索引擎

场景:
    实现稠密 + 稀疏混合搜索引擎。
    用 ChromaDB 做稠密检索 + BM25 做稀疏检索，加权融合两路结果。

要求:
    1. 实现 HybridSearchEngine 类:
       - __init__(alpha=0.5): alpha 控制稠密vs稀疏的权重
       - index(documents): 同时建立稠密和稀疏索引
       - search(query, top_k): 混合检索

    2. 融合策略:
       - 两路分别检索 top-K_candidate 个结果（K_candidate > top_k）
       - 分数归一化（Min-Max）
       - final_score = alpha * dense_score + (1-alpha) * sparse_score
       - 合并排序取 top_k

    3. 对比实验:
       - 准备测试数据集（至少 10 篇文档，3 个查询，标注 ground truth）
       - 测试三种 alpha: 0.0(纯BM25), 0.5(平衡), 1.0(纯稠密)
       - 输出每种 alpha 的 Precision@3, Recall@3, MRR

    4. 分析:
       - 哪种 alpha 在你的数据集上效果最好？
       - 混合检索相比单一策略改善了哪些查询？

TODO:
    1. 实现 HybridSearchEngine 类
    2. 实现分数归一化和加权融合
    3. 准备测试数据
    4. 运行对比实验
    5. 输出分析报告
"""

import os
import json
import math
import chromadb
from collections import Counter


# ============================================================
# TODO 1: BM25（稀疏检索）
# ============================================================
class BM25Retriever:
    """BM25 检索器（稀疏检索）"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        # TODO
        pass

    def index(self, documents: list[dict]) -> None:
        """
        建立索引。

        参数:
            documents: [{"id": "...", "text": "...", "metadata": {...}}, ...]
        """
        # TODO: 分词 + 计算 TF/DF/avgdl
        pass

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        BM25 检索。

        返回: [{"id": "...", "text": "...", "score": 5.23}, ...]
        """
        # TODO
        pass


# ============================================================
# TODO 2: DenseRetriever（稠密检索）
# ============================================================
class DenseRetriever:
    """稠密检索（ChromaDB + Embedding）"""

    def __init__(self, collection_name: str = "hybrid_dense"):
        # TODO: 初始化 ChromaDB + embedding function
        pass

    def index(self, documents: list[dict]) -> None:
        # TODO: 添加到 ChromaDB
        pass

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        稠密检索。

        返回: [{"id": "...", "text": "...", "score": 0.95}, ...]
        """
        # TODO: ChromaDB query
        pass


# ============================================================
# TODO 3: HybridSearchEngine
# ============================================================
class HybridSearchEngine:
    """
    混合搜索引擎。

    final_score = alpha * dense_norm + (1-alpha) * sparse_norm
    """

    def __init__(self, alpha: float = 0.5):
        """
        参数:
            alpha: 稠密检索权重 (0=纯BM25, 1=纯稠密)
        """
        self.alpha = alpha
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        # TODO: 初始化

    def index(self, documents: list[dict]) -> None:
        """建立双向索引"""
        # TODO: 同时索引到 bm25 和 dense
        pass

    def search(self, query: str, top_k: int = 5, candidate_k: int = 20) -> list[dict]:
        """
        混合检索。

        步骤:
        1. 两路各检索 candidate_k 个候选
        2. 分数归一化 (Min-Max)
        3. 加权融合
        4. 排序返回 top_k
        """
        # TODO: 实现
        pass

    def _normalize_scores(self, results: list[dict], score_key: str) -> list[dict]:
        """Min-Max 归一化"""
        # TODO
        pass


# ============================================================
# TODO 4: 对比实验
# ============================================================
def run_experiment():
    """
    运行 alpha 对比实验。

    测试 alpha = 0.0, 0.3, 0.5, 0.7, 1.0
    输出各 alpha 下的 Precision@3, Recall@3, MRR
    """
    # TODO:
    # 1. 准备测试数据
    # 2. 对每个 alpha 值:
    #    a. 创建 HybridSearchEngine(alpha)
    #    b. 索引文档
    #    c. 对所有查询搜索并评估
    # 3. 打印对比表格
    # 4. 分析最佳 alpha
    pass


if __name__ == "__main__":
    print("=== 混合搜索引擎 ===\n")
    # run_experiment()
    print("请完成 TODO 后运行。")
