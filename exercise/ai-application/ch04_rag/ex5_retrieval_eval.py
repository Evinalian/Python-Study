"""
练习 5: 检索策略对比评估

场景:
    设计评估实验，对比不同检索策略在相同测试集上的效果。

评估指标:
    - Precision@K: 前K个结果中相关的比例
    - Recall@K: 前K个结果中相关数 / 总相关数
    - MRR (Mean Reciprocal Rank): 第一个相关结果排名的倒数均值
    - NDCG@K (Normalized Discounted Cumulative Gain)

检索策略:
    - Dense (Embedding + 余弦相似度)
    - BM25 (关键词匹配)
    - Hybrid (Dense + BM25 加权融合)

要求:
    1. 准备测试数据集:
       - 至少 20 篇文档
       - 至少 5 个查询
       - 每个查询标注相关文档列表（ground truth）

    2. 实现评估函数:
       - evaluate_strategy(strategy, queries, ground_truth, K)
       - 返回 Precision@K, Recall@K, MRR

    3. 对比三种策略:
       - 生成对比表格
       - 分析各策略的优劣场景

    4. 调参实验:
       - 测试不同 alpha 值对 Hybrid 的影响

TODO:
    1. 准备测试数据和 ground truth
    2. 实现评估指标函数
    3. 实现 DenseRetriever（使用 OpenAI embeddings）
    4. 实现 HybridRetriever（可调 alpha）
    5. 运行对比实验，输出报告
"""

import json
import math
import numpy as np
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# ============================================================
# TODO 1: 准备测试数据
# ============================================================
def create_test_data() -> tuple[list[dict], list[dict], dict]:
    """
    创建测试数据。

    返回:
        documents: [{"id": "d1", "text": "...", "category": "..."}, ...]
        queries: [{"id": "q1", "text": "..."}, ...]
        ground_truth: {"q1": ["d1", "d3"], "q2": ["d2"], ...}
    """
    # TODO: 至少 20 篇文档，5 个查询，每个查询至少 2 个相关文档
    # 提示: 可以设计不同主题的文档，让某些查询明显与某些文档相关
    pass


# ============================================================
# TODO 2: 评估指标
# ============================================================
def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Precision@K: 前K个中相关的比例"""
    # TODO
    pass


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Recall@K: 前K个中相关的 / 总共相关的"""
    # TODO
    pass


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """
    Mean Reciprocal Rank: 1 / (第一个相关结果的排名)
    如果前K个中没有相关结果，MRR = 0
    """
    # TODO
    pass


def evaluate_strategy(
    search_fn,
    queries: list[dict],
    ground_truth: dict,
    k: int = 5,
) -> dict:
    """
    评估检索策略。

    参数:
        search_fn: 检索函数 (query_text) -> [{"id": "d1", ...}, ...]
        queries: 查询列表
        ground_truth: {"q1": ["d1", "d3"], ...}
        k: 取前K个评估

    返回:
        {"precision@k": 0.7, "recall@k": 0.8, "mrr": 0.75}
    """
    # TODO: 对每个查询执行检索，计算指标，取平均
    pass


# ============================================================
# TODO 3: DenseRetriever
# ============================================================
class DenseRetriever:
    """稠密检索（使用 OpenAI embeddings）"""

    def __init__(self, model: str = "text-embedding-3-small"):
        # TODO
        pass

    def index(self, documents: list[dict]) -> None:
        """建立索引"""
        # TODO: 计算所有文档的 embedding 并存储
        pass

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """检索"""
        # TODO: 计算查询 embedding，余弦相似度排序
        pass


# ============================================================
# TODO 4: HybridRetriever (支持 alpha 调节)
# ============================================================
class HybridRetriever:
    """混合检索"""

    def __init__(self, dense: DenseRetriever, bm25, alpha: float = 0.5):
        # TODO
        pass

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """混合检索 = alpha * dense_norm + (1-alpha) * sparse_norm"""
        # TODO
        pass


# ============================================================
# TODO 5: 对比实验
# ============================================================
def run_comparison():
    """
    运行三种检索策略的对比实验。
    输出对比表格。
    """
    # TODO:
    # 1. 加载测试数据
    # 2. 初始化三种检索器
    # 3. 对各策略调用 evaluate_strategy
    # 4. 打印对比表格
    # 5. 分析结论
    pass


if __name__ == "__main__":
    print("=== 检索策略对比评估 ===\n")
    # run_comparison()
    print("请完成 TODO 后运行。")
