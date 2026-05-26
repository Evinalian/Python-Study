"""
练习 3: 语义搜索 vs 关键词搜索对比

场景:
    对比语义搜索（ChromaDB + Embedding）和关键词搜索（BM25）的检索效果。

要求:
    1. 构建测试文档集（至少 15 篇，覆盖 3-4 个主题）
    2. 实现两种搜索:
       - SemanticSearch: ChromaDB + sentence-transformers embedding
       - KeywordSearch: BM25（用 rank_bm25 库或自己实现）
    3. 设计 5 个测试查询，标注每个查询的相关文档
    4. 评估指标: Precision@3, Recall@3, MRR
    5. 分析:
       - 哪种查询语义搜索更好？（同义词、口语化）
       - 哪种查询关键词搜索更好？（精确匹配、专有名词）
       - 两者的结果重叠度

TODO:
    1. 准备测试文档和 ground truth
    2. 实现 SemanticSearch 类
    3. 实现 KeywordSearch 类（BM25）
    4. 实现评估函数
    5. 输出对比报告
"""

import os
import json
import math
from collections import Counter

# ============================================================
# TODO 1: 准备测试数据
# ============================================================
def create_test_data() -> tuple[list[dict], list[dict], dict]:
    """
    创建测试数据。

    返回:
        documents: [{"id": "d1", "text": "...", "topic": "..."}, ...]  (至少 15 篇)
        queries: [{"id": "q1", "text": "...", "type": "semantic/keyword/mixed"}, ...]  (至少 5 个)
        ground_truth: {"q1": ["d1", "d3"], ...}  (每个查询 2-5 个相关文档)

    注意事项:
    - 包含需要语义理解的查询（同义词、改写）
    - 包含需要精确关键词匹配的查询（专有名词、数字编号）
    """
    # TODO: 创建测试数据，确保文档涵盖不同主题
    pass


# ============================================================
# TODO 2: SemanticSearch
# ============================================================
class SemanticSearch:
    """基于 ChromaDB 的语义搜索"""

    def __init__(self):
        # TODO: 初始化 ChromaDB + embedding function
        pass

    def index(self, documents: list[dict]) -> None:
        # TODO: 将文档加入 ChromaDB
        pass

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """
        语义搜索。

        返回: ["d1", "d3", "d5"] (按相关度排序的文档 ID)
        """
        # TODO
        pass


# ============================================================
# TODO 3: KeywordSearch (BM25)
# ============================================================
class KeywordSearch:
    """基于 BM25 的关键词搜索"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        # TODO: 初始化 BM25 参数
        pass

    def index(self, documents: list[dict]) -> None:
        # TODO: 分词 + 建立倒排索引
        # 使用 jieba 分词（如果可用）
        pass

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """
        关键词搜索。

        返回: ["d1", "d3", "d5"] (按 BM25 分数排序的文档 ID)
        """
        # TODO: 实现 BM25 检索
        pass

    def _tokenize(self, text: str) -> list[str]:
        """分词"""
        # TODO
        pass


# ============================================================
# TODO 4: 评估函数
# ============================================================
def evaluate(
    semantic_results: dict,
    keyword_results: dict,
    ground_truth: dict,
    k: int = 3,
) -> dict:
    """
    评估两种搜索策略。

    参数:
        semantic_results: {"q1": ["d1", "d3"], ...}
        keyword_results: {"q1": ["d2", "d1"], ...}
        ground_truth: {"q1": ["d1", "d3"], ...}
        k: 取前 K 个结果评估

    返回:
        {
            "semantic": {"precision@k": 0.7, "recall@k": 0.8, "mrr": 0.75},
            "keyword": {"precision@k": 0.5, "recall@k": 0.6, "mrr": 0.50},
        }
    """
    # TODO:
    # 对每个查询:
    #   - precision@k = 前K个中相关的数量 / K
    #   - recall@k = 前K个中相关的数量 / 所有相关的数量
    #   - MRR = 平均(1 / 第一个相关结果的排名)
    pass


# ============================================================
# TODO 5: 对比报告
# ============================================================
def run_comparison():
    """运行完整的对比实验并打印报告"""
    # TODO:
    # 1. 创建测试数据
    # 2. 初始化两种搜索
    # 3. 索引文档
    # 4. 对所有查询运行搜索
    # 5. 评估
    # 6. 打印对比表格和分析
    pass


if __name__ == "__main__":
    print("=== 语义搜索 vs 关键词搜索 ===\n")
    # run_comparison()
    print("请完成 TODO 后运行。")
