"""
练习 3: BM25 检索器

场景:
    从头实现一个 BM25 检索器，使用 jieba 进行中文分词。

要求:
    1. 实现 BM25Retriever 类:
       - add_documents(docs): 建立索引
       - search(query, top_k): 检索
       - _tokenize(text): 中文分词（使用 jieba）

    2. BM25 公式:
       score(D, Q) = sum(IDF(qi) * TF(qi, D) * (k1+1) / (TF(qi, D) + k1*(1-b+b*|D|/avgdl)))
       其中:
       - IDF(qi) = log((N - df + 0.5) / (df + 0.5))
       - |D| = 文档长度
       - avgdl = 平均文档长度
       - k1 = 1.5 (词频饱和度), b = 0.75 (长度归一化)

    3. 对比实验:
       - 准备一个文档集（至少 10 篇不同主题的短文）
       - 用 BM25 搜索 3 个不同查询
       - 手动标注"相关/不相关"（或基于预设的 ground truth）
       - 输出 Precision@3 和 Recall@3

TODO:
    1. 实现 BM25Retriever 类
    2. 准备测试文档集（至少 10 篇）
    3. 实现 evaluate 函数
    4. 运行评估并输出结果
"""

import math
import re
from collections import Counter


# 尝试导入 jieba
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("建议安装 jieba: pip install jieba")


class BM25Retriever:
    """BM25 检索器"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        # TODO: 初始化参数和索引结构
        pass

    def add_documents(self, documents: list[dict]) -> None:
        """
        添加文档并建立索引。

        参数:
            documents: [{"id": "doc1", "text": "...", "metadata": {...}}, ...]

        需要计算:
        - 每篇文档的词频 (TF)
        - 每个词的文档频率 (DF)
        - 平均文档长度 (avgdl)
        """
        # TODO:
        # 1. 对每篇文档分词
        # 2. 计算 TF
        # 3. 计算 DF
        # 4. 计算 avgdl
        pass

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        BM25 检索。

        返回:
        [{"id": "doc1", "text": "...", "score": 5.23, "metadata": {...}}, ...]
        """
        # TODO:
        # 1. 对查询分词
        # 2. 对每篇文档计算 BM25 分数
        # 3. 排序并返回 top_k
        pass

    def _tokenize(self, text: str) -> list[str]:
        """
        中文分词。

        如果 jieba 可用，使用 jieba.lcut。
        否则回退到简单的字符级分词（单字 + 双字）。
        """
        # TODO:
        if JIEBA_AVAILABLE:
            pass  # return list(jieba.cut(text))
        else:
            # 简单回退: 按字符切分
            pass


# ============================================================
# TODO: 准备测试文档集
# ============================================================
def create_test_documents() -> list[dict]:
    """
    创建至少 10 篇不同主题的测试文档。

    每篇文档:
    {
        "id": "doc_001",
        "text": "文档内容...",
        "metadata": {"category": "技术/体育/财经/..."},
        "relevant_queries": ["查询1", "查询2"],  # 本文档被认为相关的查询
    }
    """
    # TODO: 创造测试数据
    pass


# ============================================================
# TODO: 评估函数
# ============================================================
def evaluate_bm25(retriever: BM25Retriever, queries: list[str], ground_truth: dict) -> dict:
    """
    评估 BM25 检索效果。

    参数:
        retriever: 已建好索引的检索器
        queries: 测试查询列表
        ground_truth: {"查询1": ["doc_001", "doc_003"], "查询2": [...]}

    返回:
    {
        "precision_at_3": 0.67,
        "recall_at_3": 0.80,
        "mrr": 0.75,  # Mean Reciprocal Rank
        "per_query": {...}
    }
    """
    # TODO:
    # Precision@K = 前K个结果中相关的数量 / K
    # Recall@K = 前K个结果中相关的数量 / 总共相关的数量
    # MRR = 平均(1 / 第一个相关结果的排名)
    pass


if __name__ == "__main__":
    print("=== BM25 检索器测试 ===\n")

    # 创建测试数据
    # docs = create_test_documents()
    # print(f"测试文档数: {len(docs)}")

    # 建立索引
    # bm25 = BM25Retriever()
    # bm25.add_documents(docs)

    # 搜索
    # results = bm25.search("Python数据分析", top_k=5)

    # 评估
    # report = evaluate_bm25(bm25, ["查询1", "查询2"], {...})
    # print(report)

    print("请完成 TODO 后运行。")
