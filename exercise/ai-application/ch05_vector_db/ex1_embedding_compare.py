"""
练习 1: Embedding 模型对比

场景:
    对比 OpenAI text-embedding-3-small 和开源模型（如 BGE）的相似度计算结果。

要求:
    1. 实现两个 Embedding 类:
       - OpenAIEmbeddings: 封装 OpenAI API
       - LocalEmbeddings: 封装 sentence-transformers

    2. 准备测试数据:
       - 至少 5 对语义相近的文本
       - 至少 5 对语义不相近的文本
       - 包含中英文混合测试

    3. 比较:
       - 分别用两个模型计算相似度
       - 计算 Spearman 相关系数（排序一致性）
       - 分析差异（哪个模型对中文更好？对特定领域？）

TODO:
    1. 实现 OpenAIEmbeddings 类
    2. 实现 LocalEmbeddings 类
    3. 准备测试文本对
    4. 实现 compare_models 函数
    5. 输出对比报告
"""

import os
import time
import numpy as np
from typing import List, Tuple
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# ============================================================
# TODO 1: OpenAIEmbeddings 类
# ============================================================
class OpenAIEmbeddings:
    """OpenAI Embedding 模型封装"""

    def __init__(self, model: str = "text-embedding-3-small"):
        # TODO: 初始化
        pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        # TODO: 调用 client.embeddings.create，分批处理
        pass

    def similarity(self, a: list[float], b: list[float]) -> float:
        # TODO: 余弦相似度
        pass


# ============================================================
# TODO 2: LocalEmbeddings 类
# ============================================================
class LocalEmbeddings:
    """本地 Embedding 模型封装（sentence-transformers）"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        # TODO: 初始化 SentenceTransformer
        pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        # TODO: model.encode(texts, normalize_embeddings=True)
        pass

    def similarity(self, a: list[float], b: list[float]) -> float:
        # TODO: 余弦相似度
        pass


# ============================================================
# TODO 3: 测试数据
# ============================================================
def create_test_pairs() -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """
    创建测试文本对。

    返回:
        similar_pairs: [(text_a, text_b, description), ...]  语义相近的对
        dissimilar_pairs: [(text_a, text_b, description), ...]  语义不相近的对

    提示: 每个 pair 包含 (文本A, 文本B, 描述)
    例如:
        ("Python编程入门", "学习Python基础知识", "同义改写")
        ("Python编程入门", "今天天气很好", "无关话题")
    """
    # TODO: 至少各 5 对
    pass


# ============================================================
# TODO 4: 对比函数
# ============================================================
def compare_models():
    """
    对比两个模型的相似度计算结果。

    输出内容:
    1. 每对文本的相似度对比表
    2. 相似对 vs 不相干对的区分度
    3. 平均相似度差异
    4. （可选）Spearman 相关系数
    """
    # TODO:
    # 1. 加载测试数据
    # 2. 用两个模型分别计算所有文本对的相似度
    # 3. 计算统计指标
    # 4. 打印对比报告
    pass


if __name__ == "__main__":
    print("=== Embedding 模型对比 ===\n")
    # compare_models()
    print("请完成 TODO 后运行。")
