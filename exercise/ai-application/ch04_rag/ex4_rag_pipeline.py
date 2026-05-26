"""
练习 4: 端到端 RAG 文档问答系统

场景:
    构建一个完整的文档问答系统，整合文档加载、文本切块、检索、生成。

要求:
    1. 实现 RAGPipeline 类:
       - index_documents(file_paths): 离线索引
       - query(question): 在线查询

    2. 离线索引流程:
       加载文档 → 切块 → 计算 embedding → 存入向量存储

    3. 在线查询流程:
       问题向量化 → 检索 top-k → 构建 prompt → LLM 生成

    4. 支持:
       - 加载 PDF/TXT 文档
       - 回答问题时标注来源
       - 当资料不足时明确告知

    5. 测试: 用几篇文档构建知识库，回答与文档内容相关的问题

TODO:
    1. 实现 RAGPipeline 类（含 index_documents 和 query）
    2. 准备测试文档（可以用教程内容或新闻文章）
    3. 运行问答测试
"""

import os
import json
import numpy as np
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# ============================================================
# TODO: SimpleVectorStore（内存向量存储）
# ============================================================
class SimpleVectorStore:
    """
    简单的内存向量存储。

    存储: [{"text": "...", "embedding": [...], "metadata": {...}}, ...]
    """

    def __init__(self):
        # TODO: 初始化 _documents 列表
        pass

    def add(self, texts: list[str], embeddings: list[list[float]], metadatas: list[dict] = None):
        """批量添加文档"""
        # TODO
        pass

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        余弦相似度检索。

        返回: [{"text": "...", "score": 0.95, "metadata": {...}}, ...]
        """
        # TODO: 计算余弦相似度，排序返回 top_k
        pass

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """余弦相似度"""
        # TODO
        pass


# ============================================================
# TODO: TextSplitter（复用练习2的实现或简化版）
# ============================================================
class SimpleTextSplitter:
    """简化的文本切分器"""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        # TODO
        pass

    def split(self, text: str) -> list[str]:
        # TODO: 按段落切分，确保每块在 chunk_size 以内
        pass


# ============================================================
# TODO: RAGPipeline
# ============================================================
class RAGPipeline:
    """
    RAG 管道。

    离线:
        index_documents(file_paths) → 加载 → 切块 → embedding → 存储

    在线:
        query(question) → embedding → 检索 → 生成
    """

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embedding_model = embedding_model
        self.vector_store = SimpleVectorStore()
        self.splitter = SimpleTextSplitter()
        # TODO: 初始化
        pass

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        使用 OpenAI embedding API 计算向量。

        提示:
        response = client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [item.embedding for item in response.data]
        """
        # TODO: 实现
        pass

    def index_documents(self, file_paths: list[str]) -> None:
        """
        离线索引文档。

        步骤:
        1. 加载每个文件（支持 .txt, .md）
        2. 切块
        3. 计算 embedding
        4. 存入 vector_store

        提示:
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = self.splitter.split(text)
            embeddings = self.get_embeddings(chunks)
            metadatas = [{"source": file_path, "chunk_index": i} for i in range(len(chunks))]
            self.vector_store.add(chunks, embeddings, metadatas)
        """
        # TODO: 实现
        pass

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        在线查询。

        返回:
        {
            "answer": "基于文档的回答",
            "sources": ["源文件1", "源文件2"],
            "retrieved_chunks": [{"text": "...", "score": 0.95}, ...]
        }

        提示:
        1. 计算问题的 embedding
        2. 检索 top_k 相关文档块
        3. 构建 RAG prompt
        4. 调用 LLM 生成回答
        """
        # TODO: 实现
        pass

    def _build_rag_prompt(self, question: str, chunks: list[dict]) -> str:
        """构建 RAG prompt"""
        # TODO:
        # context = "\n\n---\n\n".join([c['text'] for c in chunks])
        # prompt = f"参考资料:\n{context}\n\n问题: {question}\n\n基于以上资料回答，标注来源。"
        pass


if __name__ == "__main__":
    print("=== RAG 文档问答系统 ===\n")

    # TODO: 准备测试文档（创建几个 .txt 文件）
    # 例如: 准备三篇关于不同主题的短文档

    # TODO: 初始化管道
    # pipeline = RAGPipeline()

    # TODO: 索引文档
    # pipeline.index_documents(["doc1.txt", "doc2.txt"])

    # TODO: 提问
    # result = pipeline.query("你的问题...")
    # print(f"回答: {result['answer']}")
    # print(f"来源: {result['sources']}")

    print("请完成 TODO 后运行。")
