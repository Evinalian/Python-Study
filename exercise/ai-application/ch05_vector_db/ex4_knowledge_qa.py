"""
练习 4: 完整知识库问答系统

场景:
    构建一个完整的知识库问答系统，整合第04章和第05章的知识。

系统架构:
    文档 → 加载 → 切块 → Embedding → ChromaDB → 检索 → LLM生成 → 来源标注

要求:
    1. 实现 KnowledgeBase 类:
       - index_directory(dir_path): 索引目录中所有文档
       - index_texts(texts, metadatas): 索引文本列表
       - ask(question): 检索 + LLM 生成回答
       - stats(): 统计信息

    2. 文档处理:
       - 加载 .txt, .md 文件
       - 按段落切块（chunk_size=800, overlap=150）
       - 保留来源元数据

    3. 检索 + 生成:
       - 检索 top-5 相关文档块
       - 构建 RAG prompt（含来源标注要求）
       - 调用 OpenAI 生成回答

    4. 回答质量:
       - 如果资料充足: 基于资料回答 + 标注来源
       - 如果资料不足: 明确告知 "根据现有资料无法回答"
       - 回答简洁准确

TODO:
    1. 实现 KnowledgeBase 类
    2. 实现文档加载和切块
    3. 实现 ask() 函数
    4. 准备测试文档并测试
"""

import os
import json
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# ============================================================
# TODO 1: KnowledgeBase 类
# ============================================================
class KnowledgeBase:
    """
    知识库问答系统。

    集成: 文档管理 + 向量存储 + 语义检索 + LLM 生成
    """

    def __init__(
        self,
        name: str = "default_kb",
        persist_path: str = "./kb_data",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        """
        初始化知识库。

        参数:
            name: 知识库名称
            persist_path: ChromaDB 持久化路径
            chunk_size: 切块大小
            chunk_overlap: 切块重叠
        """
        # TODO:
        # 1. 初始化 ChromaDB PersistentClient
        # 2. 设置 OpenAI embedding function
        # 3. 获取或创建 collection
        # 4. 创建文本切分器
        pass

    def index_directory(self, dir_path: str) -> int:
        """
        索引目录中所有 .txt 和 .md 文件。

        返回: 索引的文档块数量
        """
        # TODO:
        # 1. 扫描目录
        # 2. 读取每个文件
        # 3. 切块
        # 4. 批量添加到 ChromaDB
        # 5. 返回块数量
        pass

    def index_texts(self, texts: list[str], metadatas: list[dict] = None) -> None:
        """
        直接索引文本列表。

        参数:
            texts: 文本列表
            metadatas: 元数据列表
        """
        # TODO: 对每段文本切块 + 添加到 ChromaDB
        pass

    def search(self, question: str, top_k: int = 5) -> list[dict]:
        """
        语义检索。

        返回: [{"text": "...", "metadata": {...}, "score": 0.95}, ...]
        """
        # TODO: ChromaDB query
        pass

    def ask(self, question: str, top_k: int = 5) -> dict:
        """
        问答。

        流程:
        1. 检索相关文档块
        2. 如果无结果 → fallback 回答
        3. 构建 RAG prompt
        4. 调用 OpenAI 生成
        5. 提取来源

        返回:
        {
            "answer": "...",
            "sources": ["file1.txt", "file2.txt"],
            "chunks_used": 3,
            "has_answer": True
        }
        """
        # TODO: 实现完整问答流程
        pass

    def stats(self) -> dict:
        """统计信息"""
        # TODO
        pass


# ============================================================
# TODO 2: 文本切分（简化版）
# ============================================================
class TextSplitter:
    """简化的递归文本切分器"""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        # TODO
        pass

    def split(self, text: str) -> list[str]:
        # TODO: 按段落 + 句子边界切分，保持语义完整
        pass


if __name__ == "__main__":
    print("=== 知识库问答系统 ===\n")

    # kb = KnowledgeBase(name="tech_kb")

    # 示例: 索引一些技术文档
    # kb.index_texts([
    #     "Python是一种解释型编程语言...",
    #     "Django是Python的Web框架...",
    # ], metadatas=[{"source": "python_intro.txt"}, {"source": "django_intro.txt"}])

    # 问答
    # result = kb.ask("Python有哪些Web框架？")
    # print(f"回答: {result['answer']}")
    # print(f"来源: {result['sources']}")

    print("请完成 TODO 后运行。")
