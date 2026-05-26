"""
练习 2: ChromaDB 文档管理系统

场景:
    实现一个基于 ChromaDB 的文档存储和搜索系统。

要求:
    1. 实现 DocumentStore 类:
       - add(documents): 批量添加文档（自动生成 embedding）
       - search(query, top_k, where): 语义搜索（支持元数据过滤）
       - get(doc_id): 获取单个文档
       - update(doc_id, text, metadata): 更新文档
       - delete(doc_ids): 删除文档
       - stats(): 统计信息

    2. 支持以下元数据过滤条件:
       - 相等: {"category": "技术"}
       - 范围: {"year": {"$gte": 2023, "$lte": 2024}}
       - 组合: {"$and": [{"category": "技术"}, {"level": "高级"}]}

    3. 持久化到磁盘，程序重启后数据不丢失

    4. 实现一个简单的命令行交互界面（add/search/exit）

TODO:
    1. 实现 DocumentStore 类
    2. 测试 CRUD 操作
    3. 测试元数据过滤
    4. 实现命令行界面
"""

import os
import json
import chromadb
from chromadb.utils import embedding_functions


class DocumentStore:
    """
    基于 ChromaDB 的文档管理系统。

    功能: 增删改查 + 语义搜索 + 元数据过滤 + 持久化
    """

    def __init__(self, collection_name: str, persist_path: str = "./docstore_data"):
        """
        初始化文档存储。

        参数:
            collection_name: 集合名称
            persist_path: 持久化路径
        """
        # TODO:
        # 1. 创建 PersistentClient
        # 2. 设置 embedding function（默认用 sentence-transformers）
        # 3. 获取或创建 collection
        # 4. 设置 hnsw:space 为 cosine
        pass

    def add(self, documents: list[dict]) -> list[str]:
        """
        批量添加文档。

        参数:
            documents: [{"text": "...", "metadata": {...}, "id": "..."}, ...]
                       id 可选，不提供则自动生成

        返回:
            添加的文档 ID 列表
        """
        # TODO:
        # 1. 提取 texts, metadatas, ids
        # 2. 如果没有 id，自动生成 UUID
        # 3. 调用 collection.add()
        pass

    def search(self, query: str, top_k: int = 5, where: dict = None) -> list[dict]:
        """
        语义搜索。

        返回:
        [{"id": "...", "text": "...", "metadata": {...}, "score": 0.95}, ...]
        """
        # TODO:
        # 1. 调用 collection.query()
        # 2. 转换距离为相似度分数
        # 3. 返回格式化的结果
        pass

    def get(self, doc_id: str) -> dict:
        """获取单个文档"""
        # TODO: collection.get(ids=[doc_id])
        pass

    def get_all(self) -> list[dict]:
        """获取所有文档"""
        # TODO: collection.get()
        pass

    def update(self, doc_id: str, text: str = None, metadata: dict = None) -> None:
        """更新文档"""
        # TODO: collection.update()
        pass

    def delete(self, doc_ids: list[str]) -> None:
        """删除文档"""
        # TODO: collection.delete()
        pass

    def stats(self) -> dict:
        """统计信息"""
        # TODO: 返回文档数量、集合名称等
        pass


def interactive_cli():
    """
    命令行交互界面。

    命令:
    - add <text> [--category <cat>] [--level <level>]
    - search <query> [--top <k>] [--category <cat>]
    - list
    - delete <id>
    - exit
    """
    # TODO: 实现简单的命令循环
    # store = DocumentStore("cli_docs")
    # while True:
    #     cmd = input("> ").strip()
    #     ...
    pass


if __name__ == "__main__":
    print("=== ChromaDB 文档管理系统 ===\n")

    # TODO: 测试 CRUD
    # store = DocumentStore("test_docs")

    # # 添加
    # ids = store.add([
    #     {"text": "Python教程", "metadata": {"category": "编程", "level": "入门"}},
    #     {"text": "机器学习入门", "metadata": {"category": "AI", "level": "入门"}},
    #     {"text": "深度学习实战", "metadata": {"category": "AI", "level": "高级"}},
    # ])

    # # 搜索
    # results = store.search("AI入门教程", top_k=2)

    # interactive_cli()

    print("请完成 TODO 后运行。")
