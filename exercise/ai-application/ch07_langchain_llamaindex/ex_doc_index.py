"""
练习: LlamaIndex 文档索引与查询
==============================

需求:
  加载一个目录中的所有 Markdown 文档，用 LlamaIndex 建立索引，
  实现关键词匹配 + 语义检索的双路召回，并支持按元数据过滤。

要求:
  1. 实现 load_documents(directory: str) -> list[Document]:
     - 使用 SimpleDirectoryReader 加载目录中的 .md 文件
     - 为每个文档自动提取元数据: 文件名、文件路径、修改时间
     - 递归加载子目录

  2. 实现 create_index(documents: list[Document]) -> VectorStoreIndex:
     - 配置 Settings.llm 和 Settings.embed_model
     - 创建 VectorStoreIndex
     - 保存索引到 ./llama_index_storage

  3. 实现 dual_retrieval(index, query: str, k: int = 5) -> list:
     - 双路召回: 关键词检索(TF-IDF/Bag-of-Words) + 语义检索(向量相似度)
     - 合并两路结果，去重后返回 Top-K
     - 两路权重: 关键词 0.3, 语义 0.7 (可通过参数调整)

  4. 实现 query_with_filter(index, query: str, **filters) -> str:
     - 支持按文件名、修改时间、文件大小等元数据过滤
     - 例如: query_with_filter(index, "Python教程", file_name="intro.md")
     - 使用 MetadataFilters

  5. 实现 main():
     - 接收 --dir 参数指定文档目录
     - 接收 --query 参数做单次查询（不交互）
     - 无 --query 时进入交互模式
     - 支持 --rebuild 参数强制重建索引
     - 交互模式下支持命令:
       /filter key=value  设置元数据过滤
       /clear             清除过滤条件
       /source            显示上次回答的引用来源

TODO:
  - [ ] 实现 load_documents(dir)
  - [ ] 实现 create_index(docs, rebuild)
  - [ ] 实现 keyword_search(docs, query, k) 基于简单的 TF-IDF 或词匹配
  - [ ] 实现 dual_retrieval(index, query, k, keyword_weight)
  - [ ] 实现 query_with_filter(index, query, **filters)
  - [ ] 实现 main()

提示:
  - SimpleDirectoryReader 需要 pip install llama-index-readers-file
  - 元数据过滤使用 MetadataFilters 和 MetadataFilter
  - TF-IDF 可用 sklearn.feature_extraction.text.TfidfVectorizer
  - 简单的关键词匹配可以直接用 set intersection（适合小数据集）
  - LlamaIndex 的 source_nodes 可以获取引用来源
"""
import os
import argparse
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
    Document,
)
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


def load_documents(directory: str) -> list[Document]:
    # TODO: 使用 SimpleDirectoryReader 加载 .md 文件
    pass


def create_index(documents: list[Document], persist_dir: str, rebuild: bool = False) -> VectorStoreIndex:
    # TODO: 创建或加载索引
    pass


def keyword_search(documents: list[Document], query: str, k: int = 5) -> list[Document]:
    # TODO: 基于关键词匹配检索（如简单的词频统计或 TF-IDF）
    pass


def dual_retrieval(index: VectorStoreIndex, query: str, k: int = 5, keyword_weight: float = 0.3):
    # TODO: 合并关键词检索和语义检索的结果
    pass


def query_with_filter(index: VectorStoreIndex, query: str, **filters) -> str:
    # TODO: 使用 MetadataFilters 按元数据过滤查询
    pass


def main():
    # TODO: 解析参数，创建索引，进入交互循环
    pass


if __name__ == "__main__":
    main()
