"""
进阶练习: 多源 RAG（文档 + 网页 + 数据库）
=========================================

需求:
  构建一个从三个异构数据源检索的 RAG 系统:
    1. 本地 TXT/MD 文件
    2. 指定的 URL 网页内容
    3. SQLite 数据库表
  使用 LangChain 的 EnsembleRetriever 合并不同检索器的结果。

要求:
  1. 实现 FileRetrieverBuilder:
     - load_files(directory: str, glob_pattern: str) -> list[Document]
     - 使用 DirectoryLoader 批量加载
     - 切分: RecursiveCharacterTextSplitter(chunk_size=500, overlap=100)
     - embed + 存入 Chroma（不同数据源用不同 collection）
     - as_retriever(k: int) -> Retriever

  2. 实现 WebRetrieverBuilder:
     - load_urls(urls: list[str]) -> list[Document]
     - 使用 WebBaseLoader 加载网页
     - 清理 HTML，只保留正文
     - 切分 + embed + Chroma
     - as_retriever(k: int) -> Retriever

  3. 实现 SQLiteRetrieverBuilder:
     - load_table(db_path: str, table: str, text_column: str) -> list[Document]
     - 从 SQLite 表中读取指定列的内容
     - 每行作为一篇 Document
     - 切分 + embed + Chroma
     - as_retriever(k: int) -> Retriever

  4. 实现 MultiSourceRAG:
     - 接收多个 Retriever 和对应权重
     - 使用 EnsembleRetriever 合并
     - RAG Chain: 检索 → 格式化 → prompt → LLM → 回答
     - 在回答中引用来源（标明来自文件/网页/数据库）

  5. 实现 main():
     - 通过命令行参数配置各数据源
     - 交互式问答
     - 支持 /sources 命令查看引用

TODO:
  - [ ] 实现 FileRetrieverBuilder 类
  - [ ] 实现 WebRetrieverBuilder 类
  - [ ] 实现 SQLiteRetrieverBuilder 类
  - [ ] 实现 MultiSourceRAG 类 (build_chain, query 方法)
  - [ ] 实现 main() 解析参数、构建、交互

提示:
  - EnsembleRetriever 会自动去重合并不同检索器的结果
  - 每个数据源用不同的 Chroma collection（避免相互干扰）
  - 在 Document.metadata 中添加 "source_type" 字段标记来源
  - WebBaseLoader 需要 bs4: pip install beautifulsoup4 lxml
  - SQLite 不需要额外安装，Python 标准库自带 sqlite3
  - RAG Prompt 中应该包含每条上下文的来源信息
"""
import os
import argparse
import sqlite3
from langchain_community.document_loaders import DirectoryLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from dotenv import load_dotenv

load_dotenv()


class FileRetrieverBuilder:
    # TODO: 实现本地文件检索器构建器
    pass


class WebRetrieverBuilder:
    # TODO: 实现网页检索器构建器
    pass


class SQLiteRetrieverBuilder:
    # TODO: 实现 SQLite 检索器构建器
    pass


class MultiSourceRAG:
    # TODO: 实现多源 RAG 系统
    pass


def main():
    # TODO: 解析参数 → 构建各检索器 → 创建 MultiSourceRAG → 交互循环
    pass


if __name__ == "__main__":
    main()
