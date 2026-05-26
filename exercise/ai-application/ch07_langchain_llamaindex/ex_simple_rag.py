"""
练习: 简单 RAG 问答系统 (LangChain + LCEL)
===========================================

需求:
  加载本地 TXT 文件作为知识库，用 LangChain 构建一个完整的 RAG 流水线，
  用户输入问题后返回基于知识库生成的答案。

要求:
  1. 实现 load_and_split(file_path: str) -> list[Document]:
     - 使用 TextLoader 加载文件
     - 使用 RecursiveCharacterTextSplitter 切分（chunk_size=500, overlap=100）

  2. 实现 create_vectorstore(chunks: list[Document]) -> Chroma:
     - 使用 OpenAIEmbeddings (text-embedding-3-small)
     - 使用 Chroma 存储在内存中

  3. 实现 build_rag_chain(vectorstore: Chroma) -> Runnable:
     - 使用 LCEL 构建链
     - Prompt 模板要求: 基于提供的上下文回答问题，如果不知道就说不知道
     - 检索 Top-3 相关片段
     - 将检索到的文档片段格式化为上下文字符串

  4. 实现 main():
     - 接收文件路径作为命令行参数
     - 构建 RAG 链
     - 进入交互式 Q&A 循环（输入问题 → 输出答案，输入 'quit' 退出）
     - 显示每条答案所引用的来源（source metadata）

  5. 额外要求:
     - 启动时打印加载的块数和向量库大小
     - 支持 --k 参数控制检索片段数量
     - 用 tiktoken 估算每次查询消耗的 token 数

TODO:
  - [ ] 实现 load_and_split(file_path, chunk_size, chunk_overlap)
  - [ ] 实现 create_vectorstore(chunks)
  - [ ] 实现 format_docs(docs) 将文档列表转为上下文字符串
  - [ ] 实现 build_rag_chain(vectorstore, k) 用 LCEL 构建 RAG 链
  - [ ] 实现 estimate_tokens(text) 用 tiktoken 估算 token 数
  - [ ] 实现 main() 串联并进入交互循环

提示:
  - TextLoader 加载时指定 encoding="utf-8"
  - Prompt 中要求 LLM 引用来源可以提高可信度
  - 使用 RunnablePassthrough 透传原始 question 到 Prompt
  - tiktoken 使用 "cl100k_base" 编码（GPT-4 系列）
"""
import os
import argparse
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


def load_and_split(file_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[Document]:
    # TODO: 加载文件并切分为文档块
    pass


def create_vectorstore(chunks: list[Document]) -> Chroma:
    # TODO: 创建 Chroma 向量库
    pass


def format_docs(docs: list[Document]) -> str:
    # TODO: 将文档列表格式化为 LLM 可读的上下文字符串
    pass


def build_rag_chain(vectorstore: Chroma, k: int = 3):
    # TODO: 用 LCEL 构建 {context, question} → prompt → llm → parser 链
    pass


def estimate_tokens(text: str, model_encoding: str = "cl100k_base") -> int:
    # TODO: 使用 tiktoken 估算 token 数
    pass


def main():
    # TODO: 解析参数，构建 RAG 链，进入交互循环
    pass


if __name__ == "__main__":
    main()
