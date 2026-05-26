"""
进阶练习: Agentic RAG — 智能研究助手
====================================

需求:
  构建一个"智能研究助手"，用户给出一个研究主题后，
  Agent 自动规划并执行: 分解子问题 → 查知识库 → Web搜索 → 汇总生成研究报告。

  这是 Agent + RAG + Web Search 的综合练习。

要求:
  1. 实现 KnowledgeBase 类:
     - 封装 LlamaIndex 的索引和查询
     - load(directory: str): 从目录加载文档并建索引
     - search(query: str, k: int) -> str: 搜索知识库
     - 持久化索引到磁盘

  2. 实现 WebSearchTool(工具):
     - 模拟网络搜索（调用 DuckDuckGo 或 SerpAPI 或简单的 requests）
     - 输入搜索词，返回搜索结果摘要列表
     - 建议使用 duckduckgo_search 库: pip install duckduckgo-search

  3. 实现 ResearchAgent:
     - 使用 LangChain Agent (create_tool_calling_agent)
     - 配备工具: knowledge_search, web_search, save_finding
     - save_finding 工具: 将关键发现保存到临时文件，避免丢失
     - 系统提示: 指导 Agent 如何做研究（分解→搜索→记录→汇总）

  4. 实现 generate_report(findings: str, topic: str) -> str:
     - 将 Agent 收集到的所有发现汇总
     - 调用 LLM 生成结构化的研究报告:
       标题、摘要、关键发现(编号列表)、详细分析、结论、参考资料
     - 保存为 Markdown 文件

  5. 实现 main():
     - 参数: --topic（研究主题）、--kb-dir（知识库目录）
     - 显示 Agent 的完整研究过程
     - 输出最终研究报告的文件路径

TODO:
  - [ ] 实现 KnowledgeBase 类
  - [ ] 实现 web_search(query) 工具函数
  - [ ] 实现 save_finding(content) 工具函数
  - [ ] 实现 ResearchAgent 类
  - [ ] 实现 generate_report(findings, topic) 函数
  - [ ] 实现 main()

提示:
  - duckduckgo_search 的 DDGS().text() 返回搜索结果列表
  - save_finding 使用追加模式写入文件，内容需要包含时间戳
  - Agent 可能需要 5-10 步才能完成一个完整的研究任务
  - max_iterations 建议设为 20
  - verbose=True 对于观察 Agent 行为非常重要
  - 研究报告用 Markdown 格式保存便于阅读和分享
"""
import os
import argparse
import datetime
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


class KnowledgeBase:
    # TODO: 封装 LlamaIndex 的知识库（加载、索引、搜索、持久化）
    pass


@tool
def knowledge_search(query: str) -> str:
    """搜索本地知识库。输入查询问题，返回相关文档片段。"""
    # TODO: 调用 KnowledgeBase 的搜索
    pass


@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息。输入搜索关键词，返回搜索结果摘要。
    适用于需要最新信息（如新闻、最新研究、近期事件）的问题。"""
    # TODO: 使用 duckduckgo_search 进行搜索
    pass


@tool
def save_finding(content: str) -> str:
    """保存一条研究发现到报告草稿。当你发现重要信息时调用此工具。
    content 应该是一段完整的文字描述（含来源）。"""
    # TODO: 追加写入临时文件
    pass


class ResearchAgent:
    # TODO: 创建 Agent，配备工具，管理研究流程
    pass


def generate_report(findings: str, topic: str) -> str:
    # TODO: 根据研究发现生成结构化 Markdown 报告
    pass


def main():
    # TODO: 解析参数 → 创建 KB → 创建 Agent → 执行研究 → 生成报告
    pass


if __name__ == "__main__":
    main()
