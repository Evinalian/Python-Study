"""
进阶练习: 完整 RAG Web 应用 (FastAPI + Streamlit + Docker)
=========================================================

需求:
  构建一个完整的 RAG（检索增强生成）Web 应用:
    - 后端: FastAPI 提供 API（文档管理 + RAG 问答）
    - 前端: Streamlit 提供用户界面
    - 数据库: Chroma 向量数据库
    - 容器化: Docker Compose 编排

后端功能:
  - POST /upload: 上传文档（TXT/PDF/MD），支持批量上传
  - GET /documents: 列出已索引的文档列表（含元数据）
  - DELETE /documents/{doc_id}: 删除指定文档
  - POST /query: RAG 问答（非流式），返回答案+引用来源
  - POST /query/stream: RAG 问答（流式 SSE）
  - GET /health: 健康检查

前端功能 (Streamlit):
  - 侧边栏:
    - 文档上传区域（文件上传 + 上传按钮 + 进度条）
    - 已索引文档列表（可删除）
    - 设置（模型选择、温度、Top-K）
  - 主区域:
    - 聊天界面（类似 ChatGPT）
    - 每条回复下方显示引用来源（可折叠）
    - "清空对话"按钮

TODO:
  - [ ] 实现 FastAPI 后端:
       - app/main.py: FastAPI 应用入口
       - app/routers/documents.py: 文档管理路由
       - app/routers/query.py: RAG 问答路由
       - app/services/rag.py: RAG 核心逻辑（加载、切分、embedding、检索、生成）
       - app/models/schemas.py: Pydantic 模型

  - [ ] 实现 Streamlit 前端:
       - app_frontend.py: 完整的 Streamlit 界面

  - [ ] 编写 Dockerfile (后端 + 前端各一个)
  - [ ] 编写 docker-compose.yml (api + frontend + chroma)
  - [ ] 编写 .env.example 和 requirements.txt

提示:
  - 使用 LangChain 的 TextLoader + RecursiveCharacterTextSplitter + Chroma
  - 流式 SSE 参考第08章 ex_8_2_fastapi_streaming.py
  - Streamlit 使用 requests 库调用 FastAPI 后端
  - 后端地址通过环境变量配置（在 docker-compose 中用服务名）
  - 在 Docker 中 Chroma 用独立的服务容器（客户端-服务器模式）
"""
# 这是一个多文件项目，练习文件包含项目结构和各文件的 TODO 框架。
# 实际练习时需要在多个文件中完成。

import os


# ====== 项目结构 ======
PROJECT_STRUCTURE = """
rag-app/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI 应用入口
│   │   ├── config.py            # 配置
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py       # Pydantic 模型
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── documents.py     # 文档管理路由
│   │   │   └── query.py         # 问答路由
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   └── rag.py           # RAG 核心服务
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── helpers.py       # 工具函数
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py                   # Streamlit 前端
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
└── .env.example
"""


# ====== TODO: backend/app/models/schemas.py ======
SCHEMAS_TODO = """
from pydantic import BaseModel, Field
from typing import Optional, List

class DocumentUploadResponse(BaseModel):
    # TODO: doc_id, filename, chunk_count, status
    pass

class DocumentInfo(BaseModel):
    # TODO: doc_id, filename, upload_time, chunk_count, status
    pass

class QueryRequest(BaseModel):
    # TODO: question, model, temperature, max_tokens, top_k
    pass

class QueryResponse(BaseModel):
    # TODO: answer, sources (List[dict]), usage
    pass
"""


# ====== TODO: backend/app/services/rag.py ======
RAG_SERVICE_TODO = """
class RAGService:
    def __init__(self, persist_dir: str, embedding_model: str, llm_model: str):
        # TODO: 初始化 embedding 模型、LLM、Chroma 客户端
        pass

    def add_document(self, content: str, metadata: dict) -> str:
        # TODO: 加载文档 → 切分 → embedding → 存入 Chroma → 返回 doc_id
        pass

    def remove_document(self, doc_id: str) -> bool:
        # TODO: 从 Chroma 中删除指定 doc_id 的所有 chunk
        pass

    def list_documents(self) -> list[dict]:
        # TODO: 列出所有已索引的文档及其元数据
        pass

    def query(self, question: str, top_k: int = 5) -> dict:
        # TODO: 检索 → 构建 prompt → LLM 生成 → 返回答案+来源
        pass

    def query_stream(self, question: str, top_k: int = 5):
        # TODO: 流式版本的 query，yield 每个 token 的 SSE 事件
        pass
"""


# ====== TODO: backend/app/routers/documents.py ======
DOCUMENTS_ROUTER_TODO = """
from fastapi import APIRouter, UploadFile, File

router = APIRouter(prefix="/v1/documents", tags=["Documents"])

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # TODO: 上传并索引文档
    pass

@router.get("/")
async def list_documents():
    # TODO: 列出所有文档
    pass

@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    # TODO: 删除文档
    pass
"""


# ====== TODO: frontend/app.py ======
FRONTEND_TODO = """
import streamlit as st
import requests

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

def main():
    # TODO: 侧边栏（上传、文档列表、设置）
    # TODO: 主区域（聊天界面、消息历史）
    # TODO: 流式显示（模拟打字效果）
    pass
"""


print("请按照以上 TODO 指引，在 rag-app/ 目录下完成各文件的实现。")
print(f"项目结构:\n{PROJECT_STRUCTURE}")
