"""
练习: FastAPI Chat API
======================

需求:
  实现一个完整的 FastAPI Chat API 后端，包含:
    - POST /v1/chat: 非流式聊天（一次返回完整回复）
    - POST /v1/chat/stream: 流式聊天（SSE 格式，逐 token 推送）
    - GET /v1/models: 返回支持的模型列表
    - GET /health: 健康检查

要求:
  1. 使用 Pydantic 定义请求体和响应体:
     - ChatRequest: messages, model, temperature, max_tokens
     - ChatResponse: content, model, usage, finish_reason
     - 使用 Field 设置校验规则（长度限制、温度范围等）

  2. API Key 鉴权:
     - 从请求头 X-API-Key 读取
     - 从环境变量 API_KEYS（多个 key 用逗号分隔）校验
     - 未设置环境变量时放行（开发模式）
     - 鉴权失败返回 401

  3. 流式端点:
     - 返回 Content-Type: text/event-stream
     - 每个 token 用 SSE 格式: data: {"type":"content","content":"文字"}\n\n
     - 结束时发送: data: {"type":"done"}\n\n
     - 错误时发送: data: {"type":"error","message":"..."}\n\n
     - 设置 no-cache 和 no-buffering 头

  4. 请求日志中间件:
     - 记录每个请求的方法、路径、耗时、状态码
     - 添加到响应头 X-Process-Time

  5. 错误处理:
     - 使用全局异常处理器捕获所有未处理异常
     - 返回统一格式: {"error": "...", "detail": "...", "timestamp": "..."}
     - 区分: 400（参数错误）、401（认证）、429（限流）、500（服务器错误）

  6. 启动方式:
     uvicorn ex_fastapi_chat:app --reload --port 8000

TODO:
  - [ ] 定义 Pydantic 模型（ChatRequest, ChatResponse, ErrorResponse）
  - [ ] 实现 verify_api_key 依赖
  - [ ] 实现非流式 /v1/chat 端点
  - [ ] 实现流式 /v1/chat/stream 端点
  - [ ] 实现 /v1/models 端点
  - [ ] 实现 /health 端点
  - [ ] 实现请求日志中间件
  - [ ] 实现全局异常处理器

提示:
  - 使用 StreamingResponse 实现 SSE
  - async def + yield 实现异步生成器
  - 使用 Header(...) 从请求头提取 API Key
  - 使用 Depends 注入鉴权依赖
  - @app.exception_handler 注册全局异常处理
"""
import os
import time
import json
from typing import Optional, List, AsyncGenerator
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


# ====== TODO: 定义 Pydantic 模型 ======
class ChatRequest(BaseModel):
    # TODO
    pass


class ChatResponse(BaseModel):
    # TODO
    pass


class ErrorResponse(BaseModel):
    # TODO
    pass


# ====== TODO: FastAPI 应用 ======
app = FastAPI(title="AI Chat API", version="1.0.0")

# TODO: 添加 CORS 中间件


# ====== TODO: 鉴权 ======
def verify_api_key(x_api_key: str = Header(...)):
    # TODO
    pass


# ====== TODO: 请求日志中间件 ======
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # TODO
    pass


# ====== TODO: 全局异常处理器 ======
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # TODO
    pass


# ====== TODO: 路由 ======
@app.get("/health")
async def health():
    # TODO
    pass


@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    # TODO
    pass


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    # TODO
    pass


@app.post("/v1/chat/stream")
async def chat_stream(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    # TODO
    pass


# ====== 启动: uvicorn ex_fastapi_chat:app --reload --port 8000 ======
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
