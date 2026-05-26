"""
练习 3: FastAPI 流式 API 服务

场景:
    构建一个完整的 FastAPI 流式聊天后端 API。

要求:
    1. 创建 FastAPI 应用，包含以下端点:
       - POST /chat/stream: 流式聊天(SSE)
       - GET /health: 健康检查

    2. 实现 SSE 流式生成器:
       - 使用 AsyncOpenAI 客户端
       - async generator 产出 SSE 格式的事件
       - 格式: data: {"token": "...", "done": false}\n\n
       - 结束时: data: [DONE]\n\n

    3. 添加错误处理:
       - API 调用失败时返回 error 事件
       - 断开连接时优雅处理

    4. 添加 CORS 支持（允许浏览器跨域访问）

    5. （可选）添加一个简单的 HTML 测试页面(GET /)

运行:
    uvicorn ex3_fastapi_stream:app --reload
    浏览器访问 http://localhost:8000/

TODO:
    1. 创建 FastAPI app
    2. 实现 sse_generator(messages)
    3. 实现 POST /chat/stream 端点
    4. 实现 GET / 测试页面
    5. 测试: curl 命令或浏览器
"""

import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from typing import AsyncGenerator

# TODO: 初始化 FastAPI app 和 AsyncOpenAI client

# ============================================================
# TODO 1: 创建 FastAPI app + CORS
# ============================================================
app = FastAPI(title="TODO: 流式聊天API")

# TODO: 添加 CORS 中间件
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# TODO: 初始化 AsyncOpenAI 客户端
# client = AsyncOpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
# )


# ============================================================
# TODO 2: SSE 事件生成器
# ============================================================
async def sse_generator(messages: list[dict]) -> AsyncGenerator[str, None]:
    """
    生成 SSE 格式的流式事件。

    产出格式:
        data: {"token": "你", "done": false}\n\n
        data: {"token": "好", "done": false}\n\n
        data: {"token": "", "done": true}\n\n
        data: [DONE]\n\n

    如果出错:
        data: {"error": "错误信息"}\n\n
        data: [DONE]\n\n

    提示:
    1. stream = await client.chat.completions.create(model="gpt-4o", messages=messages, stream=True)
    2. async for chunk in stream:
    3.     delta = chunk.choices[0].delta
    4.     if delta.content:
    5.         yield f"data: {json.dumps({'token': delta.content, 'done': False})}\n\n"
    6. yield "data: [DONE]\n\n"
    """
    # TODO: 实现
    try:
        # stream = await client.chat.completions.create(...)
        # async for chunk in stream:
        #     ...
        pass
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"


# ============================================================
# TODO 3: POST /chat/stream 端点
# ============================================================
@app.post("/chat/stream")
async def chat_stream(request: Request):
    """
    流式聊天端点。

    请求体 JSON:
    {
        "messages": [{"role": "user", "content": "你好"}],
        "system_prompt": "可选的系统提示"  // 可选
    }

    响应: SSE 流

    提示:
    1. body = await request.json()
    2. messages = body.get("messages", [])
    3. 如果没有 system message，添加默认的
    4. return StreamingResponse(
           sse_generator(messages),
           media_type="text/event-stream",
           headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
       )
    """
    # TODO: 实现
    return {"message": "TODO"}


# ============================================================
# TODO 4: GET / 测试页面
# ============================================================
@app.get("/")
async def root():
    """返回用于测试流式 API 的 HTML 页面"""
    # TODO: 返回一个简单的 HTML，包含输入框和 EventSource 连接的 JS
    # 或者先简单地返回 {"message": "..."}
    return {"message": "TODO: 流式聊天 API"}


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    import uvicorn
    # TODO: 完成后取消注释
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    print("请完成 TODO 后运行: uvicorn ex3_fastapi_stream:app --reload")
