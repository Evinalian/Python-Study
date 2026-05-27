# 第08章 AI 应用 Web 化

## 学习目标

1. 掌握用 FastAPI 构建异步、高性能的 AI API 后端
2. 掌握流式 API（SSE）的实现，支持 LLM 流式输出
3. 能使用 Gradio 零代码快速搭建 AI 演示界面
4. 能使用 Streamlit 构建数据驱动的 AI 应用
5. 理解 Docker 容器化部署的基本操作
6. 了解不同规模和成本要求的部署方案

---

## 前置知识

- Python 异步编程基础（`async/await`、`asyncio`）
- HTTP 协议基础（GET/POST、请求/响应、状态码）
- Pydantic 数据模型（第07章已接触）
- 已完成 ai-application 第01-05章：OpenAI SDK 流式调用

---

## 8.0 前言：为什么你的 AI 应用需要一个"壳"

到现在为止，你写的所有 AI 代码都是在终端里运行的 Python 脚本——输入靠 `input()`，输出靠 `print()`。这在学习和调试阶段足够，但要让别人能用上你的 AI 能力，它需要一个"壳"——一个能让用户通过浏览器或 API 访问的界面。

本章介绍三种不同的"壳"，它们解决的是同一个问题（把 Python 函数暴露给用户）但适用于完全不同的场景。**FastAPI** 提供的是最底层的 HTTP API，适合作为后端服务被其他程序调用；**Gradio** 提供的是自动生成的 ML 演示界面，适合快速展示模型效果；**Streamlit** 提供的是数据驱动的交互式 Web 应用，适合构建分析仪表盘和工具面板。加在一起，**Docker** 让这些服务能在任意环境一致性运行。

理解这三者不是"三选一"的关系很重要——一个完整的 AI 应用通常同时使用 FastAPI（后端）和 Streamlit/Gradio（前端），通过 Docker 统一部署。

---

## 8.1 FastAPI 后端 —— 工业级 AI API 的首选

### 8.1.1 为什么 FastAPI 是 AI 应用的最佳后端框架

在 Python Web 框架的"三国演义"（FastAPI、Flask、Django）中，FastAPI 是为 AI 时代量身定制的。它的三个核心优势直接对应了 AI 应用的特殊需求：

**原生异步（async/await）**：AI 应用的瓶颈几乎总是在 I/O——LLM API 调用通常需要 2-30 秒返回。在 Flask 中，一个慢请求会阻塞整个线程，只能用多线程或多进程来弥补。在 FastAPI 中，`async/await` 让服务器在等待 LLM 响应时可以处理其他请求。这就像咖啡师在等浓缩咖啡萃取的时候给下一位客人点单——不用傻站着等。

**自动数据校验（Pydantic）**：AI API 的请求参数往往很复杂——需要校验 temperature 在 0-2 之间、max_tokens 不超过上限、model 名称是合法的。FastAPI 用 Pydantic 的 `Field` 声明式定义校验规则，框架自动生成校验逻辑和错误提示。如果参数不合法，API 返回的不再是模糊的 500 错误，而是精确的"temperature 必须在 0.0 到 2.0 之间"。

**自动文档（OpenAPI）**：FastAPI 从代码自动生成 Swagger UI 交互式文档（访问 `/docs` 即可）。这意味着你写完 API 的同时，一份可交互测试的文档就自动存在了——前端同事可以直接在文档页面尝试各种参数组合调用你的 API，而不需要翻代码或问你怎么用。

| 特性 | FastAPI | Flask | Django |
|------|---------|-------|--------|
| 异步支持 | 原生 async/await | 需插件 | 3.1+ 支持 |
| 自动 API 文档 | Swagger + ReDoc | 需插件 | 需插件 |
| 数据校验 | Pydantic 自动校验 | 手动 | DRF Serializer |
| 性能 | 极高（Starlette + Uvicorn） | 中 | 中 |
| 学习曲线 | 低 | 低 | 高 |
| AI 应用适配度 | 极佳（异步+流式） | 一般 | 一般 |

```bash
pip install fastapi uvicorn[standard] pydantic python-multipart
```

### 8.1.2 FastAPI 基础 —— 从零构建一个 Chat API

FastAPI 的应用结构和它的哲学一样简洁：定义 Pydantic 模型描述请求/响应格式 → 用装饰器定义路由 → 在函数中处理逻辑。下面这段代码构建了一个完整的 Chat API，包含了请求校验（`ChatRequest` 中 `temperature` 限制在 0-2、`max_tokens` 最大 16384）、API Key 鉴权、错误统一处理和 CORS 配置。

几个值得注意的设计细节：`@field_validator` 让你添加自定义校验逻辑（比如 system 消息必须放在第一条）；`Depends(verify_api_key)` 实现依赖注入式的鉴权；`response_model` 参数告诉 FastAPI 用哪个 Pydantic 模型来序列化响应（同时自动生成文档）。

```python
"""
ex_8_1_fastapi_basics.py: FastAPI 基础——路由、请求体、参数校验
演示：构建一个最小的 AI Chat API 后端

启动方式: uvicorn ex_8_1_fastapi_basics:app --reload --port 8000
访问文档: http://localhost:8000/docs
"""
import os
import time
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ====== 1. 创建 FastAPI 应用 ======
app = FastAPI(
    title="AI Chat API",              # API 文档标题
    description="一个简单的 AI 对话接口",  # 描述
    version="1.0.0",                  # 版本号
    docs_url="/docs",                 # Swagger 文档路径
    redoc_url="/redoc",               # ReDoc 文档路径
)

# ====== 2. CORS 配置 ======
# 允许前端跨域访问（开发阶段用 * 允许所有来源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # 生产环境改为具体域名列表
    allow_credentials=True,
    allow_methods=["*"],              # 允许所有 HTTP 方法
    allow_headers=["*"],              # 允许所有请求头
)


# ====== 3. Pydantic 模型定义 ======
# FastAPI 使用 Pydantic 自动校验请求体和生成文档

class ChatMessage(BaseModel):
    """单条对话消息"""
    role: str = Field(
        default="user",
        description="消息角色: system / user / assistant",
        pattern="^(system|user|assistant)$"  # 正则校验
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=32768,  # 32K 字符上限
        description="消息内容",
    )


class ChatRequest(BaseModel):
    """聊天请求体"""
    messages: List[ChatMessage] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="对话消息列表（至少1条，至多100条）",
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="模型名称",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,   # 大于等于 0
        le=2.0,   # 小于等于 2
        description="采样温度 (0.0-2.0)",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=16384,
        description="最大输出 token 数",
    )
    stream: bool = Field(
        default=False,
        description="是否使用流式输出",
    )

    # 自定义校验器
    @field_validator("messages")
    @classmethod
    def check_message_order(cls, v):
        """校验消息顺序：system 消息必须在最前面"""
        for i, msg in enumerate(v):
            if msg.role == "system" and i != 0:
                raise ValueError("system 消息必须放在 messages 列表的第一条")
        return v


class ChatResponse(BaseModel):
    """聊天响应体"""
    content: str = Field(description="模型回复内容")
    model: str = Field(description="实际使用的模型")
    usage: dict = Field(
        default_factory=dict,
        description="Token 用量信息 {prompt_tokens, completion_tokens, total_tokens}"
    )
    finish_reason: str = Field(
        default="stop",
        description="结束原因: stop / length / content_filter"
    )


class ErrorResponse(BaseModel):
    """错误响应体（统一格式）"""
    error: str = Field(description="错误信息")
    detail: Optional[str] = Field(default=None, description="详细错误描述")
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S")
    )


# ====== 4. API Key 鉴权 ======
# 简单的 token 验证（生产环境应用 JWT 或 OAuth2）

from fastapi import Header, Depends

def verify_api_key(x_api_key: str = Header(..., description="API Key")):
    """
    从请求头 X-API-Key 中读取并进行校验。
    如果环境变量中设置了 API_KEYS（逗号分隔），则校验请求中的 key 是否在列表中。
    如果未设置，则放行所有请求（开发模式）。
    """
    valid_keys_str = os.getenv("API_KEYS", "")
    if valid_keys_str:
        valid_keys = set(valid_keys_str.split(","))
        if x_api_key not in valid_keys:
            raise HTTPException(status_code=401, detail="无效的 API Key")
    # 如果未设置 API_KEYS 环境变量，则不做校验（开发模式）
    return x_api_key


# ====== 5. 路由定义 ======

@app.get("/")
async def root():
    """根路径：健康检查"""
    return {
        "status": "ok",
        "service": "AI Chat API",
        "version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/health")
async def health_check():
    """健康检查端点（Kubernetes / Docker 探针会调用）"""
    # 可以在这里检查数据库连接、Redis 连接等
    # 如果一切正常返回 200
    return {"status": "healthy"}


@app.post(
    "/v1/chat",
    response_model=ChatResponse,
    responses={
        200: {"description": "成功"},
        400: {"model": ErrorResponse, "description": "请求参数错误"},
        401: {"model": ErrorResponse, "description": "认证失败"},
        429: {"model": ErrorResponse, "description": "请求过于频繁"},
        500: {"model": ErrorResponse, "description": "服务器内部错误"},
    },
    tags=["Chat"],  # 在文档中分组显示
)
async def chat(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key),  # 鉴权依赖
):
    """
    聊天接口（非流式）。

    - 接收对话消息列表
    - 调用 OpenAI Chat Completion API
    - 返回模型回复
    """
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # 将 ChatMessage 列表转为 OpenAI API 需要的 dict 格式
        messages = [msg.dict() for msg in request.messages]

        # 如果不是流式，直接调用
        response = client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        choice = response.choices[0]

        return ChatResponse(
            content=choice.message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=choice.finish_reason,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 调用失败: {str(e)}")


# ====== 6. 启动说明 ======
# 运行: uvicorn ex_8_1_fastapi_basics:app --reload --port 8000
# --reload: 代码修改后自动重启（仅开发环境）
# --port: 监听端口
# 访问 http://localhost:8000/docs 查看交互式 API 文档
```

### 8.1.3 流式 API（SSE）—— LLM 应用的关键用户体验

为什么流式输出如此重要？因为 LLM 的响应延迟通常在 2-30 秒之间。如果采用传统的"请求-响应"模式，用户提交问题后面对一片空白等待十几秒，然后一次性看到完整答案。流式输出让文字像打字机一样逐字出现——用户感觉"它在帮我思考"，而不是"它在傻等"。这种心理体验的差异对用户留存率有直接的影响。

技术实现上，SSE（Server-Sent Events）是最简单有效的方案。它本质上是在 HTTP 长连接上持续发送 `data: {...}\n\n` 格式的文本流。与 WebSocket 相比，SSE 是单向的（服务器到客户端），这恰好匹配 LLM 流式输出的场景。

实现时有两个关键点：第一，必须用 `AsyncOpenAI`（异步客户端）而不是同步的 `OpenAI`，否则在生成期间整个事件循环会被阻塞；第二，响应头中的 `X-Accel-Buffering: no` 是为了告诉 Nginx 不要缓冲这个响应——否则流式数据会被 Nginx 攒到一定量再转发，用户又回到了"等待后一次性看到"的体验。

```python
"""
ex_8_2_fastapi_streaming.py: FastAPI 流式 API——SSE（Server-Sent Events）
演示：将 LLM 的流式输出实时推送给客户端

启动方式: uvicorn ex_8_2_fastapi_streaming:app --reload --port 8001

客户端测试:
  curl -X POST http://localhost:8001/v1/chat/stream \
    -H "Content-Type: application/json" \
    -H "X-API-Key: test-key" \
    -d '{"messages": [{"role": "user", "content": "讲一个笑话"}], "stream": true}' \
    --no-buffer
"""
import os
import json
import asyncio
from typing import AsyncGenerator
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Streaming Chat API")

# 初始化异步客户端（支持 await）
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class StreamChatRequest(BaseModel):
    """流式聊天请求"""
    messages: list[dict] = Field(..., description="对话消息列表")
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024)


async def generate_stream(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    """
    异步生成器：从 OpenAI API 获取流式响应并逐块 yield。

    核心概念:
    - SSE (Server-Sent Events) 是一种简单的服务器推送协议
    - 每个事件以 "data: {json}\n\n" 格式发送
    - 客户端用 EventSource API 或 fetch + ReadableStream 接收
    """
    try:
        # 调用 OpenAI 流式 API
        stream = await async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,  # 启用流式
            stream_options={"include_usage": True},  # 最后返回 usage 信息
        )

        # 逐块读取
        async for chunk in stream:
            if len(chunk.choices) == 0:
                # usage 块（最后一个 chunk）
                continue

            delta = chunk.choices[0].delta

            if delta.content is not None:
                # 将每个文本片段包装为 SSE 格式
                event_data = json.dumps({
                    "type": "content",
                    "content": delta.content,
                }, ensure_ascii=False)
                yield f"data: {event_data}\n\n"

            # finish_reason 的检查（当模型生成结束时）
            if chunk.choices[0].finish_reason is not None:
                event_data = json.dumps({
                    "type": "finish",
                    "reason": chunk.choices[0].finish_reason,
                })
                yield f"data: {event_data}\n\n"

        # 发送结束标记
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        # 错误也要通过 SSE 发送，这样客户端可以区分网络错误和业务错误
        error_data = json.dumps({
            "type": "error",
            "message": str(e),
        }, ensure_ascii=False)
        yield f"data: {error_data}\n\n"


@app.post("/v1/chat/stream")
async def chat_stream(request: StreamChatRequest):
    """
    流式聊天接口。

    返回类型: text/event-stream (SSE)
    客户端应使用 EventSource 或 fetch + reader 来读取流式数据。

    响应格式:
      data: {"type": "content", "content": "文本片段"}\n\n
      data: {"type": "finish", "reason": "stop"}\n\n
      data: {"type": "done"}\n\n
      data: {"type": "error", "message": "..."}\n\n
    """
    messages = request.messages

    return StreamingResponse(
        generate_stream(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ),
        media_type="text/event-stream",          # SSE MIME 类型
        headers={
            "Cache-Control": "no-cache",          # 禁止缓存
            "Connection": "keep-alive",           # 保持连接
            "X-Accel-Buffering": "no",            # 禁用 Nginx 缓冲（如果有反向代理）
        },
    )


# ====== 客户端测试代码（不运行时注释掉） ======
"""
import httpx

async def test_stream():
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            "http://localhost:8001/v1/chat/stream",
            json={
                "messages": [{"role": "user", "content": "用 Python 写一个斐波那契函数"}],
                "model": "gpt-4o-mini",
            },
            headers={"X-API-Key": "test-key"},
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if data["type"] == "content":
                        print(data["content"], end="", flush=True)
                    elif data["type"] == "done":
                        print("\\n[完成]")
"""
```

### 8.1.4 FastAPI 进阶 —— 中间件、限流、后台任务

当你的 API 开始有真实用户时，需要三个额外的机制：**中间件**（在请求处理前后插入通用逻辑，如日志、限流）、**后台任务**（响应返回后异步执行清理工作，不阻塞用户）、**WebSocket**（需要双向实时通信时使用，如带上下文的持续对话）。

限流器的价值在 AI API 中尤为突出——LLM 调用不是免费的，如果不做限流，单个用户可以无限制地调用，导致账单失控。下面的 `RateLimiter` 实现了滑动窗口限流（每分钟最多 N 次请求），生产环境中通常用 Redis 实现分布式限流。

```python
"""
ex_8_3_fastapi_advanced.py: FastAPI 进阶——中间件、依赖注入、后台任务
演示：日志中间件、请求限流、后台清理任务
"""
import os
import time
import asyncio
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict

app = FastAPI(title="Advanced AI API")


# ====== 1. 自定义中间件：请求日志 ======
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """记录每个请求的方法、路径、耗时和状态码"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # 在请求处理前
        print(f"[{time.strftime('%H:%M:%S')}] to {request.method} {request.url.path}")

        # 处理请求
        response = await call_next(request)

        # 在请求处理后
        elapsed = time.time() - start_time
        print(f"[{time.strftime('%H:%M:%S')}] -- {response.status_code} "
              f"({elapsed:.3f}s)")

        response.headers["X-Process-Time"] = str(elapsed)
        return response


# ====== 2. 简单的内存限流器 ======
# 生产环境用 Redis 实现分布式限流
class RateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        """
        参数:
            max_requests: 时间窗口内允许的最大请求数
            window_seconds: 时间窗口大小（秒）
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def __call__(self, client_ip: str) -> bool:
        """检查是否允许请求。返回 True = 允许，False = 限流"""
        now = time.time()
        window_start = now - self.window_seconds

        # 清理时间窗口之外的旧记录
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if t > window_start
        ]

        if len(self.requests[client_ip]) >= self.max_requests:
            return False  # 超过限制

        self.requests[client_ip].append(now)
        return True


rate_limiter = RateLimiter(max_requests=30, window_seconds=60)  # 每分钟 30 次


# ====== 3. 注册中间件 ======
app.add_middleware(RequestLoggingMiddleware)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """限流中间件"""
    client_ip = request.client.host if request.client else "unknown"

    if not rate_limiter(client_ip):
        return JSONResponse(
            status_code=429,
            content={
                "error": "请求过于频繁",
                "detail": f"每分钟最多 {rate_limiter.max_requests} 次请求",
                "retry_after": 60,  # 建议客户端 60 秒后重试
            },
        )

    response = await call_next(request)
    return response


# ====== 4. 后台任务 ======
def cleanup_temp_files():
    """后台清理任务：删除超过 1 小时的临时文件"""
    temp_dir = "./temp"
    if not os.path.exists(temp_dir):
        return

    now = time.time()
    one_hour_ago = now - 3600

    for filename in os.listdir(temp_dir):
        filepath = os.path.join(temp_dir, filename)
        if os.path.getmtime(filepath) < one_hour_ago:
            os.remove(filepath)
            print(f"已清理: {filepath}")


@app.post("/generate")
async def generate(background_tasks: BackgroundTasks):
    """
    生成内容的同时，注册一个后台清理任务。
    后台任务在响应返回后执行，不阻塞用户请求。
    """
    # ... 主要的生成逻辑 ...

    # 注册后台任务（在响应返回后异步执行）
    background_tasks.add_task(cleanup_temp_files)
    # 也可以传参: background_tasks.add_task(some_func, arg1, arg2)

    return {"status": "ok", "message": "开始生成..."}


# ====== 5. WebSocket 端点（实时双向通信） ======
@app.websocket("/ws/chat")
async def websocket_chat(websocket):
    """
    WebSocket 聊天端点。
    与 SSE 的区别: WebSocket 是双向通信，SSE 是单向（服务器to客户端）。

    消息格式:
      客户端 to 服务器: {"type": "message", "content": "..."}
      服务器 to 客户端: {"type": "content", "content": "..."}
    """
    await websocket.accept()
    print("WebSocket 连接已建立")

    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_json()

            if data.get("type") == "message":
                user_message = data["content"]

                # 调用 LLM 流式 API
                stream = await async_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": user_message}],
                    stream=True,
                )

                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        # 通过 WebSocket 发送每个 token
                        await websocket.send_json({
                            "type": "content",
                            "content": chunk.choices[0].delta.content,
                        })

                # 发送完成信号
                await websocket.send_json({"type": "done"})

            elif data.get("type") == "close":
                break

    except Exception as e:
        print(f"WebSocket 错误: {e}")
    finally:
        await websocket.close()
        print("WebSocket 连接已关闭")
```

### 8.1.5 生产级 FastAPI 项目结构

当项目从单个 `app.py` 发展到需要多人协作时，需要一个清晰的分层结构：

```
ai-api/
+-- app/
|   +-- __init__.py
|   +-- main.py            # FastAPI 应用入口
|   +-- config.py           # 配置（环境变量、Settings）
|   +-- dependencies.py     # 依赖注入（API Key 校验等）
|   +-- middleware/
|   |   +-- __init__.py
|   |   +-- logging.py      # 日志中间件
|   |   +-- rate_limit.py   # 限流中间件
|   +-- routers/
|   |   +-- __init__.py
|   |   +-- chat.py         # 聊天路由
|   |   +-- embeddings.py   # Embedding 路由
|   |   +-- admin.py        # 管理路由
|   +-- models/
|   |   +-- __init__.py
|   |   +-- request.py      # 请求 Pydantic 模型
|   |   +-- response.py     # 响应 Pydantic 模型
|   +-- services/
|   |   +-- __init__.py
|   |   +-- llm.py          # LLM 调用服务
|   |   +-- cache.py        # 缓存服务
|   +-- utils/
|       +-- __init__.py
|       +-- helpers.py      # 工具函数
+-- tests/
|   +-- test_chat.py
|   +-- test_embeddings.py
+-- Dockerfile
+-- docker-compose.yml
+-- requirements.txt
+-- .env.example
```

---

## 8.2 Gradio 快速原型 —— 让 ML 模型立刻拥有界面

### 8.2.1 Gradio 的设计哲学 —— "演示至上"

Gradio 和 FastAPI 解决的是同一个问题链的两端。FastAPI 解决的是"后端"，它的消费者是程序。Gradio 解决的是"演示"，它的消费者是人。Gradio 的设计哲学是：**ML 工程师不应该为了展示模型效果而学习前端开发**。只需定义一个函数 `fn(input) -> output`，Gradio 自动推断输入输出的类型并生成相应的 UI 组件——`str` 变成文本框，`list[str]` 变成下拉选项，`PIL.Image` 变成图片上传和显示区域。

`gr.ChatInterface` 是这个哲学的巅峰——一行代码创建一个带聊天气泡、对话历史、输入框的完整聊天界面。内部的 `history` 参数（格式为 `[(user_msg, bot_msg), ...]`）由 Gradio 自动管理。你只需要专注于"如何根据消息生成回复"这个核心逻辑。

```bash
pip install gradio
```

```python
"""
ex_8_4_gradio_chat.py: Gradio ChatInterface——几行代码创建聊天气泡界面
演示：最简单的 Gradio 聊天应用
"""
import os
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat_fn(message: str, history: list) -> str:
    """
    Gradio ChatInterface 的回调函数。

    参数:
        message: 用户当前输入的消息（字符串）
        history: 对话历史，格式为 [(user_msg, bot_msg), ...]

    返回:
        模型的文字回复（字符串）

    注意: Gradio 会自动把返回的字符串添加到聊天界面中。
    不需要手动管理 history——Gradio 内部处理。
    """
    # 将历史对话转为 OpenAI API 的 messages 格式
    messages = [
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"}
    ]

    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    # 调用 OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )

    return response.choices[0].message.content


# ====== 创建 ChatInterface（一行代码搞定） ======
demo = gr.ChatInterface(
    fn=chat_fn,                              # 处理函数
    title="AI 聊天助手",                      # 标题
    description="基于 GPT-4o-mini 的智能助手",  # 描述
    theme="soft",                            # 主题: "soft", "default", "monochrome", "glass"
    examples=["介绍 Python", "写一首诗", "解释相对论"],  # 预设示例
    # chatbot=gr.Chatbot(height=500),        # 自定义聊天窗口高度
    # textbox=gr.Textbox(placeholder="请输入..."),  # 自定义输入框
)

# 启动: python ex_8_4_gradio_chat.py to 浏览器自动打开
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # 允许局域网访问
        server_port=7860,        # 默认端口
        share=False,             # True = 生成公网链接（临时，72小时有效）
        # auth=("admin", "password"),  # 简单的用户名密码认证
    )
```

### 8.2.2 多 Tab 工具箱 —— Gradio Blocks

单个聊天界面是 Gradio 的"自动挡"，`gr.Blocks` 是"手动挡"——让你精确控制布局（行、列、Tab、展开区域等）。下面的代码展示了如何用一个 Python 文件创建三个独立的 AI 工具 Tab：聊天、翻译、代码生成。每个 Tab 有独立的输入输出组件和事件绑定。

```python
"""
ex_8_5_gradio_advanced.py: Gradio 进阶——自定义布局、多Tab、流式输出
演示：构建一个多功能的 AI 工具箱界面
"""
import os
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ====== 功能1: 聊天 ======
def chat_fn(message, history):
    messages = [{"role": "system", "content": "你是一个有帮助的 AI 助手。"}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
    )

    # 流式输出
    partial = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            partial += chunk.choices[0].delta.content
            yield partial  # Gradio 的 yield 实现流式更新


# ====== 功能2: 翻译 ======
def translate(text: str, target_lang: str) -> str:
    """将文本翻译为指定语言"""
    if not text.strip():
        return "请输入需要翻译的文本。"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"将以下文本翻译为{target_lang}，只输出译文，不要加任何解释:\n{text}",
        }],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ====== 功能3: 代码生成 ======
def generate_code(description: str, language: str) -> str:
    """根据描述生成代码"""
    if not description.strip():
        return "# 请输入代码描述"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"写一段{language}代码: {description}\n只输出代码，不要解释。",
        }],
        temperature=0.3,
        max_tokens=2048,
    )
    return response.choices[0].message.content


# ====== 构建界面 ======
with gr.Blocks(title="AI 工具箱", theme="soft") as demo:
    gr.Markdown("# AI 工具箱")
    gr.Markdown("集成了聊天、翻译、代码生成功能的 AI 助手。")

    # Tab 1: 聊天
    with gr.Tab(" 聊天"):
        gr.ChatInterface(
            fn=chat_fn,
            chatbot=gr.Chatbot(height=500),
            examples=["介绍一下 Python 的装饰器", "帮我写一封求职邮件"],
        )

    # Tab 2: 翻译
    with gr.Tab(" 翻译"):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="原文",
                    placeholder="请输入需要翻译的文本...",
                    lines=8,
                )
                target_lang = gr.Dropdown(
                    label="目标语言",
                    choices=["英文", "中文", "日文", "韩文", "法文", "德文", "西班牙文"],
                    value="英文",
                )
                translate_btn = gr.Button("翻译", variant="primary")

            with gr.Column():
                output_text = gr.Textbox(
                    label="译文",
                    lines=8,
                    interactive=False,
                )

        translate_btn.click(
            fn=translate,
            inputs=[input_text, target_lang],
            outputs=output_text,
        )

    # Tab 3: 代码生成
    with gr.Tab(" 代码生成"):
        with gr.Row():
            with gr.Column():
                code_desc = gr.Textbox(
                    label="代码描述",
                    placeholder="例如: 读取 CSV 文件并画柱状图",
                    lines=3,
                )
                code_lang = gr.Dropdown(
                    label="编程语言",
                    choices=["Python", "JavaScript", "Java", "Go", "Rust", "SQL", "HTML/CSS"],
                    value="Python",
                )
                code_btn = gr.Button("生成代码", variant="primary")

            with gr.Column():
                code_output = gr.Code(
                    label="生成的代码",
                    language="python",
                    lines=15,
                )

        code_btn.click(
            fn=generate_code,
            inputs=[code_desc, code_lang],
            outputs=code_output,
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

### 8.2.3 Gradio 部署到 HuggingFace Spaces

HuggingFace Spaces 提供免费的 Gradio/Streamlit 应用托管，非常适合个人项目和学习演示。

**步骤**：

1. 在 [huggingface.co](https://huggingface.co) 注册账号
2. 创建新 Space：New Space to 选择 Gradio SDK to 命名
3. 准备 `app.py`、`requirements.txt` 文件
4. 在 Space 的 Settings 中添加 `OPENAI_API_KEY` 等环境变量（Secrets）
5. Git push 到 Space 仓库

```
# requirements.txt
gradio>=4.0.0
openai>=1.0.0
python-dotenv>=1.0.0

# app.py 就是你的 Gradio 代码
# 注意：不需要 demo.launch()，HuggingFace 会自动处理
# 最后一行用 demo.launch() 或什么都不写，Space 会在启动时自动调用
```

---

## 8.3 Streamlit 数据应用 —— Python 脚本即 Web 界面

### 8.3.1 Streamlit 的设计哲学 —— "脚本即界面"

Streamlit 的核心理念比 Gradio 更激进：**你不用写任何 Web 代码，Python 脚本本身就是 Web 应用**。`st.title()` 生成标题，`st.button()` 生成按钮，`st.chart()` 生成图表。每次用户交互（点击按钮、输入文字、拖动滑块），Streamlit 从顶到底重新执行整个 Python 脚本——这听起来很浪费，但实际上确保了"脚本状态 = 页面状态"的一致性。

这种"全量重跑"模式带来了一个独特的问题：如果你在脚本里写了 `messages = []`，每次用户点击都会把 `messages` 重置为空。解决方案是 `st.session_state`——一个跨 rerun 持久化的字典，用于保存对话历史、分析结果等需要"记住"的数据。

Streamlit 与 Gradio 的核心区别在于：Gradio 是"回调驱动"（你定义 fn，Gradio 在用户操作时调用 fn），Streamlit 是"脚本驱动"（整个脚本从上到下执行，通过条件判断决定哪些代码块执行）。Gradio 适合"输入 → 输出"型的任务（上传图片 → 得到分类结果），Streamlit 适合"探索式"场景（选择参数 → 查看图表 → 调整参数 → 查看变化）。

```bash
pip install streamlit
```

```python
"""
ex_8_6_streamlit_chat.py: Streamlit 聊天应用
演示：st.chat_message + st.chat_input + session_state

启动方式: streamlit run ex_8_6_streamlit_chat.py
"""
import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# ====== 页面配置（必须在最前面） ======
st.set_page_config(
    page_title="AI Chat",
    page_icon="💬",
    layout="wide",    # "centered" or "wide"
    initial_sidebar_state="expanded",
)

load_dotenv()


# ====== 初始化 session_state ======
# session_state 是跨 rerun 持久化的字典
# Streamlit 每次交互都会重新执行整个脚本，session_state 是唯一的持久化方法

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"}
    ]

if "display_messages" not in st.session_state:
    # 用于显示的消息（不包含 system 消息）
    st.session_state.display_messages = []


# ====== 侧边栏 ======
with st.sidebar:
    st.title(" 设置")

    model = st.selectbox(
        "模型",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        index=0,
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
    )

    max_tokens = st.slider(
        "Max Tokens",
        min_value=100,
        max_value=4096,
        value=1024,
        step=100,
    )

    if st.button(" 清空对话"):
        st.session_state.messages = [
            {"role": "system", "content": "你是一个有帮助的 AI 助手。"}
        ]
        st.session_state.display_messages = []
        st.rerun()


# ====== 主界面标题 ======
st.title(" AI Chat")
st.caption("基于 OpenAI API 的智能对话助手")


# ====== 显示历史消息 ======
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):  # st.chat_message 创建聊天气泡
        st.markdown(msg["content"])


# ====== 用户输入 ======
if prompt := st.chat_input("输入你的消息..."):
    # 1. 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 添加到状态
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.display_messages.append({"role": "user", "content": prompt})

    # 3. 调用 LLM
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with st.chat_message("assistant"):
        # 流式显示（用空占位符 + 逐步填充）
        message_placeholder = st.empty()
        full_response = ""

        # 调用 OpenAI 流式 API
        stream = client.chat.completions.create(
            model=model,
            messages=st.session_state.messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                # 逐步更新占位符内容（模拟打字效果）
                message_placeholder.markdown(full_response + "▌")

        # 去掉光标效果
        message_placeholder.markdown(full_response)

    # 4. 保存到状态
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.display_messages.append({"role": "assistant", "content": full_response})
```

### 8.3.2 缓存 —— Streamlit 的"记忆"

Streamlit 的缓存机制是理解其性能的关键。`@st.cache_data` 和 `@st.cache_resource` 看起来相似但作用不同：前者缓存数据（如 LLM 的分析结果），当输入相同时直接返回缓存值而不重新调用昂贵的 API；后者缓存不可序列化的对象（如数据库连接、模型实例）。`ttl` 参数设置缓存有效期——超过这个时间缓存自动失效，确保用户不会看到永远不变的过期数据。

```python
"""
ex_8_7_streamlit_data.py: Streamlit 数据应用——状态管理、缓存、图表
演示：构建一个 AI 文档分析仪表盘
"""
import os
import time
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="文档分析仪表盘", layout="wide")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ====== 1. 缓存装饰器 ======
# @st.cache_data: 缓存函数返回值（数据层面）
# @st.cache_resource: 缓存外部资源（数据库连接、模型加载等）
# 区别: cache_data 用于纯数据（JSON/DataFrame），cache_resource 用于不可序列化的对象

@st.cache_data(ttl=3600)  # TTL=3600秒(1小时)后缓存失效
def analyze_document(text: str) -> dict:
    """
    分析文档内容，返回关键词、摘要、情绪等。
    ttl 参数确保 1 小时后缓存自动失效，获取新鲜结果。

    注意: 相同的 text 输入只会调用一次 API（缓存命中时直接返回）。
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""分析以下文本，返回 JSON:
{{
  "summary": "100字以内的摘要",
  "keywords": ["关键词1", "关键词2", ...],
  "sentiment": "positive/neutral/negative",
  "word_count": 总词数,
  "language": "语言名称"
}}

文本:
{text[:3000]}

只返回 JSON。""",
        }],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


@st.cache_data(ttl=600)
def embed_texts(texts: tuple) -> list:
    """批量 embedding（用 tuple 才能 hash）"""
    # 实际项目中应使用 batch embedding API
    embeddings = []
    for text in texts:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        embeddings.append(resp.data[0].embedding)
    return embeddings


# ====== 2. 文件上传 ======
st.title(" AI 文档分析仪表盘")

uploaded_files = st.file_uploader(
    "上传文档（支持 TXT, CSV, JSON）",
    type=["txt", "csv", "json"],
    accept_multiple_files=True,
)


# ====== 3. 状态管理：跨文件聚合分析 ======
if "analyzed_docs" not in st.session_state:
    st.session_state.analyzed_docs = []  # 保存所有已分析的文档结果

if uploaded_files:
    st.info(f"已上传 {len(uploaded_files)} 个文件")

    # 文件选择器
    file_names = [f.name for f in uploaded_files]
    selected_file = st.selectbox("选择要分析的文件", file_names)

    if st.button(" 分析此文件", type="primary"):
        # 找到选中的文件
        target_file = next(f for f in uploaded_files if f.name == selected_file)

        # 读取内容
        if selected_file.endswith(".txt"):
            content = target_file.read().decode("utf-8")
        elif selected_file.endswith(".csv"):
            df = pd.read_csv(target_file)
            content = df.to_string()
        elif selected_file.endswith(".json"):
            content = json.dumps(json.load(target_file), ensure_ascii=False, indent=2)
        else:
            content = target_file.read().decode("utf-8")

        # 调用分析（带缓存 + 进度条）
        with st.spinner("分析中..."):
            progress_bar = st.progress(0, text="正在调用 AI...")

            start_time = time.time()
            result = analyze_document(content)
            elapsed = time.time() - start_time

            progress_bar.progress(100, text=f"完成！（耗时 {elapsed:.2f}s）")

        # 保存结果
        result["file_name"] = selected_file
        result["analyzed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.analyzed_docs.append(result)

        # 显示结果
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("词数", result.get("word_count", 0))
        col2.metric("情绪", result.get("sentiment", "N/A"))
        col3.metric("语言", result.get("language", "N/A"))
        col4.metric("耗时", f"{elapsed:.2f}s")

        with st.expander(" 摘要", expanded=True):
            st.write(result.get("summary", ""))

        with st.expander(" 关键词"):
            keywords = result.get("keywords", [])
            st.write(", ".join(keywords))


    # ====== 4. 聚合仪表盘 ======
    if len(st.session_state.analyzed_docs) > 0:
        st.divider()
        st.subheader(" 聚合分析")

        analyzed = st.session_state.analyzed_docs

        # 数据表格
        df_display = pd.DataFrame(analyzed)
        st.dataframe(
            df_display[["file_name", "sentiment", "word_count", "language", "analyzed_at"]],
            use_container_width=True,
        )

        # 情绪分布饼图
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.write("**情绪分布**")
            sentiment_counts = {}
            for doc in analyzed:
                s = doc.get("sentiment", "unknown")
                sentiment_counts[s] = sentiment_counts.get(s, 0) + 1

            fig, ax = plt.subplots()
            ax.pie(
                sentiment_counts.values(),
                labels=sentiment_counts.keys(),
                autopct="%1.1f%%",
                colors=["#2ecc71", "#e74c3c", "#95a5a6"],
            )
            st.pyplot(fig)

        with col_chart2:
            st.write("**文档词数对比**")
            names = [d["file_name"][:15] for d in analyzed]
            words = [d.get("word_count", 0) for d in analyzed]

            fig, ax = plt.subplots()
            ax.bar(names, words, color="#3498db")
            ax.set_ylabel("Word Count")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)

        # 所有关键词汇总
        all_keywords = []
        for doc in analyzed:
            all_keywords.extend(doc.get("keywords", []))
        from collections import Counter
        top_kw = Counter(all_keywords).most_common(10)

        if top_kw:
            st.write("**热门关键词 TOP 10**")
            kw_df = pd.DataFrame(top_kw, columns=["关键词", "出现次数"])
            st.bar_chart(kw_df.set_index("关键词"))
```

### 8.3.3 三种工具的选型总结

选择 FastAPI、Gradio 还是 Streamlit，取决于你的目标用户和使用场景：

| 需求 | 推荐方案 |
|------|----------|
| 快速原型/演示（"看，我的模型能做这个！"） | Gradio |
| 数据仪表盘/分析报告（"让我探索这些数据"） | Streamlit |
| 生产级 API 后端（"让其他程序调用我的服务"） | FastAPI |
| 生产级全栈应用 | FastAPI + 前端框架 (React/Vue) |

---

## 8.4 Docker 容器化 —— "在我电脑上能跑"的终极解决方案

### 8.4.1 理解 Docker：从"环境差异"到"环境一致"

如果你曾经把 Python 项目从自己的 Mac 部署到 Linux 服务器上，大概率遇到过这些问题：Python 版本不一致、某些依赖包的 C 扩展编译失败、环境变量没配全、Chroma 用 pip 装你电脑上但服务器上需要 Docker 镜像……Docker 的核心思想很简单：**把应用和它依赖的一切（Python 解释器、系统库、第三方包、环境变量）打包成一个"镜像"**。这个镜像可以在任何安装了 Docker 的机器上原封不动地运行，不再有"我电脑上能跑啊"的尴尬。

三个核心概念：
- **Dockerfile**：构建镜像的"配方"。声明基础镜像（`FROM python:3.12-slim`）、安装依赖（`RUN pip install`）、复制代码（`COPY`）、启动命令（`CMD`）。
- **镜像（Image）**：根据 Dockerfile 构建出来的"模板"。镜像是只读的。
- **容器（Container）**：镜像的运行实例。容器是可读写的，但重启后写入的数据会丢失——因此需要持久化的数据（如向量库、日志）必须通过 `volumes` 挂载到宿主机目录。

### 8.4.2 Dockerfile 编写 —— 从最简到生产

```dockerfile
# ====== Dockerfile: FastAPI 应用 ======
# 文件名: Dockerfile

# 第一步: 选择基础镜像
FROM python:3.12-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 设置工作目录
WORKDIR /app

# 第二步: 安装依赖
# 先复制 requirements.txt（利用 Docker 缓存层）
# 如果 requirements.txt 没变，Docker 会复用这一层的缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 第三步: 复制应用代码
COPY . .

# 第四步: 暴露端口
EXPOSE 8000

# 第五步: 启动命令
# 使用 uvicorn 启动 FastAPI
# --host 0.0.0.0: 允许外部访问
# --port 8000: 监听 8000 端口
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# ====== Dockerfile: Streamlit 应用 ======
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Streamlit 默认端口 8501
# --server.address 0.0.0.0 允许从容器外访问
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
```

### 8.4.3 docker-compose.yml —— 编排多个服务

docker-compose 解决的是"多服务协作"的问题。你不可能只有一个 FastAPI 进程——至少还有向量数据库（Chroma/Milvus）、缓存（Redis）、反向代理（Nginx）。docker-compose 用 YAML 声明这些服务之间的关系（启动顺序、网络连通、环境变量共享），一条 `docker-compose up -d` 启动整个技术栈。

```yaml
# ====== docker-compose.yml: FastAPI + Chroma 向量数据库 ======
version: "3.8"

services:
  # 应用服务
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ai-api
    ports:
      - "8000:8000"
    environment:
      # 从宿主机环境变量注入，或直接写在这里（后者不推荐）
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_BASE_URL=${OPENAI_BASE_URL:-}
      - API_KEYS=${API_KEYS:-dev-key-123}
      - CHROMA_HOST=chroma
      - CHROMA_PORT=8001
      - LOG_LEVEL=INFO
    volumes:
      # 挂载代码目录（开发时方便热重载）
      - ./app:/app/app
      # 持久化向量库数据
      - ./data/chroma:/data/chroma
    depends_on:
      chroma:
        condition: service_healthy  # 等 chroma 健康检查通过后再启动
    restart: unless-stopped
    networks:
      - ai-network

  # Chroma 向量数据库
  chroma:
    image: chromadb/chroma:latest
    container_name: chroma-db
    ports:
      - "8001:8001"
    environment:
      - IS_PERSISTENT=TRUE
      - PERSIST_DIRECTORY=/chroma/data
    volumes:
      - ./data/chroma:/chroma/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/api/v2/heartbeat"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    networks:
      - ai-network

  # Nginx 反向代理（可选，用于 SSL 终止、负载均衡、静态文件服务）
  nginx:
    image: nginx:alpine
    container_name: ai-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro  # SSL 证书
    depends_on:
      - api
    restart: unless-stopped
    networks:
      - ai-network

networks:
  ai-network:
    driver: bridge
```

### 8.4.4 Docker 常用命令

```bash
# 构建镜像
docker build -t ai-api:latest .

# 运行容器
docker run -d -p 8000:8000 --name ai-api --env-file .env ai-api:latest

# 使用 docker-compose
docker-compose up -d              # 启动所有服务（后台运行）
docker-compose down               # 停止并删除所有服务
docker-compose logs -f api        # 查看 api 服务的日志（实时跟踪）
docker-compose restart api        # 重启 api 服务
docker-compose exec api bash      # 进入 api 容器的终端

# 查看
docker ps                         # 运行中的容器
docker images                     # 本地镜像列表
docker logs ai-api                # 查看容器日志

# 清理
docker system prune -a            # 清理所有未使用的镜像/容器/网络（慎用）
```

---

## 8.5 部署方案 —— 从免费到企业级

### 8.5.1 不同规模与成本的部署方案

部署方案的选择不是一个技术问题，而是一个"成本 vs 便利 vs 控制"的权衡问题。以下是四级方案：

**方案一：HuggingFace Spaces（免费，适合演示）**。最简单的方式——把你的 Gradio/Streamlit 代码 push 到 HuggingFace Spaces，自动部署并分配一个公网 URL。限制是只能用 CPU（推理速度慢）、冷启动延迟、无自定义域名。适合展示个人项目。

**方案二：Railway / Render（每月 $5-25，适合小规模应用）**。比 Spaces 更专业——自动从 GitHub 部署、内置 HTTPS、环境变量管理、可以选择 CPU/内存配置。适合小团队内部工具和 MVP。

**方案三：VPS + Docker（每月 $10-50，适合成长中的产品）**。租一台云服务器，安装 Docker，用 docker-compose 部署。你有完全的控制权，但需要自己管理 SSL 证书、域名解析、系统更新。下面的 Nginx 配置和 systemd 配置展示了典型的中型部署结构。

**方案四：Kubernetes 集群（每月 $100+，适合大规模生产）**。当你的应用需要自动扩缩容、滚动更新、多区域部署时，Kubernetes 是业界标准。但除非你的日活用户超过 1000，否则不需要。

```nginx
# nginx.conf 示例
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE 流式配置（关键！否则流式会被缓冲）
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
}
```

```bash
# 用 systemd 管理 Uvicorn（自动重启、开机启动）
# /etc/systemd/system/ai-api.service
[Unit]
Description=AI API Service
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/ai-api
ExecStart=/opt/ai-api/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 4
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

#### 方案四：Kubernetes 集群（高成本，适合大规模生产）

- **成本**: $100+/月 起步
- **服务商**: AWS EKS、GCP GKE、阿里云 ACK
- **适用**: 高可用、弹性伸缩、多服务编排

基本架构：
```
Load Balancer to Ingress Controller to Service to Deployment (N个Pod)
                                                 |
                                          ConfigMap (配置) + Secret (密钥)
```

### 8.5.2 环境变量注入模式

```python
"""
ex_8_8_config_management.py: 配置管理——多环境、多来源
演示：统一的环境变量/配置文件管理
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    应用配置类。Pydantic Settings 自动从以下来源读取配置（优先级从高到低）：
    1. 环境变量（或 .env 文件）
    2. 代码中设置的默认值

    使用方式:
        settings = Settings()
        print(settings.openai_api_key)
    """
    model_config = SettingsConfigDict(
        env_file=".env",           # 读取 .env 文件
        env_file_encoding="utf-8",
        case_sensitive=False,      # 环境变量名不区分大小写
        extra="ignore",            # 忽略未定义的额外环境变量
    )

    # OpenAI 配置
    openai_api_key: str = ""
    openai_base_url: Optional[str] = None

    # 应用配置
    app_name: str = "AI API"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # 鉴权配置
    api_keys: str = ""  # 逗号分隔的合法 API Key 列表

    # 数据库配置
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # 限流配置
    rate_limit_per_minute: int = 60

    @property
    def valid_api_keys(self) -> set[str]:
        """将逗号分隔的 key 字符串转为集合"""
        if not self.api_keys:
            return set()
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def is_development(self) -> bool:
        """是否开发模式"""
        return self.debug or os.getenv("ENV", "production") == "development"


# 全局单例
settings = Settings()


if __name__ == "__main__":
    print(f"应用名称: {settings.app_name}")
    print(f"调试模式: {settings.debug}")
    print(f"Chroma: {settings.chroma_host}:{settings.chroma_port}")
    print(f"API Keys 数量: {len(settings.valid_api_keys)}")
```

---

## 基础练习

### 练习 1: FastAPI Chat API

实现一个完整的 FastAPI Chat API，包含非流式和流式两个端点，支持 API Key 鉴权，并在 Swagger 文档中有完整的请求/响应说明。

**练习文件**: `exercise/ai-application/ch08_web_ai/ex_fastapi_chat.py`

### 练习 2: Gradio 多模型对比工具

构建一个 Gradio 应用，允许用户同时向两个不同的模型（如 gpt-4o-mini 和 gpt-4o）发送相同的问题，并排展示两个模型的回答，方便对比。

**练习文件**: `exercise/ai-application/ch08_web_ai/ex_model_compare.py`

### 练习 3: Streamlit Token 用量仪表盘

构建一个 Streamlit 应用，读取本地日志文件中的 API 调用记录，展示每日/每周的 Token 用量趋势图、模型使用分布饼图、成本统计。

**练习文件**: `exercise/ai-application/ch08_web_ai/ex_token_dashboard.py`

---

## 进阶练习

### 练习 4: 完整 RAG Web 应用

用 FastAPI + Streamlit 构建一个完整的 RAG Web 应用：
- **后端 (FastAPI)**: 提供 `/upload`（上传文档）、`/query`（RAG 问答，支持流式）、`/list`（列出已索引的文档）等接口
- **前端 (Streamlit)**: 提供文档上传、问答输入、文档列表、对话历史等界面
- 使用 Docker Compose 编排应用和 Chroma 向量数据库

**练习文件**: `exercise/ai-application/ch08_web_ai/ex_fullstack_rag.py`

### 练习 5: 多用户 LLM Gateway

构建一个简易的 LLM API Gateway：
- FastAPI 提供统一的 `/v1/chat/completions` 端点（兼容 OpenAI API 格式）
- 支持多个 LLM 后端（OpenAI、Azure OpenAI、本地模型）
- 按 API Key 路由到不同的后端
- 按 API Key 做速率限制和 Token 配额管理
- Docker 部署

**练习文件**: `exercise/ai-application/ch08_web_ai/ex_llm_gateway.py`

---

## 常见错误

### 错误 1: SSE 流式被 Nginx 缓冲

```nginx
# 错误配置（流式数据被缓冲，用户等很久才一次性看到结果）
location / {
    proxy_pass http://127.0.0.1:8000;
}

# 正确配置（关闭缓冲）
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_buffering off;       # 关闭代理缓冲
    proxy_cache off;           # 关闭缓存
    proxy_read_timeout 300s;   # 延长超时（LLM 生成可能需要较长时间）
    chunked_transfer_encoding on;
}
```

### 错误 2: CORS 未配置导致前端请求被拦截

```python
# 错误: 前端 fetch("http://localhost:8000/v1/chat") 被浏览器拦截
# 浏览器控制台: "No 'Access-Control-Allow-Origin' header is present"

# 正确: 在 FastAPI 中添加 CORS 中间件
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 明确指定前端域名
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 错误 3: Streamlit 每次操作都重置变量

```python
# 错误: 把数据放在普通变量中
messages = []  # -- 每次点击按钮/输入文本，脚本 rerun，messages 被重置

# 正确: 使用 st.session_state
if "messages" not in st.session_state:
    st.session_state.messages = []
```

### 错误 4: Docker 容器中 localhost 指向容器自身

```python
# 错误: 在 docker-compose 中，服务之间的通信不能用 localhost
# chroma.HTTPClient(host="localhost", port=8001)
# to localhost 指的是 api 容器自己，不是 chroma 容器

# 正确: 使用 docker-compose 的服务名作为主机名
# chroma.HTTPClient(host="chroma", port=8001)
# docker-compose 默认创建网络，服务名即 DNS 名
```

### 错误 5: Python slim 镜像缺少编译依赖

```dockerfile
# 错误: python:3.12-slim 缺少 gcc 等编译工具
# 导致某些 Python 包（如 chroma-hnswlib）安装失败

# 解决方案1: 在 Dockerfile 中安装编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make && \
    pip install -r requirements.txt && \
    apt-get remove -y gcc g++ make && apt-get autoremove -y

# 解决方案2: 使用更完整的镜像
FROM python:3.12  # 而不是 slim 版
```

### 错误 6: Uvicorn workers 设置不当

```python
# 错误: 使用多个 worker 运行 uvicorn
# uvicorn app.main:app --workers 4
# 结合 SSE 流式，多 worker 可能导致某些请求被路由到无响应的 worker

# 正确: 生产环境用 gunicorn + uvicorn workers 管理
# gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
# 每个 worker 独立运行一个 FastAPI 实例
```

---

## 本章小结

本章系统介绍了将 AI 应用转化为可访问的 Web 服务的完整技术栈：

1. **FastAPI** 是生产级 AI 后端的首选：
   - 原生异步支持 LLM 的高延迟调用
   - Pydantic 自动校验请求参数
   - StreamingResponse 实现 SSE 流式推送
   - 自动生成交互式 API 文档

2. **Gradio** 是原型验证的最快路径：
   - `gr.ChatInterface` 一行代码创建聊天 UI
   - `gr.Blocks` 支持多 Tab、自定义布局
   - 一键部署到 HuggingFace Spaces（免费）

3. **Streamlit** 是数据驱动 AI 应用的最佳选择：
   - 脚本式编程，Python 就是 UI
   - `st.session_state` 解决无状态问题
   - `@st.cache_data` 缓存昂贵的 API 调用
   - 丰富的内置组件：DataFrame、图表、指标卡片

4. **Docker** 是实现环境一致性和可移植性的关键：
   - Dockerfile 定义应用运行环境
   - docker-compose 编排多服务（应用 + 数据库 + 反向代理）
   - 环境变量注入管理配置

5. **部署方案** 按预算和规模分级：
   - 免费: HuggingFace Spaces
   - 小规模: Railway / Render
   - 中规模: VPS + Docker + Nginx
   - 大规模: Kubernetes 集群

---

**下一章**: 第09章 生产环境最佳实践——Token 成本追踪、缓存、限流、监控、安全、测试。
