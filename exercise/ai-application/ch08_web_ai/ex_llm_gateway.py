"""
进阶练习: 多用户 LLM Gateway
============================

需求:
  构建一个简易的 LLM API Gateway，提供与 OpenAI 兼容的接口，
  内部路由到不同的 LLM 后端，按 API Key 做鉴权和配额管理。

要求:
  1. 兼容 OpenAI API 格式:
     - POST /v1/chat/completions (核心，兼容 OpenAI 请求/响应格式)
     - GET /v1/models
     - 确保客户端可以用 OpenAI SDK 直接调用（修改 base_url 即可）

  2. 多后端路由:
     - 根据 API Key 决定使用哪个 LLM 后端
     - 支持的 backends:
       a. OpenAI (gpt-4o, gpt-4o-mini)
       b. Azure OpenAI
       c. 本地 mock 模型（用于测试，直接返回固定文本）
     - 每个 API Key 绑定一个 backend + model 组合
     - 在配置中定义 api_key → backend mapping

  3. 速率限制:
     - 每个 API Key 单独的速率限制
     - 默认: 每分钟 30 次请求
     - 超过限制返回 429 + Retry-After 头
     - 用内存字典存储（可选：Redis 实现分布式限流）

  4. Token 配额管理:
     - 每个 API Key 有月度的 Token 配额上限
     - 每次请求后从配额中扣除实际使用的 Token 数
     - 配额不足时返回 429（或 402 Payment Required）
     - 返回头中包含: X-RateLimit-Remaining-Tokens

  5. 请求/响应日志:
     - 记录每次请求: api_key(脱敏), model, prompt_tokens, completion_tokens, latency
     - 输出到结构化 JSON 日志文件
     - 每个请求有唯一的 request_id (UUID)

  6. 配置管理:
     - 从 YAML/JSON 配置文件读取 API Keys 和 Backends 的映射
     - 支持热重载配置（监听文件变化或提供 API 端点）

TODO:
  - [ ] 实现 GatewayConfig 类（加载和管理配置）
  - [ ] 实现 APIRateLimiter 类（每个 Key 独立的速率限制）
  - [ ] 实现 TokenQuotaManager 类（每个 Key 的 Token 配额管理）
  - [ ] 实现 BackendRouter 类（根据 API Key 选择后端）
  - [ ] 实现 OpenAI 兼容的 /v1/chat/completions 路由
  - [ ] 实现请求日志记录（结构化 JSON）
  - [ ] 实现配置热重载
  - [ ] 编写 api_keys.json 配置示例

提示:
  - 用 Pydantic 解析 OpenAI 兼容的请求/响应格式
  - 速率限制: 滑动窗口算法（记录时间戳，清理过期记录）
  - Token 配额: 从响应的 usage 字段减去使用的 token 数
  - 脱敏: api_key 只记录前4后4位，中间用 *** 替换
  - 配置热重载: 使用文件 mtime 检测变化
  - uuid.uuid4() 生成唯一请求 ID
  - 在响应头中添加自定义头: X-Request-ID, X-RateLimit-Remaining, X-Quota-Remaining
"""
import os
import json
import time
import uuid
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv

load_dotenv()


class GatewayConfig:
    # TODO: 加载配置（api_keys → backend mapping）
    pass


class APIRateLimiter:
    # TODO: 每个 API Key 独立的滑动窗口限流
    pass


class TokenQuotaManager:
    # TODO: 每个 API Key 的月度 Token 配额管理
    pass


class BackendRouter:
    # TODO: 根据 API Key 选择后端并转发请求
    pass


# TODO: FastAPI 应用
app = FastAPI(title="LLM Gateway", version="1.0.0")

# TODO: 中间件（请求ID、日志）

# TODO: /v1/chat/completions (POST)

# TODO: /v1/models (GET)

# TODO: /admin/config/reload (POST) — 配置热重载

# TODO: /admin/stats (GET) — 统计信息


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
