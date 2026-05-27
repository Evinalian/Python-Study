# 第03章 流式处理与SSE

## 学习目标

完成本章后，你将能够：

1. 理解流式响应的核心价值：让用户"看到进度"而非"干等结果"
2. 掌握 SSE（Server-Sent Events）协议的原理和数据格式
3. 使用 OpenAI SDK 的 stream=True 实现客户端流式消费
4. 解析流式 chunk 的结构，正确处理 delta、finish_reason、usage
5. 在流式模式下正确处理 Function Calling 的分块传输
6. 使用 FastAPI 构建流式 API 服务，前端通过 EventSource 消费
7. 实现高级流式处理：多流并行、中断控制、断点续传

## 前置知识

- OpenAI SDK 的基础用法和 messages 结构
- Function Calling 的基本原理（第02章内容）
- Python 异步编程基础（async/await, async generator）
- 基本的 HTTP 协议和 Web API 概念
- JSON 数据结构

---

## 1. 为什么需要流式

### 1.1 两种模式的本质差异 —— 不只是"快一点"

非流式（stream=False）和流式（stream=True）的区别不仅仅是"速度快慢"的问题，而是两种完全不同的**用户体验模型**和**系统架构模式**。

非流式模式下，用户发送请求后，LLM 开始在服务器端逐 token 生成完整回复，但这整个过程对用户是不可见的。用户看到的是一片空白，直到全部生成完毕才一次性收到结果。这带来了两个问题：第一是**感知延迟**——用户不知道系统是在工作还是已经崩溃；第二是**超时风险**——如果生成内容较长，期间没有任何数据传输，HTTP 连接或负载均衡器可能判定超时而断开。

流式模式下，每生成一个 token，服务器就立即推送一个数据片段到客户端。用户看到的是逐字蹦出的效果——就像打字一样。这不仅仅是"好看"，它有真实的工程价值：**首 token 延迟（TTFT, Time To First Token）**成为用户体验的核心指标。用户看到第一个字就确认了系统在工作，剩余等待时间的"痛苦感"大大降低。

更深一层看，流式还改变了系统的容错模型。非流式是"全有或全无"——要么拿到完整结果，要么超时报错。流式是"增量交付"——即使中途断开，用户至少已经看到了部分内容，你的系统可以基于已接收的内容做断点续传。

```python
"""
非流式(stream=False) vs 流式(stream=True) 的用户体验对比。

非流式:
  用户: "写一篇关于AI的1000字文章"
  → 等待 20 秒 ... (用户看着空白屏幕，不知道发生了什么)
  → 一次性返回整篇文章

流式:
  用户: "写一篇关于AI的1000字文章"
  → 立即看到第一个字，然后像打字一样逐字蹦出
  → 用户可以边读边思考
"""
import os
import time
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


def non_streaming_demo(prompt: str):
    """非流式调用 —— 等到全部生成完才返回"""
    print("[非流式] 正在生成...")
    t0 = time.time()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )

    elapsed = time.time() - t0
    content = response.choices[0].message.content
    print(f"[非流式] 生成完成，耗时 {elapsed:.1f}s")
    print(f"[非流式] 前100字: {content[:100]}...")


def streaming_demo(prompt: str):
    """流式调用 —— 逐 token 返回"""
    print("[流式] 正在生成...")
    t0 = time.time()
    first_token_time = None
    full_content = ""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in response:
        # 记录首 token 时间
        if first_token_time is None and chunk.choices[0].delta.content:
            first_token_time = time.time() - t0
            print(f"[流式] 首token延迟: {first_token_time:.3f}s")

        # 获取当前 chunk 的文本增量
        delta = chunk.choices[0].delta.content
        if delta:
            full_content += delta

    total_elapsed = time.time() - t0
    print(f"[流式] 全部完成，总耗时 {total_elapsed:.1f}s")
    print(f"[流式] 首token: {first_token_time:.3f}s, 剩余: {total_elapsed - first_token_time:.3f}s")
    print(f"[流式] 前100字: {full_content[:100]}...")

    return {
        "first_token_time": first_token_time,
        "total_time": total_elapsed,
        "content_length": len(full_content),
    }


if __name__ == "__main__":
    prompt = "请用200字介绍人工智能的发展历史"

    print("=" * 60)
    non_streaming_demo(prompt)
    print()

    print("=" * 60)
    result = streaming_demo(prompt)
    print(f"\n总结: 首token {result['first_token_time']:.3f}s 即开始显示，而非流式要等 {result['total_time']:.1f}s")
```

代码分析：注意首 token 延迟（first_token_time）和总耗时的区别。在流式模式下，first_token_time 通常远小于总耗时——用户可能在 0.3 秒内就看到第一个字，而完整生成需要 3 秒。这 0.3 秒是用户感知到的"响应时间"，而非流式下用户感知到的是整个 3 秒。这就是流式最核心的用户体验价值。

### 1.2 流式处理的核心价值

| 维度 | 非流式 (stream=False) | 流式 (stream=True) |
|------|----------------------|-------------------|
| 用户感知延迟 | 等待全部完成 | 首 token 即刻显示 |
| 用户体验 | 空白等待 | 逐字显示，类似打字 |
| 超时风险 | 长时间无响应可能超时 | 持续有数据流，不易超时 |
| 内存占用 | 完整响应存在内存中 | 逐 token 处理，可及时释放 |
| 可中断性 | 无法中途停止 | 可以在任意 token 处停止 |
| 处理复杂度 | 简单 | 需要处理 delta 和 finish_reason |

---

## 2. SSE（Server-Sent Events）协议

### 2.1 SSE 是什么 —— 单向推送的轻量协议

SSE（Server-Sent Events）是一种基于 HTTP 的服务器向客户端推送数据的技术。它与普通 HTTP 请求的本质区别在于：**普通 HTTP 是"请求-响应"模式，一个请求对应一个响应，连接用完即关；SSE 是"长连接推送"模式，客户端发起一个请求后，服务器保持连接不关闭，持续向客户端推送数据**。

OpenAI 选择 SSE 作为流式 API 的底层协议，而不是 WebSocket，有几个关键原因：

**第一，单向通信足够**。LLM 的流式输出只需要服务器向客户端推送 token，客户端在生成过程中不需要向服务器发送数据。SSE 天然就是单向的（服务器 -> 客户端），完美匹配这个场景。WebSocket 是双向的，能力过剩，带来了额外的握手开销和复杂度。

**第二，基于 HTTP，基础设施友好**。SSE 走的是标准 HTTP 端口和协议，不需要像 WebSocket 那样进行 `Upgrade` 握手。这意味着现有的 HTTP 代理、负载均衡器、CDN 无需特殊配置就能支持 SSE。而 WebSocket 在部分企业网络环境中可能被防火墙拦截。

**第三，浏览器原生支持自动重连**。`EventSource` API 内置了自动重连机制——连接断开后浏览器会自动重新建立连接。这对于生产环境中的流式服务非常重要，因为网络波动是常态。WebSocket 需要手动实现重连逻辑。

**第四，简单**。SSE 的数据格式就是纯文本的 `data: <json>\n\n`，比 WebSocket 的帧协议简单得多，调试也更容易（可以直接用 curl 看原始数据流）。

### 2.2 SSE 的数据格式

OpenAI 的流式 API 底层使用的是 SSE 协议。理解其数据格式有助于自己构建流式服务和调试问题。

```
SSE 响应格式（HTTP response body）:

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":...,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"你"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":...,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"好"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":...,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

每条消息的格式是：

```
data: <JSON内容>\n\n
```

注意几个关键细节：
- 以 `data: ` 开头（注意冒号后有空格）
- 每行以 `\n` 结尾
- 每条消息之间用空行 `\n\n` 隔开——这是 SSE 协议的分帧方式
- 最后一条是 `data: [DONE]`——这是一个规范信号，不是 JSON
- 可以包含 `event:`、`id:`、`retry:` 字段来控制事件类型和重连策略

### 2.3 手动解析 SSE 流 —— 不用 SDK 访问 OpenAI

理解底层协议的最好方式是不用 SDK，直接用 HTTP 库手动解析 SSE 流。下面用 `requests` 库发送 HTTP POST 请求，手动解析 SSE 格式。

```python
"""
手动解析 SSE（Server-Sent Events）流。

理解底层协议有助于:
1. 调试流式 API 问题
2. 自己实现流式服务
3. 处理非 OpenAI 的 SSE 流
"""
import requests
import json
import os


def raw_sse_demo():
    """
    不使用 OpenAI SDK，直接用 requests 发送 HTTP 请求，
    手动解析 SSE 流。
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "说一句你好"}],
        "stream": True,  # <-- 关键: 开启流式
    }

    # stream=True 告诉 requests 不要立即下载整个响应体
    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
        stream=True,
    )

    full_content = ""
    buffer = ""

    for chunk_bytes in response.iter_content(chunk_size=None):
        # chunk_bytes 是 bytes，解码为文本
        chunk_text = chunk_bytes.decode("utf-8")
        buffer += chunk_text

        # 按 \n\n 分割消息
        while "\n\n" in buffer:
            message, buffer = buffer.split("\n\n", 1)

            # 跳过空行或注释行（以 : 开头）
            if not message.strip() or message.startswith(":"):
                continue

            # 按 \n 分割每一行
            for line in message.split("\n"):
                if line.startswith("data: "):
                    data_str = line[6:]  # 去掉 "data: " 前缀

                    if data_str == "[DONE]":
                        print("\n[流结束]")
                        break

                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0]["delta"]
                        if "content" in delta and delta["content"]:
                            full_content += delta["content"]
                            print(delta["content"], end="", flush=True)

                        # 检查 finish_reason
                        finish = data["choices"][0].get("finish_reason")
                        if finish:
                            print(f"\n[完成原因: {finish}]")

                    except json.JSONDecodeError:
                        pass  # 忽略无法解析的行

    print(f"\n\n完整内容 ({len(full_content)} 字): {full_content}")


if __name__ == "__main__":
    # 注意: 这个示例会实际发起 API 调用
    # raw_sse_demo()
    print("手动 SSE 解析示例（取消注释以运行）")
    print("\nSSE 格式说明:")
    print("  1. Content-Type: text/event-stream")
    print("  2. 每条数据以 'data: ' 开头")
    print("  3. 数据之间以 '\\n\\n' 分割")
    print("  4. 结束时发送 'data: [DONE]'")
```

代码分析：`buffer` 机制是手动解析 SSE 的关键。网络传输中，一个 TCP 包可能包含半条 SSE 消息，也可能包含一条半——你不能假设每次 `iter_content` 返回的字节恰好是一条完整的 `data: ...\n\n`。因此需要用 buffer 累积接收到的字节，按 `\n\n` 分割出完整消息，未完成的部分留在 buffer 中等待下一波数据。这种"缓冲-分割"模式是所有流协议解析的基础。

### 2.4 SSE vs WebSocket vs 普通 HTTP —— 三者的定位

很多初学者会困惑：为什么不用 WebSocket？为什么不用普通 HTTP 轮询？下面理清三者的差异和适用场景。

**普通 HTTP（请求-响应）**：客户端发请求，服务器返回完整响应后关闭连接。适合一次性的数据获取——如获取用户信息、提交表单。不适合实时推送，因为客户端不知道服务器何时有新数据。

**SSE（单向推送）**：客户端发起一个 HTTP 连接后，服务器持续推送数据，连接保持打开。适合服务器向客户端单向推送的场景——如股票行情、AI 流式输出、日志推送。优势是简单、HTTP 基础设施兼容、浏览器原生自动重连。

**WebSocket（双向全双工）**：客户端和服务器通过 `ws://` 协议建立持久连接，双方可以随时互发数据。适合实时双向通信场景——如在线聊天、协同编辑、多人游戏。复杂度和开销比 SSE 高。

为什么 OpenAI 选择 SSE 而非 WebSocket？核心原因：LLM 流式输出只需要"服务器推送给客户端"，不需要"客户端推送给服务器"。用 WebSocket 相当于杀鸡用牛刀——多了一次协议升级握手，增加了客户端和服务端的实现复杂度，却没有获得任何实际收益。

```python
"""
SSE vs WebSocket —— 为什么 OpenAI 选择 SSE？

SSE (Server-Sent Events):
  - 单向: 服务器 → 客户端
  - 基于 HTTP 协议（不需要特殊握手）
  - 自动重连（浏览器原生支持）
  - 简单，适合"推送更新"场景
  - Content-Type: text/event-stream

WebSocket:
  - 双向: 服务器 ↔ 客户端
  - 独立的 ws:// 协议（需要 Upgrade 握手）
  - 需要自己实现重连逻辑
  - 更复杂但更灵活
  - 适合实时双向通信（如聊天、游戏）
"""

SSE_VS_WEBSOCKET = """
┌──────────────────┬───────────────────┬───────────────────┐
│      特性        │        SSE        │     WebSocket     │
├──────────────────┼───────────────────┼───────────────────┤
│ 通信方向         │ 单向(服务→客户端) │ 双向              │
│ 底层协议         │ HTTP/1.1 或 HTTP/2│ 独立协议(ws://)   │
│ 浏览器支持       │ EventSource API   │ WebSocket API     │
│ 自动重连         │ 内置              │ 需手动实现        │
│ 防火墙友好       │ 是(同HTTP端口)    │ 可能被拦截        │
│ 文本/二进制      │ 仅文本            │ 文本+二进制       │
│ 适用场景         │ 流式输出、推送    │ 聊天、协同编辑    │
│ 复杂度           │ 低                │ 中               │
│ OpenAI 的选择    │ ✓                 │ ✗                │
└──────────────────┴───────────────────┴───────────────────┘
"""

print(SSE_VS_WEBSOCKET)
```

---

## 3. 客户端流式消费

### 3.1 基础流式消费 —— stream=True 的三个关键点

使用 OpenAI SDK 进行流式消费非常简单，只需要设置 `stream=True`。但有几个容易忽略的细节：

**第一个关键点：`flush=True`**。Python 的 `print()` 默认会将输出缓冲在内存中，等到积累一定量或遇到换行时才刷新到终端。在流式场景下，如果不加 `flush=True`，用户可能看到的是几个 token 一起蹦出来，失去了"逐字显示"的效果。

**第二个关键点：`delta.content` 可能为 None**。在流式模式下，有些 chunk 的 content 为 None——比如第一个 chunk 只包含 `delta.role = "assistant"`，或者包含 tool_calls 的 chunk（此时 content 为 None，tool_calls 有值）。不做 None 检查会导致打印出 "None" 字样。

**第三个关键点：`response.close()`**。当你提前终止流式（如用户点了停止按钮），应该显式调用 `response.close()` 关闭底层连接。否则连接会保持打开直到超时，浪费服务器资源。

```python
"""
OpenAI SDK 的 stream=True 使用详解。

关键参数:
- stream=True: 返回一个可迭代的流对象
- stream_options: 控制流中是否包含 usage 信息
"""
import os
import sys
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


def basic_stream(prompt: str):
    """
    基础流式消费: 逐 token 打印。
    flush=True 确保每个 token 立即显示（不缓冲）。
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        # stream_options={"include_usage": True},  # 在最后一个 chunk 中包含 usage
    )

    print("助手: ", end="", flush=True)

    for chunk in response:
        # chunk 的结构:
        # chunk.choices[0].delta.content       ← 本次增量的文本
        # chunk.choices[0].delta.role          ← 仅第一个 chunk 有 "assistant"
        # chunk.choices[0].finish_reason       ← 最后一个 chunk 有 "stop"/"length"/...
        # chunk.usage                           ← 仅最后一个 chunk（如果 include_usage=True）

        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)  # flush=True 强制刷新缓冲区

        # 检查是否结束
        if chunk.choices[0].finish_reason:
            print(f"\n[生成完成, 原因: {chunk.choices[0].finish_reason}]")

    print()  # 最后的换行


def stream_with_stop(prompt: str, stop_after: int = 50):
    """
    流式处理 + 提前终止。

    场景: 用户点了"停止生成"按钮，或检测到敏感内容。
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    char_count = 0
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            char_count += len(delta.content)

            # 达到字数上限，主动停止
            if char_count >= stop_after:
                print(f"\n[已停止，达到 {stop_after} 字限制]")
                # 注意: 直接 break 退出循环，连接会关闭
                # 但已生成的 token 仍然会被计费
                response.close()  # 显式关闭连接
                break


if __name__ == "__main__":
    print("=== 基础流式消费 ===\n")
    # basic_stream("用50字介绍Python")

    print("\n\n=== 提前终止 ===\n")
    # stream_with_stop("列出100个Python标准库名", 100)
    print("(取消注释以运行)")
```

### 3.2 chunk 的完整结构解析 —— 每个字段在哪出现、什么时候出现

理解流式 chunk 的结构是处理复杂流式场景（如流式+Function Calling）的前提。每个 chunk 是一个 `ChatCompletionChunk` 对象，它的内部字段随着流的推进而变化：

- **第1个 chunk**：通常 `delta.role = "assistant"`，`delta.content` 可能为 None 或第一个 token。这是模型在"打招呼"——告诉客户端"接下来是我的回复"。
- **中间 chunk**：`delta.content` 包含本次增量文本（可能 1-3 个 token）。`delta.role` 为 None（角色的赋值只在第一个 chunk）。
- **tool_calls chunk**：如果模型决定调用工具，`delta.tool_calls` 包含分块的 tool_call 信息（见 4.1 节）。此时 `delta.content` 为 None。
- **最后一个 chunk**：`finish_reason` 从 None 变为 "stop"（正常结束）、"length"（达到 max_tokens）、"tool_calls"（需要调用工具）或 "content_filter"（被内容过滤）。如果设置了 `stream_options={"include_usage": True}`，这个 chunk 还包含 token 使用量统计。

```python
"""
深入理解流式 chunk 的数据结构。

每个 chunk 是一个 ChatCompletionChunk 对象，关键在于:
1. choices[0].delta 的内容随着流推进而变化
2. 第一个 chunk: delta.role = "assistant"
3. 中间 chunk: delta.content = "文字增量"
4. 如果涉及 function calling: delta.tool_calls 包含分块的 tool_call 信息
5. 最后一个 chunk: finish_reason 不再是 None
"""
import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


def inspect_chunks(prompt: str, max_chunks: int = 8):
    """
    打印前 max_chunks 个 chunk 的详细结构。
    帮助理解每个 chunk 里到底有什么。
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    chunk_count = 0
    for chunk in response:
        if chunk_count >= max_chunks:
            print(f"\n... (省略剩余 chunks)")
            break

        choice = chunk.choices[0]

        info = {
            "chunk_index": chunk_count,
            "delta_role": choice.delta.role,
            "delta_content": choice.delta.content,
            "delta_tool_calls": choice.delta.tool_calls,
            "finish_reason": choice.finish_reason,
        }

        print(f"Chunk {chunk_count}: {json.dumps(info, ensure_ascii=False)}")
        chunk_count += 1

    print(f"\n共检查了 {chunk_count} 个 chunks")
    print("\n观察要点:")
    print("  - Chunk 0 通常 delta.role = 'assistant'")
    print("  - 后续 chunk 的 delta.content 逐 token 增长")
    print("  - 最后一个 chunk 的 finish_reason 从 None 变为 'stop'")
    print("  - delta.tool_calls 在本例中始终为 None（因为没有调用工具）")


def collect_full_response(prompt: str):
    """
    收集完整的流式响应：累积 content 和使用量信息。
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},  # 在最后的 chunk 包含 usage
    )

    full_content = ""
    usage = None

    for chunk in response:
        delta = chunk.choices[0].delta

        # 累积文本内容
        if delta.content:
            full_content += delta.content

        # 收集 usage（仅最后一个 chunk 有）
        if chunk.usage:
            usage = {
                "prompt_tokens": chunk.usage.prompt_tokens,
                "completion_tokens": chunk.usage.completion_tokens,
                "total_tokens": chunk.usage.total_tokens,
            }

    print(f"完整内容 ({len(full_content)} 字符):")
    print(full_content[:200] + "..." if len(full_content) > 200 else full_content)
    print(f"\nToken 使用量: {usage}")

    return full_content, usage


if __name__ == "__main__":
    print("=== Chunk 结构详解 ===\n")
    # inspect_chunks("你好", max_chunks=5)

    print("\n\n=== 收集完整响应 ===\n")
    # collect_full_response("用50字介绍Python")
    print("(取消注释以运行)")
```

### 3.3 流式消费的三种实用模式

根据使用场景不同，流式消费有三种常见模式。它们不是互斥的，实际项目中经常组合使用。

- **实时打印模式**：边接收边显示，同时累积完整文本。适合终端聊天、Web 聊天界面。
- **缓冲收集模式**：只收集不显示，拿到完整文本后再处理。适合需要整体理解后才能进行下一步操作的场景——如翻译后润色、生成后做敏感词检测。
- **实时回调模式**：每个 token 触发自定义逻辑。适合敏感词实时检测、每个 token 做一次翻译显示、动态更新 UI 进度条等场景。

```python
"""
流式消费的三种实用模式。

模式1: 实时打印（聊天界面）
模式2: 缓冲收集（需要完整文本再处理）
模式3: 实时回调（每个 token 触发自定义逻辑）
"""
import os
from openai import OpenAI
from typing import Callable
from collections.abc import Generator

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# ============================================================
# 模式1: 实时打印 —— 最常用的聊天界面模式
# ============================================================
def stream_print(prompt: str) -> str:
    """
    边接收边打印，同时累积完整文本。
    适合终端聊天、实时显示。
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    full_content = ""
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            full_content += delta.content
            print(delta.content, end="", flush=True)

    print()  # 最后换行
    return full_content


# ============================================================
# 模式2: 缓冲收集 —— 需要完整文本后才能处理
# ============================================================
def stream_collect(prompt: str) -> str:
    """
    不打印，只收集完整文本。
    适合: 翻译后需要整体润色、生成后需要二次处理。
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    parts = []
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            parts.append(delta.content)

    full_content = "".join(parts)
    return full_content


# ============================================================
# 模式3: 实时回调 —— 每个 token 触发自定义逻辑
# ============================================================
def stream_callback(
    prompt: str,
    on_token: Callable[[str], None] = None,
    on_complete: Callable[[str, dict], None] = None,
) -> str:
    """
    每个 token 触发回调函数。

    适用场景:
    - 敏感词检测: on_token 中检查每个词，触发过滤
    - 实时翻译: 每个 token 立即翻译并显示
    - 进度条: 统计 token 数更新进度
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
    )

    full_content = ""
    token_count = 0
    usage = None

    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            full_content += delta.content
            token_count += 1

            # 触发 token 回调
            if on_token:
                on_token(delta.content)

        if chunk.usage:
            usage = {
                "prompt": chunk.usage.prompt_tokens,
                "completion": chunk.usage.completion_tokens,
            }

    # 触发完成回调
    if on_complete:
        on_complete(full_content, usage)

    return full_content


if __name__ == "__main__":
    # 演示模式3: 敏感词检测
    SENSITIVE_WORDS = {"密码", "暴力", "违禁"}

    def detect_sensitive(token: str):
        """回调: 检测敏感词"""
        for word in SENSITIVE_WORDS:
            if word in token:
                print(f"\n[警告] 检测到敏感词: {word}")

    def on_done(content: str, usage: dict):
        """回调: 生成完成"""
        print(f"\n[完成] 共生成 {len(content)} 字符, usage={usage}")

    print("=== 模式3: 实时回调 ===")
    # result = stream_callback(
    #     "用50字介绍Python的安全性",
    #     on_token=detect_sensitive,
    #     on_complete=on_done,
    # )
    print("(取消注释以运行)")
```

---

## 4. 流式 + Function Calling

### 4.1 流式模式下 tool_calls 的分块传输 —— 最复杂的场景

流式模式下的 Function Calling 是相对复杂的场景，因为 tool_calls 不是一次性到达的，而是像文本一样分块传输。理解这一点对于正确处理流式+FC 的组合至关重要。

在非流式模式下，当模型决定调用工具时，你得到的是一个完整的 tool_call 对象，其中 `function.name` 和 `function.arguments` 都是完整的字符串。但在流式模式下：

- **第一个相关 chunk**：`delta.tool_calls[0]` 出现，包含 `index=0` 和 `id="call_xxx"`，此时 `function.name` 可能还是空的。
- **后续 chunk**：`function.name` 逐步出现（可能分 2-3 个 chunk 传完"get_weather"这个名字），然后 `function.arguments` 分段传输（`{"city"`, `":`, `"北京"`, `"}`）。
- **最后一个相关 chunk**：`finish_reason` 变为 `"tool_calls"`，标志着 tool_call 传输完成。

这意味着你需要维护一个**累积结构**——在循环中不断将 chunk 中的 tool_call 片段拼接到对应的 index 上，直到 finish_reason 标志着传输完成。注意可能有多个 tool_call（index 0, 1, 2...）同时在传输。

```python
"""
流式 + Function Calling: 理解和处理分块传输的 tool_calls。

在流式模式下，tool_call 不是一次性给出的，而是分多个 chunk 传输:
- 第一个相关 chunk: delta.tool_calls[0].function.name = "get_weather"
- 后续 chunk: delta.tool_calls[0].function.arguments = '{"city"' (逐段传输)
- 你需要累积这些片段，拼成完整的 tool_call
"""
import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"],
            },
        },
    }
]


def stream_with_tool_calls(prompt: str):
    """
    流式模式下的 Function Calling 处理。

    关键点:
    1. tool_calls 是分块传输的 —— 需要累积
    2. 同一个 tool_call 的 index 保持不变
    3. function.name 和 function.arguments 分段到达
    4. 所有 tool_calls 到齐后，finish_reason 变为 "tool_calls"
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你有权调用 get_weather 工具。"},
            {"role": "user", "content": prompt},
        ],
        tools=TOOLS,
        stream=True,
    )

    # 用于累积 tool_calls 的临时结构
    accumulated_tool_calls: dict[int, dict] = {}  # {index: {"id": ..., "name": ..., "arguments": ""}}

    for chunk in response:
        delta = chunk.choices[0].delta

        # --- 处理文本内容（如果有） ---
        if delta.content:
            print(delta.content, end="", flush=True)

        # --- 处理 tool_calls 分块 ---
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index  # tool_call 的索引（支持多个并行调用）

                # 首次出现这个 index，初始化累积结构
                if idx not in accumulated_tool_calls:
                    accumulated_tool_calls[idx] = {
                        "id": tc_delta.id or "",
                        "function": {"name": "", "arguments": ""},
                    }

                # 累积 id
                if tc_delta.id:
                    accumulated_tool_calls[idx]["id"] = tc_delta.id

                # 累积 function name
                if tc_delta.function and tc_delta.function.name:
                    accumulated_tool_calls[idx]["function"]["name"] += tc_delta.function.name

                # 累积 function arguments (分段传输)
                if tc_delta.function and tc_delta.function.arguments:
                    accumulated_tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments

        # --- 检查完成状态 ---
        if chunk.choices[0].finish_reason == "tool_calls":
            print(f"\n[流式 tool_calls 接收完成]")
            for idx, tc in accumulated_tool_calls.items():
                args = json.loads(tc["function"]["arguments"])
                print(f"  Tool {idx}: {tc['function']['name']}({args})")

    return accumulated_tool_calls


if __name__ == "__main__":
    print("=== 流式 Function Calling ===\n")
    # result = stream_with_tool_calls("北京今天天气怎么样？")
    print("\n(取消注释以运行)")
```

### 4.2 流式+FC 的完整交互流程 —— 用文字描述两阶段

流式+Function Calling 的组合场景可以分为两个阶段，整个过程可以用文字清楚描述：

**第一阶段（流式接收 tool_calls）**：
1. 你发起流式 API 请求（`stream=True`），messages 中包含 system + user，附带 tools 定义。
2. 模型开始流式输出。前半部分可能先流式输出一些文本（如"让我查一下..."），然后 delta.content 变为 None，delta.tool_calls 开始出现。
3. tool_calls 分多个 chunk 传输——每个 chunk 携带一个片段（index, id, function.name 的一小段, function.arguments 的一小段）。你需要按 index 将片段累积到对应的结构中。
4. 当 finish_reason 变为 "tool_calls"，表示所有 tool_calls 传输完毕。此时你的累积结构中有完整的 id、name 和 arguments（arguments 现在是合法的 JSON 字符串）。

**第二阶段（执行+非流式回复，或执行+流式回复）**：
5. 你用累积的参数执行实际的工具函数，获取结果。
6. 将 assistant 消息（含 tool_calls）和 tool 消息（含结果）追加到 messages。
7. 再次发起 API 请求（这次可以是非流式或流式，取决于用户体验需求）。如果继续流式，模型会逐 token 生成基于工具结果的最终回复。

**关键难点**：第一阶段中 arguments 是分段传输的 JSON 字符串。在 finish_reason="tool_calls" 之前，arguments 是不完整的——你不能在中间尝试 `json.loads()`。必须等到 finish_reason 信号，然后再统一 parse。

```python
"""
流式模式下完整的 Function Calling 循环。

流程:
1. 流式接收 → 累积 tool_calls
2. 解析完整的 tool_calls
3. 执行工具函数
4. 将结果追加到 messages
5. 再次调用 API（可以是流式或非流式）
6. 流式输出最终回复
"""
import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


def get_weather(city: str) -> dict:
    weather = {"北京": "晴 25°C", "上海": "小雨 20°C"}
    return {"city": city, "weather": weather.get(city, "未知")}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

AVAILABLE_FUNCTIONS = {"get_weather": get_weather}


def stream_with_fc_loop(user_query: str):
    """
    流式 + 完整的 Function Calling 循环。

    步骤:
    1. 流式调用（可能出现 tool_calls）
    2. 累积 tool_calls
    3. 执行工具
    4. 非流式获取最终回复（简化处理）
    5. 流式打印最终回复
    """
    # --- 第一阶段: 流式调用，检查是否需要工具 ---
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是天气助手。"},
            {"role": "user", "content": user_query},
        ],
        tools=TOOLS,
        stream=True,
    )

    messages = [
        {"role": "system", "content": "你是天气助手。"},
        {"role": "user", "content": user_query},
    ]

    accumulated_tool_calls: dict[int, dict] = {}
    assistant_content = ""

    for chunk in response:
        delta = chunk.choices[0].delta

        if delta.content:
            assistant_content += delta.content
            print(delta.content, end="", flush=True)

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in accumulated_tool_calls:
                    accumulated_tool_calls[idx] = {
                        "id": tc_delta.id or "",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc_delta.id:
                    accumulated_tool_calls[idx]["id"] = tc_delta.id
                if tc_delta.function and tc_delta.function.name:
                    accumulated_tool_calls[idx]["function"]["name"] += tc_delta.function.name
                if tc_delta.function and tc_delta.function.arguments:
                    accumulated_tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments

    # --- 如果没有 tool_calls，直接返回 ---
    if not accumulated_tool_calls:
        print()
        return assistant_content

    print(f"\n[调用工具: {len(accumulated_tool_calls)} 个]")

    # --- 构建 assistant 消息（含 tool_calls） ---
    assistant_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [],
    }
    for idx, tc in accumulated_tool_calls.items():
        assistant_msg["tool_calls"].append(
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                },
            }
        )
    messages.append(assistant_msg)

    # --- 执行工具 ---
    for idx, tc in accumulated_tool_calls.items():
        func_name = tc["function"]["name"]
        func_args = json.loads(tc["function"]["arguments"])
        result = AVAILABLE_FUNCTIONS[func_name](**func_args)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": func_name,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )
        print(f"  {func_name}({func_args}) → {result}")

    # --- 第二阶段: 流式获取最终回复 ---
    print("\n助手: ", end="", flush=True)
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True,
    )

    final_content = ""
    for chunk in final_response:
        delta = chunk.choices[0].delta
        if delta.content:
            final_content += delta.content
            print(delta.content, end="", flush=True)

    print()
    return final_content


if __name__ == "__main__":
    print("=== 流式 Function Calling 完整循环 ===\n")
    # stream_with_fc_loop("北京天气怎么样？")
    print("(取消注释以运行)")
```

---

## 5. 构建流式 API 服务

### 5.1 FastAPI + StreamingResponse —— 从后端到前端的完整链路

构建流式 API 的完整架构是：**浏览器 EventSource <-SSE- FastAPI StreamingResponse <- stream - OpenAI API**。三个环节环环相扣：

1. **OpenAI API**：设置 `stream=True`，返回 async generator
2. **FastAPI**：用 `StreamingResponse` 包装 async generator，设置 `media_type="text/event-stream"`
3. **浏览器**：用 `EventSource` 或 `fetch + ReadableStream` 消费 SSE 事件

中间的 FastAPI 层充当**协议转换器**——把 OpenAI 的流式 chunk 转换成标准的 SSE 事件格式 `data: {...}\n\n`。每个 chunk 对应一个 SSE 事件。

```python
"""
使用 FastAPI 构建流式 API 服务。

架构:
  浏览器 (EventSource) ←SSE← FastAPI (StreamingResponse) ←← OpenAI API (stream)

完整流程:
1. 浏览器建立 SSE 连接
2. FastAPI 接收请求，调用 OpenAI API（stream=True）
3. 每个 token 通过 SSE 格式推送给浏览器
4. 生成完成后发送 [DONE] 信号
"""
# 文件名: streaming_api_server.py
# 运行: uvicorn streaming_api_server:app --reload

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from openai import OpenAI
import os
import json
import asyncio
from typing import AsyncGenerator

# OpenAI 客户端（同步版本也可在 async 中使用，但建议用 AsyncOpenAI）
from openai import AsyncOpenAI

app = FastAPI(title="流式聊天 API")

# 使用 AsyncOpenAI 支持异步调用
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


async def sse_event_generator(messages: list[dict]) -> AsyncGenerator[str, None]:
    """
    SSE 事件生成器 —— 从 OpenAI 流式获取 token 并转换为 SSE 格式。

    这是一个 async generator，yield 的每个字符串是一条 SSE 消息。

    SSE 格式:
        data: {"token": "你"}\n\n
        data: {"token": "好"}\n\n
        data: [DONE]\n\n
    """
    try:
        # 调用 OpenAI 流式 API
        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                # 每个 token 封装为一个 SSE 事件
                event_data = json.dumps(
                    {"token": delta.content, "finish_reason": None},
                    ensure_ascii=False,
                )
                yield f"data: {event_data}\n\n"

            # 检查是否结束
            if chunk.choices[0].finish_reason:
                event_data = json.dumps(
                    {
                        "token": "",
                        "finish_reason": chunk.choices[0].finish_reason,
                    },
                    ensure_ascii=False,
                )
                yield f"data: {event_data}\n\n"
                yield "data: [DONE]\n\n"

    except Exception as e:
        error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
        yield f"data: {error_data}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/chat/stream")
async def chat_stream(request: Request):
    """
    流式聊天端点。

    请求体:
    {
        "messages": [{"role": "user", "content": "你好"}]
    }

    响应: SSE 格式的流
    """
    body = await request.json()
    messages = body.get("messages", [])

    # 可选: 添加 system prompt
    if not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": "你是一个有帮助的AI助手。"})

    return StreamingResponse(
        sse_event_generator(messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )


@app.get("/")
async def root():
    """返回一个简单的 HTML 测试页面"""
    return {
        "message": "流式聊天 API 服务",
        "endpoints": {
            "POST /chat/stream": "流式聊天",
            "GET /": "API 信息",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

关键配置说明：(1) `media_type="text/event-stream"` 告诉浏览器这是 SSE 流，触发 EventSource 的 onmessage 事件；(2) `Cache-Control: no-cache` 防止中间代理缓冲 SSE 数据；(3) `X-Accel-Buffering: no` 针对 nginx 的特殊头，禁用 nginx 的响应缓冲（否则 nginx 会等整个响应完成后才转发给客户端，流式效果全毁）；(4) `Connection: keep-alive` 保持连接不关闭。

### 5.2 前端 EventSource 消费

```python
"""
前端 JavaScript 代码 —— 使用 EventSource 消费 SSE 流。

由于本章是 Python 教程，这里给出 HTML + JavaScript 代码作为参考。
你可以将此 HTML 保存为文件，用浏览器打开后连接上面的 FastAPI 服务。
"""

HTML_CLIENT = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>AI 流式聊天</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        #chat-box { border: 1px solid #ccc; padding: 20px; min-height: 300px; margin-bottom: 20px; border-radius: 8px; overflow-y: auto; }
        .user-msg { color: #2563eb; margin: 8px 0; }
        .ai-msg { color: #059669; margin: 8px 0; }
        #input-area { display: flex; gap: 10px; }
        #input-area input { flex: 1; padding: 10px; font-size: 16px; border: 1px solid #ccc; border-radius: 6px; }
        #input-area button { padding: 10px 20px; font-size: 16px; background: #2563eb; color: white; border: none; border-radius: 6px; cursor: pointer; }
        #input-area button:disabled { background: #93c5fd; cursor: not-allowed; }
        .cursor { animation: blink 1s step-end infinite; }
        @keyframes blink { 50% { opacity: 0; } }
    </style>
</head>
<body>
    <h1>AI 流式聊天</h1>
    <div id="chat-box"></div>
    <div id="input-area">
        <input type="text" id="user-input" placeholder="输入你的问题..." onkeydown="if(event.key==='Enter')sendMessage()">
        <button onclick="sendMessage()">发送</button>
    </div>

    <script>
        const chatBox = document.getElementById('chat-box');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.querySelector('#input-area button');

        function appendMessage(role, text) {
            const div = document.createElement('div');
            div.className = role === 'user' ? 'user-msg' : 'ai-msg';
            div.innerHTML = `<strong>${role === 'user' ? '你' : 'AI'}:</strong> <span class="content">${text}</span>`;
            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
            return div;
        }

        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;

            // 显示用户消息
            appendMessage('user', message);
            userInput.value = '';
            sendBtn.disabled = true;

            // 创建 AI 消息容器
            const aiDiv = appendMessage('ai', '<span class="cursor">|</span>');
            const contentSpan = aiDiv.querySelector('.content');

            try {
                // 发送 POST 请求到流式 API
                const response = await fetch('http://localhost:8000/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        messages: [{ role: 'user', content: message }]
                    })
                });

                // 使用 ReadableStream 读取 SSE 流
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let fullText = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });

                    // 按 \\n\\n 分割 SSE 消息
                    const lines = buffer.split('\\n\\n');
                    buffer = lines.pop() || '';  // 最后不完整的一行保留在 buffer

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const dataStr = line.substring(6);
                            if (dataStr === '[DONE]') {
                                // 移除闪烁光标
                                const cursor = contentSpan.querySelector('.cursor');
                                if (cursor) cursor.remove();
                                break;
                            }

                            try {
                                const data = JSON.parse(dataStr);
                                if (data.token) {
                                    fullText += data.token;
                                    contentSpan.textContent = fullText;
                                    chatBox.scrollTop = chatBox.scrollHeight;
                                }
                            } catch (e) {
                                // 忽略解析错误
                            }
                        }
                    }
                }
            } catch (error) {
                contentSpan.textContent = `错误: ${error.message}`;
            } finally {
                sendBtn.disabled = false;
                userInput.focus();
            }
        }
    </script>
</body>
</html>
"""

print(HTML_CLIENT)
```

---

## 6. 高级流式处理

### 6.1 多流并行处理

在实际应用中，经常需要同时发起多个流式请求——如 A/B 测试同时对比两个 prompt、多语言翻译、多模型对比。使用 `asyncio.gather` 可以轻松实现并发流式处理。

```python
"""
多流并行处理 —— 同时调用多个模型或发送多个请求。

适用场景:
- A/B 测试: 同时对两个 prompt 版本生成结果
- 多模型对比: GPT-4o vs GPT-4
- 翻译多语言: 同时翻译成英文、日文、法文
"""
import os
import asyncio
import time
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


async def stream_one(prompt: str, label: str) -> str:
    """
    流式获取一个响应，并标注来源标签。

    参数:
        prompt: 用户提示
        label: 标签（用于区分不同流）
    返回:
        完整的响应文本
    """
    print(f"\n[{label}] 开始生成...")

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    full_content = ""
    async for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            full_content += delta.content
            print(f"[{label}] {delta.content}", end="", flush=True)

    print(f"\n[{label}] 完成 ({len(full_content)} 字)")
    return full_content


async def parallel_streams(prompt: str, count: int = 3):
    """
    并行运行多个流式请求。

    所有流同时运行，各自独立打印。
    使用 asyncio.gather 并发执行。
    """
    t0 = time.time()

    tasks = [
        stream_one(prompt, f"Stream-{i + 1}")
        for i in range(count)
    ]

    results = await asyncio.gather(*tasks)

    elapsed = time.time() - t0
    print(f"\n全部完成，总耗时 {elapsed:.1f}s")
    print(f"每个结果长度: {[len(r) for r in results]}")

    return results


if __name__ == "__main__":
    print("=== 多流并行处理 ===\n")
    # asyncio.run(parallel_streams("用30字描述AI的未来", 3))
    print("(取消注释以运行)")
```

### 6.2 中断流式输出

```python
"""
中断流式输出 —— 模拟用户点击"停止"按钮。

场景:
- 用户对当前生成不满意，点击停止
- 检测到敏感内容，立即中断
- 达到了最大 token 限制
"""
import os
import asyncio
import signal
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


class StreamController:
    """
    流式控制器 —— 支持外部中断。

    使用方式:
        controller = StreamController()
        # 在其他线程/协程中调用 controller.stop()
        await stream_with_control(prompt, controller)
    """

    def __init__(self):
        self._stopped = False
        self._stop_reason = ""

    def stop(self, reason: str = "用户中断"):
        """请求停止流式输出"""
        self._stopped = True
        self._stop_reason = reason

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    @property
    def stop_reason(self) -> str:
        return self._stop_reason


async def stream_with_control(prompt: str, controller: StreamController) -> str:
    """
    可中断的流式响应。

    每隔 5 个 token 检查是否应停止。
    """
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    full_content = ""
    token_count = 0

    async for chunk in response:
        if controller.is_stopped:
            print(f"\n[已停止: {controller.stop_reason}]")
            # 关闭底层连接
            await response.aclose()
            break

        delta = chunk.choices[0].delta
        if delta.content:
            full_content += delta.content
            print(delta.content, end="", flush=True)
            token_count += 1

            # 模拟: 检测到特定内容就停止
            if "敏感词" in delta.content:
                controller.stop("检测到敏感内容")

    print(f"\n共生成 {token_count} 个 token")
    return full_content


async def simulate_stop():
    """
    模拟用户按停止按钮: 1.5秒后自动停止。
    """
    controller = StreamController()
    prompt = "列出50种动物名称"

    # 1.5 秒后自动停止
    async def auto_stop():
        await asyncio.sleep(1.5)
        controller.stop("模拟用户点击停止")

    # 并行运行：生成 + 定时停止
    task_gen = asyncio.create_task(stream_with_control(prompt, controller))
    task_stop = asyncio.create_task(auto_stop())

    await asyncio.gather(task_gen, task_stop)


if __name__ == "__main__":
    print("=== 可中断流式输出 ===\n")
    # asyncio.run(simulate_stop())
    print("(取消注释以运行)")
```

### 6.3 重连与断点续传

```python
"""
流式输出的断点续传机制。

场景: 网络中断导致流断开，如何从断点继续。

方案:
1. 重试机制: 捕获网络异常，重新发起请求
2. 上下文恢复: 将已收到的内容作为上下文传给模型，让它从断点继续
"""
import os
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


async def stream_with_retry(
    prompt: str,
    max_retries: int = 3,
    received_so_far: str = "",
) -> str:
    """
    带重试的流式调用。

    策略:
    1. 正常流式接收
    2. 如果中途断开（网络异常），记录已收到的内容
    3. 重新发起请求，在 prompt 中告诉模型"你之前已经写了xxx，请继续"
    4. 拼接两次的内容

    参数:
        prompt: 初始提示
        max_retries: 最大重试次数
        received_so_far: 之前已收到的内容（用于续传）
    """
    full_content = received_so_far
    retry_count = 0

    while retry_count <= max_retries:
        try:
            # 如果是续传，修改 prompt
            if received_so_far:
                continuation_prompt = (
                    f"{prompt}\n\n[注意] 你之前已经回复了以下内容，请从断开处继续，不要重复已回复的内容:\n"
                    f"<已回复>{received_so_far}</已回复>\n\n"
                    f"请继续完成剩余内容。"
                )
            else:
                continuation_prompt = prompt

            print(f"[尝试 {retry_count + 1}] 开始流式生成...")

            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": continuation_prompt}],
                stream=True,
            )

            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_content += delta.content
                    print(delta.content, end="", flush=True)

            # 正常完成
            print(f"\n[完成] 总长度: {len(full_content)} 字")
            return full_content

        except (ConnectionError, asyncio.TimeoutError, Exception) as e:
            retry_count += 1
            print(f"\n[错误] 流中断: {e}")
            print(f"[重试 {retry_count}/{max_retries}] 已接收 {len(full_content)} 字")

            if retry_count > max_retries:
                print("[失败] 达到最大重试次数")
                return full_content

            # 将已收到的内容作为上下文继续
            received_so_far = full_content
            await asyncio.sleep(1)  # 等待1秒后重试

    return full_content


if __name__ == "__main__":
    print("=== 流式断点续传 ===\n")
    # asyncio.run(stream_with_retry("请从1数到100，每个数字一行"))
    print("(取消注释以运行)")
```

---

## 基础练习

### 练习 1: 终端流式对话
**场景**: 实现一个简单的终端聊天程序，支持流式显示和多轮对话。
**要求**: 逐 token 打印，记录对话历史，支持 `/exit` 退出。
**文件**: `exercise/ai-application/ch03_streaming/ex1_terminal_chat.py`

### 练习 2: 流式 + 工具调用
**场景**: 实现流式天气查询助手，在流式模式下正确处理 tool_calls。
**要求**: 累积分块的 tool_calls，执行后流式返回结果。
**文件**: `exercise/ai-application/ch03_streaming/ex2_stream_fc.py`

### 练习 3: FastAPI 流式 API
**场景**: 构建一个完整的 FastAPI 流式聊天后端。
**要求**: 支持 SSE 输出，前端可用 EventSource 或 fetch 消费。
**文件**: `exercise/ai-application/ch03_streaming/ex3_fastapi_stream.py`

## 进阶练习

### 练习 4: 并行翻译对比
**场景**: 同时将一段文本翻译成英文、日文、韩文，三个流并行显示。
**要求**: 使用 asyncio.gather，每个翻译独立打印并标注语言标签。
**文件**: `exercise/ai-application/ch03_streaming/ex4_parallel_translate.py`

### 练习 5: 流式对话客户端
**场景**: 用 Python 的 Textual 或 curses 库构建一个终端流式对话客户端。
**要求**: 支持 Markdown 渲染、多轮对话历史滚动、Ctrl+C 中断。
**文件**: `exercise/ai-application/ch03_streaming/ex5_tui_chat.py`

---

## 常见错误

### 错误 1: 忘记 flush=True 导致输出延迟

```python
# 错误: 终端缓冲导致 token 不即时显示
for chunk in response:
    print(delta.content, end="")  # 没有 flush=True，输出可能被缓冲

# 修正:
for chunk in response:
    print(delta.content, end="", flush=True)  # 强制刷新缓冲区
    # 或
    sys.stdout.write(delta.content)
    sys.stdout.flush()
```

Python 的 `print()` 默认使用行缓冲（line buffering）——遇到 `\n` 才刷新。流式输出中很多 token 不包含换行符，导致多个 token 堆在一起后才突然显示。`flush=True` 强制每次 print 立即刷新缓冲区。在 Web 服务中，如果你用 WSGI 而非 ASGI，也可能遇到类似问题——中间件可能在缓冲你的输出。

### 错误 2: 流式模式 + response_format 不兼容

```python
# 错误: 流式模式不支持 response_format={"type": "json_object"}
# 因为模型需要看到完整的 JSON 才能保证格式正确
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    stream=True,
    response_format={"type": "json_object"},  # 可能报错或行为异常
)

# 修正: 在非流式模式下使用 JSON Mode
# 或者在流式模式下用 prompt 约束输出 JSON（但格式不保证）
```

深层原因：JSON Mode 要求输出是完整合法的 JSON，但在流式模式下，模型逐 token 生成时无法预知后续内容，无法保证括号闭合等 JSON 结构约束。因此 JSON Mode 和流式模式是互斥的——这是 API 层面的限制，不是 prompt 层面的问题。

### 错误 3: 在流式 loop 中修改 messages 导致状态混乱

```python
# 错误: 在流式循环中修改 messages
for chunk in response:
    messages.append(...)  # 错误! 这会导致混乱

# 修正: 先收集完流式内容，再修改 messages
full_content = ""
for chunk in response:
    if delta.content:
        full_content += delta.content
# 收集完成后再修改
messages.append({"role": "assistant", "content": full_content})
```

### 错误 4: 忘记处理 delta.content 为 None 的情况

```python
# 错误: 假设每个 chunk 都有 content
for chunk in response:
    print(chunk.choices[0].delta.content)  # tool_calls chunk 中 content 为 None

# 修正: 始终检查 None
for chunk in response:
    delta = chunk.choices[0].delta
    if delta.content is not None:
        print(delta.content, end="", flush=True)
```

### 错误 5: 异步生成器中忘记 catch 异常

```python
# 错误: SSE generator 中不处理异常，导致连接中断时客户端收到 500
async def generator():
    async for chunk in stream:  # 如果这里抛异常...
        yield f"data: {chunk}\n\n"  # 客户端连接断开

# 修正: 用 try/except 捕获
async def generator():
    try:
        async for chunk in stream:
            yield f"data: {json.dumps(chunk)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        yield "data: [DONE]\n\n"
```

### 错误 6: 混用同步和异步客户端

```python
# 错误: 在 async def 中使用同步 OpenAI 客户端
async def handler():
    client = OpenAI(...)  # 同步客户端
    response = client.chat.completions.create(...)  # 阻塞事件循环!
    # 在 async 上下文中，这会阻塞整个事件循环

# 修正: 使用 AsyncOpenAI
async def handler():
    client = AsyncOpenAI(...)  # 异步客户端
    response = await client.chat.completions.create(...)  # 非阻塞
```

---

## 本章小结

本章深入学习了流式处理与 SSE 的完整体系。回顾关键理解：

- **流式的本质价值**：不仅仅是"逐字显示"的视觉效果，更是用户体验模型的根本改变——首 token 延迟（TTFT）成为核心指标，系统从"全有或全无"变为"增量交付"。非流式让用户等待，流式让用户看到进度。

- **SSE 协议的定位**：SSE 是单向（服务器->客户端）、基于 HTTP、浏览器原生支持的推送协议。OpenAI 选择 SSE 而非 WebSocket，是因为 LLM 流式输出恰好是单向推送场景——方向匹配、复杂度更低、基础设施兼容性更好。普通 HTTP（请求-响应）、SSE（单向推送）、WebSocket（双向全双工）各有其适用场景，不存在绝对的好坏。

- **chunk 结构与生命周期**：每个 chunk 在不同阶段携带不同数据——第一个 chunk 带 role，中间 chunk 带 content 或 tool_calls，最后一个 chunk 带 finish_reason 和可选的 usage。理解这个生命周期是正确处理流式响应（特别是流式+FC 组合）的前提。

- **流式+Function Calling**：最复杂的流式场景。tool_calls 分块传输，需要按 index 累积 id/name/arguments，等到 finish_reason="tool_calls" 后才能完整 parse arguments。整个过程分为两阶段：流式接收 tool_calls + 执行工具 + (流式或非流式)获取最终回复。

- **FastAPI 流式服务架构**：浏览器 <-SSE- StreamingResponse <-async generator- AsyncOpenAI。中间层做协议转换，三个关键响应头（Content-Type: text/event-stream, Cache-Control: no-cache, X-Accel-Buffering: no）缺一不可。

- **高级流式处理**：多流并行（asyncio.gather）、可中断流（StreamController + aclose()）、断点续传（捕获异常 -> 记录已接收内容 -> 从断点重试）覆盖了生产环境的主要需求。

下一章将学习 RAG（检索增强生成）架构，让模型能够基于外部知识库回答专业问题。
