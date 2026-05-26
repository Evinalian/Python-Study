# 21 HTTP 请求与 API 调用

## 学习目标

- 理解 HTTP 协议的基本概念：请求、响应、方法、状态码
- 熟练使用 requests 库发送 GET/POST 请求，解析文本和 JSON 响应
- 能处理超时、异常和 HTTP 错误
- 理解 API 认证的基本方式，能用 GitHub API 实战
- 能处理 SSE（Server-Sent Events）流式响应
- 了解 httpx 作为 requests 替代品的优势

## 前置知识

- 熟练使用 Python 字典、列表和文件操作（第 02-08 章）
- 了解命令行基本操作（第 20 章）
- 了解 .env 文件和环境变量管理（第 20 章）

---

## 21.1 HTTP 是什么——用你的直觉理解

你每天用浏览器上网，背后发生的事情其实很简单：

```text
你在地址栏输入 https://api.github.com
        │
        ▼
浏览器（客户端）向 GitHub 服务器发起一个"请求"
        │
        ▼
GitHub 服务器收到请求，处理，返回一个"响应"
        │
        ▼
浏览器把响应的内容显示给你看
```

这个"请求-响应"的对话规则，就是 **HTTP**（HyperText Transfer Protocol）。

把 HTTP 想象成你去图书馆借书：

- **GET**："老师，我想看《Python编程》"——你只是想**获取**信息，不改变任何东西
- **POST**："老师，我交作业"——你在**提交**新的东西
- **PUT**："老师，我重写了一遍论文"——你把旧的东西**完整替换**成新的
- **DELETE**："老师，我想撤回我的作文"——你要**删除**某样东西

每次借书，图书管理员都会给你一个"答复号"：

- **200**：好的，这是你要的书。（成功）
- **404**：没找到这本书。（你请求的东西不存在）
- **500**：对不起，书架塌了。（服务器内部出错）

这就是 HTTP 的核心：**方法** + **路径** + **状态码**。不需要背诵，用多了自然记住。

> **与 C 语言对比**：C 里你要跟远程服务器通信，得手动创建 socket、构造 HTTP 报文、解析响应。Python 的 requests 库把这些全部封装好了——就像 `printf` 封装了底层的 buffer 操作一样。

---

## 21.2 你的第一个 HTTP 请求

### 21.2.1 发一个 GET 请求

```bash
pip install requests
```

```python
import requests

# 向 GitHub API 发一个最简单的 GET 请求
resp = requests.get("https://api.github.com")

# 服务器说什么？（状态码）
print(f"状态码: {resp.status_code}")        # 200 表示一切正常
# 状态码含义速记：
#   2xx = 成功，4xx = 你的请求有问题，5xx = 服务器有问题

# 响应的内容长什么样？（原始文本）
print(f"前 200 个字符:\n{resp.text[:200]}")
```

运行后你会看到类似这样的输出：

```text
状态码: 200
前 200 个字符:
{"current_user_url":"https://api.github.com/user",...}
```

`resp.text` 返回的是一个**字符串**——服务器返回的原始内容。你在浏览器里"查看网页源代码"看到的东西，和 `resp.text` 拿到的是同一回事。

### 21.2.2 从文本到结构化数据：.json()

大部分 API 返回的不是给人看的网页，而是给程序读的 **JSON** 格式数据。

```python
import requests

resp = requests.get("https://api.github.com")

# .text：返回原始字符串
print(type(resp.text))   # <class 'str'>
print(resp.text[:50])    # '{"current_user_url":"https://api.github.com/user...'

# .json()：自动把 JSON 字符串解析成 Python 字典
data = resp.json()
print(type(data))        # <class 'dict'>
print(data.keys())       # dict_keys(['current_user_url', 'authorizations_url', ...])
print(data["current_user_url"])  # 'https://api.github.com/user'
```

**关键区别**：
- `resp.text` 是 `str`——你可以打印、搜索、切片，但要用 `data["key"]` 取值还得先 `json.loads()`
- `resp.json()` 帮你做了解析，返回 `dict` 或 `list`——可以直接用 Python 字典语法访问字段

这就像你收到一封 JSON 格式的信：`.text` 是信的原始纸张，`.json()` 是帮你翻译好的内容。

### 21.2.3 带查询参数的 GET 请求

搜索 GitHub 上 star 最多的 Python 项目：

```python
import requests

resp = requests.get(
    "https://api.github.com/search/repositories",
    # params 字典会被自动拼接到 URL 后面
    params={
        "q": "language:python",          # 搜索条件：Python 语言
        "sort": "stars",                  # 按星数排序
        "order": "desc",                  # 降序
    },
    timeout=10,
)

# 看一下最终请求的完整 URL
print(f"请求的 URL: {resp.url}")
# 输出类似：https://api.github.com/search/repositories?q=language%3Apython&sort=stars&order=desc

data = resp.json()
print(f"共找到 {data['total_count']} 个仓库")
for repo in data["items"][:3]:           # 只看前 3 个
    print(f"  {repo['full_name']}: {repo['stargazers_count']} stars")
```

`params` 字典会自动处理 URL 编码——空格变 `%20`、冒号变 `%3A`——你不用手动拼 URL。

---

## 21.3 POST 请求——提交数据

GET 是"要数据"，POST 是"交数据"。比如提交一个表单、发送一条消息。

```python
import requests

resp = requests.post(
    "https://httpbin.org/post",          # httpbin.org 是一个测试用的回声服务
    json={"name": "张三", "age": 20},    # json= 参数自动序列化为 JSON 并设置 Content-Type
    timeout=10,
)

print(f"状态码: {resp.status_code}")     # 200
echo = resp.json()
print(f"你发过来的 JSON: {echo['json']}")
# 输出：你发过来的 JSON: {'name': '张三', 'age': 20}
```

> **注意**：`requests.post(url, json=...)` 会自动做三件事：
> 1. 把字典变成 JSON 字符串
> 2. 设置 `Content-Type: application/json`
> 3. 放进请求体
>
> 如果你自己用 `data=json.dumps(payload)`，必须手动设置 Content-Type。

---

## 21.4 超时——永远要设置，永远

这是 Python 面试和实际开发中都反复出现的问题。

### 21.4.1 不设 timeout 的危险

```python
# ❌ 不设 timeout —— 如果服务器不响应，程序会永远等下去
resp = requests.get("https://一个永远不响应的地址.com")
print("这行代码可能永远不会执行到……")
```

设想你的程序是一个 Web 服务，正在处理用户的请求。如果不设 timeout，一个卡住的 API 调用就能让你的整个服务瘫痪——用户在浏览器前一直转圈，直到放弃。

### 21.4.2 正确做法

```python
import requests

try:
    resp = requests.get(
        "https://api.github.com",
        timeout=10,          # 总共等待时间不超过 10 秒
    )
    print(resp.status_code)
except requests.exceptions.Timeout:
    print("请求超时！服务器可能在睡觉，或者网络有问题。")
```

也可以分别设连接超时和读取超时：

```python
# (连接超时, 读取超时)
resp = requests.get(
    "https://api.github.com",
    timeout=(3.05, 30),    # 3 秒连不上就放弃，数据传输最多等 30 秒
)
```

> **对比 C 语言**：C 的 socket 编程中，超时要用 `setsockopt()` + `select()` 手动实现，非常繁琐。requests 只用一个 `timeout=` 参数就搞定了。

---

## 21.5 一个完整的 API 调用：GitHub 用户信息

下面用 GitHub API 演示一个完整的流程——从发请求到处理错误。

### 21.5.1 不带 token 被限流

GitHub API 对未认证的请求限制为每小时 60 次（很抠门）。试一下：

```python
import requests

resp = requests.get("https://api.github.com/users/torvalds", timeout=10)
print(f"状态码: {resp.status_code}")
print(f"剩余请求次数: {resp.headers.get('X-RateLimit-Remaining')}")
# 输出类似：剩余请求次数: 58

data = resp.json()
print(f"用户名: {data['login']}")
print(f"公开仓库数: {data['public_repos']}")
```

### 21.5.2 带 token 解除限流

去 GitHub Settings > Developer settings > Personal access tokens 创建一个 token（不需要勾选任何权限，只为了认证即可）。然后：

```python
import os
import requests
from dotenv import load_dotenv

load_dotenv()  # 假设 .env 里有 GITHUB_TOKEN=ghp_xxxx

resp = requests.get(
    "https://api.github.com/users/torvalds",
    headers={
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github.v3+json",
    },
    timeout=10,
)
print(f"状态码: {resp.status_code}")
print(f"剩余请求次数: {resp.headers.get('X-RateLimit-Remaining')}")
# 输出类似：剩余请求次数: 4999  ← 带了 token，限额从 60 涨到 5000！
```

这就是 API Key / Token 的价值：**证明你是谁，从而获得更高的调用限额**。

---

## 21.6 异常处理——让程序更健壮

调用外部 API 时，什么都会发生：网络断了、服务器挂了、被限流了。一个专业的程序必须优雅地处理这些情况。

```python
import requests
import time


def safe_api_call(url: str, max_retries: int = 3) -> dict | None:
    """一个比较完备的 API 调用封装。"""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=10)

            # 检查 HTTP 错误（4xx 或 5xx）
            resp.raise_for_status()

            # 一切顺利，返回解析后的数据
            return resp.json()

        except requests.exceptions.Timeout:
            print(f"超时 (第 {attempt + 1}/{max_retries} 次尝试)")

        except requests.exceptions.ConnectionError:
            print(f"连接失败 (第 {attempt + 1}/{max_retries} 次尝试)")
            print("  请检查网络连接和 URL 是否正确")

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            if status == 429:  # 被限流
                wait = int(e.response.headers.get("Retry-After", 5))
                print(f"被限流，等待 {wait} 秒后重试...")
                time.sleep(wait)
                continue
            elif status >= 500:  # 服务器错误，可以重试
                wait = 2 ** attempt  # 指数退避：1s, 2s, 4s
                print(f"服务器错误 {status}，{wait} 秒后重试...")
                time.sleep(wait)
                continue
            else:  # 4xx 客户端错误不重试
                print(f"请求错误 {status}: {e}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"未知错误: {e}")
            return None

    print(f"已重试 {max_retries} 次，全部失败")
    return None


# 使用
result = safe_api_call("https://api.github.com/users/torvalds")
if result:
    print(f"用户名: {result['login']}")
```

**指数退避**（exponential backoff）是处理重试的标准策略：
- 第 1 次重试等 1 秒，第 2 次等 2 秒，第 3 次等 4 秒……
- 避免在服务器已经过载时还疯狂重试，给服务器恢复的时间

---

## 21.7 SSE 流式处理——AI 开发必备

### 21.7.1 什么是 SSE

当你用 ChatGPT 时，回答是一个字一个字"蹦"出来的，而不是等 10 秒后突然出现一整段文字。这种技术就叫 **SSE（Server-Sent Events，服务器推送事件）**。

传统请求：客户端问一次 → 服务器回答一次 → 结束。
SSE：客户端问一次 → 服务器持续不断地推送数据片段 → 直到结束。

SSE 的原始数据格式：

```text
data: {"choices":[{"delta":{"content":"你"}}]}

data: {"choices":[{"delta":{"content":"好"}}]}

data: {"choices":[{"delta":{"content":"！"}}]}

data: [DONE]
```

每一行 `data: {...}` 是一个事件，空行分隔不同事件，`data: [DONE]` 表示结束。

### 21.7.2 动手解析一段模拟的 SSE 数据

不依赖真实 API，我们用一段模拟数据来理解解析逻辑：

```python
import json

# 模拟 SSE 原始数据（每行是一条 data:）
mock_sse_lines = [
    'data: {"token":"你"}',
    'data: {"token":"好"}',
    'data: {"token":"，"}',
    'data: {"token":"世"}',
    'data: {"token":"界"}',
    'data: [DONE]',
]


def parse_sse(lines: list[str]) -> str:
    """解析 SSE 格式的流式数据，累积 content 并返回完整文本。"""
    full_text = []
    for line in lines:
        # 只处理 "data: " 开头的行
        if not line.startswith("data: "):
            continue

        # 切掉 "data: " 前缀，拿到后面的 JSON 字符串
        data_str = line[6:]

        # 遇到 [DONE] 表示流结束
        if data_str.strip() == "[DONE]":
            break

        # 解析 JSON，提取内容
        try:
            chunk = json.loads(data_str)
            token = chunk.get("token", "")
            if token:
                print(token, end="", flush=True)   # 逐字打印，模拟流式体验
                full_text.append(token)
        except json.JSONDecodeError:
            continue

    print()  # 换行
    return "".join(full_text)


result = parse_sse(mock_sse_lines)
print(f"完整文本: {result}")
# 输出：你好，世界
```

`flush=True` 很重要——Python 的 `print` 默认有缓冲，不加 `flush=True` 可能把多个字符攒在一起才输出，就失去了"逐字蹦出"的效果。

---

## 21.8 httpx —— requests 的现代替代

requests 是 Python 最成功的第三方库，但它有两个不足：
1. 不支持 HTTP/2
2. 没有原生异步支持

httpx 提供了几乎相同的 API，同时支持这两点：

```bash
pip install httpx
```

**同步用法**（和 requests 几乎一模一样）：

```python
import httpx

resp = httpx.get("https://api.github.com", timeout=10)
print(resp.json()["current_user_url"])
```

**异步用法**（结合第 22 章的 asyncio）：

```python
import httpx
import asyncio


async def fetch_all():
    urls = [
        "https://api.github.com/users/torvalds",
        "https://api.github.com/users/gvanrossum",
    ]
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url, timeout=10) for url in urls]
        responses = await asyncio.gather(*tasks)
        for resp in responses:
            print(resp.json()["login"])


asyncio.run(fetch_all())
# 两个请求同时发出，总耗时接近单个请求的耗时
```

**什么时候选什么**：
- 简单脚本、学习、原型：**requests**（生态最成熟，文档最多）
- 新项目，需要异步或 HTTP/2：**httpx**
- 大量并发的爬虫/服务：**aiohttp**（第 22 章详述）

---

## 21.9 本章小结

1. **HTTP**：客户端发请求，服务端返回响应。GET 是要数据，POST 是交数据。
2. **requests 核心用法**：`requests.get(url, params=..., timeout=...)` 发请求；`resp.status_code` 看状态；`resp.text` 拿字符串；`resp.json()` 拿字典。
3. **超时**：永远设 `timeout=`。不设的话，一个卡住的 API 就能让你的程序永远等下去。
4. **异常处理**：`raise_for_status()` 检查 HTTP 错误，`try/except` 捕获超时、连接错误，指数退避处理限流。
5. **SSE 流式响应**：逐行解析 `data: {...}` 格式的数据，`flush=True` 实现逐字输出。这是 AI 大模型调用的核心技能。
6. **httpx**：requests 的现代版，支持 HTTP/2 和异步，新项目推荐。

---

## 基础练习

### 练习 1：获取网页内容并统计大小

用 requests 获取 `https://www.example.com` 的网页内容，打印状态码、Content-Type 响应头、以及网页内容的字符数。

**参考答案：**

```python
import requests

resp = requests.get("https://www.example.com", timeout=10)
print(f"状态码: {resp.status_code}")
print(f"Content-Type: {resp.headers.get('Content-Type')}")
print(f"网页内容字符数: {len(resp.text)}")
# 输出示例：
# 状态码: 200
# Content-Type: text/html; charset=UTF-8
# 网页内容字符数: 1256
```

### 练习 2：调用公开 API 并解析 JSON

调用 `https://api.github.com/users/python` 这个公开 API，获取并打印该用户的以下信息：用户名、头像 URL、公开仓库数、粉丝数。

**参考答案：**

```python
import requests

resp = requests.get("https://api.github.com/users/python", timeout=10)
resp.raise_for_status()  # 状态码不是 2xx 就报错
data = resp.json()

print(f"用户名: {data['login']}")
print(f"头像: {data['avatar_url']}")
print(f"公开仓库: {data['public_repos']}")
print(f"粉丝: {data['followers']}")
print(f"关注: {data['following']}")
# 输出示例：
# 用户名: python
# 头像: https://avatars.githubusercontent.com/u/1525981?v=4
# 公开仓库: 6
# 粉丝: 106
# 关注: 0
```

### 练习 3：处理 HTTP 错误状态码

写一个函数，接受一个 URL，尝试 GET 请求它。根据不同的状态码给出不同的友好提示（200 成功，404 资源不存在，403 没有权限，其他状态码打印数字）。

**参考答案：**

```python
import requests


def check_url(url: str) -> None:
    """检查 URL 是否可访问，根据状态码给出提示。"""
    try:
        resp = requests.get(url, timeout=10)
    except requests.exceptions.Timeout:
        print("错误：请求超时，网站可能无法访问")
        return
    except requests.exceptions.ConnectionError:
        print("错误：连接失败，请检查 URL 是否正确")
        return

    if resp.status_code == 200:
        print(f"成功！页面大小: {len(resp.text)} 字符")
    elif resp.status_code == 404:
        print("错误：页面不存在 (404)")
    elif resp.status_code == 403:
        print("错误：没有权限访问 (403)")
    elif resp.status_code >= 500:
        print(f"服务器错误 ({resp.status_code})，稍后重试")
    else:
        print(f"其他状态码: {resp.status_code}")


# 测试
check_url("https://www.example.com")           # 成功
check_url("https://api.github.com/not-exist")  # 404
```

---

## 进阶练习

### 练习 4：调用 GitHub API 获取用户的仓库列表并处理分页

调用 GitHub API 获取某个用户（比如 `torvalds`）的所有公开仓库列表。GitHub 每页默认返回 30 条，需要处理分页直到拿完所有仓库。打印每个仓库的名称、星数、描述。

**提示**：GitHub 的分页信息在响应头的 `Link` 字段中。也可以直接用 `?page=N&per_page=100` 的方式翻页。

**参考答案：**

```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()


def get_all_repos(username: str) -> list[dict]:
    """获取一个 GitHub 用户的所有公开仓库，处理分页。"""
    all_repos = []
    page = 1

    # 准备认证头（带了 token 限额更高，不带也能跑但只有 60 次/小时）
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    while True:
        url = f"https://api.github.com/users/{username}/repos"
        resp = requests.get(
            url,
            params={"page": page, "per_page": 100},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        repos = resp.json()

        if not repos:          # 返回空列表说明没有更多页了
            break

        all_repos.extend(repos)
        print(f"第 {page} 页：获取到 {len(repos)} 个仓库")
        page += 1

    return all_repos


# 获取 torvalds 的所有仓库
repos = get_all_repos("torvalds")
print(f"\n共 {len(repos)} 个仓库：")
for repo in sorted(repos, key=lambda r: r.get("stargazers_count", 0), reverse=True):
    name = repo["full_name"]
    stars = repo["stargazers_count"]
    desc = repo.get("description", "无描述")
    print(f"  {name} ({stars} stars) - {desc}")
```

---

## 常见错误

### 错误 1：忘了设 timeout

```python
# ❌ 错误：没有 timeout，网络出问题时程序永久卡住
resp = requests.get("https://api.github.com")
```

```python
# ✓ 正确：永远加上 timeout
resp = requests.get("https://api.github.com", timeout=10)
```

### 错误 2：resp.json() 在错误时直接调用

```python
# ❌ 错误：如果返回的不是 JSON（比如服务器返回了 HTML 错误页），直接崩溃
resp = requests.get("https://api.example.com/data")
data = resp.json()  # 如果状态码是 500，返回的可能是 HTML，JSON 解析失败
```

```python
# ✓ 正确：先检查状态码，再解析 JSON
resp = requests.get("https://api.example.com/data", timeout=10)
resp.raise_for_status()  # 不是 2xx 就直接报错，后面不会执行
data = resp.json()       # 到这里说明状态码 OK，可以安全解析
```

### 错误 3：混淆 .text 和 .json() 的用法

```python
# ❌ 错误：想要字典，但拿到的却是字符串
resp = requests.get("https://api.github.com", timeout=10)
print(resp.text["current_user_url"])  # TypeError: string indices must be integers

# ✓ 正确：先用 .json() 解析成字典再取字段
data = resp.json()
print(data["current_user_url"])       # https://api.github.com/user
```

### 错误 4：流式输出忘记 flush

```python
# ❌ 错误：print 默认有缓冲，字符不会即时显示
for chunk in stream:
    print(chunk, end="")   # 可能攒了好多行才一次性输出
```

```python
# ✓ 正确：加 flush=True 强制立即输出
for chunk in stream:
    print(chunk, end="", flush=True)   # 逐字即时显示
```

### 错误 5：429 限流后立即重试

```python
# ❌ 错误：被限流了还立刻重试，只会继续收到 429
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 429:
        resp = requests.get(url, timeout=10)  # 马上重试，继续被限流
```

```python
# ✓ 正确：等待后再重试，优先看服务器给的 Retry-After
import time
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 429:
        wait = int(e.response.headers.get("Retry-After", 5))
        print(f"被限流，等待 {wait} 秒...")
        time.sleep(wait)
        continue
```
