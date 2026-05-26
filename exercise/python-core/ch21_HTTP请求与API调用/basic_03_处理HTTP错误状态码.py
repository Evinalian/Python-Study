"""
章节：第21章 HTTP 请求与 API 调用
题目：处理 HTTP 错误状态码
类型：基础练习

题目描述：
写一个函数 `check_url(url: str)`，接受一个 URL，尝试 GET 请求它。
根据不同的状态码给出不同的友好提示：
- 200：成功，打印页面大小
- 404：资源不存在
- 403：没有权限
- 5xx：服务器错误，建议稍后重试
- 其他状态码：打印数字

同时处理超时（Timeout）和连接错误（ConnectionError）两种异常。

前置准备：
需要安装的包：requests

提示：
- HTTP 状态码分类：2xx=成功，3xx=重定向，4xx=客户端错误，5xx=服务器错误
- resp.raise_for_status() 自动检查状态码，非 2xx 抛出 HTTPError
- 网络请求需要处理多种异常：Timeout、ConnectionError、HTTPError 等
- 测试时可以用 https://httpbin.org/status/404 来模拟各种状态码
"""

import requests

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 定义函数 check_url(url: str) -> None
# 2. 用 requests.get() 请求 URL，设置 timeout=10
# 3. 根据 resp.status_code 给出不同的友好中文提示
# 4. 额外处理 301/302（重定向）、401（未授权）、429（限流）
# 5. 处理 Timeout、ConnectionError、SSLError 等异常
#
# 提示：
# - 用 if/elif 分支处理状态码，注意 5xx 用范围判断 500 <= status < 600
# - 用 https://httpbin.org/status/XXX 测试各种状态码
# - 注意处理网络错误和超时
# - 可参考第 21.3 节"错误处理"
