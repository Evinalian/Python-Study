"""
章节：第21章 HTTP 请求与 API 调用
题目：获取网页内容并统计大小
类型：基础练习

题目描述：
用 requests 获取 `https://www.example.com` 的网页内容，打印：
1. 状态码
2. Content-Type 响应头
3. 网页内容的字符数（即 len(resp.text)）

前置准备：
需要安装的包：requests

提示：
- 永远设置 timeout 参数，防止网络问题导致程序永远卡住
- resp.text 返回的是字符串，resp.content 返回的是原始字节
- resp.headers 是一个类似字典的对象，可以像字典一样取值
- 如果网络不通，requests 会抛出 ConnectionError 异常
"""

import requests

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 用 requests.get() 获取 https://www.example.com，设置 timeout=10
# 2. 打印 resp.status_code 和 resp.headers.get("Content-Type")
# 3. 打印 len(resp.text) 统计网页内容字符数
# 4. 处理 Timeout 和 ConnectionError 异常
#
# 提示：
# - resp.text 返回解码后的字符串，resp.content 返回原始字节
# - 注意处理网络错误和超时
# - 可参考第 21.1 节"发送 HTTP 请求"
