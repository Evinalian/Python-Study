"""
章节：第22章 异步编程基础
题目：用 aiohttp 并发请求并对比同步版本耗时
类型：进阶练习

题目描述：
用 aiohttp 并发请求 10 个 URL，统计总耗时。再写一个同步版本（用 requests），
对比两者的耗时差异，计算加速比。

要求：
1. 同步版本：用 requests 串行请求 10 个 URL，记录耗时
2. 异步版本：用 aiohttp 并发请求同样的 10 个 URL，记录耗时
3. 计算并打印加速比（同步耗时 / 异步耗时）
4. 两个版本要对同样的 URL 列表做测试，保证公平对比

前置准备：
需要安装的包：aiohttp, requests

提示：
- 测试用 httpbin.org/delay/N 来模拟有延迟的响应（N 是秒数）
- 如果 httpbin.org 不可用，可以用其他测试 URL 替代
- 注意 aiohttp 的 ClientSession 用在 async with 上下文中
- time.perf_counter() 比 time.time() 更适合做性能测量
- 加速比 = 同步耗时 / 异步耗时，越高说明异步优势越大
"""

import asyncio
import time
import sys
import requests
import aiohttp

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 准备 10 个 httpbin.org/delay/1 测试 URL（模拟 1s 网络延迟）
# 2. 实现 sync_fetch_all()：用 requests 串行请求，记录耗时
# 3. 实现 async_fetch_all()：用 aiohttp 并发请求，用 asyncio.gather
# 4. 分别运行两个版本，记录各自的耗时和成功请求数
# 5. 计算加速比 = sync_time / async_time，打印对比结果
#
# 前置准备：
# - 需要安装的包：aiohttp, requests
#
# 提示：
# - aiohttp.ClientSession 使用 async with，设置 ClientTimeout
# - 同步串行总耗时 ≈ 10s（10 个 × 1s），异步并发 ≈ 1s，加速比约 10x
# - 对比同步和异步的执行时间，理解 I/O 密集型任务中并发的巨大优势
# - 可参考第 22.4 节"aiohttp 实战"
