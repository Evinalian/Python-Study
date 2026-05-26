"""
章节：第22章 异步编程基础
题目：用 ThreadPoolExecutor 并发下载
类型：基础练习

题目描述：
用 ThreadPoolExecutor 并发获取 5 个不同的 GitHub 用户信息（使用预先给定的 URL 列表）。
对每个用户，打印其 login 和 public_repos。

要求：
1. 使用 ThreadPoolExecutor，max_workers=3
2. 使用 as_completed 按完成顺序处理结果（谁先返回就先打印谁）
3. 捕获每个 future 可能发生的异常
4. 不使用 async/await（这就是线程池的优势——不改代码结构）

前置准备：
需要安装的包：requests

提示：
- ThreadPoolExecutor 适合 I/O 密集型任务，因为 I/O 等待时 GIL 被释放
- executor.submit() 返回一个 Future 对象，可以后续获取结果
- as_completed() 按完成顺序 yield future，而不是按提交顺序
- executor.map() 保持输入顺序，as_completed() 不保持顺序
- 这是 async/await 之外的另一条路：如果代码库是同步的，线程池改动最小
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 准备 5 个 GitHub API URL（如 torvalds、gvanrossum、kennethreitz 等）
# 2. 定义 get_user(url) 函数，用 requests 获取并返回 JSON
# 3. 用 ThreadPoolExecutor(max_workers=3) 提交所有任务
# 4. 用 as_completed() 按完成顺序处理，打印 login 和 public_repos
# 5. 捕获 Timeout、HTTPError 等异常，失败的任务打印失败原因
# 6. 记录并发总耗时
#
# 提示：
# - future_to_url 字典映射 future -> url，便于打印进度
# - as_completed() 按完成顺序返回，不是提交顺序
# - 对比同步和异步的执行时间：5 个请求串行 vs 3 线程并发
