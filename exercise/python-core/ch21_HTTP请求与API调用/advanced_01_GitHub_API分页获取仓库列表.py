"""
章节：第21章 HTTP 请求与 API 调用
题目：调用 GitHub API 获取用户的仓库列表并处理分页
类型：进阶练习

题目描述：
调用 GitHub API 获取某个用户（比如 `torvalds`）的所有公开仓库列表。
GitHub 每页默认返回 30 条，需要处理分页直到拿完所有仓库。

要求：
1. 编写函数获取指定用户的所有仓库
2. 支持分页参数 `?page=N&per_page=100`
3. 打印每个仓库的名称、星数、描述
4. 按星数降序排列显示
5. （可选）使用 GitHub Token 提升 API 限额

前置准备：
需要安装的包：requests, python-dotenv
可选：在 .env 中设置 GITHUB_TOKEN=ghp_xxxx 以提升 API 限额（不设置也能跑，但限额较低）

提示：
- GitHub 每页最多 100 条（per_page=100）
- 当返回的列表为空时，说明已经拿完所有页
- 带了 token 后，限额从 60 次/小时 提升到 5000 次/小时
- token 通过 Authorization 请求头发送：Authorization: Bearer <token>
- 注意处理网络错误和 API 限流
"""

import os
import sys
import requests
import time

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 定义 get_all_repos(username, token=None) 函数，用 while 循环处理分页
# 2. 使用 params={"page": page, "per_page": 100} 传参
# 3. 通过 Authorization: Bearer <token> 请求头提升 API 限额
# 4. 当返回空列表或 len(repos) < per_page 时停止分页
# 5. 定义 display_repos() 按 stargazers_count 降序排列显示
# 6. 从 .env 读取 GITHUB_TOKEN 环境变量（可选）
#
# 提示：
# - 处理 403 限流：读取 Retry-After 头，用 time.sleep() 等待
# - 处理超时：等待后重试，已获取的数据保留
# - 用 os.getenv("GITHUB_TOKEN") 读取 token
# - 可参考第 21.4 节"分页与认证"
