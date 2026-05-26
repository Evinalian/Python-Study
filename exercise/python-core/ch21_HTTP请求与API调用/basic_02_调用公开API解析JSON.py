"""
章节：第21章 HTTP 请求与 API 调用
题目：调用公开 API 并解析 JSON
类型：基础练习

题目描述：
调用 `https://api.github.com/users/python` 这个公开 API，获取并打印该用户的以下信息：
1. 用户名（login）
2. 头像 URL（avatar_url）
3. 公开仓库数（public_repos）
4. 粉丝数（followers）
5. 关注数（following）

前置准备：
需要安装的包：requests

提示：
- GitHub API 是公开的，不需要 token 也能访问（但限额较低：每小时 60 次）
- resp.json() 将 JSON 字符串自动解析为 Python 字典
- 先用 resp.raise_for_status() 检查状态码是否为 2xx
- API 返回的 JSON 是字典，可以直接用 data["key"] 访问字段
- 区别：resp.text 返回字符串，resp.json() 返回字典/列表
"""

import requests

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 定义函数 get_github_user(username: str) -> dict | None
# 2. 用 requests.get() 请求 https://api.github.com/users/{username}
# 3. 用 resp.raise_for_status() 检查状态码，用 resp.json() 解析
# 4. 打印 login、avatar_url、public_repos、followers、following
# 5. 处理 Timeout、ConnectionError、HTTPError 异常
#
# 提示：
# - 处理 404（用户不存在）和 403（频率限制）
# - 查看 X-RateLimit-Remaining 响应头了解 API 配额剩余
# - 注意处理网络错误和超时
# - 可参考第 21.2 节"解析 JSON 响应"
