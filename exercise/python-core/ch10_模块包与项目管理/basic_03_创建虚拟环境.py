"""
章节：第10章 模块、包与项目管理
题目：创建虚拟环境并安装第三方库
类型：基础练习

题目描述：
本练习需要执行终端命令和编写 Python 代码两部分。

第一步：在终端中创建虚拟环境并安装 requests 库
    python -m venv .venv
    .venv\Scripts\Activate.ps1    (Windows PowerShell)
    pip install requests
    pip freeze > requirements.txt

第二步：运行本 Python 代码验证安装是否成功。

示例输入/输出：
    请求成功！
    {'slideshow': {'author': ...}, ...}

提示：
- 确保先激活虚拟环境再运行本文件
- requests.get() 发送 HTTP GET 请求
- response.json() 将 JSON 响应解析为 Python 字典
"""

import requests


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 先创建并激活虚拟环境，安装 requests
# 2. 实现 main() 函数：发送 GET 请求到 https://httpbin.org/json
# 3. 用 response.json() 解析 JSON 并打印
#
# 提示：参考第10章虚拟环境和第三方库示例
#
# 完成后运行: python basic_03_创建虚拟环境.py


def main():
    """发送 HTTP 请求验证 requests 安装"""
    pass  # TODO: 实现函数体


if __name__ == "__main__":
    main()
