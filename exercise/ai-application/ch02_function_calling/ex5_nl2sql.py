"""
练习 5: 对话式数据库查询 (NL2SQL)

场景:
    实现一个自然语言查询数据库的助手。
    用户用中文描述查询需求 → 模型生成 SQL → 执行 → 解释结果。

安全性要求:
    - 只允许 SELECT 语句
    - 禁止 INSERT/UPDATE/DELETE/DROP/ALTER 等写操作
    - 禁止多语句查询（用 ; 分割）

要求:
    1. 定义 query_database 工具的 Schema:
       - 参数 sql: 只读 SQL 查询语句
       - description 中说明只允许 SELECT

    2. 实现 SQL 安全校验函数:
       - 检查是否为 SELECT 语句
       - 检查是否包含危险关键词（DROP, DELETE, INSERT, UPDATE, ALTER, TRUNCATE）
       - 检查是否有多条语句（; 分割）

    3. 实现模拟数据库（用 SQLite 或纯 Python 模拟）:
       - 至少包含 2 个表（如 products 和 orders）
       - 预填充一些数据

    4. 实现 run_query(user_query):
       - 模型生成 SQL → 安全校验 → 执行 → 返回结果
       - 如果 SQL 不安全，返回错误让模型重新生成
       - 结果以友好的方式呈现给用户

    5. 支持追问: 维护对话历史，用户可以就上一个查询结果追问

TODO:
    1. 定义 TOOLS 列表（query_database 工具）
    2. 实现 validate_sql(sql)
    3. 创建模拟数据库（sqlite 或 dict 模拟）
    4. 实现 run_query(user_query)
    5. 测试（查询、不安全SQL、追问）
"""

import os
import json
import re
import sqlite3
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# ============================================================
# TODO 1: 定义 Tool Schema
# ============================================================
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "TODO: 描述查询数据库功能，强调只允许 SELECT 查询",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "TODO: 描述 SQL 必须是纯 SELECT 语句",
                }
            },
            "required": ["sql"],
        },
    },
}

# ============================================================
# TODO 2: SQL 安全校验
# ============================================================
FORBIDDEN_KEYWORDS = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE", "CREATE", "EXEC", "EXECUTE"]


def validate_sql(sql: str) -> tuple[bool, str]:
    """
    校验 SQL 是否安全。

    返回:
        (is_safe: bool, message: str)
        - (True, ""): 安全
        - (False, "原因"): 不安全
    """
    # TODO:
    # 1. 去除首尾空白
    # 2. 检查是否包含多条语句（用 ; 分割，忽略字符串内的 ;）
    # 3. 检查首位关键词是否为 SELECT（忽略前导空白）
    # 4. 检查是否包含禁止的关键词（大小写不敏感）
    # 5. 如果通过所有检查，返回 (True, "")
    pass


# ============================================================
# TODO 3: 创建模拟数据库
# ============================================================
def create_mock_database():
    """
    创建一个 SQLite 内存数据库，包含 products 和 orders 两个表，
    并预填充数据。

    表结构:
    products: id, name, category, price, stock
    orders: id, product_id, quantity, customer, order_date, status
    """
    # TODO:
    # 1. 创建 SQLite 内存数据库
    # 2. CREATE TABLE products (...)
    # 3. CREATE TABLE orders (...)
    # 4. INSERT 至少 5 条 products 数据
    # 5. INSERT 至少 8 条 orders 数据
    # 6. 返回 connection 对象
    pass


# ============================================================
# TODO 4: 数据库查询函数
# ============================================================
# 全局数据库连接
db_conn = None  # TODO: 在 main 中初始化


def query_database(sql: str) -> dict:
    """
    执行安全的 SELECT 查询。

    返回:
    - 成功: {"columns": [...], "rows": [[...], ...], "row_count": n}
    - 失败: {"error": "错误信息"}
    """
    # TODO:
    # 1. 调用 validate_sql(sql) 进行安全校验
    # 2. 如果校验失败，返回 {"error": ...}
    # 3. 执行查询
    # 4. 获取列名和数据
    # 5. 返回结构化结果
    pass


# ============================================================
# TODO 5: 对话处理
# ============================================================
def run_query(user_query: str, conversation_history: list[dict] = None) -> str:
    """
    处理用户的自然语言数据查询。

    参数:
        user_query: 用户查询文本
        conversation_history: 之前的对话历史（支持追问）

    返回:
        模型基于查询结果生成的回复
    """
    # TODO:
    # 1. 构建 system prompt:
    #    - 描述数据库表结构和字段含义
    #    - 告诉模型生成 SQL 时要遵循的规则
    #    - 要求模型只生成 SELECT 语句
    # 2. 将 user_query 加入 messages
    # 3. 调用 API (带 tools)
    # 4. 处理 tool_calls:
    #    - 执行 query_database
    #    - 如果返回 error，追加 error 信息到 messages（让模型修正 SQL）
    #    - 如果成功，追加查询结果
    # 5. 继续循环直到模型生成最终回复
    pass


# ============================================================
# TODO 6: 测试
# ============================================================
if __name__ == "__main__":
    # 初始化数据库
    db_conn = create_mock_database()
    print("=== 数据库已初始化 ===\n")

    # 场景1: 简单查询
    print("--- 场景1: 简单查询 ---")
    # print(run_query("有哪些商品类别？"))

    # 场景2: 带条件的查询
    print("\n--- 场景2: 带条件查询 ---")
    # print(run_query("价格大于200的商品有哪些？"))

    # 场景3: 聚合查询
    print("\n--- 场景3: 聚合查询 ---")
    # print(run_query("每个类别的商品平均价格是多少？"))

    # 场景4: 不安全 SQL（应该被拦截）
    print("\n--- 场景4: 不安全SQL拦截 ---")
    # print(run_query("把商品表清空"))  # 应返回错误

    # 场景5: 追问
    print("\n--- 场景5: 追问 ---")
    # history = []
    # reply1 = run_query("查询所有电子产品", history)
    # print(reply1)
    # reply2 = run_query("这些产品中有库存低于5的吗？", history)
    # print(reply2)

    print("\n请完成 TODO 后取消注释运行。")
