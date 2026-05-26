"""
练习: Token 成本分析工具
========================

需求:
  写一个命令行工具，读取 JSONL 格式的 Token 使用日志文件，
  按日、按模型、按用户生成成本分析报告。

要求:
  1. JSONL 日志格式（每行一个 JSON 对象）:
     {
       "timestamp": "2025-01-15T14:30:00",
       "model": "gpt-4o-mini",
       "endpoint": "chat",
       "prompt_tokens": 150,
       "completion_tokens": 80,
       "total_tokens": 230,
       "cost_usd": 0.0005,
       "user_id": "user_001",
       "metadata": {"feature": "chat"}
     }

  2. 命令行参数:
     --log-file: 日志文件路径（必需）
     --start-date: 开始日期 YYYY-MM-DD（可选）
     --end-date: 结束日期 YYYY-MM-DD（可选）
     --user: 按用户筛选（可选）
     --model: 按模型筛选（可选）
     --format: 输出格式 "text" (默认) / "csv" / "json"
     --output: 输出文件路径（可选，默认打印到控制台）

  3. 分析内容（text 格式）:
     a. 概览: 总调用数、总 Token、总成本、平均每次成本
     b. 每日趋势表: 日期 | 调用次数 | Token | 成本
     c. 模型分布表: 模型 | 调用次数 | Token | 成本 | 占比
     d. 用户分布表（如果 user_id 不是全都 anonymous）
     e. TOP 10 最贵请求

  4. 支持导出为 CSV（每行一条原始记录，带筛选后的结果）

  5. 实现函数:
     - load_logs(file, start, end, user, model) -> list[dict]
     - aggregate_daily(records) -> list[dict]
     - aggregate_by_model(records) -> dict
     - aggregate_by_user(records) -> dict
     - format_text_report(stats) -> str
     - export_csv(records, output_path)

TODO:
  - [ ] 实现 load_logs(file, start, end, user, model) 加载和筛选日志
  - [ ] 实现 aggregate_daily(records) 按日聚合
  - [ ] 实现 aggregate_by_model(records) 按模型聚合
  - [ ] 实现 aggregate_by_user(records) 按用户聚合
  - [ ] 实现 format_text_report(stats) 格式化文本报告
  - [ ] 实现 export_csv(records, output_path) 导出 CSV
  - [ ] 实现 main() 解析参数并生成报告

提示:
  - 用 csv.DictWriter 写 CSV
  - 日期解析: datetime.fromisoformat(ts).date()
  - 分组用 itertools.groupby 或手写 dict 循环
  - 格式化金额: f"${cost:.4f}"
  - TOP N 用 sorted + 切片: sorted(records, key=lambda r: r["cost_usd"], reverse=True)[:N]
"""
import os
import json
import csv
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Optional


def load_logs(
    log_file: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user: Optional[str] = None,
    model: Optional[str] = None,
) -> list[dict]:
    # TODO: 加载并筛选日志
    pass


def aggregate_daily(records: list[dict]) -> list[dict]:
    # TODO: 按日聚合
    pass


def aggregate_by_model(records: list[dict]) -> dict:
    # TODO: 按模型聚合
    pass


def aggregate_by_user(records: list[dict]) -> dict:
    # TODO: 按用户聚合
    pass


def format_text_report(records: list[dict]) -> str:
    # TODO: 生成格式化的文本报告
    pass


def export_csv(records: list[dict], output_path: str):
    # TODO: 导出 CSV
    pass


def main():
    # TODO: 解析参数 → 加载 → 聚合 → 格式化 → 输出
    pass


if __name__ == "__main__":
    main()
