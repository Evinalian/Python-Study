"""
练习: Streamlit Token 用量仪表盘
================================

需求:
  构建一个 Streamlit 应用，读取本地 JSON 日志文件中的 API 调用记录，
  展示 Token 用量的可视化仪表盘。

要求:
  1. 数据加载:
     - 从本地 JSON/JSONL 文件读取 API 调用日志
     - 日志格式: 每行一个 JSON 对象，字段包含:
       timestamp, model, prompt_tokens, completion_tokens,
       total_tokens, cost_usd, endpoint, user_id (可选)
     - 支持上传日志文件（st.file_uploader）
     - 支持加载示例数据（如果没有日志文件）

  2. 概览指标（页面顶部一行 4 个 st.metric）:
     - 总调用次数
     - 总 Token 消耗
     - 总成本（美元）
     - 平均响应 Token 数

  3. 时间趋势图:
     - 日期范围选择器（st.date_input）
     - 每日 Token 用量折线图
     - 每日成本折线图
     - 每日调用次数柱状图

  4. 模型分布:
     - 饼图: 各模型的调用次数占比
     - 柱状图: 各模型的 Token 消耗对比
     - 表格: 各模型的详细统计（次数、总Token、平均Token、总成本）

  5. 交互功能:
     - 模型筛选器（多选下拉框）
     - 时间范围筛选
     - 数据导出按钮（导出过滤后的数据为 CSV）

  6. 外观:
     - 使用 st.set_page_config 设置页面标题和图标
     - 使用 columns 布局
     - 图表使用 Matplotlib 或 Plotly

TODO:
  - [ ] 实现 load_data(file) 解析 JSON/JSONL 日志
  - [ ] 实现 generate_sample_data() 生成模拟数据
  - [ ] 实现 plot_daily_tokens(df) 绘制每日 Token 趋势
  - [ ] 实现 plot_daily_cost(df) 绘制每日成本趋势
  - [ ] 实现 plot_model_distribution(df) 绘制模型分布饼图
  - [ ] 实现 plot_model_comparison(df) 绘制模型对比柱状图
  - [ ] 实现 main() 构建完整的仪表盘

提示:
  - 用 pandas 处理数据: df = pd.DataFrame(records)
  - 日期转换: pd.to_datetime(df['timestamp']).dt.date
  - 分组聚合: df.groupby('model').agg({...})
  - 图表: Matplotlib -> st.pyplot(fig) 或直接用 st.bar_chart / st.line_chart
  - 缓存: @st.cache_data 缓存加载和聚合结果
  - 导出 CSV: df.to_csv(index=False).encode('utf-8') 然后 st.download_button
"""
import os
import json
import random
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Token 用量仪表盘", page_icon="📊", layout="wide")


def generate_sample_data(num_records: int = 200) -> pd.DataFrame:
    """生成模拟的 API 调用记录，用于演示（没有真实日志时使用）"""
    # TODO: 生成模拟数据
    pass


def load_data(uploaded_file=None) -> pd.DataFrame:
    """从上传文件或示例数据加载"""
    # TODO: 加载数据
    pass


def plot_daily_tokens(df: pd.DataFrame):
    """每日 Token 用量趋势"""
    # TODO
    pass


def plot_daily_cost(df: pd.DataFrame):
    """每日成本趋势"""
    # TODO
    pass


def plot_model_distribution(df: pd.DataFrame):
    """模型使用分布"""
    # TODO
    pass


def plot_model_comparison(df: pd.DataFrame):
    """模型对比"""
    # TODO
    pass


def main():
    # TODO: 构建完整仪表盘
    pass


if __name__ == "__main__":
    main()
