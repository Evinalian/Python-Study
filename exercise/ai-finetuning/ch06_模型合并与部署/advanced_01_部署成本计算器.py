"""
进阶练习 1: 自部署 vs API 调用 —— 成本计算器

场景:
    你需要决定是自己部署微调模型还是使用商业 API（GPT-4、Qwen API 等）。
    这个决策需要综合考虑调用量、延迟要求、数据安全和成本。

    你需要在具体数字的基础上做出决策。

要求:
    1. 设计一个命令行成本计算器
    2. 对比自部署和 API 调用的月度/年度成本
    3. 计算"盈亏平衡点"（什么时候自部署更便宜）

TODO:
    1. 实现 SelfHostedCost 数据类:
       - gpu_cost: GPU 购置成本（一次性）
       - gpu_lifespan_months: GPU 预期使用年限（月数）
       - electricity_cost_per_kwh: 电价（元/度）
       - gpu_power_watts: GPU 功耗（瓦）
       - server_cost_per_month: 服务器租赁费（月）
       - maintenance_hours_per_month: 月均维护工时
       - labor_cost_per_hour: 人工时薪

    2. 实现 APICost 数据类:
       - input_price_per_1m_tokens: 输入价格（元/百万 token）
       - output_price_per_1m_tokens: 输出价格（元/百万 token）
       - 支持多个 API 提供商的定价:
         * gpt-4o: $2.50/$10.00 per 1M tokens
         * gpt-4o-mini: $0.15/$0.60
         * qwen-max: ¥20/¥60 (示例)
         * deepseek-v2: ¥1/¥2 (示例)

    3. 实现 calculate_self_hosted_monthly(params) 函数:
       - 月度成本 = GPU 折旧 + 电费 + 服务器费 + 人工成本
       - GPU 月折旧 = gpu_cost / gpu_lifespan_months
       - 月电费 = gpu_power_watts / 1000 * 24 * 30 * electricity_cost_per_kwh
       - 月人工 = maintenance_hours_per_month * labor_cost_per_hour

    4. 实现 calculate_api_monthly(params, daily_calls, avg_input_tokens, avg_output_tokens) 函数:
       - 月度 API 成本 = 30 * daily_calls * (
           avg_input_tokens / 1e6 * input_price +
           avg_output_tokens / 1e6 * output_price
         )

    5. 实现 find_breakeven_point(self_hosted_params, api_params, usage_range) 函数:
       - usage_range: 要分析的日均调用量列表 [100, 500, 1000, 5000, 10000, ...]
       - 对每个调用量，计算自部署和 API 的月成本
       - 找到 API 成本超过自部署成本的最小调用量（盈亏平衡点）
       - 打印对比表和盈亏平衡分析

    6. 实现 print_cost_comparison_table(self_hosted_monthly, api_costs, usage_levels):
       - 打印格式化表格:
         | 日均调用量 | 自部署月成本 | GPT-4o月成本 | Qwen-Max月成本 | 最优方案 |
         | 100       | ¥3,200      | ¥150        | ¥50           | Qwen API |
         | 10000     | ¥3,200      | ¥15,000     | ¥5,000        | 自部署   |

    7. 思考题（注释回答）:
       - 为什么"盈亏平衡点"在不同 API 提供商之间差别很大？
       - 除了纯成本，还有哪些因素影响自部署 vs API 的决策？
       - 如果公司有闲置的 GPU 服务器，计算方式应该怎么调整？
"""
from dataclasses import dataclass, field


# ============================================================
# TODO 1: 自部署成本模型
# ============================================================
@dataclass
class SelfHostedCost:
    """自部署方案的各项成本参数。"""
    gpu_cost: float = 15000             # GPU 购置成本（元）
    gpu_lifespan_months: int = 36       # GPU 使用年限（月），通常 3 年
    electricity_cost_per_kwh: float = 0.8  # 电价（元/度），商业用电约 0.8-1.2
    gpu_power_watts: int = 300           # GPU 功耗（瓦），RTX 4090 约 300W
    server_cost_per_month: float = 500   # 服务器托管/租赁费（月）
    maintenance_hours_per_month: float = 10  # 月均维护时间（小时）
    labor_cost_per_hour: float = 100     # 人工时薪（元）


# ============================================================
# TODO 2: API 成本模型
# ============================================================
@dataclass
class APIPricing:
    """单个 API 提供商的定价。"""
    name: str = ""
    input_price_per_1m: float = 0.0    # 输入价格（元/百万 token）
    output_price_per_1m: float = 0.0   # 输出价格（元/百万 token）


# 常见 API 提供商的定价（2024 年参考价，单位：元/百万 token）
# TODO: 完善以下定价表
API_PROVIDERS = {
    # "gpt-4o": APIPricing(
    #     name="GPT-4o",
    #     input_price_per_1m=...,   # $2.50 ≈ ¥18
    #     output_price_per_1m=...,  # $10.00 ≈ ¥72
    # ),
    # "gpt-4o-mini": APIPricing(...),
    # "qwen-max": APIPricing(...),
    # "deepseek-v2": APIPricing(...),
}


# ============================================================
# TODO 3: 计算自部署月成本
# ============================================================
def calculate_self_hosted_monthly(params: SelfHostedCost) -> dict:
    """
    计算自部署方案的月度成本。

    返回:
        {
            "gpu_depreciation": float,    # GPU 月折旧
            "electricity": float,         # 月电费
            "server": float,              # 月服务器费
            "labor": float,               # 月人工成本
            "total_monthly": float,       # 月度总成本
        }
    """
    # TODO: GPU 月折旧 = gpu_cost / gpu_lifespan_months
    # TODO: 月电费 = gpu_power_watts / 1000 * 24 * 30 * electricity_cost_per_kwh
    # TODO: 月人工 = maintenance_hours_per_month * labor_cost_per_hour
    # TODO: 总成本 = 折旧 + 电费 + 服务器费 + 人工
    pass


# ============================================================
# TODO 4: 计算 API 月成本
# ============================================================
def calculate_api_monthly(
    pricing: APIPricing,
    daily_calls: int,
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 300,
) -> float:
    """
    计算使用某个 API 的月度成本。

    参数:
        pricing: API 定价
        daily_calls: 日均调用次数
        avg_input_tokens: 平均每次输入的 token 数
        avg_output_tokens: 平均每次输出的 token 数

    返回: 月度成本（元）
    """
    # TODO: 月总输入 tokens = 30 * daily_calls * avg_input_tokens
    # TODO: 月总输出 tokens = 30 * daily_calls * avg_output_tokens
    # TODO: 月度成本 = 输入 tokens/1e6 * input_price + 输出 tokens/1e6 * output_price
    pass


# ============================================================
# TODO 5: 盈亏平衡分析
# ============================================================
def find_breakeven_point(
    self_hosted_params: SelfHostedCost,
    api_providers: dict[str, APIPricing],
    usage_range: list[int] = None,
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 300,
) -> dict:
    """
    找到每种 API 的盈亏平衡点（自部署更便宜的日均调用量）。

    返回:
        {
            "self_hosted_monthly": float,  # 自部署月成本（固定）
            "breakeven_points": {
                "GPT-4o": 5000,           # GPT-4o 的盈亏平衡点
                "Qwen-Max": 20000,        # Qwen-Max 的盈亏平衡点
            },
            "comparison_table": [...],    # 详细对比数据
        }
    """
    if usage_range is None:
        usage_range = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]

    # TODO: 计算自部署月成本（不随调用量变化）
    # TODO: 对每个 usage_level:
    #   计算每个 API 提供商的月成本
    #   判断自部署是否更便宜
    # TODO: 对每个 API 提供商:
    #   找到 API 成本超过自部署成本的最小调用量
    #   如果永远不超过，标记为 "无盈亏平衡点（API 始终更便宜）"
    pass


# ============================================================
# TODO 6: 打印对比表
# ============================================================
def print_cost_comparison_table(
    self_hosted_monthly: float,
    api_costs: dict[str, list[float]],
    usage_levels: list[int],
    api_names: list[str],
):
    """
    打印自部署 vs API 成本对比表。

    输出格式:
    日均调用量 | 自部署月成本 | GPT-4o月成本 | Qwen-Max月成本 | 最优方案
    -----------|------------|-------------|---------------|---------
    100        | ¥3,200     | ¥150        | ¥50           | Qwen API
    1000       | ¥3,200     | ¥1,500      | ¥500          | Qwen API
    10000      | ¥3,200     | ¥15,000     | ¥5,000        | 自部署
    50000      | ¥3,200     | ¥75,000     | ¥25,000       | 自部署
    """
    # TODO: 打印表头
    # TODO: 对每个 usage_level，打印一行
    # TODO: "最优方案"列根据成本最低的方案标注
    # TODO: 用 ¥ 符号和千分位格式化金额
    pass


# ============================================================
# TODO 7: 思考题
# ============================================================
"""
Q1: 为什么"盈亏平衡点"在不同 API 提供商之间差别很大？
A1: TODO
    提示: GPT-4 的价格远高于 Qwen/DeepSeek 等国产 API。
    自部署的月成本约 ¥2000-4000，对于便宜的 API 可能永远不需要自部署
    （除非有数据安全或延迟要求）。

Q2: 除了纯成本，还有哪些因素影响自部署 vs API 的决策？
A2: TODO
    提示: 数据安全（医疗/金融数据不能出境）、延迟要求（<100ms vs API 的 500ms+）、
    可用性 SLA（API 可能宕机）、模型行为可控性（API 模型可能随时更新改变行为）。

Q3: 如果公司有闲置的 GPU 服务器，计算方式应该怎么调整？
A3: TODO
    提示: 闲置 GPU 的"机会成本"——如果不用它部署模型，它也没有其他用途，
    那么 GPU 折旧可以视为零（沉没成本），自部署月成本大幅降低。
"""


if __name__ == "__main__":
    print("=" * 60)
    print("  自部署 vs API 调用 —— 成本计算器")
    print("=" * 60)

    # 默认的自部署参数
    self_hosted = SelfHostedCost()

    # 计算自部署月成本
    monthly = calculate_self_hosted_monthly(self_hosted)
    print(f"\n自部署月度成本估算:")
    print(f"  GPU 折旧:        ¥{monthly.get('gpu_depreciation', 0):,.0f}/月")
    print(f"  电费:            ¥{monthly.get('electricity', 0):,.0f}/月")
    print(f"  服务器费:        ¥{monthly.get('server', 0):,.0f}/月")
    print(f"  人工成本:        ¥{monthly.get('labor', 0):,.0f}/月")
    print(f"  月度总成本:      ¥{monthly.get('total_monthly', 0):,.0f}/月")

    # 盈亏平衡分析
    usage_range = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    breakeven = find_breakeven_point(self_hosted, API_PROVIDERS, usage_range)

    if breakeven:
        print(f"\n盈亏平衡分析:")
        for provider, point in breakeven.get("breakeven_points", {}).items():
            if point == float("inf"):
                print(f"  {provider}: API 始终更便宜（无盈亏平衡点）")
            else:
                print(f"  {provider}: 日均 > {point:,} 次调用时自部署更便宜")

    print("\n请完成以上 TODO 以看到完整的成本分析。")
    print("提示: 需要填写 API_PROVIDERS 的实际定价数据。")
