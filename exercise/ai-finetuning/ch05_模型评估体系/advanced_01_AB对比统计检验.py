"""
进阶练习 1: A/B 对比与统计显著性检验

场景:
    你微调了一个模型，在 50 个测试问题上对比了微调前后的生成结果。
    GPT-4 Judge 给出了每个回复的评分（1-5 分）。
    你需要用统计方法判断：微调后真的变好了吗？还是只是随机波动？

要求:
    1. 使用 LLM-as-Judge 获得微调前后的评分
    2. 用多种统计方法检验差异的显著性
    3. 生成专业的统计报告

TODO:
    1. 实现 collect_scores(test_cases, model_before, model_after, judge_model):
       - 对每个 test case:
         1) 用微调前模型生成回复
         2) 用微调后模型生成回复
         3) 用 GPT-4 Judge 分别评分
         4) 用双向评判方法对比两个回复
       - 返回 {"before_scores": [...], "after_scores": [...], "pairwise_wins": {...}}

    2. 实现 paired_ttest(before_scores, after_scores):
       - 使用 scipy.stats.ttest_rel 进行配对 t 检验
       - 计算 Cohen's d 效应量
       - 返回 {t_statistic, p_value, significant, effect_size, interpretation}

    3. 实现 mcnemar_test(wins_before, wins_after, ties):
       - 统计微调后更好的样本数 vs 微调前更好的样本数
       - 使用 McNemar 检验判断比例是否显著
       - 返回 {chi2, p_value, significant}

    4. 实现 bootstrap_ci(before_scores, after_scores, n_bootstrap=10000):
       - 使用 Bootstrap 方法估计评分差异的 95% 置信区间
       - 如果置信区间不包含 0，判定为显著
       - 返回 {mean_diff, ci_lower, ci_upper, significant}

    5. 实现 generate_statistical_report(all_results):
       - 整合所有检验结果
       - 生成格式化的统计报告（ASCII 表格）
       - 包含以下部分:
         * 描述性统计（均值、标准差、样本量）
         * t 检验结果
         * McNemar 检验结果
         * Bootstrap 置信区间
         * 综合结论（是否推荐部署微调模型）

    6. 实现 visualize_results(before_scores, after_scores):
       - 用 ASCII 绘制:
         * 评分分布直方图（微调前 vs 微调后）
         * Bootstrap 差异分布的直方图
         * 均值差异的置信区间图

    7. 思考题（注释回答）:
       - 如果 p 值 = 0.051（刚好不显著），但 effect size = 0.5（中等效应），
         应该怎么决策？
       - 统计显著性（p < 0.05）和实际重要性（effect size 大）之间是什么关系？
"""
import numpy as np
from scipy import stats
from typing import Optional


# ============================================================
# TODO 1: 收集评分数据
# ============================================================
def collect_scores(
    test_cases: list[dict],
    model_before,  # 微调前模型
    model_after,   # 微调后模型
    tokenizer,
    judge_model: str = "gpt-4o",
) -> dict:
    """
    用微调前后模型生成回复，并用 LLM-as-Judge 评分。

    test_cases 格式:
    [{"instruction": "...", "reference_answer": "..."}, ...]

    返回:
        {
            "before_scores": [3.5, 4.0, 2.5, ...],  # 每个测试用例的评分
            "after_scores": [4.0, 4.5, 3.0, ...],
            "pairwise_results": {
                "after_better": 30,
                "before_better": 12,
                "tie": 8,
            },
            "individual_judgments": [
                {"instruction": "...", "before_score": 3.5, "after_score": 4.0, "winner": "after"},
                ...
            ]
        }
    """
    # TODO: 对每个 test case:
    #   1) 用 model_before 生成回复
    #   2) 用 model_after 生成回复
    #   3) 用 GPT-4 分别对两个回复评分（1-5 分，四个维度取平均）
    #   4) 用双向评判对比两个回复
    # TODO: 收集所有评分
    # TODO: 返回结构化结果
    pass


# ============================================================
# TODO 2: 配对 t 检验
# ============================================================
def paired_ttest(
    before_scores: list[float],
    after_scores: list[float],
    alpha: float = 0.05,
) -> dict:
    """
    配对 t 检验。

    返回:
        {
            "n": int,               # 样本量
            "mean_before": float,   # 微调前均值
            "mean_after": float,    # 微调后均值
            "mean_diff": float,     # 均值差
            "std_diff": float,      # 差异的标准差
            "t_statistic": float,   # t 统计量
            "p_value": float,       # p 值
            "significant": bool,    # p < alpha?
            "effect_size": float,   # Cohen's d
            "effect_size_interpretation": str,  # "small"/"medium"/"large"
        }
    """
    # TODO: 使用 scipy.stats.ttest_rel
    # TODO: 计算 Cohen's d = mean_diff / std_diff
    # TODO: 解释效应量:
    #   |d| < 0.2: negligible
    #   0.2 <= |d| < 0.5: small
    #   0.5 <= |d| < 0.8: medium
    #   |d| >= 0.8: large
    pass


# ============================================================
# TODO 3: McNemar 检验
# ============================================================
def mcnemar_test(
    wins_after: int,   # 微调后更好的样本数
    wins_before: int,  # 微调前更好的样本数
    ties: int = 0,     # 平局数
) -> dict:
    """
    McNemar 检验：判断"更好"的比例是否显著偏向一边。

    返回:
        {
            "total": int,
            "after_better": int,
            "before_better": int,
            "tie": int,
            "chi2": float,
            "p_value": float,
            "significant": bool,
        }
    """
    # TODO: 使用连续性校正的 McNemar 检验
    #   chi2 = (|b - c| - 1)^2 / (b + c)  (with Yates correction)
    #   p_value = 1 - chi2.cdf(chi2, df=1)
    pass


# ============================================================
# TODO 4: Bootstrap 置信区间
# ============================================================
def bootstrap_ci(
    before_scores: list[float],
    after_scores: list[float],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Bootstrap 方法估计评分差异的置信区间。

    不假设数据服从正态分布。

    返回:
        {
            "mean_diff": float,       # 原始数据的均值差异
            "bootstrap_mean": float,  # Bootstrap 均值差异
            "ci_lower": float,        # 置信区间下界
            "ci_upper": float,        # 置信区间上界
            "ci_level": float,        # 置信水平
            "significant": bool,      # CI 不包含 0 则为显著
            "bootstrap_diffs": [...], # 所有 Bootstrap 差异（用于绘图）
        }
    """
    # TODO: 将 before 和 after 转为 numpy 数组
    # TODO: 对于 n_bootstrap 次:
    #   1) 有放回地随机抽样 n 个索引
    #   2) 计算抽样的 before 和 after 的均值
    #   3) 记录差异
    # TODO: 计算分位数得到置信区间
    # TODO: 判断区间是否不包含 0
    pass


# ============================================================
# TODO 5: 统计报告
# ============================================================
def generate_statistical_report(all_results: dict):
    """
    生成完整的统计报告（ASCII 格式）。

    报告结构:
    ========================================
              A/B 对比统计报告
    ========================================

    1. 描述性统计
       - 样本量、均值、标准差

    2. 配对 t 检验
       - t 值、p 值、显著性、效应量及解释

    3. McNemar 检验（二分类）
       - 微调后更好 N, 微调前更好 M, 平局 K
       - chi2, p 值, 显著性

    4. Bootstrap 置信区间
       - 均值差异的 95% CI
       - CI 区间和显著性

    5. 综合结论
       - 是否推荐部署微调模型
       - 推荐的信心水平
    """
    # TODO: 提取各部分结果
    # TODO: 格式化打印（使用对齐的 ASCII 表格）
    # TODO: 生成综合结论
    pass


# ============================================================
# TODO 6: 可视化
# ============================================================
def visualize_results(before_scores: list[float], after_scores: list[float]):
    """
    用 ASCII 绘制结果可视化。

    包括:
    1. 评分分布直方图（微调前 vs 微调后）
    2. Bootstrap 差异分布直方图
    3. 均值差异的置信区间图（error bar）
    """
    # TODO: 计算直方图数据
    # TODO: 用 ASCII 字符绘制
    #   █ = 高频率, ▓ = 中频率, ▒ = 低频率, ░ = 很低频率
    # TODO: 绘制置信区间图
    #   微调前均值: 3.5 [====*====]
    #   微调后均值: 4.2 [=======*==]
    pass


# ============================================================
# TODO 7: 思考题
# ============================================================
"""
Q1: 如果 p 值 = 0.051（刚好不显著），但 effect size = 0.5（中等效应），
    应该怎么决策？
A1: TODO
    提示: p 值受样本量影响——样本量小可能导致真实效应不显著。
    考虑: 增加样本量？用贝叶斯方法？分析实际业务影响？

Q2: 统计显著性（p < 0.05）和实际重要性（effect size 大）之间是什么关系？
A2: TODO
    提示: p 值告诉你"差异是否可能由随机因素产生"，
    effect size 告诉你"差异有多大"。
    大样本下很小的差异也可能显著，但不一定有意义。

Q3: Bootstrap 方法相比传统 t 检验有什么优势？
A3: TODO
"""


if __name__ == "__main__":
    print("=" * 50)
    print("  A/B 对比统计检验")
    print("=" * 50)

    # 模拟评分数据（用于测试）
    np.random.seed(42)
    n_samples = 50

    # 模拟：微调前评分 ~ N(3.2, 0.8)
    before_scores = np.random.normal(3.2, 0.8, n_samples).clip(1, 5).tolist()
    # 模拟：微调后评分 ~ N(3.8, 0.7)（略微提升）
    after_scores = np.random.normal(3.8, 0.7, n_samples).clip(1, 5).tolist()

    print(f"\n模拟数据: {n_samples} 个样本")
    print(f"微调前均值: {np.mean(before_scores):.3f} +/- {np.std(before_scores):.3f}")
    print(f"微调后均值: {np.mean(after_scores):.3f} +/- {np.std(after_scores):.3f}")

    # 1. t 检验
    ttest_result = paired_ttest(before_scores, after_scores)
    print(f"\nt 检验: t={ttest_result.get('t_statistic', 'TODO'):.3f}, "
          f"p={ttest_result.get('p_value', 'TODO'):.4f}, "
          f"显著={'是' if ttest_result.get('significant') else '否'}")

    # 2. McNemar 检验
    # 统计"更好的"样本数
    wins_after = sum(1 for b, a in zip(before_scores, after_scores) if a > b)
    wins_before = sum(1 for b, a in zip(before_scores, after_scores) if b > a)
    ties = n_samples - wins_after - wins_before
    mn_result = mcnemar_test(wins_after, wins_before, ties)
    print(f"McNemar: after_better={wins_after}, before_better={wins_before}, "
          f"tie={ties}, p={mn_result.get('p_value', 'TODO'):.4f}")

    # 3. Bootstrap CI
    bootstrap_result = bootstrap_ci(before_scores, after_scores)
    print(f"Bootstrap CI: mean_diff={bootstrap_result.get('mean_diff', 'TODO'):.3f}, "
          f"95% CI=[{bootstrap_result.get('ci_lower', 'TODO'):.3f}, "
          f"{bootstrap_result.get('ci_upper', 'TODO'):.3f}]")

    # 4. 可视化
    visualize_results(before_scores, after_scores)

    print("\n请完成以上 TODO 以看到真实结果。")
