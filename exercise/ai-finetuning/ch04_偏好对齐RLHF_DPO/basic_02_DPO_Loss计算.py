"""
练习 2: DPO Loss 手动计算与可视化

场景:
    你想深入理解 DPO 的 loss 函数是如何工作的——不同的参数（beta、概率差异）
    如何影响 loss 值。手动实现 DPO loss 有助于建立直觉。

要求:
    1. 手动实现 DPO loss 函数（不依赖 TRL 库）
    2. 模拟不同的概率差异和 beta 值
    3. 可视化 loss 随参数的变化

TODO:
    1. 实现 dpo_loss(chosen_logp, rejected_logp, ref_chosen_logp, ref_rejected_logp, beta):
        DPO Loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
        其中:
        - log_ratio_chosen = chosen_logp - ref_chosen_logp
        - log_ratio_rejected = rejected_logp - ref_rejected_logp

    2. 实现 simulate_dpo_losses(beta_values, policy_diff_values):
        - policy_diff = chosen_logp - rejected_logp（当前策略下 chosen 和 rejected 的概率差）
        - ref_diff = ref_chosen_logp - ref_rejected_logp（参考模型下的概率差）
        - 对于多组 (beta, policy_diff) 组合，计算 DPO loss
        - 返回二维数组

    3. 实现 plot_dpo_loss_heatmap(beta_values, policy_diff_values, losses):
        - 用 ASCII 绘制 DPO loss 的热力图
        - X 轴: policy_diff（当前策略的 chosen-rejected 概率差）
        - Y 轴: beta 值
        - 颜色深度表示 loss 值

    4. 实现 analyze_dpo_gradient(chosen_logp, rejected_logp, ref_chosen_logp,
                                  ref_rejected_logp, beta):
        - 计算 DPO loss 对 chosen_logp 和 rejected_logp 的梯度
        - 用 pytorch 的 autograd 或手动推导
        - 分析梯度方向: 对于 chosen 回复，模型是增加还是减小其概率？

    5. 实现 experiment_ref_model_effect():
        - 固定 beta=0.1
        - 变化 ref_chosen_logp - ref_rejected_logp（参考模型对 chosen/rejected 的区分度）
        - 观察 DPO loss 如何随参考模型的"初始偏好"变化
        - 解释: 如果参考模型本身已经偏好 chosen，DPO 还需要做什么？

    6. 思考题（注释回答）:
       - 如果 chosen_logp 和 rejected_logp 都很低（模型对两者都不自信），
         DPO loss 会怎样？这说明了什么？
       - beta 的直觉含义是什么？为什么它被称为"温度参数"？
"""
import math
import torch
import torch.nn.functional as F


# ============================================================
# TODO 1: DPO Loss 实现
# ============================================================
def dpo_loss(
    chosen_logp: float,
    rejected_logp: float,
    ref_chosen_logp: float,
    ref_rejected_logp: float,
    beta: float = 0.1,
) -> float:
    """
    手动计算 DPO loss。

    参数:
        chosen_logp: 当前模型对 chosen 回复的 log probability
        rejected_logp: 当前模型对 rejected 回复的 log probability
        ref_chosen_logp: 参考模型对 chosen 回复的 log probability
        ref_rejected_logp: 参考模型对 rejected 回复的 log probability
        beta: KL 惩罚系数（温度参数）

    返回:
        DPO loss (标量)
    """
    # TODO: 计算 log_ratio_chosen = chosen_logp - ref_chosen_logp
    # TODO: 计算 log_ratio_rejected = rejected_logp - ref_rejected_logp
    # TODO: 计算 scaled_diff = beta * (log_ratio_chosen - log_ratio_rejected)
    # TODO: 计算 loss = -log(sigmoid(scaled_diff))
    # TODO: 使用 math.log1p 提高数值稳定性
    #   log(sigmoid(x)) = -log(1 + exp(-x)) = -log1p(exp(-x)) 当 x > 0
    #                    = x - log(1 + exp(x)) = x - log1p(exp(x)) 当 x <= 0
    pass


def dpo_loss_batch(
    chosen_logps: torch.Tensor,     # (batch_size,)
    rejected_logps: torch.Tensor,   # (batch_size,)
    ref_chosen_logps: torch.Tensor, # (batch_size,)
    ref_rejected_logps: torch.Tensor, # (batch_size,)
    beta: float = 0.1,
) -> torch.Tensor:
    """
    批量版本的 DPO loss（使用 PyTorch）。
    """
    # TODO: 实现 PyTorch 版本的批量 DPO loss
    # TODO: 使用 F.logsigmoid 提高数值稳定性
    pass


# ============================================================
# TODO 2: 模拟不同参数组合的 Loss
# ============================================================
def simulate_dpo_losses(
    beta_values: list[float],
    policy_diffs: list[float],  # chosen_logp - rejected_logp (当前策略)
    ref_diff: float = 0.0,      # ref_chosen_logp - ref_rejected_logp (参考模型)
) -> list[list[float]]:
    """
    模拟不同 (beta, policy_diff) 组合下的 DPO loss。

    返回: losses[b_idx][p_idx] = loss for beta_values[b_idx] and policy_diffs[p_idx]
    """
    # TODO: 初始化二维数组
    # TODO: 对于每组 (beta, policy_diff)，计算 DPO loss
    #   chosen_logp = policy_diff (为简化，设 rejected_logp = 0)
    #   ref_chosen_logp = ref_diff (为简化，设 ref_rejected_logp = 0)
    # TODO: 返回二维数组
    pass


# ============================================================
# TODO 3: ASCII Loss 热力图
# ============================================================
def plot_dpo_loss_heatmap(
    beta_values: list[float],
    policy_diffs: list[float],
    losses: list[list[float]],
):
    """
    用 ASCII 字符绘制 DPO loss 热力图。

    输出示例:
              policy_diff (chosen - rejected log prob)
              -5.0  -2.5   0.0   2.5   5.0
    beta 0.01  ████  ███   ██    █     ·
    beta 0.10  ███   ██    █     ·     ·
    beta 0.50  ██    █     █     ·     ·
    beta 1.00  █     █     ·     ·     ·
    """
    # TODO: 归一化 losses 到 [0, 1]
    # TODO: 映射到 ASCII 字符: " ·░▒▓█"
    # TODO: 绘制 X 轴标签（policy_diff 值）
    # TODO: 绘制 Y 轴标签（beta 值）
    # TODO: 填充热力图格子
    pass


# ============================================================
# TODO 4: 梯度分析
# ============================================================
def analyze_dpo_gradient(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    ref_chosen_logp: float,
    ref_rejected_logp: float,
    beta: float = 0.1,
) -> dict:
    """
    分析 DPO loss 对 chosen_logp 和 rejected_logp 的梯度。

    梯度方向告诉我们：
    - chosen 回复应该被"鼓励"还是"抑制"
    - rejected 回复应该被"鼓励"还是"抑制"

    返回:
        {
            "grad_chosen": float,    # d(loss)/d(chosen_logp)
            "grad_rejected": float,  # d(loss)/d(rejected_logp)
            "grad_chosen_sign": int, # +1 表示减小 chosen_logp 会减小 loss（即鼓励 chosen）
            "grad_rejected_sign": int, # -1 表示增大 rejected_logp 会增大 loss（即抑制 rejected）
        }
    """
    # TODO: 用 torch.autograd 计算梯度
    #   - 将 chosen_logp 和 rejected_logp 设为 requires_grad=True
    #   - 计算 loss
    #   - loss.backward()
    #   - 读取 .grad 属性
    # TODO: 分析梯度符号
    #   - grad_chosen < 0: 增大 chosen_logp 减小 loss → 鼓励 chosen
    #   - grad_rejected > 0: 增大 rejected_logp 增大 loss → 抑制 rejected
    pass


# ============================================================
# TODO 5: 参考模型的影响
# ============================================================
def experiment_ref_model_effect():
    """
    实验: 参考模型本身的"初始偏好"如何影响 DPO loss。

    场景分析:
    1. 参考模型本身就强烈偏好 chosen（ref_chosen_logp >> ref_rejected_logp）
       → DPO 的"改进空间"很小，loss 较低但提升有限
    2. 参考模型对两者没有明显偏好（ref_chosen_logp ≈ ref_rejected_logp）
       → DPO 需要从零开始学习偏好，loss 初始较高但提升空间大
    3. 参考模型偏好 rejected（ref_chosen_logp < ref_rejected_logp）
       → DPO 需要先纠正参考模型的"错误偏好"，loss 最高

    打印这三种场景下的 DPO loss 和所需的"校正力度"。
    """
    # TODO: 固定 beta=0.1, chosen_logp=-2.0, rejected_logp=-4.0
    #   (当前策略已经偏好 chosen)
    # TODO: 变化 ref_chosen_logp 和 ref_rejected_logp:
    #   场景 1: ref_chosen=-2.0, ref_rejected=-4.0 (参考模型也偏好 chosen)
    #   场景 2: ref_chosen=-3.0, ref_rejected=-3.0 (参考模型无偏好)
    #   场景 3: ref_chosen=-4.0, ref_rejected=-2.0 (参考模型偏好 rejected)
    # TODO: 计算每种场景的 DPO loss
    # TODO: 分析每种场景下 DPO 需要"克服"的初始偏差
    pass


# ============================================================
# TODO 6: 思考题
# ============================================================
"""
Q1: 如果 chosen_logp 和 rejected_logp 都很低（模型对两者都不自信），
    DPO loss 会怎样？这说明了什么？
A1: TODO
    提示: 代入公式计算。如果两者都很低且差值为0，policy_diff=0，loss 会接近 -log(0.5) ≈ 0.693。
    这表示 DPO 需要一个"区分度"来降低 loss。

Q2: beta 的直觉含义是什么？为什么它被称为"温度参数"？
A2: TODO
    提示: 类比 softmax 中的 temperature。beta 控制了 log_ratio 差异的"放大程度"。
    beta 小 → 即使偏好差异大，loss 反馈也弱 → 模型变化小
    beta 大 → 即使偏好差异小，loss 反馈也强 → 模型变化大

Q3: 如果 DPO loss 在训练过程中一直不下降，可能是什么原因？
A3: TODO
"""


if __name__ == "__main__":
    print("=" * 50)
    print("  DPO Loss 手动计算与可视化")
    print("=" * 50)

    # 测试基础 DPO loss
    print("\n1. 基础 DPO Loss 测试:")
    loss1 = dpo_loss(
        chosen_logp=-2.0,
        rejected_logp=-5.0,
        ref_chosen_logp=-3.0,
        ref_rejected_logp=-3.0,
        beta=0.1,
    )
    print(f"   chosen 更好 (当前策略已有偏好): loss = {loss1:.4f}")

    loss2 = dpo_loss(
        chosen_logp=-4.0,
        rejected_logp=-2.0,
        ref_chosen_logp=-3.0,
        ref_rejected_logp=-3.0,
        beta=0.1,
    )
    print(f"   rejected 更好 (当前策略偏好错误): loss = {loss2:.4f}")
    print(f"   (错误偏好的 loss 应该 > 正确偏好的 loss)")

    # 模拟不同参数
    print("\n2. 参数扫描:")
    beta_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    policy_diffs = [-5.0, -2.5, 0.0, 2.5, 5.0]
    losses = simulate_dpo_losses(beta_values, policy_diffs)

    print("\n3. Loss 热力图:")
    plot_dpo_loss_heatmap(beta_values, policy_diffs, losses)

    print("\n4. 参考模型影响实验:")
    experiment_ref_model_effect()

    print("\n请完成以上 TODO 以看到完整结果。")
