"""
练习 5（进阶）：实现流水线并行
==============================

实现一个简化版的流水线并行训练（概念演示）。

你需要完成:
1. 实现模型分片: 将 n_layers 均匀分配到 pp_size 个 GPU 上
2. 实现 micro-batch 调度:
   - 简单版本: 逐个 micro-batch 依次流经所有 GPU
   - 进阶版本: 1F1B 调度
3. 计算流水线气泡比例:
   - 气泡 = (micro_batch数 < pp_size × 2) 时 GPU 空闲的时间比例
4. 分析气泡与 micro-batch 数量、pp_size 的关系

简化说明:
    本练习在单 GPU 上模拟多 GPU 的时序，不真正涉及跨进程通信。
    重点是理解流水线的时序和气泡概念。

示例（pp_size=4, num_micro_batches=8，简单调度）:
    GPU0: F0 F1 F2 F3 F4 F5 F6 F7 B7 B6 B5 B4 B3 B2 B1 B0
    GPU1:    F0 F1 F2 F3 F4 F5 F6 F7 B7 B6 B5 B4 B3 B2 B1 B0
    ...
    气泡 = 每端的空闲时间 / 总时间

思考题:
- 气泡比例与 pp_size 和 num_micro_batches 的关系是什么？
- 为什么 1F1B 调度能减少激活值显存？
- 什么情况下流水线并行优于张量并行？
"""

import time


# TODO: 模拟单 GPU 上的一个"层"的计算
def simulate_layer_computation(gpu_id, micro_batch_id, is_forward):
    """
    TODO: 模拟一个 GPU 上对一个 micro-batch 的计算。

    参数:
        gpu_id: 当前 GPU 编号
        micro_batch_id: micro-batch 编号
        is_forward: True=前向, False=反向

    返回:
        latency: 模拟的计算耗时（固定值即可，如 0.1 秒）
    """
    pass


# TODO: 简单流水线调度
def simulate_naive_pipeline(pp_size, num_micro_batches):
    """
    TODO: 模拟简单的流水线调度（先全部前向，再全部反向）。

    计算:
    - 总时间（最后一个 GPU 完成反向的时间）
    - 每张 GPU 的利用率 = 活跃时间 / 总时间
    - 气泡比例 = 1 - 平均利用率

    返回:
        schedule: dict，{gpu_id: [(step_type, micro_batch_id, start_time, end_time), ...]}
        bubble_ratio: float
    """
    pass


# TODO: 1F1B 调度
def simulate_1f1b_pipeline(pp_size, num_micro_batches):
    """
    TODO: 模拟 1F1B 调度。

    规则:
    - warmup 阶段: 只做前向，积累 micro-batch
    - 稳定阶段: 交替 1 个前向 + 1 个反向
    - cooldown 阶段: 只做反向，清空积累的 micro-batch
    """
    pass


if __name__ == "__main__":
    pass
