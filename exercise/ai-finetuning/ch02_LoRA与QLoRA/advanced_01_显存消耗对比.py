"""
进阶练习 1: 显存消耗对比实验

场景:
    你想要定量地比较不同微调方案（全量微调/LoRA/QLoRA）在各种配置下
    （不同 rank、不同 batch size、不同序列长度）的实际显存消耗。
    这能帮助你根据自己的 GPU 资源选择合适的方案。

要求:
    1. 使用 torch.cuda.memory_stats() 或 nvidia-smi 测量实际显存占用
    2. 比较多种方案的显存消耗
    3. 生成可视化的对比报告

TODO:
    1. 实现 get_gpu_memory_used() 函数:
       - 返回当前 GPU 显存使用量（MB）
       - 使用 torch.cuda.memory_allocated() 或 torch.cuda.max_memory_allocated()

    2. 实现 measure_model_memory(model) 函数:
       - 测量模型的静态显存占用（权重 + LoRA adapter）
       - 重置 CUDA memory stats
       - 返回显存使用量

    3. 实现 simulate_training_step(model, batch_size, seq_length) 函数:
       - 模拟一次训练步骤（前向 + 反向传播）
       - 使用随机生成的 input_ids（模拟真实 batch）
       - 测量前向传播后的显存增量、反向传播后的显存增量
       - 返回详细的显存使用分解

    4. 实现 compare_memory_usage(model_name, configs) 函数:
       - configs 列表包含多种配置:
         [
           {"name": "Full Fine-tune", "method": "full", ...},
           {"name": "LoRA (r=8)",   "method": "lora", "r": 8, ...},
           {"name": "QLoRA (r=8)",  "method": "qlora", "r": 8, ...},
           ...
         ]
       - 对每种配置测量静态显存和训练显存
       - 打印对比表格

    5. 实现 generate_report(results) 函数:
       - results 是 compare_memory_usage 的输出
       - 生成包含以下内容的报告:
         * 模型信息（名称、总参数量）
         * 每种方案的显存消耗表格
         * 每种方案能否在主流 GPU 上运行（RTX 3060 12GB, RTX 4090 24GB, A100 40GB, A100 80GB）
         * 推荐的性价比方案
       - 用 ASCII 表格输出

    6. 思考题（注释回答）:
       - 为什么 QLoRA 的训练显存比 LoRA 小，但差距没有 4 倍那么大？
       - 序列长度对显存的影响为什么是平方级的？
"""
import torch
import gc
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ============================================================
# TODO 1: GPU 显存测量
# ============================================================
def get_gpu_memory_used() -> int:
    """返回当前 GPU 已分配显存（MB）。"""
    # TODO: 检查 CUDA 是否可用
    # TODO: 使用 torch.cuda.memory_allocated() 获取已分配显存（字节）
    # TODO: 转换为 MB 并返回
    # TODO: 如果没有 GPU，返回 0
    pass


def reset_memory_stats():
    """重置 CUDA 显存统计。"""
    # TODO: torch.cuda.reset_peak_memory_stats()
    # TODO: torch.cuda.empty_cache()
    # TODO: gc.collect()
    pass


# ============================================================
# TODO 2: 模型显存测量
# ============================================================
def measure_model_memory(model) -> dict:
    """
    测量模型的静态显存占用。

    返回:
        {
            "total_allocated_mb": float,
            "peak_allocated_mb": float,
        }
    """
    # TODO: 重置显存统计
    # TODO: 获取已分配显存
    # TODO: 获取峰值显存
    pass


# ============================================================
# TODO 3: 模拟训练步骤
# ============================================================
def simulate_training_step(
    model,
    batch_size: int = 2,
    seq_length: int = 512,
    vocab_size: int = 32000,
) -> dict:
    """
    模拟一次训练步骤（前向 + 反向），测量显存增量。

    返回:
        {
            "before_fwd_mb": 前向传播前显存,
            "after_fwd_mb": 前向传播后显存,
            "after_bwd_mb": 反向传播后显存,
            "activation_mb": 激活值显存估值 (= after_fwd - before_fwd),
            "gradient_mb": 梯度显存估值 (= after_bwd - after_fwd),
        }
    """
    # TODO: 生成随机 input_ids: shape=(batch_size, seq_length)，值在 [0, vocab_size)
    # TODO: 创建 attention_mask: 全 1
    # TODO: 重置显存统计
    # TODO: 记录前向传播前显存
    # TODO: 执行前向传播 model(input_ids, attention_mask=attention_mask)
    # TODO: 记录前向传播后显存
    # TODO: 对 loss 执行 backward()
    # TODO: 记录反向传播后显存
    # TODO: 清理（model.zero_grad(), 删除中间变量, gc.collect()）
    # TODO: 返回分解结果
    pass


# ============================================================
# TODO 4: 多种方案显存对比
# ============================================================
def compare_memory_usage(model_name: str, configs: list[dict]) -> list[dict]:
    """
    对比不同配置方案的显存消耗。

    configs 格式:
    [
        {
            "name": "LoRA r=8",
            "method": "lora",       # "full", "lora", "qlora"
            "r": 8,
            "lora_alpha": 16,
            "target_modules": [...],
        },
        ...
    ]

    返回:
        [
            {
                "name": "LoRA r=8",
                "static_memory_mb": float,      # 模型加载后显存
                "training_peak_mb": float,      # 训练时峰值显存
                "activation_mb": float,         # 激活值显存
                "gradient_optimizer_mb": float, # 梯度 + 优化器显存
                "runs_on": ["RTX 4090", "A100"],  # 能运行此方案的 GPU 列表
            },
            ...
        ]
    """
    # TODO: 对每种配置:
    #   1) 根据 method 选择加载方式（full / lora / qlora）
    #   2) 测量静态显存
    #   3) 模拟训练步骤，测量训练显存
    #   4) 判断能在哪些 GPU 上运行
    #   5) 卸载模型，清理显存，继续下一配置
    pass


# ============================================================
# TODO 5: 生成对比报告
# ============================================================
GPU_SPECS = {
    "RTX 3060": 12 * 1024,    # 12 GB -> MB
    "RTX 4070": 12 * 1024,
    "RTX 4090": 24 * 1024,
    "A10": 24 * 1024,
    "A100": 40 * 1024,
    "A100 80GB": 80 * 1024,
}

def generate_report(results: list[dict]):
    """
    生成显存消耗对比报告（ASCII 表格格式）。

    报告内容:
    1. GPU 规格表
    2. 每种方案的显存分解表
    3. GPU 兼容性矩阵（方案 x GPU）
    4. 推荐方案
    """
    # TODO: 打印 GPU 规格
    # TODO: 打印每种方案的详细显存分解
    #   | 方案           | 静态显存 | 训练峰值 | 激活值 | 梯度+优化器 |
    #   |---------------|---------|---------|-------|-----------|
    #   | LoRA r=8      | 14.2 GB | 18.5 GB | 2.1GB | 2.2GB     |
    # TODO: 打印 GPU 兼容性矩阵
    #   | 方案           | RTX3060 | RTX4090 | A100 | A10080G |
    #   | LoRA r=8      |   NO    |   YES   | YES  |  YES    |
    # TODO: 推荐最佳性价比方案
    pass


# ============================================================
# TODO 6: 思考题
# ============================================================
"""
Q1: 为什么 QLoRA 的训练显存比 LoRA 小，但差距没有 4 倍那么大？
A1: TODO
    提示: 考虑哪些显存组成部分不受量化影响（激活值、梯度、优化器状态）

Q2: 序列长度对显存的影响为什么是平方级的？
A2: TODO
    提示: 思考注意力机制的 O(N^2) 复杂度

Q3: 如果你只有一张 RTX 3060 12GB，微调 7B 模型的最佳策略是什么？
A3: TODO
    提示: QLoRA + gradient checkpointing + 小 batch size + 短序列长度
"""


if __name__ == "__main__":
    print("=" * 50)
    print("  显存消耗对比实验")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("警告: 未检测到 GPU，本练习需要 CUDA 环境。")
        print("可以在 CPU 上加载小模型测试代码逻辑，但显存测量功能不可用。")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        print(f"GPU: {gpu_name} ({total_mb} MB)")

    # 示例配置（使用小模型测试代码逻辑）
    # configs = [
    #     {"name": "LoRA r=8",  "method": "lora",  "r": 8,  "lora_alpha": 16},
    #     {"name": "QLoRA r=8", "method": "qlora", "r": 8,  "lora_alpha": 16},
    # ]
    # results = compare_memory_usage("Qwen/Qwen2.5-0.5B", configs)
    # generate_report(results)

    print("\n请完成以上 TODO 后取消注释运行。")
