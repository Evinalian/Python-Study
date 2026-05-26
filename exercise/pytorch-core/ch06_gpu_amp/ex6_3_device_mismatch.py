"""
练习 6-3：CPU/GPU Tensor 混用错误复现与修复

目标：
  故意写出 4-5 种常见的 CPU/GPU Tensor 混用错误，用 try/except 捕获并打印错误信息，
  然后给出正确写法。

要求：
  演示以下错误场景（每种先展示错误代码及报错信息，再展示正确代码）：

  场景 1：模型在 GPU，输入数据在 CPU
  场景 2：两个参与运算的 tensor 在不同 GPU 上（cuda:0 和 cuda:1）
  场景 3：loss 计算中引入 CPU tensor（如自定义权重）
  场景 4：optimizer.step() 后尝试将 GPU tensor 和 CPU tensor 比较
  场景 5：DataLoader 加载的数据（CPU）直接与 GPU 上的模型参数做运算

  对每种场景：
    1. 打印场景描述
    2. 用 try/except 捕获 RuntimeError，打印错误信息（截取前 200 字符）
    3. 打印正确的修复代码

建议步骤：
  1. 定义 5 个函数，每个函数对应一个错误场景
  2. 每个函数内部：
     a. 打印 "=== 场景 N: 描述 ==="
     b. 打印 "--- 错误代码 ---"
     c. try: 错误代码 except RuntimeError as e: print(str(e)[:200])
     d. 打印 "--- 正确代码 ---"
     e. 执行正确代码（验证无报错）
  3. 主函数中依次调用 5 个函数

提示：
  - 场景 2 需要至少 2 张 GPU，如果只有 1 张或没有 GPU，则需要判断并提示
  - 如果 CUDA 不可用，打印 "需要 CUDA 环境" 并退出
  - 使用 torch.cuda.device_count() 检查 GPU 数量
"""

# TODO: 导入 torch, torch.nn


# TODO: 定义一个简单的模型类用于演示


# TODO: 实现 demo_scene_1() —— 模型在 GPU，输入在 CPU


# TODO: 实现 demo_scene_2() —— 两个 tensor 在不同 GPU（需要 device_count >= 2）


# TODO: 实现 demo_scene_3() —— loss 计算引入 CPU tensor


# TODO: 实现 demo_scene_4() —— 将 GPU tensor 和 CPU tensor 比较


# TODO: 实现 demo_scene_5() —— DataLoader 数据与 GPU 模型参数混用


# TODO: 主函数：依次调用 5 个函数
