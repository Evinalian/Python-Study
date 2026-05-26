"""
练习 5-3：TorchScript 导出 —— script vs trace

目标：
  定义一个含控制流（if/else 分支）的简单模型，分别用 torch.jit.script 和 torch.jit.trace 导出。
  用不同的输入验证 trace 的局限性（只记录一条执行路径）。

要求：
  1. 定义一个模型 ControlFlowNet，forward 中根据 x.sum() 的正负选择不同分支：
     - 如果 sum > 0：x = x * 2 + 1
     - 如果 sum <= 0：x = x * (-1) + 5
  2. 用 torch.jit.script 导出，保存为 scripted_model.pt
  3. 用 torch.jit.trace 导出（使用 sum > 0 的输入追踪），保存为 traced_model.pt
  4. 准备两个测试输入：
     - test_input_a: sum > 0（应触发分支 A）
     - test_input_b: sum < 0（应触发分支 B）
  5. 分别用原始模型、scripted 模型、traced 模型对两个测试输入做推理
  6. 对比输出，验证：
     - scripted 模型对所有输入都正确
     - traced 模型对 test_input_b 输出错误（因为它只记录了分支 A 的逻辑）

建议步骤：
  1. 定义 ControlFlowNet 类
  2. 创建模型实例，设为 eval 模式
  3. 准备追踪用输入 trace_input = torch.tensor([1.0, 2.0, 3.0])（sum > 0）
  4. 用 torch.jit.script 导出并保存
  5. 用 torch.jit.trace 导出并保存
  6. 准备测试输入 test_a = torch.tensor([1.0, 2.0])（sum > 0）、test_b = torch.tensor([-5.0, -3.0])（sum < 0）
  7. 对三个模型（原始、scripted、traced）分别用 test_a 和 test_b 做推理
  8. 格式化打印对比结果
  9. 输出结论：说明 trace 不适用于含控制流的模型

提示：
  - traced_model.code 可以查看 trace 生成的图代码
  - scripted_model.code 可以查看 script 生成的代码
  - 比较输出时使用 torch.allclose 或直接打印数值
"""

# TODO: 导入 torch, torch.nn


# TODO: 定义 ControlFlowNet(nn.Module)，forward 含 if/else 分支


# TODO: 创建模型实例并设置为 eval 模式


# TODO: 准备 trace 用输入 trace_input（sum > 0）


# TODO: 使用 torch.jit.script 导出，保存为 "scripted_model.pt"


# TODO: 使用 torch.jit.trace 导出，保存为 "traced_model.pt"


# TODO: 准备测试输入 test_a（sum > 0）和 test_b（sum < 0）


# TODO: 分别用原始模型、scripted 模型、traced 模型对 test_a 和 test_b 做推理


# TODO: 打印对比结果表格（模型类型、输入、输出）


# TODO: 输出结论
