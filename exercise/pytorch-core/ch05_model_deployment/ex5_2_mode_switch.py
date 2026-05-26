"""
练习 5-2：train/eval 模式切换的影响

目标：
  创建一个包含 Dropout 和 BatchNorm 的模型。
  分别用 model.train() 和 model.eval() 对同一批数据推理两次，
  观察 Dropout 产生的随机性以及 BatchNorm 统计量的变化。

要求：
  1. 定义模型：Linear -> BatchNorm1d -> ReLU -> Dropout(p=0.5) -> Linear
  2. 对同一批输入数据做多次前向传播
  3. 在 train() 模式下：每次前向传播结果不同（Dropout 随机），bn.running_mean 会更新
  4. 在 eval() 模式下：每次前向传播结果完全相同（Dropout 关闭），bn.running_mean 不更新
  5. 打印每轮输出以及前后的 running_mean 值，验证上述行为

建议步骤：
  1. 定义含 BatchNorm 和 Dropout 的模型
  2. 设置随机种子，生成一批输入数据
  3. 使用 model.train()，对同一批数据连续 forward 3 次，打印每次输出和 running_mean
  4. 使用 model.eval()，对同一批数据连续 forward 3 次，打印每次输出和 running_mean
  5. 计算 train 模式下两次输出之间的差异（应 > 0），eval 模式下两次输出之间的差异（应 == 0）

提示：
  - print(f"running_mean: {model.bn.running_mean}") 可以观察 BatchNorm 统计量
  - torch.equal(a, b) 可以判断两个张量是否完全相同
  - 注意 Dropout 在 train 模式下对元素值的缩放（除以 1-p）
"""

# TODO: 导入 torch, torch.nn


# TODO: 定义模型 ModeDemoModel（Linear -> BN1d -> ReLU -> Dropout(0.5) -> Linear）


# TODO: 固定随机种子 torch.manual_seed(42)


# TODO: 生成一批输入数据 x = torch.randn(8, 16)


# TODO: 切换到 train() 模式，连续 forward 3 次同一批数据，打印输出和 running_mean


# TODO: 计算 train 模式下两两输出之间的差异（L2 距离）


# TODO: 切换到 eval() 模式，连续 forward 3 次同一批数据，打印输出和 running_mean


# TODO: 计算 eval 模式下两两输出之间的差异（应为 0）


# TODO: 打印总结：train 模式下的差异（应 > 0），eval 模式下的差异（应 == 0）
