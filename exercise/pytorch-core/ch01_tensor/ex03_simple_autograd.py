"""
练习 1-3: 简单 autograd 求偏导数

任务：
对函数 f(x, y) = x^3 + 3 * x^2 * y + y^2

1. 手算偏导数：
   - df/dx = 3x^2 + 6xy
   - df/dy = 3x^2 + 2y

2. 在点 (x=2, y=3) 处：
   - 手算结果: df/dx = 3*4 + 6*2*3 = 12 + 36 = 48
   - 手算结果: df/dy = 3*4 + 2*3 = 12 + 6 = 18

3. 用 PyTorch autograd 验证：
   - 创建 x=2.0, y=3.0，均设置 requires_grad=True
   - 构建 f 的计算表达式
   - 调用 f.backward()
   - 打印 x.grad 和 y.grad，与手算结果对比

提示：
- 确保 f 最终是标量（它应该自然就是），否则 .backward() 会报错
"""

import torch


# TODO: 1. 创建 x=2.0, y=3.0，均 requires_grad=True


# TODO: 2. 构建 f(x, y) = x^3 + 3*x^2*y + y^2


# TODO: 3. 调用 .backward()


# TODO: 4. 打印 x.grad 和 y.grad，验证与手算结果一致


# TODO: 5. 手算验证（在注释中写出计算过程）


