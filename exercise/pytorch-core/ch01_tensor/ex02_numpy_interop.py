"""
练习 1-2: NumPy 与 Tensor 互转与内存共享

任务：
1. 创建一个 NumPy 数组 arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
2. 用 torch.from_numpy() 转为 Tensor t_shared（共享内存）
3. 修改 t_shared 中某个元素的值，打印 arr 验证 arr 是否同步变化
4. 用 torch.tensor() 深拷贝一份 t_copy（不共享内存），修改 t_copy 中的值，打印 arr 验证 arr 是否变化
5. 创建一个 Tensor，通过 .numpy() 转为 NumPy 数组，修改 Tensor 验证 NumPy 是否同步变化

预期观察结果：
- 步骤 3: arr 的值会随 t_shared 变化（共享内存）
- 步骤 4: arr 的值不会随 t_copy 变化（深拷贝）
- 步骤 5: NumPy 的值会随 Tensor 变化（共享内存）
"""

import torch
import numpy as np


# TODO: 1. 创建 NumPy 数组 arr


# TODO: 2. 用 torch.from_numpy() 创建共享内存的 Tensor


# TODO: 3. 修改 Tensor 中的值，打印 arr 验证共享内存


# TODO: 4. 用 torch.tensor() 深拷贝，修改后验证 arr 不受影响


# TODO: 5. 从 Tensor 转 NumPy（.numpy()），验证共享内存


