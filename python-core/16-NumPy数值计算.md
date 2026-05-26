# 第16章 NumPy 数值计算

> NumPy 是 Python 科学计算和 AI 的基石。它用 C 语言实现数组运算，速度比 Python 原生列表快几十到上百倍。理解 NumPy 的 ndarray 是学习 Pandas、PyTorch 和 TensorFlow 的前提。

---

## 学习目标

- 理解为什么 NumPy 比 Python 列表快（C 层向量化 vs Python 逐元素解释）
- 掌握创建数组的多种方式（从列表、零值、序列、随机数）
- 理解索引与切片中的"视图"概念（修改切片会影响原数组）
- 理解广播机制的逐步推导（标量→一维→二维）
- 掌握 axis 参数的含义（沿哪个轴操作，就消除哪个轴）
- 能使用布尔索引和花式索引筛选数据
- 使用 ufunc 和聚合函数进行向量化计算

---

## 前置知识

- Python 列表与切片操作
- 基本的数学概念（均值、标准差、矩阵）

---

## 1. 为什么需要 NumPy

### 1.1 Python 列表的瓶颈

假设你需要计算 100 万个数的平方。用纯 Python 列表：

```python
import time

# 纯 Python：用 for 循环逐元素计算
data = list(range(1_000_000))

start = time.time()
squared = [x ** 2 for x in data]   # 列表推导式
elapsed = time.time() - start
print(f"纯 Python 耗时: {elapsed:.3f} 秒")
# 典型输出: 纯 Python 耗时: 0.120 秒
```

这对 Python 来说已经很快了（用了列表推导式，比普通 for 循环要快）。但对比 NumPy：

```python
import numpy as np
import time

data = np.arange(1_000_000)        # NumPy 数组

start = time.time()
squared = data ** 2                 # 一行，向量化运算
elapsed = time.time() - start
print(f"NumPy 耗时: {elapsed:.3f} 秒")
# 典型输出: NumPy 耗时: 0.003 秒  —— 快了约 40 倍
```

**为什么 NumPy 这么快？** 三个原因：

1. **C 语言实现**：NumPy 的核心运算在编译好的 C 代码中执行，没有 Python 解释器介入。
2. **连续内存**：ndarray 在内存中是一片连续区域（C 数组），CPU 缓存友好。Python 列表存的是对象指针，每个元素可能散落在内存各处。
3. **同类型**：ndarray 中所有元素类型相同，运算时不需要像 Python 列表那样每次检查"这个元素是什么类型"。

**类比 C/Java**：NumPy 的 ndarray 更像 C 的数组（`int arr[1000000]`）——连续内存、同类型、无装箱开销。Python 列表则像 Java 的 `ArrayList<Object>`——每个元素都是独立对象。

### 1.2 导入 NumPy

```python
import numpy as np   # np 是社区的通用别名，几乎所有代码都这么写
```

---

## 2. 创建 ndarray

### 2.1 从 Python 列表创建

```python
import numpy as np

# 一维数组
arr1d = np.array([1, 2, 3, 4, 5])
print(arr1d)
# [1 2 3 4 5]

# 二维数组（矩阵）
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2d)
# [[1 2 3]
#  [4 5 6]]

# 查看基本属性
print(arr2d.ndim)      # 2 —— 维度数（几维数组）
print(arr2d.shape)     # (2, 3) —— 各维的长度（2 行, 3 列）
print(arr2d.size)      # 6 —— 总元素数
print(arr2d.dtype)     # int64（或 int32，取决于系统）—— 元素类型
```

### 2.2 创建固定值数组

```python
# 全 0
print(np.zeros((3, 4)))
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# 全 1
print(np.ones((2, 3)))
# [[1. 1. 1.]
#  [1. 1. 1.]]

# 全指定值
print(np.full((2, 2), 3.14))
# [[3.14 3.14]
#  [3.14 3.14]]

# 单位矩阵
print(np.eye(3))
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

### 2.3 创建序列数组

```python
# arange：类似 Python 的 range
print(np.arange(5))          # [0 1 2 3 4]
print(np.arange(2, 10, 3))   # [2 5 8]  —— 起始=2, 步长=3

# linspace：均匀间隔（含终点）
print(np.linspace(0, 1, 5))
# [0.   0.25 0.5  0.75 1.  ]  —— 在 [0, 1] 之间均匀取 5 个点

print(np.linspace(0, 1, 5, endpoint=False))
# [0.  0.2 0.4 0.6 0.8]  —— 不包含终点 1.0
```

### 2.4 创建随机数组

```python
rng = np.random.default_rng(seed=42)   # 新版推荐方式

# [0, 1) 均匀分布的随机数
print(rng.random(5))
# [0.77395605 0.43887844 0.85859792 0.69736803 0.09417735]

# 指定形状
arr = rng.random((2, 3))
print(arr.shape)  # (2, 3)

# 正态分布：均值=0, 标准差=1
samples = rng.normal(0, 1, 1000)
print(f"均值: {samples.mean():.3f}, 标准差: {samples.std():.3f}")
# 均值约为 0, 标准差约为 1

# 随机整数
print(rng.integers(1, 101, 10))  # 1-100 之间 10 个随机整数
# 例如: [87 44 82 73 20 63 59 94 36 71]
```

**注意**：不再使用 `np.random.seed()` 和 `np.random.rand()` 等旧 API。新 API（`default_rng`）更安全，特别是在多线程环境中。

---

## 3. 基本运算：向量化

### 3.1 逐元素运算

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# 四则运算——逐元素执行
print(a + b)    # [11 22 33 44]
print(a - b)    # [-9 -18 -27 -36]
print(a * b)    # [10 40 90 160]  —— 注意：这是逐元素乘，不是矩阵乘法！
print(a / b)    # [0.1 0.1 0.1 0.1]

# 与标量运算
print(a + 10)   # [11 12 13 14]
print(a * 2)    # [2 4 6 8]
print(a ** 2)   # [1 4 9 16]
```

**重要区分**：`*` 是逐元素乘，`@` 或 `np.dot()` 才是矩阵乘法：

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A * B)    # [[ 5 12]   逐元素乘
                #  [21 32]]

print(A @ B)    # [[19 22]   矩阵乘法
                #  [43 50]]
```

学过线性代数的同学：NumPy 的 `*` 不是矩阵乘法。想算矩阵乘，请用 `@`。

### 3.2 数学函数（ufunc）

NumPy 提供了大量数学函数，它们都在 C 层执行，速度极快：

```python
x = np.array([0, np.pi/2, np.pi])

# 三角函数
print(np.sin(x))    # [0. 1. 0.]
print(np.cos(x))    # [1. 0. -1.]

# 指数和对数
print(np.exp([1, 2, 3]))        # [ 2.718 7.389 20.086]
print(np.log([1, np.e, np.e**2]))  # [0. 1. 2.]

# 取整
print(np.floor([1.2, 2.8, -1.5]))  # [ 1.  2. -2.]
print(np.ceil([1.2, 2.8, -1.5]))   # [ 2.  3. -1.]
print(np.round([1.2, 2.8, -1.5]))  # [ 1.  3. -2.]
```

---

## 4. 索引与切片：小心"视图"陷阱

### 4.1 基本切片

```python
arr = np.array([10, 20, 30, 40, 50])

# 切片语法和 Python 列表一样
print(arr[1:4])      # [20 30 40]
print(arr[:3])       # [10 20 30]
print(arr[::2])      # [10 30 50]  —— 每隔 1 个取
```

### 4.2 关键区别：NumPy 切片是视图！

**这是 NumPy 和 Python 列表最重要的区别之一**——NumPy 的切片返回的是**视图（view）**而非副本：

```python
# NumPy：切片是视图——修改切片会影响原数组
arr = np.array([10, 20, 30, 40, 50])
sub = arr[1:4]           # [20, 30, 40]
sub[0] = 999             # 修改切片
print(arr)               # [10 999 30 40 50]  ← 原数组被修改了！

# Python 列表：切片是副本——修改切片不影响原列表
py_list = [10, 20, 30, 40, 50]
py_sub = py_list[1:4]
py_sub[0] = 999
print(py_list)           # [10, 20, 30, 40, 50]  ← 原列表不变
```

**为什么会这样？** Python 列表的切片总是创建新列表（分配新内存）。NumPy 的切片只是为了效率——大数组上创建副本很昂贵，所以默认返回"指向原数组同一块内存的视图"。

**如何安全地创建独立副本？** 显式调用 `.copy()`：

```python
arr = np.array([10, 20, 30, 40, 50])
sub_copy = arr[1:4].copy()   # 显式创建副本
sub_copy[0] = 999
print(arr)                    # [10 20 30 40 50]  ← 原数组不变
```

**注意**：花式索引（用列表或数组作为索引）返回的是**副本**，不是视图。

### 4.3 二维数组的切片

```python
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# 单独一行: arr2d[行索引, :]
print(arr2d[0, :])      # [1 2 3]  —— 第 0 行

# 单独一列: arr2d[:, 列索引]
print(arr2d[:, 1])      # [2 5 8]  —— 第 1 列

# 子矩阵
print(arr2d[:2, 1:])    # [[2 3]
                        #  [5 6]]  —— 前 2 行 × 后 2 列
```

### 4.4 布尔索引：用条件筛选

```python
scores = np.array([85, 92, 58, 76, 95, 61])

# 生成布尔掩码
mask = scores >= 60
print(mask)  # [True True False True True True]

# 用掩码筛选
print(scores[mask])                  # [85 92 76 95]  —— 及格的分数

# 组合条件（注意：用 & 和 |，不是 and 和 or！）
print(scores[(scores >= 60) & (scores < 90)])  # [85 76]  —— 60-89 分
print(scores[(scores < 60) | (scores > 90)])  # [58 92 95] —— 不及格或 >90
```

**为什么用 `&` 而不是 `and`？** `and` 是 Python 关键字，不能被 NumPy 重载。`&` 被 NumPy 重载为逐元素的"按位与"，在布尔数组上等价于"逻辑与"。

---

## 5. 形状变换

```python
arr = np.arange(12)
print(arr)  # [0 1 2 3 4 5 6 7 8 9 10 11]

# reshape：改变形状，总元素数必须一致
print(arr.reshape(3, 4))
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# -1 表示"自动推导此维长度"
print(arr.reshape(3, -1))     # (3, 4) —— -1 自动算出 12/3=4
print(arr.reshape(2, 2, -1))  # (2, 2, 3) —— -1 自动算出 12/(2*2)=3

# ravel()：展平为一维（返回视图）
flat = arr2d.ravel()  # 从 (2,3) 变回 (6,)
flat[0] = 999
print(arr2d)           # [[999 2 3] [4 5 6]] —— 视图，原数组被修改

# flatten()：展平为一维（返回副本）
flat_copy = arr2d.flatten()  # 安全副本，不影响原数组
```

---

## 6. 广播机制

广播是 NumPy 最强大的特性之一：不同形状的数组进行运算时，NumPy 会自动扩展维度，无需你手动复制数据。

### 6.1 最简单的情况：数组和标量

```python
arr = np.array([1, 2, 3, 4])

# 标量 10 被"广播"成和 arr 一样的形状 [10, 10, 10, 10]，然后逐元素相加
print(arr + 10)  # [11 12 13 14]

# 实际上是 arr + np.array([10, 10, 10, 10])
# 但 NumPy 不会真的创建那个 [10,10,10,10] 的数组——它在运算时"虚拟"地重复标量值
```

这很直观，你在写普通算式时也是这么想的。但广播还能处理更复杂的情况。

### 6.2 一维数组和二维数组

```python
# 矩阵：3 行 × 4 列
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])
print(f"matrix 形状: {matrix.shape}")  # (3, 4)

# 行向量：1 行 × 4 列
row = np.array([10, 20, 30, 40])
print(f"row 形状: {row.shape}")        # (4,)

# 广播：(4,) 自动扩展为 (1, 4)，再扩展到 (3, 4)
print(matrix + row)
# [[11 22 33 44]
#  [15 26 37 48]
#  [19 30 41 52]]
# row 被"广播"到每一行——相当于每行都加了 [10, 20, 30, 40]
```

### 6.3 列向量广播

```python
# 列向量：3 行 × 1 列
col = np.array([[100], [200], [300]])
print(f"col 形状: {col.shape}")        # (3, 1)

# 广播：(3, 1) 扩展到 (3, 4)
print(matrix + col)
# [[101 102 103 104]
#  [205 206 207 208]
#  [309 310 311 312]]
# col 被广播到每一列
```

### 6.4 广播规则总结

从后往前（最内维到最外维）逐维比较：

1. 如果两个维度相等 → 兼容
2. 如果其中一个维度是 1 → 可以扩展为另一个维度的大小
3. 否则 → 广播失败，报错

```python
# 示例：(3, 4) 和 (4,) —— 从后往前比
# 轴 1:  4 == 4  ✓
# 轴 0:  3 vs (无) → (4,) 被看作 (1, 4)，轴 0 为 1 可扩展  ✓
# 结果: (3, 4)

# 不兼容的例子
a = np.ones((3, 4))
b = np.ones((5,))
# a + b  → ValueError！
# 从后往前: 轴 1 上 4 != 5，且都不是 1 → 无法广播
```

---

## 7. 聚合统计与 axis 参数

### 7.1 基本聚合函数

```python
rng = np.random.default_rng(42)
scores = rng.normal(70, 15, 1000)  # 均值 70，标准差 15，1000 个成绩

print(f"总和: {scores.sum():.2f}")
print(f"均值: {scores.mean():.2f}")
print(f"标准差: {scores.std():.2f}")
print(f"最小值: {scores.min():.2f}")
print(f"最大值: {scores.max():.2f}")
print(f"中位数: {np.median(scores):.2f}")
print(f"25% 分位: {np.percentile(scores, 25):.2f}")
print(f"75% 分位: {np.percentile(scores, 75):.2f}")
```

### 7.2 axis 参数：沿哪个方向聚合

对于二维数组（矩阵），聚合函数可以沿行或沿列操作。`axis` 参数的含义是 **"消灭哪个轴"**：

用下面这个 2x3 矩阵来理解：

```
      列0  列1  列2
行0  [ 1    2    3  ]
行1  [ 4    5    6  ]

shape: (2, 3)
        轴0  轴1
```

- **`axis=0`**：沿行方向操作 → 对每**列**的元素进行聚合 → 结果消灭行维度 → shape 变成 `(3,)`
- **`axis=1`**：沿列方向操作 → 对每**行**的元素进行聚合 → 结果消灭列维度 → shape 变成 `(2,)`

```python
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])

# axis=0：沿着行方向（上下方向）求和 → 每列的和
print(arr2d.sum(axis=0))  # [5 7 9]
# 计算过程: 列0: 1+4=5, 列1: 2+5=7, 列2: 3+6=9

# axis=1：沿着列方向（左右方向）求和 → 每行的和
print(arr2d.sum(axis=1))  # [6 15]
# 计算过程: 行0: 1+2+3=6, 行1: 4+5+6=15

# 验证形状变化
print(arr2d.sum(axis=0).shape)  # (3,) —— 原来的轴 0（行数 2）被消灭了
print(arr2d.sum(axis=1).shape)  # (2,) —— 原来的轴 1（列数 3）被消灭了
```

**口诀**：`axis=0` 是"压扁行"（结果等于各列的和）；`axis=1` 是"压扁列"（结果等于各行的和）。`axis=-1` 永远表示最后一个轴。

---

## 8. 常用操作速查

### 8.1 条件选择：np.where

```python
arr = np.array([15, 8, 22, 3, 41, 9])

# 类似三目运算：条件为 True 取 X，False 取 Y
result = np.where(arr > 10, "高", "低")
print(result)  # ['高' '低' '高' '低' '高' '低']

# 不带替换值时返回满足条件的索引
indices = np.where(arr > 10)
print(indices)  # (array([0, 2, 4]),)
print(arr[indices])  # [15 22 41]
```

### 8.2 排序

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# 返回排序后的新数组（原数组不变）
print(np.sort(arr))     # [1 1 2 3 4 5 6 9]

# argsort：返回"排序后索引"——排序后第 i 个位置的元素在原数组中的索引
idx = np.argsort(arr)
print(idx)              # [1 3 6 0 2 4 7 5]
print(arr[idx])         # [1 1 2 3 4 5 6 9] —— 等价于排序结果

# 原地排序：arr.sort() —— 修改原数组
```

### 8.3 矩阵运算

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵乘法
print(A @ B)              # [[19 22] [43 50]]

# 转置
print(A.T)                # [[1 3] [2 4]]

# 逆矩阵
print(np.linalg.inv(A))   # [[-2.   1. ] [ 1.5 -0.5]]

# 行列式
print(np.linalg.det(A))   # -2.0
```

---

## 9. 常见错误

### 错误 1：忘记切片是视图

```python
# 错误：以为切片是独立的
a = np.array([1, 2, 3, 4])
b = a[:2]
b[0] = 99
print(a)  # [99 2 3 4]  —— 原数组被修改了！

# 正确：需要独立副本时显式 .copy()
a = np.array([1, 2, 3, 4])
b = a[:2].copy()
b[0] = 99
print(a)  # [1 2 3 4]  —— 原数组不变
```

### 错误 2：布尔条件用 `and`/`or` 而不是 `&`/`|`

```python
# 错误：Python 的 and/or 不能用于数组
arr[(arr > 0) and (arr < 5)]   # ValueError: 数组的真值二义性

# 正确：用 & 和 |（注意加括号！每个条件都要用括号包起来）
arr[(arr > 0) & (arr < 5)]
```

### 错误 3：混淆 `*`（逐元素乘）和 `@`（矩阵乘）

```python
A = np.eye(2)   # 2x2 单位矩阵
B = np.eye(2)

# 在这个特例中碰巧结果看起来一样，但语义不同：
# A * B → 逐元素乘
# A @ B → 矩阵乘法
```

### 错误 4：广播规则不满足时报错

```python
a = np.ones((3, 4))
b = np.ones((3,))   # (3,) 不是 (3, 1)！

# a + b  → ValueError: operands could not be broadcast together
# 从后往前: 轴1 上 4 vs (无) → (3,) 被看作 (1,3)，但 4 != 3

# 修复：显式 reshape 为列向量
b = b[:, np.newaxis]  # 变成 (3, 1)
# 现在 a + b  → (3, 4) + (3, 1) → 广播成功
```

---

## 10. 练习题

### 基础练习

**练习 1：创建不同类型的数组**

创建以下数组：
1. 一个从 10 到 49 的一维整数数组
2. 一个 3x4 的全零浮点数组
3. 一个 2x3 的 `[0, 1)` 均匀分布随机数组

<details>
<summary>参考答案（点击展开）</summary>

```python
import numpy as np

# 1. 从 10 到 49
arr1 = np.arange(10, 50)
print(arr1[:5])   # [10 11 12 13 14]

# 2. 3x4 全零浮点
arr2 = np.zeros((3, 4))
print(arr2)
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# 3. 2x3 随机数组
rng = np.random.default_rng(42)
arr3 = rng.random((2, 3))
print(arr3)
```
</details>

**练习 2：对数组进行基本数学运算**

创建数组 `a = np.array([1, 2, 3, 4])` 和 `b = np.array([10, 20, 30, 40])`，计算：
1. `a + b`
2. `a * b`
3. `a ** 2`
4. `b / a`

<details>
<summary>参考答案（点击展开）</summary>

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(a + b)    # [11 22 33 44]
print(a * b)    # [10 40 90 160]
print(a ** 2)   # [1 4 9 16]
print(b / a)    # [10. 10. 10. 10.]  —— 自动转为浮点
```
</details>

**练习 3：计算数组的统计量**

创建 100 个均值为 70、标准差为 15 的正态分布随机数，计算总和、均值、标准差、最小值、最大值、中位数。

<details>
<summary>参考答案（点击展开）</summary>

```python
import numpy as np

rng = np.random.default_rng(42)
scores = rng.normal(70, 15, 100)

print(f"总和: {scores.sum():.2f}")
print(f"均值: {scores.mean():.2f}")
print(f"标准差: {scores.std():.2f}")
print(f"最小值: {scores.min():.2f}")
print(f"最大值: {scores.max():.2f}")
print(f"中位数: {np.median(scores):.2f}")
```
</details>

**练习 4：用布尔索引筛选数组元素**

从 `np.array([85, 92, 58, 76, 95, 61, 43, 78, 88, 69])` 中筛选出：
1. 所有及格的成绩（>= 60）
2. 所有优秀的成绩（>= 85）

<details>
<summary>参考答案（点击展开）</summary>

```python
import numpy as np

scores = np.array([85, 92, 58, 76, 95, 61, 43, 78, 88, 69])

passed = scores[scores >= 60]
print(f"及格: {passed}")          # [85 92 76 95 61 78 88 69]

excellent = scores[scores >= 85]
print(f"优秀: {excellent}")       # [85 92 95 88]
```
</details>

### 进阶练习

**练习 5：生成随机数据并找出离群值**

生成 1000 个均值为 50、标准差为 10 的正态分布随机数。然后：
1. 计算第 5、25、50、75、95 百分位数
2. 用 IQR（四分位距）方法找出离群值：小于 Q1 - 1.5*IQR 或大于 Q3 + 1.5*IQR 的值
3. 输出离群值的数量和具体值

<details>
<summary>参考答案（点击展开）</summary>

```python
import numpy as np

# 生成数据
rng = np.random.default_rng(42)
data = rng.normal(50, 10, 1000)

# 1. 百分位数
percentiles = np.percentile(data, [5, 25, 50, 75, 95])
print("百分位数:")
for p, v in zip([5, 25, 50, 75, 95], percentiles):
    print(f"  {p:>2d}%: {v:.2f}")

# 2. IQR 方法找离群值
Q1 = percentiles[1]   # 第 25 百分位
Q3 = percentiles[3]   # 第 75 百分位
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nQ1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}")
print(f"下界 = {lower_bound:.2f}, 上界 = {upper_bound:.2f}")

# 3. 筛选离群值
outliers = data[(data < lower_bound) | (data > upper_bound)]
print(f"\n离群值数量: {len(outliers)}")
print(f"离群值: {np.sort(outliers)}")
```
</details>

---

## 11. 本章小结

| 概念 | 要点 |
|------|------|
| ndarray vs list | C 连续内存、同类型、向量化运算——速度快几十倍 |
| 切片是视图 | NumPy 切片指向原数组内存，修改会影响原数组；需要副本请 `.copy()` |
| 广播 | 从后往前逐维对齐，维度相等或为 1 才兼容；标量自动广播到数组形状 |
| axis | `axis=0` 沿行方向操作（消灭行维），`axis=1` 沿列方向操作（消灭列维） |
| 布尔索引 | `arr[arr > 10]` 按条件筛选；组合条件用 `&` 和 `|`（不是 `and`/`or`） |
| 矩阵乘 | `@` 是矩阵乘法，`*` 是逐元素乘法 |
| ufunc | C 层实现的向量化函数，比 Python 循环快 1-2 个数量级 |
