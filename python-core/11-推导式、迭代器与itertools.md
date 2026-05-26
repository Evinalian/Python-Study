# 第11章 推导式、迭代器与 itertools

## 学习目标
- 熟练使用列表/字典/集合推导式替代传统 for 循环
- 理解生成器表达式的惰性求值和内存优势
- 掌握迭代器协议（`__iter__` / `__next__`）并能自定义迭代器
- 使用 itertools 中常用工具（chain、accumulate、groupby、product、combinations、islice）
- 理解 `yield from` 委托子生成器

## 前置知识
学过 for 循环、列表、字典、函数基础、生成器基本概念。

---

## 1. 列表推导式：从 4 行到 1 行

### 1.1 传统写法 vs 推导式

```python
# 任务：生成 1 到 5 每个数的平方

# === 传统 for 循环写法（4 行）===
squares = []
for n in range(1, 6):
    squares.append(n ** 2)
print(squares)    # [1, 4, 9, 16, 25]

# === 列表推导式写法（1 行）===
squares = [n ** 2 for n in range(1, 6)]
print(squares)    # [1, 4, 9, 16, 25]
```

**解读**：`[n ** 2 for n in range(1, 6)]` 可以读成"对于 1 到 5 中的每个 n，计算 n 的平方，放进列表"。推导式把"做什么"（`n ** 2`）和"遍历什么"（`for n in ...`）写在一行里，紧凑直观。

### 1.2 加过滤条件

```python
# 任务：取出 1-20 中的所有偶数，算平方

# === 传统写法 ===
result = []
for n in range(1, 21):
    if n % 2 == 0:           # 先判断
        result.append(n ** 2)  # 再计算

# === 推导式 ===
result = [n ** 2 for n in range(1, 21) if n % 2 == 0]
print(result)
# [4, 16, 36, 64, 100, 144, 196, 256, 324, 400]
```

**`if` 在末尾 = 过滤**：不符合条件的元素直接被跳过，不在结果中出现。

### 1.3 if-else 在表达式位置 = 映射

```python
numbers = [-3, -1, 0, 2, 5, -8, 7]

# if 在末尾（过滤，结果元素少）
positives = [n for n in numbers if n > 0]
print(positives)    # [2, 5, 7] —— 只有 3 个元素

# if-else 在表达式位置（映射，结果元素不变）
abs_values = [n if n >= 0 else -n for n in numbers]
print(abs_values)   # [3, 1, 0, 2, 5, 8, 7] —— 7 个元素，负数变正数
```

**记忆口诀**：`if` 在右边是过滤（丢掉不想要的）；`if-else` 在左边是映射（每个都保留，值可能变）。

**Java 对比**：Java 没有列表推导式，最接近的是 Stream API（`.filter().map().collect()`），相对冗长。

---

## 2. 字典推导式和集合推导式

### 2.1 字典推导式

```python
# 把列表转成字典：单词 -> 长度
words = ["go", "python", "rust", "javascript"]
word_len = {w: len(w) for w in words}
print(word_len)    # {'go': 2, 'python': 6, 'rust': 4, 'javascript': 10}

# 键值互换
d = {"a": 1, "b": 2, "c": 3}
swapped = {v: k for k, v in d.items()}
print(swapped)     # {1: 'a', 2: 'b', 3: 'c'}

# 过滤：只保留分数 >= 80 的学生
scores = {"小明": 85, "小红": 92, "小刚": 67, "小丽": 78}
passed = {k: v for k, v in scores.items() if v >= 80}
print(passed)      # {'小明': 85, '小红': 92}
```

### 2.2 集合推导式

```python
# 自动去重
names = ["Alice", "Bob", "alice", "Charlie", "ALICE"]
unique_lower = {n.lower() for n in names}
print(unique_lower)    # {'alice', 'bob', 'charlie'} —— 不区分大小写去重

# 取所有单词的长度集合
word_lengths = {len(w) for w in ["hello", "world", "python", "code"]}
print(word_lengths)    # {4, 5, 6} —— 自动去重
```

**三种推导式的语法就是换了个括号**：列表 `[...]`、字典 `{k:v ...}`、集合 `{...}`。

---

## 3. 生成器表达式：用内存证明它的价值

### 3.1 语法：把方括号换成圆括号

```python
# 列表推导式：一次性生成 1000 万个平方数——内存爆炸
# squares_list = [x ** 2 for x in range(10_000_000)]  # 不要运行！会吃光内存

# 生成器表达式：不立即计算，只在迭代时逐个产出
squares_gen = (x ** 2 for x in range(10_000_000))
print(squares_gen)    # <generator object <genexpr> at 0x...> —— 几乎不占内存
```

### 3.2 用 sys.getsizeof 证明内存差异

```python
import sys

# 列表推导式：所有结果都存入内存
list_comp = [x ** 2 for x in range(10000)]
print(f"列表推导式占用: {sys.getsizeof(list_comp)} 字节")

# 生成器表达式：只有生成器对象本身
gen_expr = (x ** 2 for x in range(10000))
print(f"生成器表达式占用: {sys.getsizeof(gen_expr)} 字节")

# 典型输出（具体数值因平台而异）:
# 列表推导式占用: 87616 字节
# 生成器表达式占用: 200 字节
# 差距：400 多倍！
```

**为什么生成器这么省内存？** 列表推导式是一次性把所有结果算出来存好。生成器表达式是一个"配方"——它记住了怎么算，但还没算。只有当你迭代它的时候（比如 `for x in gen_expr`），它才一个一个地算、一个一个地给。

### 3.3 函数参数中的语法糖

当生成器表达式是函数的**唯一参数**时，可以省略外层的圆括号：

```python
# 完整写法
total = sum((x ** 2 for x in range(100)))

# 省略圆括号（更常见）
total = sum(x ** 2 for x in range(100))
print(total)    # 328350

# 其他例子
any(n > 100 for n in [10, 50, 200])    # True（有一个 > 100）
all(n > 0 for n in [1, 2, 3, 4])       # True（全部 > 0）
max(len(w) for w in "hello world python".split())  # 6
```

### 3.4 生成器只能用一次

```python
gen = (x ** 2 for x in range(3))
print(list(gen))    # [0, 1, 4]
print(list(gen))    # [] —— 生成器已经耗尽了，没有东西了

# 需要重复使用？先转成列表再操作
values = list(x ** 2 for x in range(3))
print(values)       # [0, 1, 4] —— 可以反复访问
```

---

## 4. 迭代器协议：for 循环的底层

### 4.1 协议定义

迭代器协议只需两个方法：

```python
class MyIterator:
    def __iter__(self):
        """返回迭代器自身（让迭代器也可以放进 for 循环）"""
        return self

    def __next__(self):
        """返回下一个元素；没元素了抛 StopIteration"""
        raise StopIteration
```

### 4.2 一个完整示例：倒数迭代器

```python
class Countdown:
    """从 start 倒数到 0 的迭代器"""

    def __init__(self, start):
        self._current = start    # 记录当前数到哪了

    def __iter__(self):
        return self              # 迭代器自身就是迭代器

    def __next__(self):
        if self._current < 0:
            raise StopIteration  # 没东西了，通知 for 循环停止
        value = self._current    # 记住当前值
        self._current -= 1       # 移到下一个数
        return value             # 返回当前值


# 使用——放进 for 循环
for n in Countdown(3):
    print(n, end=" ")
# 输出: 3 2 1 0
```

### 4.3 for 循环底层等价代码

```python
# 你写的：
for item in container:
    print(item)

# Python 内部等价于：
iterator = iter(container)          # 1. 获取迭代器（调用 container.__iter__()）
while True:
    try:
        item = next(iterator)       # 2. 每次调用 __next__() 拿下一个
        print(item)
    except StopIteration:           # 3. 迭代器耗尽，退出循环
        break
```

**关键概念区分**：
- **可迭代对象**（Iterable）：有 `__iter__` 方法，可以放进 for 循环。如 list、str、dict、range
- **迭代器**（Iterator）：同时有 `__iter__` 和 `__next__` 方法。如生成器对象、`iter([])` 的返回值
- 所有迭代器都是可迭代的，反过来不一定

**Java 对比**：Python 的迭代器只需 `__next__` 一个方法，结束时抛 `StopIteration`。Java 需要 `hasNext()` + `next()` 两个方法，且没有异常机制来标识结束。

---

## 5. itertools：5 个最常用的工具

### 5.1 chain —— 串联多个序列

```python
from itertools import chain

# 把多个列表/元组连成一个迭代器
letters = list(chain("ABC", "DEF", "GHI"))
print(letters)    # ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

# 展平一层嵌套列表
nested = [[1, 2], [3, 4], [5, 6]]
flat = list(chain.from_iterable(nested))
print(flat)       # [1, 2, 3, 4, 5, 6]
```

### 5.2 accumulate —— 累积运算

```python
from itertools import accumulate
import operator

nums = [1, 2, 3, 4, 5]

# 默认：累加（前缀和）
print(list(accumulate(nums)))    # [1, 3, 6, 10, 15]

# 累积乘（阶乘）
print(list(accumulate(nums, operator.mul)))    # [1, 2, 6, 24, 120]

# 运行最大值
print(list(accumulate([5, 3, 8, 1, 6], max)))  # [5, 5, 8, 8, 8]
```

### 5.3 groupby —— 分组（先排序！）

```python
from itertools import groupby

# 注意：groupby 只对连续的相同 key 分组，所以必须先排序！
records = [
    ("技术部", "张三"),
    ("技术部", "李四"),
    ("市场部", "王五"),
    ("市场部", "赵六"),
]

# 已经按部门排好序了，可以直接 groupby
for dept, members in groupby(records, key=lambda x: x[0]):
    names = [m[1] for m in members]   # members 是一个迭代器
    print(f"{dept}: {names}")

# 输出:
# 技术部: ['张三', '李四']
# 市场部: ['王五', '赵六']
```

### 5.4 product —— 笛卡尔积（多层循环）

```python
from itertools import product

# 两层循环的替代品
for a, b in product([1, 2], "xy"):
    print(f"({a}, {b})", end=" ")
# (1, x) (1, y) (2, x) (2, y)

# repeat 参数：product(A, repeat=2) 等价于 product(A, A)
print(list(product("AB", repeat=2)))
# [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]
```

### 5.5 combinations 和 permutations —— 组合与排列

```python
from itertools import combinations, permutations

# 组合（顺序无关）：从 ABC 中选 2 个
print(list(combinations("ABC", 2)))
# [('A', 'B'), ('A', 'C'), ('B', 'C')]

# 排列（顺序有关）：从 ABC 中选 2 个
print(list(permutations("ABC", 2)))
# [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]
```

### 5.6 islice —— 惰性切片

```python
from itertools import islice

# 常规切片：list(gen)[5:15] 会把生成器全部展开再切片——浪费内存
# islice：只取需要的部分，不展开其余的

gen = (x ** 2 for x in range(1000))
first_5 = list(islice(gen, 5))
print(first_5)    # [0, 1, 4, 9, 16]

gen = (x ** 2 for x in range(1000))
middle = list(islice(gen, 10, 15))    # 跳过前 10 个，取 11-15
print(middle)     # [100, 121, 144, 169, 196]
```

**其他 itertools 工具速查**：

| 函数 | 作用 | 示例 |
|---|---|---|
| `count(start, step)` | 无限递增 | `count(10, 2)` -> 10, 12, 14... |
| `cycle(seq)` | 无限循环 | `cycle("AB")` -> A, B, A, B... |
| `repeat(x, n)` | 重复 n 次 | `repeat("hi", 3)` -> hi, hi, hi |
| `compress(data, selectors)` | 按布尔掩码过滤 | `compress("ABC", [1,0,1])` -> A, C |
| `takewhile(pred, seq)` | 取到条件为假 | `takewhile(lambda x: x<5, [1,4,6,2])` -> 1, 4 |
| `dropwhile(pred, seq)` | 跳过直到条件为假 | `dropwhile(lambda x: x<5, [1,4,6,2])` -> 6, 2 |
| `pairwise(seq)` | 相邻成对 | `pairwise("ABC")` -> ('A','B'), ('B','C') |
| `tee(seq, n)` | 复制 n 份 | `a, b = tee(gen, 2)` |

---

## 6. 基础练习

### 练习 1：用列表推导式生成 1-100 的完全平方数

**题目**：生成 1 到 100 之间所有完全平方数（即某个整数的平方）的列表。

参考答案：

```python
# 方法：找出 1 到 100 间所有满足 sqrt(n) 是整数的 n
import math

squares = [n for n in range(1, 101) if math.isqrt(n) ** 2 == n]
print(squares)
# [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# 或者更直接：生成 1 到 10 的平方
squares2 = [i ** 2 for i in range(1, 11)]
print(squares2)
# [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
```

### 练习 2：用字典推导式把列表转成字典

**题目**：给定一个水果列表，用字典推导式生成 `{水果名: 名称长度}` 的字典。

参考答案：

```python
fruits = ["apple", "banana", "cherry", "date", "elderberry"]

fruit_lengths = {fruit: len(fruit) for fruit in fruits}
print(fruit_lengths)
# {'apple': 5, 'banana': 6, 'cherry': 6, 'date': 4, 'elderberry': 10}

# 进阶：只保留长度 >= 5 的水果
long_fruits = {fruit: len(fruit) for fruit in fruits if len(fruit) >= 5}
print(long_fruits)
# {'apple': 5, 'banana': 6, 'cherry': 6, 'elderberry': 10}
```

### 练习 3：写一个生成器函数产生前 n 个偶数

**题目**：写一个生成器函数 `even_numbers(n)`，产生前 n 个偶数（0, 2, 4, ...）。

参考答案：

```python
def even_numbers(n):
    """生成前 n 个偶数"""
    for i in range(n):
        yield i * 2    # yield 让这个函数变成生成器


# 迭代生成器
for num in even_numbers(5):
    print(num, end=" ")
# 输出: 0 2 4 6 8

# 或者直接转列表
print(list(even_numbers(10)))
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```

---

## 7. 进阶练习

### 练习 4：用 itertools 实现滑动窗口

**题目**：写一个生成器函数 `sliding_window(seq, n)`，返回长度为 n 的滑动窗口。例如 `sliding_window([1,2,3,4], 2)` 产生 `[1,2]`, `[2,3]`, `[3,4]`。

参考答案：

```python
from itertools import islice

def sliding_window(seq, n):
    """惰性滑动窗口生成器"""
    it = iter(seq)
    # 取前 n 个元素作为第一个窗口
    window = list(islice(it, n))
    if len(window) < n:
        return    # 数据不够一个窗口，直接结束
    yield tuple(window)
    # 后续每次：弹出第一个，追加下一个
    for item in it:
        window.pop(0)
        window.append(item)
        yield tuple(window)


# 测试
print(list(sliding_window([1, 2, 3, 4, 5], 3)))
# [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

print(list(sliding_window("ABCDEF", 2)))
# [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F')]
```

### 练习 5：用生成器管道处理大文件

**题目**：写一个生成器管道，逐行读取一个大文件，过滤掉空行和注释行（以 # 开头），然后把每行转成大写。

参考答案：

```python
def read_lines(filepath):
    """生成器 1：逐行读取文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")   # 去掉末尾换行符


def filter_lines(lines):
    """生成器 2：过滤空行和注释行"""
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            yield line


def transform_lines(lines):
    """生成器 3：转大写"""
    for line in lines:
        yield line.upper()


# 管道组合
filepath = "sample.txt"

# 先创建一个测试文件
with open(filepath, "w", encoding="utf-8") as f:
    f.write("hello world\n")
    f.write("# 这是一行注释\n")
    f.write("\n")
    f.write("python is great\n")
    f.write("# 另一行注释\n")
    f.write("the end\n")

# 运行管道
pipeline = transform_lines(filter_lines(read_lines(filepath)))
for processed_line in pipeline:
    print(processed_line)

# 输出:
# HELLO WORLD
# PYTHON IS GREAT
# THE END
```

---

## 8. 常见错误

### 错误 1：混淆 if 的位置

```python
nums = [-3, -1, 0, 2, 5]

# 语法错误：if-else 不能放在末尾
# result = [n * 2 for n in nums if n > 0 else n * 3]   # SyntaxError!

# 正确：if 在末尾是过滤；if-else 在表达式位置是映射
result = [n * 2 if n > 0 else n * 3 for n in nums]
print(result)   # [-9, -3, 0, 4, 10]
```

### 错误 2：groupby 未先排序

```python
from itertools import groupby

data = [("A", 1), ("B", 2), ("A", 3)]    # 注意两个 "A" 不相邻

# 错误：直接 groupby
for key, group in groupby(data, key=lambda x: x[0]):
    print(key, list(group))
# A [('A', 1)]
# B [('B', 2)]
# A [('A', 3)]       ← "A" 被分成了两组！

# 正确：先排序
data.sort(key=lambda x: x[0])
for key, group in groupby(data, key=lambda x: x[0]):
    print(key, list(group))
# A [('A', 1), ('A', 3)]
# B [('B', 2)]
```

### 错误 3：生成器被消耗两次

```python
gen = (x for x in range(3))
print(sum(gen))     # 3 —— 生成器耗尽了
print(sum(gen))     # 0 —— 生成器为空！

# 解决方案：需要多次使用就转成列表
data = list(x for x in range(3))
print(sum(data))    # 3
print(sum(data))    # 3 —— 没问题
```

### 错误 4：推导式中做副作用

```python
# 不要这样写——推导式是用来创建数据集合的，不是用来执行操作的
[print(x) for x in range(5)]    # 虽然能工作，但不地道

# 用普通 for 循环
for x in range(5):
    print(x)
```

---

## 9. 本章小结

| 概念 | 核心要点 |
|---|---|
| 列表推导式 | `[expr for x in seq if cond]`；if 在右过滤，if-else 在左映射 |
| 字典推导式 | `{k: v for ...}`；换了个括号，用法一致 |
| 集合推导式 | `{expr for ...}`；自动去重 |
| 生成器表达式 | `(...)` 惰性求值，不占内存；作为唯一参数时括号可省略；只能遍历一次 |
| 迭代器协议 | `__iter__` 返回自身 + `__next__` 返回元素/抛 StopIteration；for 循环底层实现 |
| itertools 精选 | chain（串联）、accumulate（累积）、groupby（分组需先排序）、product（笛卡尔积）、combinations/permutations（组合排列）、islice（惰性切片） |
| 生成器管道 | 多个生成器串联处理数据流，内存恒定，适合大文件 |
