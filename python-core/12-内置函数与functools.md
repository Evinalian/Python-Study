# 第12章 内置函数与 functools

## 学习目标
- 掌握最常用的内置函数：sorted、enumerate、zip、any/all
- 理解 map 和 filter 的用法及与推导式的选择
- 掌握 sorted 的 key 参数进阶用法
- 了解 getattr、setattr 等反射函数
- 使用 functools.lru_cache 做函数结果缓存
- 使用 functools.partial 简化函数调用
- 了解 reduce、wraps、singledispatch 的用途

## 前置知识
学过 Python 函数、列表、字典、装饰器基础。

---

## 1. 内置函数分层讲解

Python 的内置函数有 70 多个，不需要全背。按使用频率分三层：

### 第一层：每天都会用到的

**sorted**、**enumerate**、**zip**、**any** / **all**、**len**、**range**、**print**、**input**、**type**、**isinstance**

### 第二层：进阶但很重要

**map**、**filter**、**reversed**、**sum**、**max** / **min**、**abs**、**round**

### 第三层：元编程 / 反射

**getattr**、**setattr**、**hasattr**、**dir**、**id**、**callable**

本章重点讲解第一层和第二层中最有技巧性的几个。

---

## 2. sorted：key 参数的逐步演进

`sorted()` 返回一个**新列表**（原数据不变），`key` 参数决定了"按什么规则排序"。

### 2.1 阶段一：key=len —— 按长度排序

```python
words = ["Python", "Go", "Rust", "C", "JavaScript"]

# 默认：按字母顺序排
print(sorted(words))
# ['C', 'Go', 'JavaScript', 'Python', 'Rust']

# 按字符串长度排（key=len）
print(sorted(words, key=len))
# ['C', 'Go', 'Rust', 'Python', 'JavaScript']
```

**`key=len` 做了什么？** Python 对每个元素调用 `len(element)`，然后按返回值排序。注意：`len` 是一个函数——`key` 参数需要一个**函数**。

### 2.2 阶段二：key=lambda —— 自定义排序规则

当内置函数不够用时，用 `lambda` 写一个简短的匿名函数：

```python
students = [
    ("张三", 85),
    ("李四", 92),
    ("王五", 85),
    ("赵六", 78),
]

# 按分数排序（取元组的第二个元素）
by_score = sorted(students, key=lambda x: x[1])
print(by_score)
# [('赵六', 78), ('张三', 85), ('王五', 85), ('李四', 92)]

# 按分数降序，分数相同按姓名升序（元组 key 实现多级排序）
by_score_desc = sorted(students, key=lambda x: (-x[1], x[0]))
print(by_score_desc)
# [('李四', 92), ('王五', 85), ('张三', 85), ('赵六', 78)]
```

**lambda x: -x[1] 是啥？** `lambda` 就是"一个没有名字的小函数"。`lambda x: -x[1]` 等价于 `def f(x): return -x[1]`，取负号是为了让大数排在前面（实现降序，或者直接用 `reverse=True`）。

### 2.3 阶段三：itemgetter —— 更专业的方式

```python
from operator import itemgetter, attrgetter

students = [("张三", 85), ("李四", 92), ("王五", 85)]

# itemgetter(1) 等价于 lambda x: x[1]，但更快（C 实现）
print(sorted(students, key=itemgetter(1)))
# [('张三', 85), ('王五', 85), ('李四', 92)]

# attrgetter：用于对象属性排序
class Student:
    def __init__(self, name, score):
        self.name = name
        self.score = score
    def __repr__(self):
        return f"Student({self.name}, {self.score})"

objs = [Student("张三", 85), Student("李四", 92), Student("王五", 78)]
print(sorted(objs, key=attrgetter("score"), reverse=True))
# [Student(李四, 92), Student(张三, 85), Student(王五, 78)]
```

**`itemgetter` / `attrgetter` 何时用？** 当你的 lambda 写的是 `lambda x: x[n]` 或 `lambda x: x.attr` 时，换成 itemgetter/attrgetter 更简洁且执行更快。

---

## 3. enumerate 和 zip

### 3.1 enumerate —— 带编号的遍历

```python
tasks = ["写需求文档", "编码实现", "单元测试", "代码审查"]

# 传统写法：手动维护一个计数器
for i in range(len(tasks)):
    print(f"{i + 1}. {tasks[i]}")

# enumerate 写法：自动生成序号
for i, task in enumerate(tasks, start=1):   # start=1 表示从 1 开始编号
    print(f"{i}. {task}")

# 输出:
# 1. 写需求文档
# 2. 编码实现
# 3. 单元测试
# 4. 代码审查
```

### 3.2 zip —— 并行遍历

```python
names = ["张三", "李四", "王五"]
scores = [85, 92, 78]

# 同时遍历两个列表
for name, score in zip(names, scores):
    print(f"{name}: {score}")
# 张三: 85
# 李四: 92
# 王五: 78

# 快捷用法：dict(zip(keys, values)) 快速创建字典
name_score_dict = dict(zip(names, scores))
print(name_score_dict)    # {'张三': 85, '李四': 92, '王五': 78}
```

**如果两个列表长度不一样呢？** `zip` 以最短的为准，多余的元素被忽略。如果想要以最长的为准，用 `itertools.zip_longest`。

---

## 4. any 和 all：批量判断

```python
# any: 只要有一个是 True，结果就是 True
print(any([False, False, True]))    # True
print(any([]))                       # False（没有真值）

# all: 所有都是 True，结果才是 True
print(all([True, True, False]))     # False
print(all([]))                       # True（没有假值）

# 最常见的用法：配合生成器表达式
scores = [85, 92, 78, 88, 95]

# 有没有人不及格？
print(any(s < 60 for s in scores))     # False —— 没有

# 所有人都及格了吗？
print(all(s >= 60 for s in scores))    # True —— 是的

# 存在至少一个大于 90 的学霸？
print(any(s > 90 for s in scores))     # True
```

---

## 5. map 和 filter

这两个函数是函数式编程的风格，但在现代 Python 中推导式通常更受欢迎。

```python
nums = [1, 2, 3, 4, 5]

# === map：对每个元素应用函数 ===
# map 写法
squared = map(lambda x: x ** 2, nums)
print(list(squared))     # [1, 4, 9, 16, 25]

# 推导式写法（更推荐）
squared = [x ** 2 for x in nums]

# === filter：保留满足条件的元素 ===
# filter 写法
evens = filter(lambda x: x % 2 == 0, nums)
print(list(evens))       # [2, 4]

# 推导式写法（更推荐）
evens = [x for x in nums if x % 2 == 0]

# filter(None, ...) 过滤掉假值（0, 空字符串, None, [] 等）
items = [0, 1, "", "hello", None, [], "world"]
truthy = list(filter(None, items))
print(truthy)    # [1, 'hello', 'world']
```

**何时用 map/filter？** 当你要传入的函数已经定义好了（比如 `map(str.upper, words)`），用 map/filter 比推导式更简洁。否则推导式更易读。

**重要提醒**：`map` 和 `filter` 返回的是**迭代器**，不是列表。如果要用下标访问，需要先 `list(...)` 转换。

---

## 6. 反射函数：getattr / setattr / hasattr

这些函数让你在运行时动态地操作对象的属性——不需要事先知道属性名。

```python
class Config:
    host = "localhost"
    port = 8080

# 从字符串动态获取属性
field = "host"
value = getattr(Config, field)
print(value)    # localhost

# 属性不存在时给默认值
print(getattr(Config, "debug", False))   # False（没有 debug 属性，返回默认值）

# 动态设置属性
setattr(Config, "timeout", 30)           # 等价于 Config.timeout = 30
print(Config.timeout)                    # 30

# 检查属性是否存在
print(hasattr(Config, "port"))           # True
print(hasattr(Config, "password"))       # False
```

**什么时候用？** 当你从配置文件、数据库或 API 拿到属性名（字符串）时，用这些函数动态访问。不用写一堆 if-else。

---

## 7. functools 精选

### 7.1 lru_cache —— 函数结果缓存

`@lru_cache` 自动记住函数的输入和输出，相同输入直接返回缓存结果，不再重复计算。用斐波那契递归演示：

```python
from functools import lru_cache
import time

# === 无缓存版本：O(2^n)，极慢 ===
def fib_no_cache(n):
    if n < 2:
        return n
    return fib_no_cache(n - 1) + fib_no_cache(n - 2)

# === 有缓存版本：O(n)，瞬间出结果 ===
@lru_cache(maxsize=None)    # maxsize=None 表示不限缓存数量
def fib_with_cache(n):
    if n < 2:
        return n
    return fib_with_cache(n - 1) + fib_with_cache(n - 2)


# 对比测试
n = 35

start = time.perf_counter()
result = fib_with_cache(n)
print(f"有缓存: fib({n}) = {result}, 耗时: {time.perf_counter() - start:.6f}s")

# 查看缓存统计
info = fib_with_cache.cache_info()
print(f"缓存命中: {info.hits} 次, 实际计算: {info.misses} 次")

# 无缓存版本对于 n=35 已经非常慢了，这里不做计时对比
# 你可以试试 fib_no_cache(35) —— 可能要等好几秒到几十秒！
```

**理解**：斐波那契递归无缓存时是"指数爆炸"——每个 fib(n) 又调用两个 fib(n-1) 和 fib(n-2)，重复计算无数遍。加上 `@lru_cache` 后，每个 n 的 fib(n) 只算一次，后续直接从缓存取。时间复杂度从 O(2^n) 降到 O(n)。

**适用条件**：函数必须是"纯函数"（相同输入永远返回相同输出），且参数必须可哈希（不能是列表、字典等可变类型）。

### 7.2 partial —— 固定部分参数

场景：你有一个函数参数很多，每次调用都要重复传同样的值：

```python
import json

# 每次都要写 ensure_ascii=False 和 indent=2，手酸
data = {"name": "小明", "age": 18}
print(json.dumps(data, ensure_ascii=False, indent=2))

# 用 partial 创建一个"记住"了默认参数的新函数
from functools import partial

json_pretty = partial(json.dumps, ensure_ascii=False, indent=2)

# 现在只需要传第一个参数
print(json_pretty({"name": "小明", "age": 18}))
print(json_pretty({"hobby": "编程", "level": 3}))
```

**`partial` 做了什么？** `partial(json.dumps, ensure_ascii=False, indent=2)` 创建了一个新函数，这个新函数在调用 `json.dumps` 时自动带上 `ensure_ascii=False, indent=2`。

更多场景：

```python
from functools import partial

# 固定指数，创建平方函数和立方函数
def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))    # 25
print(cube(5))      # 125

# 固定排序 key
by_score = partial(sorted, key=lambda x: x[1], reverse=True)
students = [("张三", 85), ("李四", 92), ("王五", 78)]
print(by_score(students))
# [('李四', 92), ('张三', 85), ('王五', 78)]
```

### 7.3 reduce —— 累积运算

`reduce` 将序列中的元素两两合并，最终聚合成一个值：

```python
from functools import reduce

# 连乘（阶乘）
numbers = [1, 2, 3, 4, 5]
product = reduce(lambda a, b: a * b, numbers)
print(product)    # 120 (= 1*2*3*4*5)

# 等价于: ((((1 * 2) * 3) * 4) * 5)

# 连加（带初始值）
total = reduce(lambda a, b: a + b, numbers, 100)
print(total)      # 115 (= 100 + 1 + 2 + 3 + 4 + 5)
```

大多数情况下 `sum()`、`math.prod()` 等专用函数比 `reduce` 更清晰，但 `reduce` 适合自定义聚合逻辑。

### 7.4 其他速查

```python
from functools import wraps, singledispatch, total_ordering

# wraps: 写装饰器时保持原函数的 __name__ 和 __doc__
# singledispatch: 根据第一个参数类型分派——Python 版的"函数重载"
# total_ordering: 只需定义 __eq__ 和 __lt__，自动生成其他比较方法
```

---

## 8. 基础练习

### 练习 1：用 sorted 对列表按不同规则排序

**题目**：有如下字符串列表，分别按长度排序、按最后一个字母排序、按字母倒序排列。

```python
words = ["banana", "apple", "cherry", "date", "elderberry", "fig"]
```

参考答案：

```python
# 按长度排序
print(sorted(words, key=len))
# ['fig', 'date', 'apple', 'banana', 'cherry', 'elderberry']

# 按最后一个字母排序
print(sorted(words, key=lambda w: w[-1]))
# ['banana', 'apple', 'date', 'elderberry', 'fig', 'cherry']

# 按字母倒序
print(sorted(words, reverse=True))
# ['fig', 'elderberry', 'date', 'cherry', 'banana', 'apple']
```

### 练习 2：用 enumerate 给列表编号输出

**题目**：对一个水果列表，用 enumerate 输出"第 X 个水果是: YYY"。

```python
fruits = ["苹果", "香蕉", "樱桃", "榴莲", "葡萄"]
for i, fruit in enumerate(fruits, start=1):
    print(f"第 {i} 个水果是: {fruit}")
# 第 1 个水果是: 苹果
# 第 2 个水果是: 香蕉
# ...
```

### 练习 3：用 zip 合并两个列表为字典

**题目**：给定两个列表，用 `dict(zip(...))` 将它们合并为字典。

```python
keys = ["name", "age", "city"]
values = ["小明", 20, "北京"]

info = dict(zip(keys, values))
print(info)    # {'name': '小明', 'age': 20, 'city': '北京'}
```

### 练习 4：用 any/all 判断列表条件

**题目**：对分数列表，判断是否全部及格（>=60）、是否存在优秀（>=90）、是否存在不及格（<60）。

```python
scores = [85, 92, 78, 55, 88, 95, 60]

print("全部及格?", all(s >= 60 for s in scores))     # False
print("存在优秀?", any(s >= 90 for s in scores))     # True
print("存在不及格?", any(s < 60 for s in scores))    # True
```

---

## 9. 进阶练习

### 练习 5：用 lru_cache 优化递归

**题目**：写一个函数 `climb_stairs(n)` 计算爬楼梯的方法数（每次可以爬 1 级或 2 级，f(1)=1, f(2)=2, f(n)=f(n-1)+f(n-2)）。先用无缓存版本，再加 `@lru_cache` 对比。

参考答案：

```python
from functools import lru_cache
import time

# 无缓存版本：与斐波那契一样是指数级复杂度
def climb_stairs_slow(n):
    if n <= 2:
        return n
    return climb_stairs_slow(n - 1) + climb_stairs_slow(n - 2)

# 有缓存版本
@lru_cache(maxsize=None)
def climb_stairs(n):
    if n <= 2:
        return n
    return climb_stairs(n - 1) + climb_stairs(n - 2)


# 测试有缓存版本
n = 50
start = time.perf_counter()
result = climb_stairs(n)
print(f"climb_stairs({n}) = {result}, 耗时: {time.perf_counter() - start:.6f}s")
print(f"缓存信息: {climb_stairs.cache_info()}")
# 无缓存版本对 n=50 几乎不可计算
```

### 练习 6：用 partial 简化回调函数

**题目**：有一个日志函数 `log(level, message)`，用 `partial` 创建 `log_info`、`log_error` 两个专用版本，分别固定了 level 为 "INFO" 和 "ERROR"。

参考答案：

```python
from functools import partial
from datetime import datetime

def log(level, message):
    """通用日志函数"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

# 用 partial 创建专用日志函数
log_info = partial(log, "INFO")      # 固定第一个参数 level = "INFO"
log_error = partial(log, "ERROR")    # 固定第一个参数 level = "ERROR"
log_warning = partial(log, "WARNING")

# 使用：调用时只需要传 message
log_info("程序启动")
log_warning("磁盘空间已使用 80%")
log_error("数据库连接失败，请检查网络")

# 输出示例:
# [14:30:22] [INFO] 程序启动
# [14:30:22] [WARNING] 磁盘空间已使用 80%
# [14:30:22] [ERROR] 数据库连接失败，请检查网络
```

---

## 10. 常见错误

### 错误 1：用 type() 做类型检查

```python
class Animal:
    pass
class Dog(Animal):
    pass

dog = Dog()

# 错误：type 不识别继承关系
if type(dog) == Animal:     # False！Dog 是 Animal 的子类
    print("是 Animal")

# 正确：用 isinstance
if isinstance(dog, Animal):   # True —— 考虑继承链
    print("是 Animal")
```

### 错误 2：map/filter 返回迭代器而不是列表

```python
result = map(str.upper, ["a", "b", "c"])
print(result)           # <map object at ...> —— 不是列表！
# print(result[0])      # TypeError: 'map' object is not subscriptable

# 需要索引：先转换
result = list(map(str.upper, ["a", "b", "c"]))
print(result[0])        # 'A'
```

### 错误 3：lru_cache 用在方法上可能导致内存泄漏

```python
class Calculator:
    @lru_cache(maxsize=128)
    def compute(self, n):    # self 也被缓存了
        return n ** 2

# 问题：如果创建大量 Calculator 实例并很快丢弃，
# 每个实例的 self 被缓存在 lru_cache 中，阻止了垃圾回收
```

**避免方法**：对于实例方法，避免使用 `@lru_cache`，或者将缓存逻辑提到模块级别的纯函数。

### 错误 4：partial 的位置参数顺序

```python
from functools import partial

def divide(a, b):
    return a / b

# 固定第一个位置参数 a = 2
half = partial(divide, 2)
print(half(10))   # divide(2, 10) = 0.2 —— 不是 divide(10, 2)!

# 要固定第二个参数，用关键字参数
inverse = partial(divide, b=2)
print(inverse(10))   # divide(10, 2) = 5.0
```

### 错误 5：sorted 的 key 返回值不可比较

```python
# 错误：key 返回了一个字典，字典不能比较大小
# sorted(data, key=lambda x: {"value": x[1]})
# TypeError: '<' not supported between instances of 'dict' and 'dict'

# key 函数的返回值必须支持 < 运算符
```

---

## 11. 本章小结

| 主题 | 核心要点 |
|---|---|
| sorted + key | `key=len`、`key=lambda x: x[1]`、`key=itemgetter(1)` 逐步演进 |
| enumerate | `enumerate(seq, start=0)` 同时取索引和值 |
| zip | `zip(a, b)` 并行遍历；`dict(zip(keys, values))` 快速创建字典 |
| any / all | 配合生成器表达式做批量条件判断 |
| map / filter | 函数式风格；返回迭代器需 list() 转换；推导式通常更优先 |
| lru_cache | 自动缓存函数结果；斐波那契从 O(2^n) 降到 O(n)；参数必须可哈希 |
| partial | 固定函数部分参数，生成专用版本；场景如固定日志级别、固定 JSON 选项 |
| reduce | 累积运算；多数场景 sum/prod 等专用函数更直观 |
| getattr/setattr | 运行时动态操作属性；根据字符串访问/修改属性 |
| isinstance vs type | 始终用 isinstance（支持继承）；type 不考虑子类关系 |
