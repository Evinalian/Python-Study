# 第1章 认识 Python 与变量

## 学习目标

- 理解 Python 程序从源码到执行的完整流程（解释器、字节码、PVM）
- 掌握"变量是对象引用"这一核心语义，能对比 Java/C 的变量模型
- 使用 `id()` 和 `is` 判断对象身份，区分引用相等与值相等
- 理解可变与不可变对象的内存行为差异
- 掌握 int、float、bool、None 的基本用法与常见陷阱
- 了解浮点数精度问题的成因与解决方案

## 前置知识

- 熟悉至少一门编程语言（C/Java）：变量、基本类型、编译运行流程
- 了解内存的基本概念（栈、堆）

---

## 1. Python 程序是怎么运行的

### 1.1 从源码到执行——三步流程

当你输入 `python hello.py` 并回车，背后发生了三件事：

```
源码(.py) ──编译──▶ 字节码(.pyc) ──执行──▶ PVM（Python 虚拟机）
```

**第一步：编译成字节码**

CPython（Python 的官方实现）首先把你的 `.py` 文件编译成字节码（bytecode）。字节码是一种介于"人类可读的源码"和"CPU 可直接执行的机器码"之间的中间表示。你可以把它理解为 Python 虚拟机的"汇编语言"。

```python
# hello.py
x = 10
y = 20
print(x + y)
```

上面这个文件被编译后，会产生类似这样的字节码指令（用 `dis` 模块可以查看）：

```
  1           0 LOAD_CONST               0 (10)    # 把常量 10 推到栈顶
              2 STORE_NAME               0 (x)     # 把栈顶的值存到变量 x

  2           4 LOAD_CONST               1 (20)    # 把常量 20 推到栈顶
              6 STORE_NAME               1 (y)     # 把栈顶的值存到变量 y

  3           8 LOAD_NAME                2 (print) # 加载 print 函数
             10 LOAD_NAME                0 (x)     # 加载 x 的值
             12 LOAD_NAME                1 (y)     # 加载 y 的值
             14 BINARY_ADD                         # 执行加法
             16 CALL_FUNCTION            1         # 调用 print
```

对学过 Java 的同学来说，这个过程类似于 `javac` 把 `.java` 编译成 `.class`（字节码）。区别在于 Python 的编译是**隐式的**——你不需要手动运行一个编译命令，解释器自动处理。

**第二步：PVM 逐条执行**

编译得到的字节码文件（`.pyc`）被交给 **Python 虚拟机（PVM, Python Virtual Machine）** 逐条执行。PVM 是一个"软件 CPU"——它读一条字节码指令，执行一条，再读下一条。

**第三步（CPython 特有）：没有 JIT**

CPython 的 PVM 纯粹是解释执行，不会像 Java 的 JVM 那样在运行时把热点代码编译成机器码（JIT, Just-In-Time Compilation）。不过 Python 生态系统中有其他实现提供了 JIT：

| 实现 | 特点 |
|------|------|
| CPython | 官方实现，最广泛使用，纯解释执行 |
| PyPy | 内置 JIT，长时间运行的计算密集型代码可快 4-10 倍 |
| Jython | 运行在 JVM 上，可与 Java 代码互操作 |

对于 AI/数据科学场景，计算密集部分通常由 C/CUDA 底层库（NumPy、PyTorch）完成，CPython 负责调度和组合逻辑，解释执行的性能瓶颈不显著。

### 1.2 动态类型——类型在对象上，不在变量上

如果你学过 C 或 Java，你习惯的是**静态类型**：变量的类型在声明时就确定了，之后不能改变。

```c
// C 语言
int x = 42;      // x 的类型是 int，编译时确定
// x = "hello";  // 编译错误！类型不兼容
```

```java
// Java
int x = 42;      // 同上
String s = "hi"; // s 的类型是 String
// s = 42;       // 编译错误！
```

Python 则完全不同——变量没有类型，类型附着在**对象**上：

```python
x = 42           # x 指向一个 int 对象
x = "hello"      # 现在 x 指向一个 str 对象——完全合法
x = [1, 2, 3]    # 现在 x 又指向一个 list 对象——依然合法
```

这被形象地称为**鸭子类型**（Duck Typing）："如果它走起来像鸭子，叫起来像鸭子，那么它就是鸭子。"翻译成代码就是——我不关心你是什么类型，我只关心你能不能做我需要的事。

```python
def double(x):
    return x + x          # 不要求 x 是 int，只要 x 支持 + 即可

print(double(3))          # 6        —— int 支持 +
print(double("abc"))      # "abcabc" —— str 支持 +（拼接）
print(double([1, 2]))     # [1, 2, 1, 2] —— list 支持 +（合并）
```

这带来了极大的灵活性，代价是类型错误要等到运行时才会暴露。

---

## 2. 变量：对象引用模型

### 2.1 Python 变量不是"盒子"，而是"标签"

这是 Python 与 C/Java **最核心的区别**，务必理解透彻。

**C 语言的视角**：变量是命名的内存存储单元。`int a = 5` 在栈上分配 4 字节，把值 5 写进去。变量 a **就是**那块内存。

**Java 的视角**：基本类型（int, double 等）和 C 一样；引用类型（String, List 等）的变量存储的是指向堆上对象的指针。变量 **存储** 指针。

**Python 的视角**：所有变量都是对堆上对象的**引用**（或者说"标签"）。变量本身不"包含"任何值——它只是"指向"某个存在于堆上的对象。

用一个简单的例子来感受：

```python
x = 10  # 这行代码做了什么？分两步：
        # 第1步：Python 在内存（堆）中创建一个 int 对象，值为 10。
        # 第2步：在当前的命名空间里创建名字 x，把它"贴"到这个 int 对象上。
        # 注意：x 本身不"包含"10——x 只是一个标签，指到了那个 int 对象上。
```

如果你学过 C 的指针，可以粗略地认为 Python 变量总是"自动解引用的指针"——你用变量名，就能访问它指向的对象，不需要手动 `*p`。

### 2.2 赋值不是复制

```python
a = [1, 2, 3]    # a 指向一个 list 对象 [1, 2, 3]
b = a             # b 指向 a 所指的同一个 list 对象！没有复制！

b.append(4)       # 通过 b 修改这个 list 对象
print(a)          # [1, 2, 3, 4] —— a 也被影响了！
print(id(a))      # 比如 2085934482112（CPython 中即内存地址）
print(id(b))      # 与上面完全相同
```

这个行为与 Java 的引用类型完全一致。你学 Java 时应该见过类似的代码：

```java
// Java
List<Integer> a = new ArrayList<>(Arrays.asList(1, 2, 3));
List<Integer> b = a;           // b 和 a 指向同一个 ArrayList 对象
b.add(4);
System.out.println(a);         // [1, 2, 3, 4]
```

但要记住：在 Python 中，**连整数都是对象引用**，不存在 Java 中 int 和 Integer 的区别。

---

## 3. id() 和 is —— 深入理解对象身份

### 3.1 基础概念

`id()` 函数返回对象的"身份证号"——在 CPython 中就是对象在内存中的地址。

`is` 运算符比较的是两个变量的 `id()` 是否相等，即它们是否指向**同一个对象**。

`==` 运算符比较的是两个对象的**值**是否相等（调用对象的 `__eq__` 方法）。

```python
a = [1, 2]
b = [1, 2]
c = a

print(a == b)   # True  —— 值相等
print(a is b)   # False —— 不是同一个对象（虽然长得一样）

print(a is c)   # True  —— c 就是 a，同一个对象
print(a == c)   # True  —— 值当然也相等
```

对比 Java 来理解：

| Python | Java | 含义 |
|--------|------|------|
| `a == b` | `a.equals(b)` | 值比较 |
| `a is b`（引用类型） | `a == b` | 引用（身份）比较 |

### 3.2 逐步深入：小整数缓存

CPython 为了性能，在解释器启动时就预先创建了 -5 到 256 之间的所有 int 对象。这会导致一些"反直觉"的现象：

```python
# 示例 1：小整数（-5 ~ 256），is 返回 True
a = 100
b = 100
print(a is b)   # True  —— CPython 复用了预创建的缓存对象

# 示例 2：大整数（超出 256），is 返回 False
a = 1000
b = 1000
print(a is b)   # False —— 每次创建新对象（但同一行字面量可能意外为 True）
```

字符串也有类似的 intern 机制——只包含字母、数字、下划线的短字符串会被自动 intern：

```python
s1 = "hello"
s2 = "hello"
print(s1 is s2)     # True —— 被 intern 了

s3 = "hello world!" # 含有空格和标点
s4 = "hello world!"
print(s3 is s4)     # 可能 False —— 不保证 intern
```

### 3.3 黄金法则

> **永远用 `==` 做值比较，用 `is` 只在与 `None` 比较时使用。**

```python
# 正确：与 None 比较必须用 is
if result is None:
    print("没有结果")

# 错误：用 == 比较 None（可能被自定义 __eq__ 干扰）
if result == None:   # 不推荐
    ...

# 判断列表是否为空——用真值测试，不要用 is 或 == []
if not my_list:      # 推荐
    print("列表为空")
if len(my_list) == 0:  # 也可以
    print("列表为空")
```

---

## 4. 可变对象 vs 不可变对象——亲手做实验

### 4.1 概念对照

| 类型 | 可变性 | 说明 |
|------|--------|------|
| `int`, `float`, `bool`, `str`, `tuple`, `frozenset` | 不可变 | 创建后内容不能修改，"修改"操作都返回新对象 |
| `list`, `dict`, `set`, `bytearray` | 可变 | 支持原地修改，不创建新对象 |

### 4.2 实验一：int 的"修改"（其实创建了新对象）

```python
x = 100
old_id = id(x)       # 记录 x 原来指向的对象的 id

x = x + 1            # 看起来是"把 100 改成 101"
                     # 实际上：计算 100+1 → 创建新 int 对象 101 → x 指向新对象

print(id(x) == old_id)  # False！x 已经指向不同的对象了
print(x)                 # 101
```

### 4.3 实验二：list 的修改（原地修改）

```python
lst = [1, 2, 3]
old_id = id(lst)     # 记录 lst 原来指向的对象的 id

lst.append(4)        # 在 list 末尾添加元素
print(lst)           # [1, 2, 3, 4]

print(id(lst) == old_id)  # True！还是同一个对象
```

### 4.4 实验三：+= 对不可变和可变的不同效果

```python
# 不可变类型（str）的 += —— 创建新对象
s = "hello"
old_id = id(s)
s += " world"              # 等价于 s = s + " world"，创建了新 str
print(id(s) == old_id)     # False

# 可变类型（list）的 += —— 原地修改
lst = [1, 2]
old_id = id(lst)
lst += [3]                 # 调用 __iadd__，原地扩展
print(id(lst) == old_id)   # True

# 注意区别：lst = lst + [3] 会创建新对象！
lst = [1, 2]
old_id = id(lst)
lst = lst + [3]            # 调用 __add__，创建新 list
print(id(lst) == old_id)   # False
```

### 4.5 为什么这很重要

如果你把列表传给一个函数，函数内部修改列表，外面的原始列表也会变：

```python
def add_item(lst):
    lst.append(99)     # 修改了 lst 指向的对象本身

my_list = [1, 2, 3]
add_item(my_list)
print(my_list)         # [1, 2, 3, 99] —— 外面也被改了！
```

这正是 Java 中引用类型的典型行为，但 Python 需要谨记：**所有类型都是引用语义**。

---

## 5. 数字类型详解

### 5.1 int：没有溢出的整数

Python 3 的 `int` 是**任意精度**整数——无论多大的数都能精确表示，只受可用内存限制。

```python
# Python int 完全不用担心溢出
big = 2 ** 1000      # 一个约 301 位的天文数字
print(big)           # 完整打印，不截断

# 对比 Java：
# int 最大约 21 亿（32 位有符号）
# long 最大约 9×10^18（64 位有符号）
# BigInteger 才支持任意精度（但很笨重）
```

Python 3 中 `int` 和 `long` 已经统一，你不需要像 Python 2 那样区分两者。

### 5.2 整除——向下取整，而非截断

这是从 C/Java 转过来的同学最容易踩的坑之一：

```python
# Python 的 // 是向下取整（floor），不是向零截断
print(7 // 3)    # 2    —— 2.33 向下取整得 2
print(-7 // 3)   # -3   —— 注意！-2.33 向下取整得 -3（不是 -2）

# Java/C 的整数除法是向零截断（truncate toward zero）
# 在 Java 中：-7 / 3 = -2（截掉小数部分）

# Python 的 % 保证：a = b * (a // b) + (a % b)
# 且余数与除数同号
print(-7 % 3)    # 2    —— 因为 -7 = (-3) * 3 + 2
```

### 5.3 浮点数精度——0.1 + 0.2 不等于 0.3

```python
print(0.1 + 0.2)            # 0.30000000000000004 —— 不是 0.3！
print(0.1 + 0.2 == 0.3)     # False！
```

**原因**：Python 的 `float` 使用 IEEE 754 双精度（64 位），底层是二进制。而 `0.1`、`0.2` 这样的十进制小数，在二进制中是**无限循环小数**，就像十进制中的 1/3 = 0.333... 一样。计算机只能存储有限位数，必然产生舍入误差。

**解决方案**：

```python
# 方案一：使用 round() 控制小数位数（适用于显示）
print(round(0.1 + 0.2, 10))  # 0.3

# 方案二：使用 math.isclose 做近似比较（适用于判断相等）
import math
print(math.isclose(0.1 + 0.2, 0.3))  # True

# 方案三：使用 Decimal 做精确十进制运算（适用于金额等场景）
from decimal import Decimal
print(Decimal("0.1") + Decimal("0.2"))  # 0.3 —— 精确！
# 注意：必须用字符串 "0.1" 构造 Decimal，用浮点数 0.1 传进去误差已经存在
```

### 5.4 round() 的银行家舍入

```python
# round() 使用"银行家舍入"（round half to even）
print(round(2.5))   # 2 —— 不是 3！因为 2 是偶数，向偶数舍入
print(round(3.5))   # 4 —— 3.5 向偶数 4 舍入
print(round(1.5))   # 2 —— 1.5 向偶数 2 舍入

# 这不是 bug——银行家舍入在大规模数据上能减少累积误差
```

### 5.5 bool 是 int 的子类

```python
print(True + 1)               # 2 —— True 就是 1
print(False + 1)              # 1 —— False 就是 0
print(isinstance(True, int))  # True —— bool 继承自 int！
print(issubclass(bool, int))  # True

# 利用这一点可以优雅地计数
conditions = [True, False, True, True]
print(sum(conditions))        # 3 —— 直接求和得到 True 的个数
```

---

## 6. 类型转换与真值测试

### 6.1 显式类型转换

Python **不会隐式转换类型**，必须显式调用：

```python
# int() —— 截断取整（不是四舍五入）
print(int(3.9))    # 3
print(int(-3.9))   # -3

# float() —— 转浮点
print(float(5))    # 5.0

# str() —— 转字符串
print(str(123))    # "123"

# bool() —— 转布尔
print(bool(1))     # True
print(bool(0))     # False
print(bool(""))    # False
print(bool("a"))   # True
print(bool([]))    # False
```

### 6.2 真值测试——什么算"真"，什么算"假"

在 `if x:` 这样的布尔上下文中，以下值被视为 **falsy（假）**，其余一切皆为 **truthy（真）**：

| 值 | 类别 |
|----|------|
| `False`, `None` | 内置假值 |
| `0`, `0.0`, `0j` | 零值 |
| `""`（空字符串）、`b""`（空字节） | 空序列 |
| `[]`, `()`, `{}`, `set()`, `range(0)` | 空容器 |

```python
if []:
    print("不会执行")    # 空列表是 falsy

if [0]:
    print("会执行")      # 非空列表是 truthy（即使里面是 0）

if " ":
    print("会执行")      # 含空格的字符串是 truthy

# None 的比较必须用 is
if x is None:            # 正确
    ...
if x == None:            # 不推荐 —— == 可能被自定义 __eq__ 覆盖
    ...
```

---

## 基础练习

### 练习 1：判断变量命名是否合法

下面哪些变量名是合法的 Python 标识符？写出合法和不合法的原因。

```
1name    _name    name_1    n@me    for    myName    __hidden__
```

**参考答案**：

```python
# 1name   —— 不合法，不能以数字开头
# _name   —— 合法，可以以下划线开头
# name_1  —— 合法，字母+数字+下划线的组合
# n@me    —— 不合法，@ 不是字母、数字或下划线
# for     —— 虽然符合命名规则，但 for 是 Python 关键字，不推荐（实际可以使用但会引发混淆）
#          注：Python 3 中 for 不是"保留字作为标识符"是语法错误
# myName  —— 合法，但不推荐（Python 约定用 snake_case: my_name）
# __hidden__ —— 合法，双下划线开头和结尾是 Python 的特殊命名约定
```

### 练习 2：类型转换结果预测

写出下列表达式的值和类型，然后在 Python 中验证：

```python
a = int(3.14)        # ？
b = float(10)        # ？
c = bool("False")    # ？
d = str(100) + str(200)  # ？
e = True + 100       # ？
```

**参考答案**：

```python
a = int(3.14)        # 3，<class 'int'> —— 截断取整
b = float(10)        # 10.0，<class 'float'>
c = bool("False")    # True，<class 'bool'> —— 非空字符串都是 True！
d = str(100) + str(200)  # "100200"，<class 'str'> —— 字符串拼接
e = True + 100       # 101，<class 'int'> —— True 就是 1
```

### 练习 3：预测 id 和 is 的输出

不运行代码，预测下列 `print` 的输出并解释原因：

```python
a = 10
b = 10
print(a is b)       # ？

lst1 = [1, 2]
lst2 = [1, 2]
print(lst1 is lst2)  # ？
print(lst1 == lst2)  # ？

x = lst1
print(x is lst1)     # ？
```

**参考答案**：

```python
a = 10
b = 10
print(a is b)       # True —— 10 在 -5~256 范围内，被 CPython 缓存，a 和 b 指向同一个对象

lst1 = [1, 2]
lst2 = [1, 2]
print(lst1 is lst2)  # False —— 两个不同的列表对象（虽然内容相同）
print(lst1 == lst2)  # True  —— == 比较值，内容相同

x = lst1
print(x is lst1)     # True  —— x 就是 lst1，同一个引用
```

---

## 进阶练习

### 练习 4：判断对象身份并解释可变/不可变行为

写一个函数 `same_object(a, b)` 返回两个变量是否指向同一对象。然后用它来验证：

1. 对 int 类型：`a = 5; b = 5` 后 `same_object(a, b)` 是什么？`a = a + 1` 后，a 和原来的 5 还在同一个对象吗？
2. 对 list 类型：`a = [1,2]; b = a` 后 `same_object(a, b)` 是什么？`a.append(3)` 后，a 和 b 还指向同一对象吗？

**参考答案**：

```python
def same_object(a, b):
    """判断 a 和 b 是否指向同一个对象"""
    return a is b


# === 测试 1：int（不可变） ===
a = 5
b = 5
print(f"a 和 b 指向同一对象？{same_object(a, b)}")  # True（小整数缓存）

old_a = a          # 保存 a 原来指向的对象引用
a = a + 1          # 计算 6，创建新对象，a 指向新对象
print(f"a+1 后，a 还指向原来的 5 吗？{same_object(a, old_a)}")  # False
# 原因：int 是不可变对象，a + 1 创建了一个新的 int 对象 6，
# 变量 a 被"贴"到了新对象上，原来的对象 5 不受影响。

# === 测试 2：list（可变） ===
a = [1, 2]
b = a              # b 指向和 a 相同的列表对象
print(f"a 和 b 指向同一对象？{same_object(a, b)}")  # True

a.append(3)        # 原地修改——不创建新对象
print(f"append 后 a 和 b 指向同一对象？{same_object(a, b)}")  # True
print(f"a = {a}, b = {b}")  # 两者都是 [1, 2, 3]
# 原因：list 是可变对象，append 直接修改对象本身，不创建新对象。
# 变量 a 和 b 始终指向同一个列表对象，所以修改对两者都可见。
```

---

## 常见错误

### 错误 1：用 `is` 做值比较

```python
# 错误写法
a = 1000
b = 1000
if a is b:          # False！大整数不在缓存范围
    print("相等")

# 正确写法
if a == b:          # True —— 比较值
    print("相等")
```

### 错误 2：忘记 Python 没有 ++/-- 运算符

```python
# 错误写法
x = 0
x++                 # SyntaxError（但 x++ 被解析为 x + (+1)，最终是语法错误）

# 正确写法
x += 1              # x = x + 1 的简写
```

### 错误 3：混淆 = 和 ==

```python
# 错误写法
if x = 5:           # SyntaxError —— 赋值是语句，不能用在 if 条件中
    print("yes")

# 正确写法
if x == 5:          # 比较
    print("yes")
```

### 错误 4：以为 round(2.5) 是 3

```python
# 预期 3，实际是 2
print(round(2.5))   # 2 —— 银行家舍入，向偶数方向舍入
print(round(3.5))   # 4

# 如果需要"四舍五入"到整数，可以这样做
import math
print(math.floor(2.5 + 0.5))  # 3
```

### 错误 5：忽略浮点数精度问题

```python
# 错误写法
if 0.1 + 0.2 == 0.3:   # False！永远不相等
    print("相等")

# 正确写法
import math
if math.isclose(0.1 + 0.2, 0.3):   # True
    print("近似相等")

# 或者用 Decimal
from decimal import Decimal
if Decimal("0.1") + Decimal("0.2") == Decimal("0.3"):
    print("精确相等")
```

### 错误 6：用 == 比较 None

```python
# 不推荐
if result == None:
    ...

# 正确（is 比较身份，不受 __eq__ 影响）
if result is None:
    ...
```

---

## 本章小结

- **CPython** 将源码编译为字节码后在 PVM 上解释执行；没有 JIT，但 PyPy 提供了 JIT 替代方案
- **动态类型**：类型在对象上而非变量上；鸭子类型不检查类型而检查对象是否支持所需操作
- **变量是对象引用**：`a = b` 不复制对象，只是多贴了一个标签；`id()` 获取对象的内存标识
- **`==` 比值，`is` 比身份**：日常用 `==`，仅在与 `None` 比较时用 `is`
- **小整数（-5~256）和标识符字符串被缓存**：不要依赖 `is` 对小整数的行为
- **不可变类型**（int, str, tuple）的"修改"创建新对象；**可变类型**（list, dict, set）支持原地修改
- **int** 任意精度不溢出；**float** 是 IEEE 754 双精度，`0.1 + 0.2 != 0.3`，金额用 Decimal
- **`//` 向下取整**（区别于 C/Java 的向零截断）；**`round()` 银行家舍入**
- **falsy 值**：`False`, `None`, 零值, 空容器；其余为 truthy
