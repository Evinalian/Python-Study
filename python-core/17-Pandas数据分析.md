# 第17章 Pandas 数据分析

> 当 Excel 打开 10 万行数据开始转圈圈时，Pandas 可以在一秒内处理百万行。本章从零开始教你用 DataFrame 做数据加载、清洗、分组统计和导出——这是数据分析师和 AI 工程师每天都在做的事情。

---

## 学习目标

- 理解 Series（带索引的一维数组）和 DataFrame（Series 的容器）的结构
- 掌握 `read_csv()` 加载数据及常见参数
- 能区分 `loc`（标签索引）和 `iloc`（位置索引）
- 学会数据清洗：缺失值、重复值、类型转换
- 理解 groupby 的 split-apply-combine 三步模式
- 使用 `merge` 和 `concat` 合并数据

---

## 前置知识

- NumPy ndarray 基本操作（建议先读第 16 章）
- Python 字典和列表操作

---

## 1. 当 Excel 不够用时

你打开一个 CSV 文件，Excel 告诉你"文件太大，无法完全加载"。或者你有一个 50 万行的销售记录，需要按月份分组求总和，Excel 的 VLOOKUP 又慢又容易出错。

Pandas 就是为解决这些问题而生的。它提供两个核心数据结构：

| 结构 | 类比 | 说明 |
|------|------|------|
| **Series** | Excel 中的一列 | 带索引的一维数组，索引 + 值 |
| **DataFrame** | Excel 中的一张表 | 多列组成的二维表格，每列是一个 Series |

下面是一个迷你 DataFrame（3 行 x 3 列），帮你建立直观印象：

```
      name   age  score     ← 列名（columns）
0    Alice    25     95     ← 第 0 行（行索引）
1      Bob    30     87     ← 第 1 行
2     Carl    28     92     ← 第 2 行
↑
行索引（index）
```

```python
import pandas as pd   # pd 是通用别名
import numpy as np
```

---

## 2. Series 和 DataFrame 的创建

### 2.1 Series：带标签的一维数组

```python
# 从列表创建，指定索引（标签）
s = pd.Series([95, 87, 92], index=["math", "english", "cs"])
print(s)
# math       95
# english    87
# cs         92
# dtype: int64

# Series 由两部分组成：索引 + 值
print(s.index)   # Index(['math', 'english', 'cs'], dtype='object')
print(s.values)  # array([95, 87, 92]) —— 底层是 NumPy 数组

# 按标签访问
print(s["math"])            # 95 —— 像字典的键
print(s[["math", "cs"]])    # 传入标签列表，返回一个子 Series
```

### 2.2 DataFrame：从字典创建

```python
# 最常用的创建方式：字典的 key → 列名，value → 该列的值
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Carl"],
    "age": [25, 30, 28],
    "score": [95, 87, 92],
})
print(df)
#     name  age  score
# 0  Alice   25     95
# 1    Bob   30     87
# 2   Carl   28     92

# 基本属性速查
print(df.shape)     # (3, 3) —— 3 行，3 列
print(df.columns)   # Index(['name', 'age', 'score'], dtype='object')
print(df.index)     # RangeIndex(start=0, stop=3, step=1) —— 默认 0,1,2...

# info()：查看每列的非空数量、类型——这是你拿到数据后的第一个操作
print(df.info())
# Data columns (total 3 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   name    3 non-null      object
#  1   age     3 non-null      int64
#  2   score   3 non-null      int64
```

---

## 3. read_csv：从文件加载数据

### 3.1 最基本用法

```python
# 最简调用——大多数情况下够用
df = pd.read_csv("data.csv")
print(df.head())  # 看前 5 行
```

### 3.2 逐步引入参数

随着你遇到实际问题，逐步加入参数：

```python
# 第一步：文件编码不对？指定 encoding
df = pd.read_csv("data.csv", encoding="utf-8")

# 第二步：只想读其中几列？用 usecols
df = pd.read_csv("data.csv", usecols=["name", "score"])

# 第三步：某列应该解析为日期而不是字符串？
df = pd.read_csv("data.csv", parse_dates=["order_date"])

# 第四步：文件里有 "N/A" 这样的缺失值标记？
df = pd.read_csv("data.csv", na_values=["N/A", "null", "缺失"])

# 第五步：id 列读成了数字但应该是字符串？
df = pd.read_csv("data.csv", dtype={"id": str})

# 第六步：文件太大想先预览前 1000 行？
df = pd.read_csv("data.csv", nrows=1000)

# 全部参数合在一起：
df = pd.read_csv(
    "data.csv",
    encoding="utf-8",
    usecols=["name", "age", "score", "order_date"],
    parse_dates=["order_date"],
    na_values=["N/A", "null"],
    dtype={"id": str},
    nrows=1000,
)
```

### 3.3 分块处理超大文件

如果文件大到内存放不下（比如 10GB 的日志），用 `chunksize` 分块读取：

```python
# 每次读 10 万行，逐块处理
chunks = []
for chunk in pd.read_csv("large.csv", chunksize=100_000):
    # 在当前块上做处理
    processed = chunk[chunk["status"] == "success"].groupby("category").size()
    chunks.append(processed)

# 合并所有块的结果
result = pd.concat(chunks).groupby(level=0).sum()
```

---

## 4. 数据探查：拿到数据后首先做什么

```python
# 1. 看前几行 / 后几行 / 随机抽样
df.head(10)           # 前 10 行
df.tail(5)            # 后 5 行
df.sample(10)         # 随机 10 行

# 2. 看结构（最重要！）
df.info()             # 列名、非空计数、dtype、内存占用
# 输出会告诉你每列有多少缺失值、类型是什么

# 3. 看数值列的统计摘要
df.describe()         # count / mean / std / min / 25% / 50% / 75% / max
# 一眼看出有没有异常的极值、均值和中位数是否偏离

# 4. 看分类列的分布
df["category"].value_counts()             # 每个值出现的次数
df["category"].value_counts(normalize=True)  # 占比形式

# 5. 缺失值统计
df.isna().sum()       # 每列有多少缺失值
```

**拿到数据的标准流程**：`info()` → `describe()` → `isna().sum()` ——三步走，不超过 30 秒就能摸清数据的大致状况。

---

## 5. loc vs iloc：最容易混淆的地方

### 5.1 一句话区分

| 方法 | 用什么选行 | 用什么选列 | 区间语义 |
|------|-----------|-----------|---------|
| `loc` | 索引标签 | 列标签 | **闭区间** `[start, end]` |
| `iloc` | 整数位置 | 整数位置 | **左闭右开** `[start, end)` |

类比：`loc` 就像按名字找人——"把 'Alice' 到 'Carl' 之间的所有人叫过来"（包括 Carl）。`iloc` 就像按排号找人——"把第 0 排到第 2 排的人叫过来"（不包括第 3 排）。

### 5.2 用示例看清区别

```python
df = pd.DataFrame({
    "A": [10, 20, 30, 40, 50],
    "B": [100, 200, 300, 400, 500],
}, index=["a", "b", "c", "d", "e"])   # 注意：索引是字母，不是 0,1,2...

# loc：用标签名，"b" 到 "d" 是闭区间——包含 d
print(df.loc["b":"d"])
#     A    B
# b  20  200
# c  30  300
# d  40  400    ← d 被包含了

# iloc：用位置编号，1 到 3 是左闭右开——不包含位置 3（即索引 "d"）
print(df.iloc[1:3])
#     A    B
# b  20  200
# c  30  300    ← 只有位置 1 和 2，不包含位置 3
```

### 5.3 选择行和列

```python
# loc[行标签, 列标签]
df.loc[["a", "c", "e"], ["A"]]    # 选 a、c、e 行的 A 列

# iloc[行位置, 列位置]
df.iloc[[0, 2, 4], [0]]           # 选第 0、2、4 行，第 0 列

# 冒号选全部
df.loc[:, "A"]                     # 所有行的 A 列
df.iloc[:, 0]                      # 所有行的第 0 列
```

---

## 6. 数据筛选：找到你想要的行

### 6.1 布尔索引

```python
# 单条件
df[df["score"] > 90]

# 多条件：每个条件用括号包起来，用 & (与) 和 | (或)
df[(df["score"] > 80) & (df["age"] < 30)]   # 分数>80 且 年龄<30
df[(df["score"] < 60) | (df["score"] > 95)] # 分数<60 或 >95

# 取反
df[~(df["status"] == "cancelled")]  # 状态不是 cancelled 的
```

**注意**：和 NumPy 一样，这里必须用 `&` 和 `|`，不能用 `and` 和 `or`。原因相同——`and`/`or` 是 Python 关键字，不能被 Pandas 重载。

### 6.2 字符串筛选

```python
# str.contains：筛选包含某子串的行
df[df["name"].str.contains("Alice")]

# str.startswith / str.endswith
df[df["email"].str.endswith("@gmail.com")]

# str.match：用正则
df[df["phone"].str.match(r"^1\d{10}$")]  # 11 位手机号
```

### 6.3 query()：更可读的筛选

```python
# 用字符串表达式筛选——引用外部变量用 @
min_score = 80
max_age = 30
result = df.query("score > @min_score and age < @max_age")
# 等价于: df[(df["score"] > 80) & (df["age"] < 30)]
# 但 query 的写法更像 SQL，可读性更好
```

---

## 7. 数据清洗

### 7.1 缺失值处理

```python
# 查看缺失
df.isna().sum()    # 每列缺失数量

# 删除含缺失的行
df.dropna()                        # 任何列有缺失就删
df.dropna(subset=["name", "age"])  # 只在 name 或 age 缺失时才删

# 填充缺失值
df.fillna(0)                        # 全部用 0 填充
df.fillna({"age": 0, "name": "未知"})  # 分列用不同值
df["score"].fillna(df["score"].mean())  # 用该列均值填充

# 前向填充（用上一行的值填）
df["value"].fillna(method="ffill")
```

### 7.2 重复值处理

```python
# 查看是否有重复
df.duplicated().sum()                  # 完全重复的行数
df.duplicated(subset=["name", "email"]).sum()  # 仅 name 和 email 重复的行数

# 删除重复行
df.drop_duplicates()                           # 保留第一次出现
df.drop_duplicates(subset=["name"], keep="last")  # 保留最后一次出现
```

### 7.3 类型转换

```python
# astype：安全类型转换
df["age"] = df["age"].astype(int)

# to_numeric：字符串转数字——转换不了的变成 NaN
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# to_datetime：字符串转日期
df["date"] = pd.to_datetime(df["date_str"])
df["date"] = pd.to_datetime(df["date_str"], format="%Y-%m-%d")  # 指定格式更快
```

---

## 8. 添加列、删除列、修改列

### 8.1 添加新列

```python
# 直接赋值
df["total"] = df["quantity"] * df["unit_price"]

# 用 apply：对 Series 逐元素应用函数
df["name_length"] = df["name"].apply(len)
df["age_group"] = df["age"].apply(lambda x: "young" if x < 30 else "senior")

# 用 assign：链式操作，返回新 DataFrame
df = df.assign(
    total=lambda d: d["quantity"] * d["unit_price"],
    name_length=lambda d: d["name"].apply(len),
)
```

### 8.2 删除列

```python
# drop：删除行或列（axis=1 表示删列）
df = df.drop("temp_column", axis=1)
df = df.drop(["col1", "col2"], axis=1)  # 删除多列
```

---

## 9. groupby：分组聚合（重点）

groupby 是 Pandas 中最强大的功能之一，也是新手最容易困惑的地方。它的核心思想叫 **split-apply-combine**（拆分-应用-合并）：

```
原始数据                split              apply               combine
┌──────┬──────┬────┐   ┌─────────────┐   ┌──────────┐   ┌──────┬──────┬────┐
│ 城市  │ 姓名 │ 年龄│   │ 北京 组      │   │ mean     │   │ 城市  │ avg  │
├──────┼──────┼────┤   │ 小明, 小刚   │   │ 20,22→21 │   ├──────┼──────┼────┤
│ 北京  │ 小明 │ 20 │→  ├─────────────┤ → ├──────────┤ → │ 北京  │  21  │
│ 上海  │ 小红 │ 25 │   │ 上海 组      │   │ mean     │   │ 上海  │  28  │
│ 北京  │ 小刚 │ 22 │   │ 小红, 小美   │   │ 25,31→28 │   │ 广州  │  24  │
│ 广州  │ 小华 │ 24 │   ├─────────────┤   ├──────────┤   └──────┴──────┴────┘
│ 上海  │ 小美 │ 31 │   │ 广州 组      │   │ mean     │
└──────┴──────┴────┘   │ 小华        │   │ 24→24    │
                       └─────────────┘   └──────────┘
```

### 9.1 基本用法

```python
# 示例数据：员工信息
df = pd.DataFrame({
    "department": ["技术", "销售", "技术", "销售", "技术", "人事"],
    "name": ["小王", "小李", "小张", "小陈", "小刘", "小赵"],
    "salary": [15000, 12000, 18000, 11000, 20000, 13000],
    "age": [25, 28, 32, 26, 35, 30],
})

# 按部门分组，求平均薪资
result = df.groupby("department")["salary"].mean()
print(result)
# department
# 技术    17666.67
# 人事    13000.00
# 销售    11500.00
```

**解读**：
1. `groupby("department")`：按部门拆成 3 组（技术、销售、人事）——这是 **split**
2. `["salary"]`：只对薪资列操作
3. `.mean()`：每组求均值——这是 **apply**
4. 结果自动合并为一个 Series ——这是 **combine**

### 9.2 多种聚合

```python
# 单列多指标聚合
print(df.groupby("department")["salary"].agg(["mean", "min", "max", "count"]))
#               mean    min    max  count
# department
# 技术      17666.67  15000  20000      3
# 人事      13000.00  13000  13000      1
# 销售      11500.00  11000  12000      2

# 多列不同指标
print(df.groupby("department").agg({
    "salary": ["mean", "sum"],    # 薪资：均值和总和
    "age": "mean",                 # 年龄：只要均值
}))
```

### 9.3 transform：保持原表行数

`agg` 把每组压缩成一行。如果你想把组级结果**广播回原表的每一行**（比如算"每个员工的薪资相对于部门均值的偏差"），用 `transform`：

```python
# transform 返回和原 DataFrame 一样长的 Series
df["dept_avg"] = df.groupby("department")["salary"].transform("mean")
df["salary_diff"] = df["salary"] - df["dept_avg"]
print(df)
#   department name  salary  age     dept_avg  salary_diff
# 0        技术   小王   15000   25  17666.67    -2666.67
# 1        销售   小李   12000   28  11500.00      500.00
# ...

# 对比 agg：agg 返回每组一行, transform 返回和原表一样多行
```

---

## 10. 排序

```python
# 按值排序
df.sort_values("score")                          # 升序
df.sort_values("score", ascending=False)          # 降序
df.sort_values(["category", "score"], ascending=[True, False])  # 多列，各列不同方向

# 按索引排序
df.sort_index()
```

---

## 11. 合并数据

### 11.1 concat：纵向或横向拼接

```python
# 纵向堆叠（行数增加）
df_combined = pd.concat([df_jan, df_feb, df_mar])
df_combined = pd.concat([df_jan, df_feb], ignore_index=True)  # 重置行索引

# 横向拼接（列数增加）
df_wide = pd.concat([df_left, df_right], axis=1)
```

### 11.2 merge：数据库风格的 JOIN

如果你学过 SQL，merge 就是 SQL 的 JOIN。如果你没学过，可以这样理解：两张表通过共同的列（键）关联起来。

```python
# 员工表和部门表通过 department_id 关联
employees = pd.DataFrame({
    "name": ["小王", "小李", "小张"],
    "dept_id": [1, 2, 1],
})
departments = pd.DataFrame({
    "dept_id": [1, 2, 3],
    "dept_name": ["技术", "销售", "人事"],
})

# 内连接：只保留两边都有的 dept_id（1 和 2）
result = pd.merge(employees, departments, on="dept_id")
print(result)
#   name  dept_id dept_name
# 0   小王        1       技术
# 1   小张        1       技术
# 2   小李        2       销售
# 注意 dept_id=3（人事）没有员工，所以不出现

# 左连接：保留 employees 的所有行，departments 匹配不上的填 NaN
result_left = pd.merge(employees, departments, on="dept_id", how="left")

# 外连接：两边的行都保留，匹配不上的填 NaN
result_outer = pd.merge(employees, departments, on="dept_id", how="outer")
```

merge 的 `how` 参数对照：

| how | 含义 | SQL 等价 |
|-----|------|---------|
| `"inner"` | 两表都有的才保留 | INNER JOIN |
| `"left"` | 保留左表全部行 | LEFT JOIN |
| `"right"` | 保留右表全部行 | RIGHT JOIN |
| `"outer"` | 两表行全部保留 | FULL OUTER JOIN |

---

## 12. 输出与保存

```python
# 保存为 CSV（最通用）
df.to_csv("output.csv", index=False)    # index=False 很重要！否则会多出一列序号
df.to_csv("output.csv", index=False, encoding="utf-8-sig")  # utf-8-sig 方便 Excel 打开

# 保存为 Excel
with pd.ExcelWriter("report.xlsx") as writer:
    df_summary.to_excel(writer, sheet_name="汇总", index=False)
    df_detail.to_excel(writer, sheet_name="明细", index=False)

# 二进制格式 Parquet（比 CSV 快 5-10 倍，保留数据类型）
df.to_parquet("data.parquet")
df = pd.read_parquet("data.parquet")
```

---

## 13. 常见错误

### 错误 1：loc 用位置索引，iloc 用标签索引

```python
# loc 基于标签，iloc 基于位置——搞混了会选出意想不到的数据
# 当索引恰好是 0,1,2...时两值碰巧一致，但当索引是字母时立刻不同
df.loc[0:5]   # 如果索引不是 0..5 而是日期，这行可能选出空 DataFrame
df.iloc[0:5]  # 永远选前 5 行——位置索引不关心标签是什么
```

### 错误 2：链式索引引发 SettingWithCopyWarning

```python
# 错误：链式索引——Pandas 不知道你是在修改视图还是副本
df[df["A"] > 0]["B"] = 10   # 警告：SettingWithCopyWarning！

# 正确：用 .loc 一步完成"筛选 + 修改"
df.loc[df["A"] > 0, "B"] = 10
```

### 错误 3：忘记 axis 参数

```python
# drop 默认删行，删列要指定 axis=1
df.drop("column_name", axis=1)

# apply 默认沿列（axis=0），如果想逐行操作要指定 axis=1
df.apply(func, axis=1)
```

### 错误 4：groupby 后结果是多级索引但不 reset_index

```python
result = df.groupby(["city", "month"])["amount"].sum()
print(type(result))    # <class 'pandas.core.series.Series'> —— 索引是 (city, month)

# 转回便于处理的 DataFrame
result = result.reset_index()  # 现在 city, month, amount 都是普通列
```

### 错误 5：循环逐行 append 到 DataFrame

```python
# 极差：每次 append 都创建新 DataFrame，O(n^2) 复杂度
df = pd.DataFrame()
for row in data:
    df = pd.concat([df, pd.DataFrame([row])])   # 死慢！

# 好：先收集到列表，一次性创建
rows = []
for row in data:
    rows.append(row)
df = pd.DataFrame(rows)
```

---

## 14. 练习题

### 基础练习

**练习 1：从字典创建 DataFrame 并查看基本信息**

用字典创建一个包含 3 个学生信息的 DataFrame（姓名、年龄、成绩），然后输出 `info()` 和 `describe()` 的结果。

<details>
<summary>参考答案（点击展开）</summary>

```python
import pandas as pd

df = pd.DataFrame({
    "name": ["小明", "小红", "小刚"],
    "age": [20, 22, 21],
    "score": [85, 92, 78],
})
print(df)
print("\n--- info ---")
print(df.info())
print("\n--- describe ---")
print(df.describe())
```
</details>

**练习 2：筛选满足条件的行**

从以下 DataFrame 中筛选出 "技术" 部门且薪资大于 15000 的员工：

```python
df = pd.DataFrame({
    "name": ["小王", "小李", "小张", "小陈", "小刘"],
    "department": ["技术", "销售", "技术", "销售", "技术"],
    "salary": [15000, 12000, 18000, 11000, 20000],
})
```

<details>
<summary>参考答案（点击展开）</summary>

```python
import pandas as pd

df = pd.DataFrame({
    "name": ["小王", "小李", "小张", "小陈", "小刘"],
    "department": ["技术", "销售", "技术", "销售", "技术"],
    "salary": [15000, 12000, 18000, 11000, 20000],
})

tech_high = df[(df["department"] == "技术") & (df["salary"] > 15000)]
print(tech_high)
#   name department  salary
# 2   小张        技术   18000
# 4   小刘        技术   20000
```
</details>

**练习 3：添加新列并删除旧列**

给上述 DataFrame 添加一列 `bonus`（= salary * 0.1），然后删除原来的 `salary` 列。

<details>
<summary>参考答案（点击展开）</summary>

```python
# 添加新列
df["bonus"] = df["salary"] * 0.1
print(df)

# 删除旧列
df = df.drop("salary", axis=1)
print(df)
```
</details>

**练习 4：按某列分组求均值**

按 `department` 分组，计算各组的平均薪资。

<details>
<summary>参考答案（点击展开）</summary>

```python
import pandas as pd

df = pd.DataFrame({
    "name": ["小王", "小李", "小张", "小陈", "小刘", "小赵"],
    "department": ["技术", "销售", "技术", "销售", "技术", "人事"],
    "salary": [15000, 12000, 18000, 11000, 20000, 13000],
})

avg_salary = df.groupby("department")["salary"].mean()
print(avg_salary)
# department
# 技术    17666.67
# 人事    13000.00
# 销售    11500.00
```
</details>

### 进阶练习

**练习 5：完整的数据分析流程——从清洗到导出**

模拟一份销售数据，完成"清洗 → 分组统计 → 排序 → 导出"的完整流程。

步骤：
1. 创建包含缺失值和重复行的模拟数据
2. 清理：去除缺失值、重复行
3. 添加 `total_amount = quantity * unit_price` 列
4. 按产品和月份分组，计算总销售额
5. 按总销售额降序排列
6. 导出为 CSV

<details>
<summary>参考答案（点击展开）</summary>

```python
import pandas as pd
import numpy as np

# 1. 创建模拟数据（含缺失值和重复行）
rng = np.random.default_rng(42)
data = pd.DataFrame({
    "date": rng.choice(pd.date_range("2024-01-01", "2024-06-30"), 100),
    "product": rng.choice(["Laptop", "Mouse", "Keyboard", "Monitor"], 100),
    "quantity": rng.integers(1, 5, 100),
    "unit_price": rng.choice([9999, 299, 899, 3499], 100),
})
# 故意插入一些脏数据
data.loc[5, "quantity"] = np.nan       # 缺失值
data.loc[10, "unit_price"] = np.nan    # 缺失值
data = pd.concat([data, data.iloc[:3]]) # 插入 3 行重复

print(f"原始行数: {len(data)}")
print(f"缺失值: {data.isna().sum().sum()} 个")
print(f"重复行: {data.duplicated().sum()} 行")

# 2. 清洗
data = data.dropna()                    # 去缺失
data = data.drop_duplicates()           # 去重复
print(f"清洗后行数: {len(data)}")

# 3. 添加总金额列
data["total_amount"] = data["quantity"] * data["unit_price"]

# 4. 按产品和月份分组
data["month"] = data["date"].dt.to_period("M")
summary = (
    data.groupby(["product", "month"])["total_amount"]
    .sum()
    .reset_index()
)

# 5. 按销售额降序
summary = summary.sort_values("total_amount", ascending=False)

print("\n=== 销售额 TOP 10 ===")
print(summary.head(10))

# 6. 导出
summary.to_csv("sales_summary.csv", index=False, encoding="utf-8-sig")
print("\n已导出到 sales_summary.csv")
```
</details>

---

## 15. 本章小结

| 操作 | 核心 API |
|------|---------|
| 创建 | `pd.DataFrame(dict)`、`pd.read_csv()` |
| 探查 | `df.info()`、`df.describe()`、`df.isna().sum()` |
| 筛选 | `df.loc[行标签, 列标签]`、`df.iloc[行位置, 列位置]` |
| 清洗 | `dropna()`、`fillna()`、`drop_duplicates()`、`astype()` |
| 分组 | `df.groupby("key")["col"].mean()` |
| 合并 | `pd.merge()`（JOIN）、`pd.concat()`（拼接） |
| 排序 | `df.sort_values("col", ascending=False)` |
| 导出 | `df.to_csv("file.csv", index=False)` |
