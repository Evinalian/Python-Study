# 第18章 Matplotlib 可视化

> "一图胜千言"——本章从画出第一张折线图开始，逐步教你用 Matplotlib 创建折线图、散点图、柱状图、直方图，并掌握标题、图例、标注等美化的技巧。

---

## 学习目标

- 用最少的代码画出第一张图
- 理解 pyplot 和面向对象（OO）两套 API 的区别
- 掌握 Figure（画布）和 Axes（绘图区）的层次关系
- 能创建折线图、散点图、柱状图、直方图
- 会添加标题、轴标签、图例、标注
- 解决中文字体显示方框的问题
- 掌握 `fig.savefig()` 保存高质量图片

---

## 前置知识

- NumPy 基础（创建数组用于生成数据）
- Pandas 基础（可选，配合使用效果更好）

---

## 1. 第一张图：一行代码的成就感

```python
import matplotlib.pyplot as plt

# 一行代码画出折线图
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.show()
```

运行这段代码，你会看到一张图：x 轴是 `[1, 2, 3, 4]`，y 轴是 `[1, 4, 9, 16]`（即 x 的平方）。Matplotlib 自动为你添加了坐标轴、刻度和连线。

> 如果你在 Jupyter Notebook 中运行，需要先执行 `%matplotlib inline`，图片会直接嵌入在 Notebook 里。如果是普通 `.py` 脚本，必须调用 `plt.show()` 才会弹出窗口显示图片。

---

## 2. 两套 API：pyplot vs 面向对象

Matplotlib 有两套 API，新手常混淆：

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)

# 方式一：pyplot API（状态机风格，适合快速探索）
plt.plot(x, np.sin(x))
plt.title("Sine Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.show()

# 方式二：面向对象 API（显式操作对象，推荐用于正式代码）
fig, ax = plt.subplots()       # 同时创建画布和绘图区
ax.plot(x, np.sin(x))          # 在绘图区上画线
ax.set_title("Sine Wave")      # 注意：ax.set_title() 而非 plt.title()
ax.set_xlabel("x")
ax.set_ylabel("sin(x)")
plt.show()
```

**区别**：
- pyplot 隐式操作"当前"的图和绘图区——适合 Jupyter 里快速看一眼
- 面向对象显式操作 `fig` 和 `ax` 对象——适合写函数、多子图、精细控制

**建议**：一开始就养成 OO 风格的习惯，后面的所有示例都用 OO 风格。

---

## 3. Figure 和 Axes：画布和子图

用生活场景理解 Matplotlib 的层次结构：

```
Figure（画布）           ← 一整张白纸
├── Axes（绘图区 #1）     ← 纸上画的第一幅小图
│   ├── x 轴（Axis）
│   ├── y 轴（Axis）
│   ├── 线条、散点……
│   └── 标题、图例……
├── Axes（绘图区 #2）     ← 纸上画的第二幅小图
│   └── ...
└── ...
```

- **Figure（画布）**：整个窗口/图片，可以包含多个 Axes
- **Axes（绘图区）**：一个坐标系 + 绘图区域，大部分绘图函数在它上面调用
- **Axis（轴）**：x 轴或 y 轴，管理刻度和标签

`plt.subplots()` 一次性创建 Figure 和 Axes：

```python
# 创建 1 行 2 列的子图
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # figsize 单位是英寸

# axes 是一个数组，包含两个 Axes 对象
axes[0].plot(x, np.sin(x), color="blue")
axes[0].set_title("sin(x)")

axes[1].plot(x, np.cos(x), color="red")
axes[1].set_title("cos(x)")

plt.tight_layout()  # 自动调整子图间距
plt.show()
```

---

## 4. 折线图（plot）

折线图用于展示数据随某个变量（通常是时间或序列）变化的趋势。

```python
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = np.cos(x)

fig, ax = plt.subplots(figsize=(8, 4))

# 画两条线
ax.plot(x, y1,
        color="steelblue",      # 颜色：可用颜色名或十六进制码 "#1f77b4"
        linewidth=2,            # 线宽
        linestyle="-",          # 线型：'-' 实线, '--' 虚线, '-.' 点划线, ':' 点线
        marker="o",             # 数据点标记：'o' 圆, 's' 方, '^' 三角, 'D' 菱形
        markersize=4,           # 标记大小
        alpha=0.8,              # 透明度：0=完全透明, 1=完全不透明
        label="sin(x)")         # 图例中的标签

ax.plot(x, y2, color="coral", linewidth=2, linestyle="--", label="cos(x)")

# 添加标题和轴标签
ax.set_title("三角函数曲线", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()                     # 显示图例

plt.show()
```

**效果描述**：图中出现两条曲线——蓝色实线（sin，带圆点标记）和橙色虚线（cos）。右上角有图例，标明每条线的含义。

---

## 5. 散点图（scatter）

散点图用于展示两个变量之间的关系——每个点代表一个观测。

```python
fig, ax = plt.subplots(figsize=(6, 5))

# 生成两组随机数据
rng = np.random.default_rng(42)
n = 50

# 第一组：集中在左下角
x1 = rng.normal(2, 0.8, n)
y1 = rng.normal(2, 0.8, n)

# 第二组：集中在右上角
x2 = rng.normal(6, 0.8, n)
y2 = rng.normal(6, 0.8, n)

# 画两组散点，用不同颜色区分
ax.scatter(x1, y1, color="steelblue", alpha=0.7, s=60, label="A 组")
ax.scatter(x2, y2, color="coral", alpha=0.7, s=60, label="B 组")

ax.set_title("两组数据的散点分布", fontsize=14)
ax.set_xlabel("X 坐标")
ax.set_ylabel("Y 坐标")
ax.legend()

plt.show()
```

**效果描述**：蓝色点群集中在左下角 (2, 2) 附近，红色点群集中在右上角 (6, 6) 附近。一眼就能看出两组数据属于不同的聚类。

---

## 6. 柱状图（bar）

柱状图用于比较不同类别的数值大小。

```python
fig, ax = plt.subplots(figsize=(7, 4))

# 数据
categories = ["语文", "数学", "英语", "物理", "化学"]
scores = [88, 95, 72, 85, 78]
colors = ["steelblue", "coral", "seagreen", "gold", "mediumpurple"]

# 画柱状图
bars = ax.bar(categories, scores, color=colors, edgecolor="white", linewidth=1.5)

# 在每个柱子上方显示数值
for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
            str(score), ha="center", va="bottom", fontsize=11)

ax.set_title("各科目成绩", fontsize=14)
ax.set_ylabel("分数")
ax.set_ylim(0, 105)  # 留出数值标签的空间

plt.show()
```

**效果描述**：5 根不同颜色的柱子代表 5 个科目，每根柱子顶部显示具体分数。数学最高（95），英语最低（72）。

---

## 7. 直方图（hist）

直方图用于展示数值数据的分布——把数据分成若干区间（bin），统计每个区间有多少个观测值。

```python
fig, ax = plt.subplots(figsize=(7, 4))

# 生成 1000 个正态分布随机数
rng = np.random.default_rng(42)
data = rng.normal(70, 15, 1000)  # 均值 70，标准差 15（模拟考试成绩）

ax.hist(data,
        bins=30,               # 分成 30 个柱子
        color="steelblue",
        edgecolor="white",
        alpha=0.7)

ax.set_title("考试成绩分布", fontsize=14)
ax.set_xlabel("分数")
ax.set_ylabel("人数")

# 添加均值和标准差标记
mean_val = data.mean()
ax.axvline(mean_val, color="red", linestyle="--", linewidth=2,
           label=f"均值 = {mean_val:.1f}")
ax.legend()

plt.show()
```

**效果描述**：柱状分布图大致呈钟形（正态分布），中间高两边低，说明大部分学生分数在 55-85 之间。红色虚线标记均值约 70 的位置。

---

## 8. 中文字体：一行代码解决方框问题

Windows 上 Matplotlib 默认字体不支持中文，所以标题和标签里的中文会显示为方框 `□□□□`。下面是可以直接复制粘贴的解决方案：

```python
import matplotlib.pyplot as plt

# 设置中文字体（Windows 推荐 SimHei 黑体）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
# 解决负号 "-" 显示为方框的问题
plt.rcParams["axes.unicode_minus"] = False

# 之后正常画图即可
fig, ax = plt.subplots()
ax.set_title("这里的中文可以正常显示了")
ax.set_xlabel("横轴")
ax.set_ylabel("纵轴")
plt.show()
```

**推荐封装成函数**，在每个脚本开头调用：

```python
def setup_chinese_font():
    """尝试设置中文字体，失败则静默忽略"""
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei",
                                            "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

setup_chinese_font()
```

---

## 9. 双纵坐标（twinx）

有时候两个指标的量纲不同（比如温度和降水量），但你想把它们画在同一个 x 轴上对比。`twinx()` 创建一个共享 x 轴的第二个 y 轴：

```python
fig, ax1 = plt.subplots(figsize=(8, 4))

months = np.arange(1, 13)
temperature = [2, 5, 12, 20, 26, 30, 33, 32, 27, 20, 12, 5]  # 月均温度
rainfall = [15, 20, 40, 60, 80, 120, 180, 160, 90, 50, 30, 15]  # 月降水量

# 左 y 轴：温度
color_temp = "tab:red"
ax1.bar(months, temperature, color=color_temp, alpha=0.5, label="温度")
ax1.set_xlabel("月份")
ax1.set_ylabel("温度 (℃)", color=color_temp)
ax1.tick_params(axis="y", labelcolor=color_temp)

# 右 y 轴：降水量
ax2 = ax1.twinx()  # 创建共享 x 轴的第二个 y 轴
color_rain = "tab:blue"
ax2.plot(months, rainfall, color=color_rain, linewidth=2, marker="o", label="降水量")
ax2.set_ylabel("降水量 (mm)", color=color_rain)
ax2.tick_params(axis="y", labelcolor=color_rain)

ax1.set_title("各月温度和降水量", fontsize=14)
plt.show()
```

**效果描述**：红色半透明柱子表示温度（左轴），蓝色折线表示降水量（右轴）。便于观察"温度高的时候降水也多"的趋势。

---

## 10. 保存图片

```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])

# 保存为 PNG（一般用途）
fig.savefig("plot.png", dpi=300, bbox_inches="tight")

# 保存为 SVG（矢量格式，适合论文/网页）
fig.savefig("plot.svg", format="svg", bbox_inches="tight")

# 保存为 PDF（矢量格式，适合 LaTeX）
fig.savefig("plot.pdf", format="pdf", bbox_inches="tight")
```

参数说明：
- `dpi=300`：分辨率，值越大图片越清晰但文件也越大
- `bbox_inches="tight"`：自动裁剪多余的白边

---

## 11. 常见错误

### 错误 1：在脚本中忘记 `plt.show()`

```python
# 错误：脚本执行完不会自动弹出窗口
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
# 脚本结束——图片窗口从未出现！

# 正确：显式调用 plt.show()
plt.show()
```

注：Jupyter Notebook 中 `%matplotlib inline` 会自动显示，但脚本中必须调用 `plt.show()`。

### 错误 2：混用 pyplot 和 OO API

```python
# 混乱的写法：plt 和 ax 混用
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title("Title")       # pyplot 函数操作"当前" Axes——碰巧对了
plt.xlabel("X label")    # 但如果有多个子图，这句是在修改哪个？

# 正确：统一用 OO 风格
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("Title")
ax.set_xlabel("X label")
```

### 错误 3：中文显示为方框

```python
# 错误：直接写中文，显示为 □□□□
ax.set_title("折线图")

# 正确：在文件开头设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
ax.set_title("折线图")  # 现在正常显示
```

### 错误 4：保存图片时内容被裁切

```python
# 错误：图片边缘的标签或标题被裁掉
fig.savefig("plot.png")

# 正确：使用 bbox_inches="tight" 自动调整
fig.savefig("plot.png", bbox_inches="tight")
```

---

## 12. 练习题

### 基础练习

**练习 1：画一个简单的折线图**

用 `plt.subplots()` 画一条折线图：
- x 轴：0 到 6.28（2*pi）
- y 轴：sin(x)
- 设置标题"正弦波"，x 轴标签"角度"，y 轴标签"sin(x)"

<details>
<summary>参考答案（点击展开）</summary>

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("正弦波")
ax.set_xlabel("角度")
ax.set_ylabel("sin(x)")
plt.show()
```
</details>

**练习 2：画一个柱状图并添加标题和轴标签**

用以下数据画柱状图：

```python
fruits = ["苹果", "香蕉", "橙子", "葡萄", "西瓜"]
sales = [120, 85, 95, 60, 45]
```

要求：每根柱子不同颜色，添加标题"水果销量"和轴标签，在每个柱子上显示数值。

<details>
<summary>参考答案（点击展开）</summary>

```python
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fruits = ["苹果", "香蕉", "橙子", "葡萄", "西瓜"]
sales = [120, 85, 95, 60, 45]
colors = ["red", "gold", "orange", "purple", "green"]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(fruits, sales, color=colors)

# 柱子上方显示数值
for bar, val in zip(bars, sales):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            str(val), ha="center", va="bottom", fontsize=11)

ax.set_title("水果销量", fontsize=14)
ax.set_ylabel("销量（箱）")
ax.set_ylim(0, 140)
plt.show()
```
</details>

**练习 3：画散点图，用不同颜色区分两组数据**

生成两组随机数（每组 50 个点），A 组以 (2, 2) 为中心，B 组以 (6, 6) 为中心。用蓝色和红色分别画出两组散点，添加图例。

<details>
<summary>参考答案（点击展开）</summary>

```python
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)

# A 组：(2, 2) 附近
xA = rng.normal(2, 0.8, 50)
yA = rng.normal(2, 0.8, 50)

# B 组：(6, 6) 附近
xB = rng.normal(6, 0.8, 50)
yB = rng.normal(6, 0.8, 50)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(xA, yA, color="steelblue", alpha=0.7, s=60, label="A 组")
ax.scatter(xB, yB, color="coral", alpha=0.7, s=60, label="B 组")
ax.set_title("两组数据散点图")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
plt.show()
```
</details>

### 进阶练习

**练习 4：折线图 + 柱状图组合图（双纵坐标）**

模拟一份月度销售数据，用柱状图表示每月销量（左轴），用折线图表示累计销量（右轴），展示在一个图中。

数据：

```python
months = list(range(1, 13))
monthly_sales = [120, 135, 148, 162, 155, 178, 190, 185, 200, 210, 195, 220]
# 累计销量需要你自己算
```

要求：柱状图（左轴）、折线图（右轴）、双 y 轴标签用不同颜色、添加标题和图例。

<details>
<summary>参考答案（点击展开）</summary>

```python
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

months = np.arange(1, 13)
monthly_sales = [120, 135, 148, 162, 155, 178, 190, 185, 200, 210, 195, 220]
cumulative_sales = np.cumsum(monthly_sales)  # 累计

fig, ax1 = plt.subplots(figsize=(10, 5))

# 左 y 轴：月销量（柱状图）
color_bar = "steelblue"
ax1.bar(months, monthly_sales, color=color_bar, alpha=0.7, label="月销量")
ax1.set_xlabel("月份")
ax1.set_ylabel("月销量（件）", color=color_bar)
ax1.tick_params(axis="y", labelcolor=color_bar)

# 右 y 轴：累计销量（折线图）
ax2 = ax1.twinx()
color_line = "coral"
ax2.plot(months, cumulative_sales, color=color_line, linewidth=2.5,
         marker="o", markersize=6, label="累计销量")
ax2.set_ylabel("累计销量（件）", color=color_line)
ax2.tick_params(axis="y", labelcolor=color_line)

# 合并两个轴的图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

ax1.set_title("月度销售趋势", fontsize=15)
ax1.set_xticks(months)
plt.tight_layout()
plt.show()
```
</details>

---

## 13. 本章小结

| 概念 | 要点 |
|------|------|
| API 选择 | 推荐 OO 风格：`fig, ax = plt.subplots()` |
| 层次结构 | Figure（画布） > Axes（绘图区） > Axis（轴）、线、点…… |
| 折线图 | `ax.plot(x, y)` |
| 散点图 | `ax.scatter(x, y)` |
| 柱状图 | `ax.bar(categories, values)` |
| 直方图 | `ax.hist(data, bins=30)` |
| 中文 | 设置 `font.sans-serif` 为 SimHei 等中文字体 |
| 双纵坐标 | `ax.twinx()` |
| 保存 | `fig.savefig("name.png", dpi=300, bbox_inches="tight")` |
