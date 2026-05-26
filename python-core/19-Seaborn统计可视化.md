# 第19章 Seaborn 统计可视化

> Matplotlib 画出的图"能用"，Seaborn 画出的图"好看"——而且一行代码就能自动处理分组着色、置信区间、核密度估计等统计需求。本章从实际对比出发，教你用 Seaborn 做专业的统计可视化。

---

## 学习目标

- 理解 Seaborn 解决了 Matplotlib 的哪些痛点（默认美观 + 统计语义 + DataFrame 原生支持）
- 掌握 `sns.set_theme()` 一键美化的效果
- 使用 `scatterplot` + `hue` 一行代码实现分组着色和图例
- 理解箱线图各元素的含义（中位数、四分位数、须线、离群点）
- 使用 `heatmap` 展示相关性矩阵
- 使用 `pairplot` 快速探索多维数据关系

---

## 前置知识

- Matplotlib 基础（Figure/Axes 概念），建议先读第 18 章
- Pandas DataFrame 操作（Seaborn 的所有函数都原生接受 DataFrame）

---

## 1. 从 Matplotlib 到 Seaborn：同样的数据，更美的图

### 1.1 对比：Matplotlib 默认 vs Seaborn 默认

用 Matplotlib 画分组散点图，你需要手动做很多事情：

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 示例数据：企鹅数据集
penguins = pd.DataFrame({
    "bill_length_mm": [39.1, 39.5, 40.3, 36.7, 49.3, 47.5, 50.0, 48.7],
    "bill_depth_mm": [18.7, 17.4, 18.0, 19.3, 15.6, 14.3, 16.1, 15.2],
    "species": ["Adelie", "Adelie", "Adelie", "Adelie",
                "Chinstrap", "Chinstrap", "Gentoo", "Gentoo"],
})

# Matplotlib：要手动分组、手动选颜色、手动写循环
species_list = penguins["species"].unique()
colors = ["steelblue", "coral", "seagreen"]
fig, ax = plt.subplots()
for sp, c in zip(species_list, colors):
    subset = penguins[penguins["species"] == sp]
    ax.scatter(subset["bill_length_mm"], subset["bill_depth_mm"],
               color=c, label=sp, alpha=0.7)
ax.set_xlabel("Bill Length (mm)")
ax.set_ylabel("Bill Depth (mm)")
ax.legend()
ax.set_title("Penguin Bill Dimensions (Matplotlib)")
plt.show()
```

现在看 Seaborn 怎么做——一行代码：

```python
import seaborn as sns

fig, ax = plt.subplots()
sns.scatterplot(
    data=penguins,
    x="bill_length_mm", y="bill_depth_mm",
    hue="species",          # 按 species 列分组着色——自动分颜色、自动加图例
    ax=ax,
)
ax.set_title("Penguin Bill Dimensions (Seaborn)")
plt.show()
```

**区别总结**：

| 方面 | Matplotlib | Seaborn |
|------|-----------|---------|
| 数据格式 | 手动传 x, y 数组 | 直接传 DataFrame，列名=变量 |
| 分组着色 | 手动循环 + 手动选色 | `hue="列名"` 一行搞定 |
| 默认配色 | 灰度朴素 | 彩色调色板，自动美观 |
| 统计功能 | 无（需手动计算） | 内置均值线、置信区间、KDE |

### 1.2 导入和主题设置

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 一键设置美观主题（推荐在每个 Seaborn 脚本开头调用）
sns.set_theme(style="darkgrid")

# 可选样式：
# "darkgrid" —— 深灰背景 + 网格线（推荐）
# "whitegrid" —— 白色背景 + 网格线
# "dark" —— 深灰背景，无网格
# "white" —— 白色背景，无网格
# "ticks" —— 白色背景 + 刻度标记
```

**效果对比**：`sns.set_theme()` 之后，所有 Matplotlib 的图也会自动变好看（字体更大、配色更柔和、网格线自动添加）。这就是为什么很多人在画 Matplotlib 图之前也会先 `import seaborn`——只为借用它的默认样式。

---

## 2. scatterplot + hue：Seaborn 最亮眼的组合

散点图是最常用的探索工具。Seaborn 的 `hue` 参数让你一行代码实现分组着色——多个分类变量的子集同时画在一张图上，自动用不同颜色区分，自动生成图例。

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 加载示例数据集
tips = sns.load_dataset("tips")  # 餐饮账单数据集，244 行
print(tips.head())
#    total_bill   tip     sex smoker  day    time  size
# 0       16.99  1.01  Female     No  Sun  Dinner     2
# 1       10.34  1.66    Male     No  Sun  Dinner     3
# ...

fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(
    data=tips,
    x="total_bill", y="tip",
    hue="time",               # 按用餐时间分组——Lunch vs Dinner 不同颜色
    size="size",              # 按就餐人数调点的大小——人越多点越大
    alpha=0.7,
    ax=ax,
)
ax.set_title("Tips: Total Bill vs Tip (hue=time, size=size)")
plt.show()
```

**效果描述**：每个点代表一次消费。x 轴是账单总额，y 轴是小费。Lunch（午餐）用一种颜色，Dinner（晚餐）用另一种。较大的点表示聚餐（多人），较小的点表示单独就餐。一眼就能看出：消费金额越高，小费越多；晚餐的小费普遍比午餐高。

**核心价值**：`hue`、`size`、`style` 三个参数让你在一个二维平面上编码多达 5 个维度的信息（x, y, hue, size, style），这是 Matplotlib 原生做不到的。

---

## 3. 箱线图：看懂这个最常用的统计图

### 3.1 箱线图怎么看

箱线图（boxplot）用五个数字概括一组数据的分布，是统计中的"瑞士军刀"。先看一个 ASCII 示意图：

```
     离群点 (异常值)
       o
       |
   ----|----  上须线（最大值，或 Q3 + 1.5*IQR）
   |       |
   |-------|  上四分位数 Q3（75% 的数据在此之下）
   |       |
   |-------|  中位数（50% 的数据在此之下）
   |       |
   |-------|  下四分位数 Q1（25% 的数据在此之下）
   |       |
   ----|----  下须线（最小值，或 Q1 - 1.5*IQR）
       |
       o     离群点 (异常值)
```

- **箱子中间的线**：中位数（median），一半数据在此之上，一半在此之下
- **箱子的上下边**：Q1（第 25 百分位）和 Q3（第 75 百分位）——箱子包含了中间 50% 的数据
- **须线（whisker）**：向外延伸到 `Q1 - 1.5*IQR` 和 `Q3 + 1.5*IQR` 范围内的最远数据点（IQR = Q3 - Q1）
- **离群点**：须线范围之外的单独点，用圆圈标记——这些是需要关注的异常值

### 3.2 用 Seaborn 画箱线图

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(
    data=tips,
    x="day", y="total_bill",    # x=分类变量（星期几）, y=数值变量（账单金额）
    hue="smoker",               # 再按是否吸烟分组
    palette="Set2",             # 调色板
    ax=ax,
)
ax.set_title("各天消费金额分布（按吸烟分组）")
ax.set_xlabel("星期")
ax.set_ylabel("账单金额 ($)")
plt.show()
```

**效果描述**：每个星期 x 吸烟/不吸烟组合都有一根箱线。对比箱子的位置（中位数高低）可以判断"周末消费更高"的趋势；对比箱子的高度（IQR 大小）可以判断"吸烟者的消费波动更大"。

---

## 4. 直方图与 KDE：数据分布的两面

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 模拟两组数据：A 组均值为 0，B 组均值为 2
rng = np.random.default_rng(42)
df = pd.DataFrame({
    "value": np.concatenate([rng.normal(0, 1, 500), rng.normal(2, 0.8, 500)]),
    "group": ["A"] * 500 + ["B"] * 500,
})

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左图：直方图 + KDE 密度曲线
sns.histplot(df, x="value", hue="group",
             bins=40, kde=True,         # kde=True 叠加核密度估计曲线
             alpha=0.4, ax=axes[0])
axes[0].set_title("直方图 + KDE（密度曲线）")

# 右图：纯 KDE 密度图
sns.kdeplot(df, x="value", hue="group",
            fill=True, alpha=0.3,       # fill=True 填充曲线下面积
            ax=axes[1])
axes[1].set_title("纯核密度估计（KDE）")

plt.tight_layout()
plt.show()
```

**效果描述**：左图用柱子展示每个区间的频数，曲线是平滑的密度估计。右图只有平滑曲线和填充——更干净，适合在多组间对比分布形状。可以看出 B 组（橙色）的分布整体右移，且更窄（标准差更小）。

---

## 5. heatmap：相关性的可视化利器

热力图用颜色深浅表示数值大小，最常用于展示**相关系数矩阵**——一眼看出哪些变量之间关系强。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 加载企鹅数据集，计算数值列的相关性
penguins = sns.load_dataset("penguins")
numeric_cols = penguins.select_dtypes("number")

# 计算相关系数矩阵
corr = numeric_cols.corr()
print(corr)
#                  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g
# bill_length_mm          1.00000       -0.23505           0.656181     0.595110
# bill_depth_mm          -0.23505        1.00000          -0.583851    -0.471916
# flipper_length_mm       0.65618       -0.58385           1.000000     0.871202
# body_mass_g             0.59511       -0.47192           0.871202     1.000000

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    corr,
    annot=True,           # 在每个格子里显示数值
    fmt=".2f",            # 数值保留 2 位小数
    cmap="RdBu_r",        # 红-蓝发散色（红色=正相关，蓝色=负相关）
    center=0,             # 0 对应白色中点
    vmin=-1, vmax=1,      # 相关系数范围固定在 -1 到 1
    square=True,          # 每个格子是正方形
    linewidths=0.5,       # 格子间距
    ax=ax,
)
ax.set_title("企鹅特征相关性热力图", fontsize=14)
plt.tight_layout()
plt.show()
```

**效果描述**：颜色越红表示正相关越强（如 flipper_length_mm 和 body_mass_g 的 0.87），颜色越蓝表示负相关越强。对角线永远是 1（自己和自己完美相关）。

**热力图在 AI 领域很常用**：相关性矩阵、混淆矩阵（分类模型评估）、注意力权重（Transformer 中的 Attention Map）、特征重要性矩阵——都是用 `sns.heatmap()` 展示的。

---

## 6. pairplot：多维数据"第一眼"的探索工具

`pairplot()` 把所有数值列两两配对画散点图，对角线上画分布图。这是你拿到新数据集后"第一眼"探索的最佳工具。

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

penguins = sns.load_dataset("penguins")

# 所有数值列两两交叉散点图，按品种着色
sns.pairplot(
    penguins,
    hue="species",              # 按品种分组着色——三种企鹅用三种颜色
    diag_kind="kde",            # 对角线：'hist'=直方图, 'kde'=密度曲线
    height=2.5,                 # 每个子图的高度（英寸）
    plot_kws={"alpha": 0.6},   # 传递给散点图的参数（透明度）
)
plt.suptitle("企鹅数据集 pairplot：各品种在多维特征上的分布", y=1.02, fontsize=14)
plt.show()
```

**效果描述**：生成一个 4x4 的矩阵图。对角线上是每个特征按品种分组的密度曲线（可以看出不同品种在该特征上的分布差异）。非对角线是两两特征的散点图（可以看出变量间的关系模式——线性、聚类、还是随机）。

**怎么看 pairplot**：
- 对对角线上看分布：同一特征，不同品种的数据是否重叠？如果不重叠，说明这个特征可以很好地分离品种。
- 看非对角线上的散点：不同品种的点是否聚成不同的簇？如果是，这两个特征的组合可以区分品种。

---

## 7. Seaborn + Matplotlib 联动

Seaborn 函数返回的 `Axes` 就是标准的 Matplotlib 对象，你可以用 Matplotlib 的全部方法进一步微调：

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

fig, ax = plt.subplots(figsize=(7, 5))

# Seaborn 画图
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", ax=ax)

# Matplotlib 微调细节
ax.set_title("消费与小费关系", fontsize=14, fontweight="bold")
ax.set_xlabel("账单总额 (USD)")
ax.set_ylabel("小费 (USD)")

# 添加一条平均小费的参考线
mean_tip = tips["tip"].mean()
ax.axhline(y=mean_tip, color="red", linestyle="--",
           label=f"平均小费 = ${mean_tip:.2f}")
ax.legend()

# 隐藏顶部和右侧的脊线（更简洁）
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.show()
```

**分工原则**：Seaborn 处理统计语义（分组、着色、回归线、误差带），Matplotlib 处理呈现细节（标题、轴范围、刻度、字体、保存）。

---

## 8. 常见错误

### 错误 1：figure-level 函数不接受 `ax` 参数

```python
# 错误：relplot, displot, catplot, lmplot 是 figure-level（返回 FacetGrid）
# 它们不接受 ax 参数——它们自己管理 Figure
fig, ax = plt.subplots()
# sns.relplot(data=df, x="a", y="b", ax=ax)  # TypeError!

# 正确：figure-level 函数独立使用
sns.relplot(data=df, x="a", y="b", col="category")

# 或者用 axes-level 版本（小写字母开头，接受 ax 参数）
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="a", y="b", hue="category", ax=ax)  # OK
```

**速记**：以 `plot` 结尾的通常是 axes-level（`scatterplot`, `lineplot`, `boxplot`, `histplot`），接受 `ax=`。以 `plot` 开头且不带 `plot` 后缀的通常是 figure-level（`relplot`, `displot`, `catplot`, `lmplot`），不接受 `ax=`。

### 错误 2：figure-level 函数返回 FacetGrid，不能直接用 Matplotlib 方法

```python
g = sns.relplot(data=df, x="a", y="b", col="category")
# g 是 FacetGrid 对象，不是 Axes

# 错误
# g.set_title("Title")  # FacetGrid 的方法不同

# 正确：用 FacetGrid 的方法访问内部 Axes
g.axes[0, 0].set_title("Custom Title")  # 访问第一个子图的 Axes
```

### 错误 3：`hue` 是数值型但希望它被当作分类

```python
# 如果 year 是整数（如 2020, 2021），Seaborn 默认用连续调色板——不好看
sns.scatterplot(data=df, x="x", y="y", hue="year")

# 解决：显式转为分类
df["year_cat"] = df["year"].astype(str)
sns.scatterplot(data=df, x="x", y="y", hue="year_cat")
```

### 错误 4：大数据上 pairplot 卡死

`pairplot` 在数据超过几千行时会很慢（因为有 n^2 个子图）。解决办法：

```python
# 抽样 2000 行
sample = df.sample(2000)
sns.pairplot(sample, hue="category")

# 或者只选关键列
sns.pairplot(df[["col1", "col2", "col3", "category"]], hue="category")
```

---

## 9. 练习题

### 基础练习

**练习 1：画一个带 hue 的散点图**

使用 `tips` 数据集（`sns.load_dataset("tips")`），画 x=`total_bill`、y=`tip`、hue=`smoker` 的散点图，观察吸烟者与不吸烟者的小费行为差异。

<details>
<summary>参考答案（点击展开）</summary>

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=tips, x="total_bill", y="tip",
                hue="smoker", alpha=0.7, ax=ax)
ax.set_title("Smoker vs Non-smoker: Tip Behavior")
plt.show()
```
</details>

**练习 2：画一个箱线图对比不同组的数据分布**

使用 `penguins` 数据集（`sns.load_dataset("penguins")`），按 `species` 分组画 `flipper_length_mm` 的箱线图，观察三种企鹅的鳍肢长度分布差异。

<details>
<summary>参考答案（点击展开）</summary>

```python
import seaborn as sns
import matplotlib.pyplot as plt

penguins = sns.load_dataset("penguins")

fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(data=penguins, x="species", y="flipper_length_mm",
            palette="Set2", ax=ax)
ax.set_title("Flipper Length Distribution by Species")
ax.set_xlabel("Species")
ax.set_ylabel("Flipper Length (mm)")
plt.show()
```
</details>

**练习 3：画一个相关系数热力图**

使用 `penguins` 数据集的数值列，计算相关系数矩阵并画出带数值标注的热力图。

<details>
<summary>参考答案（点击展开）</summary>

```python
import seaborn as sns
import matplotlib.pyplot as plt

penguins = sns.load_dataset("penguins")
numeric_cols = penguins.select_dtypes("number")
corr = numeric_cols.corr()

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True,
            linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap")
plt.tight_layout()
plt.show()
```
</details>

### 进阶练习

**练习 4：用 pairplot 探索鸢尾花数据集**

Iris（鸢尾花）是机器学习中最经典的数据集之一。请使用 `sns.load_dataset("iris")` 画 pairplot，按 `species`（三种不同的鸢尾花）着色。观察：

1. 哪些特征对区分三种鸢尾花最有帮助？
2. 哪一种鸢尾花和其他两种最容易区分？

<details>
<summary>参考答案（点击展开）</summary>

```python
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")

sns.pairplot(
    iris,
    hue="species",              # 按品种着色
    diag_kind="kde",            # 对角线显示密度曲线
    height=2.5,
    plot_kws={"alpha": 0.6},
)
plt.suptitle("Iris Dataset: Pairplot by Species", y=1.02, fontsize=14)
plt.show()

# 观察结论：
# 1. petal_length 和 petal_width 对区分三种鸢尾花最有帮助——
#    在这两个特征上，三种花的点群基本不重叠。
# 2. setosa（蓝色）与 versicolor（橙色）和 virginica（绿色）的
#    区分最明显——在所有特征上 setosa 都远离另外两种。
# 3. versicolor 和 virginica 在 petal 特征上有一定重叠，
#    但在对角线的密度曲线上仍能看出差异。
```
</details>

---

## 10. 本章小结

| 图表需求 | 推荐函数 | 关键参数 |
|---------|---------|---------|
| 二维散点 + 分组着色 | `sns.scatterplot` | `hue`, `size`, `style` |
| 数值趋势 + 误差带 | `sns.lineplot` | `errorbar`, `estimator` |
| 分布形状 | `sns.histplot` / `sns.kdeplot` | `kde=True`, `hue` |
| 分类对比（箱线图） | `sns.boxplot` | `hue`, `palette` |
| 相关性矩阵 | `sns.heatmap` | `annot=True`, `cmap="RdBu_r"` |
| 多维探索（第一眼看数据） | `sns.pairplot` | `hue`, `diag_kind="kde"` |

**核心工作流**：`DataFrame` --Seaborn--> 统计语义（分组、着色、密度） --Matplotlib--> 呈现细节（标题、轴、字体、保存）。

Seaborn 处理"统计关系"，Matplotlib 处理"美观细节"——两者是互补的伙伴，不是竞争关系。
