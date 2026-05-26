# 20 Python 开发环境与工具链

## 学习目标

- 理解虚拟环境的存在价值，能用 venv 创建和管理独立项目环境
- 熟练使用 pip 安装包、锁定依赖、恢复环境
- 理解 pyproject.toml 的结构，能手写一个完整的项目配置文件
- 学会用 .env + python-dotenv 安全管理敏感信息
- 能从零安装并使用 Jupyter Notebook/JupyterLab
- 了解 pyenv、conda、pre-commit 等周边工具的使用场景

## 前置知识

- 已掌握 Python 基础语法、函数、类和文件操作（第 01-13 章）
- 使用过命令行（终端），知道 `cd`、`ls`/`dir` 等基本命令
- 在 C/Java 课上学过编译、运行的基本流程

---

## 20.1 虚拟环境 —— 解决"在我电脑上能跑"的问题

### 20.1.1 一个一定会遇到的场景

假设你有两个 Python 项目：

- **项目 A**（课程大作业）：用了 `numpy==1.24.0`，恰好依赖这个版本的某个 API
- **项目 B**（跟同学合作的爬虫）：用了 `numpy==2.1.0`，新版 API 完全不兼容旧版

如果你把两个包都装在系统全局环境里，后装的会覆盖先装的。结果就是：总有一个项目跑不起来。

```text
系统 Python 全局 site-packages/
    安装了 numpy==1.24.0  →  项目 A 能跑，项目 B 炸了
    又装了 numpy==2.1.0   →  项目 B 能跑，项目 A 炸了
```

**虚拟环境**就是给每个项目一个独立的小隔间，每个隔间里可以装自己版本的包，互不干扰。

```text
项目A/.venv/
    └── site-packages/
        └── numpy==1.24.0          ← 项目 A 专用

项目B/.venv/
    └── site-packages/
        └── numpy==2.1.0           ← 项目 B 专用

系统全局 site-packages/          ← 干干净净，不瞎装东西
```

> **类比 Java**：虚拟环境有点像 Maven/Gradle 的本地仓库（`~/.m2/repository`）和 classpath 隔离。不同在于 Python 虚拟环境连**解释器本身**都是隔离的——每个 `.venv/` 里有一个独立的 Python 可执行文件。

### 20.1.2 用 venv 创建第一个虚拟环境

venv 是 Python 3.3+ 内置的，不需要装任何东西。打开终端，跟着一步步走。

**第 1 步：创建虚拟环境**

```bash
# 进入你的项目目录
cd my-project

# 创建虚拟环境，名字习惯叫 .venv（前面的点表示隐藏目录）
python -m venv .venv
```

执行后目录里会多出一个 `.venv/` 文件夹。看一眼里面有什么：

```bash
# Windows PowerShell
Get-ChildItem .venv

# 你会看到类似这样的结构：
# .venv/
#   Scripts/          ← 里面有 python.exe、pip.exe、activate 等
#   Lib/              ← 里面有 site-packages/，后面装的包都在这里
#   pyvenv.cfg        ← 记录这个环境是基于哪个 Python 创建的
```

**第 2 步：激活虚拟环境**

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
# 激活成功后，命令行前面会出现 (.venv) 标记
# (.venv) PS D:\my-project>
```

Linux/Mac：

```bash
source .venv/bin/activate
# 激活成功后，命令行前面会出现 (.venv) 标记
```

如果 PowerShell 报错"无法加载文件"，需要先允许执行脚本（只需做一次）：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**第 3 步：确认环境已激活**

```bash
# 查看当前用的是哪个 python —— 应该指向 .venv 里的
where python        # Windows
# 或
which python        # Linux/Mac

# 查看当前用的哪个 pip
where pip           # Windows
# 或
which pip           # Linux/Mac
# 输出路径应该包含 .venv，类似：D:\my-project\.venv\Scripts\pip.exe
```

**第 4 步：在虚拟环境中安装包**

```bash
# 现在 pip install 会装到 .venv/Lib/site-packages/ 里
pip install requests

# 验证
python -c "import requests; print(requests.__version__)"
```

**第 5 步：退出虚拟环境**

```bash
deactivate
# (.venv) 标记消失，回到全局 Python 环境
```

> **常见困惑**：`deactivate` 后你在哪个目录都无所谓——虚拟环境的激活/退出只影响"当前终端会话"中 `python` 和 `pip` 指向哪里。

---

## 20.2 pip —— Python 的包管理器

在 C 语言里你想用别人的库，可能要手动下载 .c 文件然后 `#include`。在 Java 里用 Maven/Gradle 管依赖。Python 的对应工具就是 **pip**。

### 20.2.1 最基本的操作

```bash
# 安装一个包
pip install requests

# 安装指定版本
pip install requests==2.31.0

# 安装版本范围（>=2.28 且 <3.0）
pip install "requests>=2.28,<3.0"

# 查看已安装的所有包
pip list
# 输出示例：
# Package    Version
# ---------- -------
# certifi    2024.2.2
# charset-normalizer 3.3.2
# idna       3.6
# requests   2.31.0
# urllib3    2.2.1

# 查看某个包的详细信息（版本、依赖、安装位置）
pip show requests

# 升级一个包
pip install --upgrade requests

# 卸载一个包
pip uninstall requests
```

### 20.2.2 锁定依赖：让你的项目可复现

写完一个项目，怎么确保别人（或几个月后的你自己）能用**完全相同**的包版本来运行？

**导出依赖列表：**

```bash
pip freeze > requirements.txt
```

生成的 `requirements.txt` 长这样：

```text
certifi==2024.2.2
charset-normalizer==3.3.2
idna==3.6
requests==2.31.0
urllib3==2.2.1
```

每一行是"包名==精确版本号"。这个文件应该提交到 git。

**从依赖文件恢复环境：**

```bash
# 在新的电脑或新的虚拟环境里，一条命令恢复所有依赖
pip install -r requirements.txt
```

这就是 Python 项目可复现性的基础。以后你下载别人的 Python 项目，看到 `requirements.txt` 就知道用它来装依赖。

### 20.2.3 可编辑模式安装（开发自己的库时用）

如果你自己在开发一个 Python 包，每次改代码都要重新 `pip install` 很烦。用可编辑模式：

```bash
pip install -e .
# -e 表示 editable（可编辑）
# . 表示当前目录
```

这样 pip 会创建一个"链接"指向你的源码目录，改了代码立即生效，不用重新安装。类似于 Java 里 IDE 的热部署。

### 20.2.4 配置国内镜像源（加速下载）

PyPI 官方源在国外，国内下载可能很慢。配置清华镜像源：

```bash
# 永久配置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 或者临时使用（不修改全局配置）
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

# 查看当前配置
pip config list
```

---

## 20.3 pyproject.toml —— 现代 Python 项目的身份证

### 20.3.1 为什么需要它

Python 打包历史比较乱：以前有 `setup.py`（要执行代码，不安全）、`setup.cfg`、`requirements.txt`、`MANIFEST.in`……各种配置文件散落一地。

2024 年之后，Python 社区统一用 **`pyproject.toml`** 这一个文件来管理项目的所有配置。它相当于 Java 项目里 `pom.xml` 的地位，但语法更清爽。

### 20.3.2 一个最简模板

先看一个最简版本，适合学生作业级别的项目：

```toml
# pyproject.toml —— 放在项目根目录

[project]
name = "my-homework"                    # 项目名（用 - 分隔）
version = "0.1.0"                       # 版本号
description = "Python 课程大作业"        # 一句话描述
requires-python = ">=3.11"              # 要求 Python >= 3.11
dependencies = [
    "requests>=2.28",                   # 运行时需要的包
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",                      # 开发和测试时才需要的包
]
```

逐字段解释：

| 字段 | 含义 | 对应 Java |
|------|------|-----------|
| `name` | 包名，发布到 PyPI 时用这个名字 | Maven 的 `<artifactId>` |
| `version` | 当前版本，语义化版本号 | Maven 的 `<version>` |
| `description` | 一句话说明 | Maven 的 `<description>` |
| `requires-python` | 要求的最低 Python 版本 | —— |
| `dependencies` | 运行时依赖，装这个包时自动装上 | Maven 的 `<dependencies>` |
| `[project.optional-dependencies]` | 可选的依赖组，如 `pip install -e ".[dev]"` | Maven 的 `<profiles>` |

### 20.3.3 一个完整模板

下面是一个更完整的版本，包含了常用工具的配置（ruff 做代码检查，pytest 做测试）：

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-ai-app"
version = "0.1.0"
description = "一个 AI 对话助手"
readme = "README.md"
requires-python = ">=3.11"
authors = [
    { name = "张三", email = "zhangsan@example.com" }
]
dependencies = [
    "openai>=1.35",
    "python-dotenv>=1.0",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.5",
]

# ruff 配置：保存时自动格式化和检查
[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "W"]   # 启用的检查规则

# pytest 配置
[tool.pytest.ini_options]
testpaths = ["tests"]
```

**要点**：`pyproject.toml` 替代了过去散落的多个配置文件。`[tool.xxx]` 的格式是约定——任何工具都可以在这里写自己的配置，这样项目的配置就集中在一个文件里了。

---

## 20.4 管理敏感信息：.env 文件

### 20.4.1 API Key 写死在代码里的灾难

假设你写了一个调用 OpenAI API 的程序：

```python
# ❌ 危险做法 —— 千万不要这样
import openai

client = openai.OpenAI(api_key="sk-abc123def456ghijklmn789")  # 密钥明文！
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "你好"}],
)
```

如果你把这段代码推到 GitHub 公开仓库，会发生什么：

1. 攻击者的爬虫在**几分钟内**扫描到你的 API Key
2. 用你的 Key 去调用 API，刷爆你的额度
3. OpenAI 检测到泄露后自动禁用 Key，但你已经被扣费了

这就像把自己的银行卡密码写在纸条上然后贴在宿舍门口。

### 20.4.2 正确做法：.env + python-dotenv

**第 1 步：创建 .env 文件存放密钥**

在项目根目录创建 `.env` 文件：

```bash
# .env —— 注意：这个文件绝对不能提交到 git！
OPENAI_API_KEY=sk-abc123def456ghijklmn789
DATABASE_URL=postgresql://localhost/mydb
```

**第 2 步：安装 python-dotenv**

```bash
pip install python-dotenv
```

**第 3 步：在代码中用 `load_dotenv()` 读取**

```python
import os
from dotenv import load_dotenv

# 加载 .env 文件中的内容到环境变量
load_dotenv()

# 用 os.environ 或 os.getenv 读取
api_key = os.environ["OPENAI_API_KEY"]        # Key 不存在时抛出 KeyError
db_url = os.getenv("DATABASE_URL", "默认值")   # Key 不存在时返回默认值

print(f"API Key 前 8 位: {api_key[:8]}...")   # 验证已加载
```

**第 4 步：创建 .env.example 作为"安全说明书"**

```bash
# .env.example —— 这个文件可以提交到 git
OPENAI_API_KEY=把你的-key-填在这里
DATABASE_URL=把你的-数据库地址-填在这里
```

这样别人下载你的代码后，看到 `.env.example` 就知道需要配置哪些环境变量，填好后再重命名为 `.env` 即可。

**第 5 步：.gitignore 中忽略敏感文件**

```gitignore
# .gitignore
.venv/
__pycache__/
*.pyc
.env              # ← 关键：排除 .env
```

> **与 Java 对比**：Java 里通常用 `application.properties` 配合 profiles 管理配置，敏感信息放在 `application-secret.properties` 中并加入 `.gitignore`。Python 的 `.env` 方案更轻量。

---

## 20.5 Jupyter —— 边写代码边看结果

Jupyter 是一种"笔记本"式的编程环境，你可以写一段代码，运行，立刻看到结果，然后接着写下一段。特别适合数据分析、AI 实验和学习探索。

### 20.5.1 从零安装到第一次运行

```bash
# 第 1 步：安装（推荐 JupyterLab，功能更全）
pip install jupyterlab

# 第 2 步：启动
jupyter lab
# 浏览器会自动打开 http://localhost:8888/lab
```

### 20.5.2 第一个 Notebook

打开 JupyterLab 后：

1. 点击左侧文件浏览器上方的 **"Python 3"** (Notebook) 按钮
2. 出现一个代码单元格（cell），输入：

```python
print("Hello, Jupyter!")
```

3. 按 **Shift + Enter** 运行这个 cell
4. 下面立刻出现输出：`Hello, Jupyter!`
5. 在下一个 cell 里继续写代码……可以一直追加新的 cell

### 20.5.3 让 Jupyter 使用虚拟环境中的包

Jupyter 默认使用启动它的那个 Python 环境。如果你想让 Jupyter 使用某个虚拟环境中的包：

```bash
# 第 1 步：激活目标虚拟环境
.venv\Scripts\Activate.ps1     # Windows

# 第 2 步：在虚拟环境中安装 ipykernel
pip install ipykernel

# 第 3 步：把这个环境注册为 Jupyter 的一个 kernel
python -m ipykernel install --user --name=myproject --display-name="Python (myproject)"
```

之后在 JupyterLab 里点击右上角的 kernel 名称，就能切换到 `Python (myproject)`，使用该虚拟环境中的包。

### 20.5.4 几个好用的魔法命令

```python
# 计时：这段代码跑了多久？
%timeit sum(range(1000000))
# 输出：17.2 ms ± 215 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# 在 Notebook 里直接画图
%matplotlib inline
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [4, 5, 6])
# 图表直接嵌在 cell 下方

# 列出当前所有变量
%who
%whos  # 更详细，包括类型和值
```

---

## 20.6 进阶工具速览

### 20.6.1 Python 版本管理器 pyenv

当你在不同项目间需要切换 Python 版本（比如 3.11 vs 3.12）时，pyenv 能帮你管理。

```bash
# 安装后
pyenv install 3.12.3      # 安装指定版本
pyenv global 3.12.3       # 设为全局默认
pyenv local 3.11.8        # 当前目录下用 3.11（创建 .python-version 文件）
```

> **类比 Java**：pyenv 类似于 jEnv 或 SDKMAN!，让你在不同 JDK 版本间切换。

### 20.6.2 conda/miniforge

conda 不仅能管 Python 包，还能管理**非 Python 依赖**（C 库、CUDA 等）。这对 AI/数据科学项目特别有用。例如装 PyTorch 时 conda 会自动匹配 CUDA 版本。

```bash
conda create -n myproj python=3.12    # 创建环境
conda activate myproj                 # 激活
conda install numpy pandas            # 安装包
conda deactivate                      # 退出
```

**选 venv 还是 conda？**

| 场景 | 推荐 |
|------|------|
| 纯 Python Web/API 项目 | venv（轻量，Python 自带） |
| 深度学习、需要 CUDA | conda/miniforge |
| Docker 部署 | venv（镜像更小） |

### 20.6.3 pre-commit：提交前自动检查代码

pre-commit 在你 `git commit` 之前自动运行代码检查，不通过的 commit 会被阻止。

```bash
pip install pre-commit
pre-commit install           # 安装 git hook
```

之后每次 commit 都会自动检查。如果代码有格式问题或明显的 bug，commit 直接失败，逼你在提交前修好。

---

## 20.7 本章小结

1. **虚拟环境（venv）**：每个项目一个 `.venv/`，包互不冲突。创建 → 激活 → 装包 → 退出，四步走。
2. **pip**：`pip install` 装包，`pip list` 查看，`pip freeze > requirements.txt` 锁定依赖，`pip install -r requirements.txt` 恢复环境。
3. **pyproject.toml**：现代 Python 项目的统一配置文件，替代过去散落的 `setup.py`、`setup.cfg`、各工具的独立配置。
4. **.env**：API Key 等敏感信息放在 `.env` 文件里，`.gitignore` 排除它，用 `python-dotenv` 在代码中加载。
5. **Jupyter**：边写边看结果的交互式编程环境，实验和学习的好帮手。
6. **进阶工具**：pyenv 管 Python 版本，conda 管复杂科学计算依赖，pre-commit 自动做代码检查。

---

## 基础练习

### 练习 1：创建虚拟环境并安装 requests

在桌面上创建一个 `learn-venv` 目录，用 venv 创建虚拟环境，激活它，安装 `requests` 包，验证安装成功。

**参考答案：**

```powershell
# Windows PowerShell 下执行：
mkdir $env:USERPROFILE\Desktop\learn-venv
cd $env:USERPROFILE\Desktop\learn-venv
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install requests
python -c "import requests; print(f'requests 版本: {requests.__version__}')"
# 输出类似：requests 版本: 2.31.0
```

```bash
# Linux/Mac 下执行：
mkdir ~/Desktop/learn-venv
cd ~/Desktop/learn-venv
python -m venv .venv
source .venv/bin/activate
pip install requests
python -c "import requests; print(f'requests 版本: {requests.__version__}')"
```

### 练习 2：写一个最简 pyproject.toml

在你的 `learn-venv` 目录下创建一个 `pyproject.toml`，包含基本的项目元数据和 `requests` 依赖。

**参考答案：**

```toml
# pyproject.toml —— 放在 learn-venv/ 目录下
[project]
name = "learn-venv"
version = "0.1.0"
description = "学习虚拟环境的小项目"
requires-python = ">=3.11"
dependencies = [
    "requests>=2.28",
]
```

### 练习 3：用 python-dotenv 读取环境变量

在 `learn-venv` 目录下，创建 `.env` 文件写入一个测试变量，安装 `python-dotenv`，写一个 `main.py` 读取并打印该变量。

**参考答案：**

`.env` 文件内容：

```bash
MY_NAME=张三
CLASS_NAME=计算机科学
```

`main.py` 内容：

```python
# main.py
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 读取变量
my_name = os.environ["MY_NAME"]
class_name = os.getenv("CLASS_NAME", "未知课程")

print(f"姓名: {my_name}")
print(f"课程: {class_name}")
```

运行：`python main.py`，输出：

```text
姓名: 张三
课程: 计算机科学
```

---

## 进阶练习

### 练习 4：从零初始化一个完整的 Python 项目

创建一个完整的 Python 项目 `my-weather-app`，包含以下内容：

1. 合理的目录结构
2. 虚拟环境 `.venv/`
3. `pyproject.toml`（含项目元数据、依赖、ruff 配置）
4. `.env` 和 `.env.example`
5. `.gitignore`
6. 一个 `main.py` 从 `.env` 读取 API Key 并打印"配置加载成功"

**参考答案：**

项目目录结构：

```text
my-weather-app/
├── .venv/                  # 虚拟环境（不提交到 git）
├── .env                    # 敏感信息（不提交到 git）
├── .env.example            # 模板（提交到 git）
├── .gitignore              # git 忽略规则
├── pyproject.toml          # 项目配置
├── main.py                 # 主程序入口
└── README.md               # 项目说明
```

**步骤 1：创建目录和虚拟环境**

```bash
mkdir my-weather-app
cd my-weather-app
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate  # Linux/Mac
```

**步骤 2：创建 pyproject.toml**

```toml
[project]
name = "my-weather-app"
version = "0.1.0"
description = "一个查询天气的命令行小工具"
requires-python = ">=3.11"
dependencies = [
    "requests>=2.28",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.5",
]

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

**步骤 3：创建 .env 和 .env.example**

`.env`：

```bash
OPENWEATHER_API_KEY=abc123yourkeyhere
```

`.env.example`：

```bash
OPENWEATHER_API_KEY=请在这里填入你的 API Key
```

**步骤 4：创建 .gitignore**

```gitignore
# 虚拟环境
.venv/

# Python
__pycache__/
*.pyc

# 敏感信息
.env

# IDE
.vscode/
.idea/
```

**步骤 5：写 main.py**

```python
# main.py
import os
from dotenv import load_dotenv


def main():
    # 加载 .env 中的配置
    load_dotenv()

    # 读取 API Key
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if api_key and api_key != "abc123yourkeyhere":
        print(f"配置加载成功！API Key: {api_key[:8]}...")
    else:
        print("请先在 .env 文件中设置你的 OPENWEATHER_API_KEY")

    print("项目初始化完成，可以开始开发了！")


if __name__ == "__main__":
    main()
```

运行 `python main.py`，看到"配置加载成功"说明一切就绪。

---

## 常见错误

### 错误 1：忘记激活虚拟环境

```powershell
# ❌ 错误：直接在当前目录装包
PS D:\my-project> pip install requests
# 可能装到了全局环境，而不是项目的 .venv 里
```

```powershell
# ✓ 正确：先激活虚拟环境
PS D:\my-project> .venv\Scripts\Activate.ps1
(.venv) PS D:\my-project> pip install requests
```

### 错误 2：把 .env 提交到了 git

```bash
# ❌ 错误：git add . 或忘了加 .gitignore
$ git add .
$ git commit -m "initial commit"
# .env 文件被提交了！API Key 泄露！
```

```bash
# ✓ 正确：先在 .gitignore 中加 .env，再 commit
$ echo ".env" >> .gitignore
$ git add .gitignore 其他文件
$ git commit -m "initial commit"
# .env 不会被追踪
```

### 错误 3：pip install 后 import 还是找不到包

```python
# ❌ 错误现象：
import requests  # ModuleNotFoundError: No module named 'requests'
```

**可能原因与解决：**
- 忘记激活虚境环境 → 激活后再 `pip install`
- VSCode 右下角选错了解释器 → `Ctrl+Shift+P` → `Python: Select Interpreter` → 选 `.venv/Scripts/python.exe`
- 安装到了全局但用虚拟环境运行 → 激活虚拟环境重新安装

### 错误 4：pip install 下载很慢

```bash
# ❌ 默认从 PyPI 官方源下载，国内可能只有几十 KB/s
pip install numpy
```

```bash
# ✓ 配置清华镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy  # 速度飞起
```

### 错误 5：load_dotenv() 放在读取之后

```python
# ❌ 错误：先读了 os.environ，再 load_dotenv，读不到！
api_key = os.environ["OPENAI_API_KEY"]   # KeyError!
load_dotenv()                             # 太晚了
```

```python
# ✓ 正确：先 load_dotenv()，再读取
from dotenv import load_dotenv
load_dotenv()                              # 先加载
api_key = os.environ["OPENAI_API_KEY"]     # 再读取
```

### 错误 6：用 pip freeze 的精确版本代替宽松版本

```toml
# ❌ 在 pyproject.toml 的 dependencies 里写精确版本
dependencies = [
    "requests==2.31.0",    # 太死板，别人装不上 2.32 会失败
]

# ✓ 开发给别人的库用宽松版本范围。精确版本只用于 requirements.txt
dependencies = [
    "requests>=2.28,<3.0", # 表达"兼容 2.28 到 3.0 之前的版本"
]
```
