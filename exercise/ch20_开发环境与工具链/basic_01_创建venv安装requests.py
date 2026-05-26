"""
章节：第20章 开发环境与工具链
题目：创建虚拟环境并安装 requests
类型：基础练习

题目描述：
在桌面上创建一个 `learn-venv` 目录，用 venv 创建虚拟环境，激活它，安装 `requests` 包，验证安装成功。

前置准备：
需要已安装 Python 3.3+（venv 是内置模块，无需额外安装）。

提示：
- 本练习主要是命令行操作，下面的代码用注释展示了每一步的具体命令。
- Windows PowerShell 和 Linux/Mac 的命令不同，已分别列出。
- 激活虚拟环境后，命令行的提示符前会出现 `(.venv)` 标记。
- 如果 PowerShell 报"无法加载文件"错误，需要先运行：
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
- 退出虚拟环境：deactivate
"""

# ========== 命令行操作步骤 ==========

"""
=== Windows PowerShell 下执行 ===

# 第 1 步：创建项目目录
mkdir $env:USERPROFILE\Desktop\learn-venv
cd $env:USERPROFILE\Desktop\learn-venv

# 第 2 步：创建虚拟环境
python -m venv .venv

# 第 3 步：激活虚拟环境
.venv\Scripts\Activate.ps1
# 激活成功后，提示符前会出现 (.venv) 标记

# 第 4 步：在虚拟环境中安装 requests
pip install requests

# 第 5 步：验证安装成功
python -c "import requests; print(f'requests 版本: {requests.__version__}')"
# 输出类似：requests 版本: 2.31.0

# 第 6 步：退出虚拟环境
deactivate


=== Linux/Mac 下执行 ===

# 第 1 步：创建项目目录
mkdir ~/Desktop/learn-venv
cd ~/Desktop/learn-venv

# 第 2 步：创建虚拟环境
python3 -m venv .venv

# 第 3 步：激活虚拟环境
source .venv/bin/activate

# 第 4 步：在虚拟环境中安装 requests
pip install requests

# 第 5 步：验证安装成功
python -c "import requests; print(f'requests 版本: {requests.__version__}')"

# 第 6 步：退出虚拟环境
deactivate
"""

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 按以上命令行步骤在桌面创建 learn-venv 并配置虚拟环境
# 2. 在虚拟环境中安装 requests 包
# 3. 写一个验证脚本，检查 sys.executable 和 requests.__version__
#
# 提示：
# - Windows 激活：.venv\Scripts\Activate.ps1
# - Linux/Mac 激活：source .venv/bin/activate
# - 退出虚拟环境：deactivate
# - 可参考第 20.1 节"虚拟环境入门"
