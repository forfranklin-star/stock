# Streamlit Cloud 部署指南

本文档详细说明如何将 A股智能量化分析系统部署到 Streamlit Community Cloud。

---

## 一、部署前准备

### 1.1 所需账号

| 账号 | 用途 | 注册地址 |
|------|------|----------|
| GitHub 账号 | 托管代码仓库 | https://github.com |
| Streamlit 账号 | 部署应用（可用GitHub账号直接登录） | https://share.streamlit.io |

### 1.2 本地环境（用于测试）

- Python 3.9+
- Git

---

## 二、项目文件清单

部署到 Streamlit Cloud **必须**包含以下文件，全部位于仓库根目录：

```
your-repo/
├── streamlit_app.py          # ✅ 必需：Streamlit 主入口（文件名可自定义，但需在部署时指定）
├── analysis.py               # ✅ 必需：核心分析逻辑模块（被 streamlit_app.py import）
├── requirements.txt          # ✅ 必需：Python 依赖清单
├── packages.txt              # ⭐ 推荐：系统级依赖（apt 包），用于 lxml 等编译
├── .streamlit/
│   └── config.toml           # ⭐ 推荐：Streamlit 主题和服务器配置
├── .gitignore                # ⭐ 推荐：忽略缓存、虚拟环境等
└── README.md                 # 可选：项目说明
```

### 各文件详细说明

#### 1. `streamlit_app.py` — 主入口
Streamlit Cloud 会执行 `streamlit run <文件名>`。默认查找 `streamlit_app.py`，也可在部署时手动指定其他文件名。

本文件包含：
- 页面配置（标题、布局、主题）
- 侧边栏控制面板（股票代码、日期范围、高级参数）
- 7个分析结果标签页
- Plotly 交互式图表（K线、概率曲线、相关性、资金曲线）
- 实时信号面板

#### 2. `analysis.py` — 核心分析模块
被 `streamlit_app.py` 通过 `import analysis` 引入，包含全部量化分析逻辑：
- 数据获取（akshare，前复权，磁盘缓存）
- 23个技术指标计算
- 最优买卖点（贪心峰谷法）
- 相关性分析（点二列相关 + t检验）
- 数学模型（阈值加权打分系统）
- 历史回测（T+1执行，手续费+印花税）
- 预测准确率评估
- 实时信号生成

#### 3. `requirements.txt` — Python 依赖
Streamlit Cloud 自动执行 `pip install -r requirements.txt`。

```
streamlit>=1.30.0      # Web 框架
plotly>=5.18.0         # 交互式图表（K线、折线、柱状）
akshare>=1.12.0        # A股数据接口（免费，无需Token）
pandas>=2.0.0           # 数据处理
numpy>=1.24.0           # 数值计算
scipy>=1.10.0           # 统计检验（点二列相关、t检验）
scikit-learn>=1.3.0     # 机器学习工具
```

#### 4. `packages.txt` — 系统依赖
Streamlit Cloud 基于 Debian，自动执行 `apt-get install`。akshare 依赖的 `lxml` 库编译时需要以下系统包：

```
libxml2-dev
libxslt-dev
zlib1g-dev
```

> **注意**：如果缺少此文件，`pip install lxml` 可能因编译失败而导致部署报错。虽然 lxml 通常有预编译 wheel，但添加此文件可确保万无一失。

#### 5. `.streamlit/config.toml` — 应用配置
```toml
[server]
headless = true
port = 8501

[theme]
base = "light"
primaryColor = "#1976d2"
```

#### 6. `.gitignore` — 推荐内容
```
__pycache__/
*.pyc
cache/
*.pkl
.env
.venv/
venv/
```

---

## 三、部署步骤（图文流程）

### 步骤 1：创建 GitHub 仓库

1. 登录 https://github.com
2. 点击右上角 **+** → **New repository**
3. 填写：
   - **Repository name**: `stock-analyzer`（或任意名称）
   - **Description**: A股智能量化分析系统
   - **Public**（公开仓库，Streamlit Cloud 免费部署公开应用）
   - 勾选 **Add a README file**（可选）
4. 点击 **Create repository**

### 步骤 2：上传代码到 GitHub

**方式 A：网页上传（适合新手）**

1. 进入刚创建的仓库页面
2. 点击 **Add file** → **Upload files**
3. 将以下文件拖拽到上传区域：
   - `streamlit_app.py`
   - `analysis.py`
   - `requirements.txt`
   - `packages.txt`
   - `.streamlit/config.toml`（需先创建 `.streamlit` 文件夹：点击 Add file → Create new file，文件名输入 `.streamlit/config.toml`）
4. 点击 **Commit changes**

**方式 B：Git 命令行（推荐）**

```bash
# 1. 进入项目目录
cd /path/to/stock_analyzer

# 2. 初始化 Git（如果尚未初始化）
git init
git add .
git commit -m "Initial commit: A股智能量化分析系统"

# 3. 关联远程仓库
git branch -M main
git remote add origin https://github.com/你的用户名/stock-analyzer.git
git push -u origin main
```

### 步骤 3：登录 Streamlit Cloud

1. 访问 https://share.streamlit.io
2. 点击 **Sign up** 或 **Log in**
3. 选择 **Continue with GitHub**（使用 GitHub 账号登录）
4. 首次登录需要授权 Streamlit 访问你的 GitHub 仓库，点击 **Authorize streamlit**

### 步骤 4：创建应用（Deploy）

1. 登录后进入工作台，点击 **New app** 或 **Create app**
2. 填写部署配置：

   | 配置项 | 填写内容 |
   |--------|----------|
   | **Repository** | 选择你的 GitHub 仓库（如 `your-name/stock-analyzer`） |
   | **Branch** | `main`（或 `master`，根据你的默认分支） |
   | **Main file path** | `streamlit_app.py`（主入口文件名） |

3. 点击 **Advanced settings**（高级设置，可选）：
   - **Python version**: 选择 3.10 或 3.11（推荐 3.10，兼容性最好）
   - **Secrets**: 本应用无需密钥（akshare 免费接口不需要 Token），留空即可
4. 点击 **Deploy!** 按钮

### 步骤 5：等待部署完成

Streamlit Cloud 会自动执行以下操作：
1. 从 GitHub 拉取代码
2. 安装系统依赖（`packages.txt`）
3. 安装 Python 依赖（`requirements.txt`）— 这一步通常需要 2-5 分钟
4. 启动 Streamlit 应用

部署过程中可以看到实时日志。如果一切正常，页面会自动跳转到你的应用。

**部署成功后**，你将获得一个永久免费的 URL，格式如：
```
https://your-name-stock-analyzer-streamlit-app-xxxxx.streamlit.app
```

### 步骤 6：验证应用功能

1. 打开部署后的 URL
2. 在左侧输入股票代码（如 `600519`）
3. 选择日期范围
4. 点击 **开始分析**
5. 确认各标签页图表和数据正常显示

---

## 四、代码更新与重新部署

Streamlit Cloud 支持 **自动重新部署**：

1. 将更新后的代码推送到 GitHub：
   ```bash
   git add .
   git commit -m "更新说明"
   git push
   ```
2. Streamlit Cloud 检测到 GitHub 仓库变更后，会自动重新部署（通常 1-2 分钟内完成）
3. 也可以在应用页面右下角点击 **Manage app** → **Reboot** 手动重启

---

## 五、常见问题与排查

### 5.1 部署失败：`pip install` 报错

**现象**：日志中出现 `ERROR: Could not build wheels for lxml` 或类似编译错误。

**解决**：
- 确认 `packages.txt` 存在且包含 `libxml2-dev`、`libxslt-dev`
- 在应用设置中将 Python 版本降级到 3.10
- 检查 `requirements.txt` 中版本号是否过新，适当降低版本要求

### 5.2 应用启动后报错：`ModuleNotFoundError`

**现象**：页面显示 `ModuleNotFoundError: No module named 'xxx'`

**解决**：
- 确认 `requirements.txt` 中包含所有依赖
- 确认 `analysis.py` 与 `streamlit_app.py` 在同一目录
- 点击 **Manage app** → **Reboot** 重启应用

### 5.3 数据获取失败：akshare 接口报错

**现象**：点击分析后显示 "未获取到股票数据" 或网络超时。

**原因**：
- akshare 依赖东方财富等第三方数据源，Streamlit Cloud 的服务器 IP 可能被限流
- 某些时间段数据源接口不稳定

**解决**：
- 稍后重试
- 更换股票代码测试
- 本应用内置了24小时磁盘缓存，成功获取一次后后续会更快
- 如果持续失败，可考虑在 `analysis.py` 中添加重试机制或更换数据源

### 5.4 应用休眠（Sleeping）

**现象**：一段时间不访问后，应用显示 "This app is asleep"。

**原因**：Streamlit Cloud 免费版对超过7天无访问的应用会自动休眠以节省资源。

**解决**：
- 点击页面上的 **Yes, get this app back up!** 按钮即可唤醒（通常30秒内恢复）
- 定期访问应用可防止休眠
- 如需始终在线，可升级到 Streamlit Cloud 付费版

### 5.5 内存不足（OOM）

**现象**：应用崩溃，日志显示 `Killed` 或内存溢出。

**原因**：免费版应用内存限制为 1GB，分析长时间范围（如5年以上）的大量数据可能超出限制。

**解决**：
- 缩短分析时间范围（建议2年以内）
- 本应用已对K线数据做了采样（最多800根），降低内存占用
- 缓存目录 `cache/` 中的文件不会占用运行时内存

### 5.6 中文显示问题

**现象**：图表中文字符显示为方框或乱码。

**解决**：
- 本应用使用 Plotly 图表，默认支持中文显示，无需额外配置
- 如果出现问题，确认 `.streamlit/config.toml` 中 `font = "sans serif"`

---

## 六、本地测试（部署前必做）

在部署到 Streamlit Cloud 之前，强烈建议先在本地运行测试：

```bash
# 1. 进入项目目录
cd stock_analyzer

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行应用
streamlit run streamlit_app.py

# 5. 浏览器自动打开 http://localhost:8501
```

测试要点：
- [ ] 页面正常加载，侧边栏控件可用
- [ ] 输入股票代码和日期后能正常分析
- [ ] K线图、买卖点标注正常显示
- [ ] 各标签页切换正常
- [ ] 实时信号功能正常
- [ ] 无报错或异常警告

---

## 七、应用自定义

### 7.1 修改应用名称和 URL

在 Streamlit Cloud 工作台 → 应用设置 → **Settings** 中可修改：
- 应用显示名称
- 自定义子域名（如 `my-stock-app.streamlit.app`）

### 7.2 添加访问密码（可选）

如果不希望公开访问，可在应用设置中启用 **Private app** 或添加密码保护（付费功能）。

### 7.3 连接自定义域名

在应用设置 → **Custom domain** 中可绑定自己的域名（如 `stock.yourdomain.com`）。

---

## 八、文件清单速查表

部署前确认以下文件全部存在于 GitHub 仓库根目录：

| # | 文件名 | 必需 | 作用 |
|---|--------|------|------|
| 1 | `streamlit_app.py` | ✅ 必需 | 应用主入口 |
| 2 | `analysis.py` | ✅ 必需 | 分析逻辑模块 |
| 3 | `requirements.txt` | ✅ 必需 | Python依赖 |
| 4 | `packages.txt` | ⭐ 推荐 | 系统依赖 |
| 5 | `.streamlit/config.toml` | ⭐ 推荐 | 主题配置 |
| 6 | `.gitignore` | ⭐ 推荐 | 忽略缓存文件 |
| 7 | `README.md` | 可选 | 项目说明 |

---

## 九、部署检查清单

- [ ] GitHub 仓库已创建且为 Public
- [ ] 所有必需文件已上传到仓库根目录
- [ ] `requirements.txt` 包含 streamlit、plotly、akshare、pandas、numpy、scipy
- [ ] `packages.txt` 包含 libxml2-dev、libxslt-dev
- [ ] `streamlit_app.py` 中 `import analysis` 路径正确
- [ ] 本地运行测试通过
- [ ] Streamlit Cloud 已授权 GitHub 访问
- [ ] 部署时 Repository、Branch、Main file path 填写正确
- [ ] Python 版本选择 3.10（推荐）
- [ ] 部署日志无报错
- [ ] 应用 URL 可正常访问和使用

---

> **提示**：Streamlit Community Cloud 免费版适用于个人项目和演示。如果需要更高的内存、始终在线、私有应用等企业级功能，可考虑 Streamlit for Teams 或自行部署到云服务器（AWS/GCP/Azure/阿里云等）。
