# QuantLab Student — 量化研究与回测平台

面向大学生的端到端量化研究工作流平台：数据采集 → 质量检查 → 数据清洗 → 探索性分析 → 特征工程 → 机器学习 → 策略回测 → 组合风险 → 报告导出 → 模拟交易与任务调度。

**当前进度：阶段 4 已完成（模拟交易 / 任务调度 / PDF 报告 / 均值-方差优化，181 个测试全过）。** 前序：阶段 3（组合/风险/优化/稳健性）、阶段 2（指标/ML/回测）、阶段 1（数据链路）。详见 `docs/` 下各阶段交付说明。

---

## 一、安装依赖与启动

### 1. 环境要求

- **Python 3.10 或更高版本**（推荐使用 64 位 Python 3.11 或 3.12；其他版本及操作系统尚需自行验证）。
- Windows 10/11、macOS 或 Linux；需要 pip 和可创建虚拟环境的 Python。
- 安装第三方依赖时需要能连接 Python 包镜像；**使用内置演示数据/本地 CSV 不需要 Yahoo 网络或 API Key**。
- 全功能所需依赖集中在根目录 `requirements.txt`，其中包括机器学习、组合优化、Excel 导入、PDF 导出及测试工具。依赖采用最低版本约束，不是经过锁定的可复现环境文件。

### 2. Windows（PowerShell，推荐）

在终端中执行以下命令（可以复制运行，**不需要激活虚拟环境**）：

```powershell
git clone https://github.com/glh5835/QuantLab.git
cd QuantLab

# 若未安装 Git，可从 GitHub 下载 ZIP 解压，再在解压目录打开终端。
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 检查依赖冲突并运行测试
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests/ -q

# 启动 Streamlit
.\.venv\Scripts\python.exe -m streamlit run app/Home.py
```

若机器上没有 Python 3.11，可以安装该版本，或将 `py -3.11` 改为本机实际可用的 `py -3.12` / `py -3.10`；先执行 `py -0p` 查看 Windows 已安装版本。项目首页一般由浏览器打开 `http://localhost:8501`（端口被占用时以终端实际输出为准）。

**双击 `启动QuantLab.bat` 的注意事项：**该批处理使用 PATH 中的 `python`，**不会自动安装依赖，也不会自动选用 `.venv`**。按上面的命令用虚拟环境启动最可靠；如果想通过批处理启动，应确保 PATH/当前已激活环境对应安装了依赖的 Python。

### 3. macOS / Linux

```bash
git clone https://github.com/glh5835/QuantLab.git
cd QuantLab

python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip check
./.venv/bin/python -m pytest tests/ -q
./.venv/bin/python -m streamlit run app/Home.py
```

安装完成后浏览器打开终端提示的本地地址；若 `python3` 不是兼容版本，请改用 `python3.11` 或 `python3.12` 创建虚拟环境。

### 4. 依赖对应功能

| 功能 | 关键依赖 | 说明 |
|---|---|---|
| Web 页面、数据处理、图表 | `streamlit`、`pandas`、`numpy`、`plotly` | 首页及全部工作台 |
| Yahoo 在线行情 | `yfinance`、`requests` | 网络不可用时仍可导入 CSV 或使用模拟数据 |
| Excel 数据导入 | `openpyxl`、`xlrd` | `.xlsx/.xlsm` 使用 openpyxl；旧 `.xls` 使用 xlrd |
| 机器学习实验 | `scikit-learn`、`joblib` | 特征预处理、模型训练与实验模型保存 |
| 均值—方差优化 | `scipy`（及 `scikit-learn`） | SLSQP 求解、可选 Ledoit-Wolf 协方差收缩 |
| PDF 研究报告 | `reportlab`、`matplotlib` | 报告生成、图表绘制；中文字体由模块按可用性回退 |
| 自动化测试 | `pytest`、`pypdf` | 包括 PDF 内容读取验证；非应用运行必需 |

### 5. 常见安装问题与验证口径

- **`No module named ...`**：检查启动应用和安装依赖是否使用同一个 `.venv` Python；不要混用全局 `pip` 与虚拟环境。
- **`pip` 安装失败或依赖冲突**：确认 Python 版本、网络及系统架构，使用 `python -m pip check` 检查已安装依赖；可删除失败的 `.venv` 后重新创建，**不要删除项目数据目录**。
- **Yahoo 行情采集失败**：不等于项目未安装成功；可使用演示数据或 CSV 导入。在线采集需单独确认网络可达。
- **Excel 无法读取**：检查文件实际格式，`.xlsx/.xlsm` 需 `openpyxl`，`.xls` 需 `xlrd`；仓库中的旧 `.xls` 自动化用例因缺少合法测试样本会跳过，不能视为格式已完整实测。
- **PDF 中文字体失败**：模块优先项目授权字体或系统字体，再使用内置 CID 字体；全部不可用时会拒绝生成并提示替代导出。
- **验收依据**：阶段 4 交付文档记录过 `181 passed`，**但安装教程更新本身不代表已经在全新环境复测**；请以本机 `pytest` 实际输出为准。

## 二、5 分钟演示路线（新手必读）

1. **🏠 统一工作台**：查看数据集数量、任务记录、异常提示（所有数字来自本地 SQLite 真实记录，无数据时显示空状态）。
2. **📊 数据采集中心**：
   - 有网络：选择 "Yahoo Finance"，输入代码如 `AAPL` / `600519.SS` / `SPY`，点击开始采集；
   - 无网络：选择"示例数据"标签页，采集内置模拟数据（**明确标注为模拟，非真实行情**）；
   - 观察任务面板的进度、日志、取消与失败重试。
3. **🗂️ 数据集管理**：选中 `DEMO_DIRTY_daily`（内置脏数据示例）→ 运行质量检查 → 查看评分 15/100 与 5 类问题清单 → 勾选清洗操作执行 → 查看清洗前后对比（原始数据永远保留，清洗结果另存）。
4. **📈 自动分析**：多选数据集 → 一键运行分析 → 查看价格/成交量、描述统计、相关性热力图、滚动波动率、最大回撤曲线、缺失值图 → 导出 CSV。

## 三、项目结构

```
quantlab/                  后端核心包（与界面分离，可独立测试）
├── config.py              路径与阈值配置
├── db.py                  SQLite 元数据（数据集/任务/日志）
├── storage.py             行情文件读写（raw 与 clean 物理隔离）
├── datasources/           可替换数据适配器
│   ├── base.py            适配器抽象基类 + 注册表
│   ├── yfinance_adapter.py  Yahoo Finance 在线适配器（重试/限流处理）
│   ├── csv_adapter.py     本地文件导入（CSV/Excel xlsx/xls，中英文列名）
│   └── demo_adapter.py    示例数据适配器（固定种子模拟，明确标注）
├── collector.py           采集编排（下载→校验→保存→注册，增量更新/取消）
├── tasks.py               后台任务执行器（线程 + SQLite 状态恢复）
├── quality.py             数据质量检查与评分（7 类问题）
├── cleaning.py            显式清洗（操作留痕 + 前后对比）
├── analysis.py            EDA 计算（统计/相关/回撤/滚动/夏普）
├── indicators.py          统一技术指标接口（SMA/EMA/RSI/MACD/布林带/波动率/收益族）
├── features.py            ML 特征工程（全部因果特征，防未来数据）
├── ml_lab.py              机器学习实验室（时间三分+purge / Pipeline 防泄漏 / 实验存档）
├── strategies.py          策略统一接口（买入持有 / 均线交叉 / ML 信号）
├── backtest.py            回测引擎（t+1 成交 / 费用滑点 / 风控 / 绩效 / 存档）
├── portfolio.py           多资产组合与再平衡（等权/自定义/逆波动率）
├── risk.py                高级风险（指标/回撤区间/分段/压力测试/预警）
├── optimizer.py           参数网格搜索（验证段选参，测试段仅终评）
├── robustness.py          滚动样本外验证（规则+ML，每窗口独立训练）
├── reports.py             实验比较与 HTML/CSV 研究报告
├── paper_trading.py       虚拟账户与模拟交易（仅虚拟资金）
├── scheduler.py           任务调度（配置/执行分离+租约单实例）
├── pdf_report.py          PDF 报告（三级中文字体回退）
└── mvo.py                 均值-方差优化（SLSQP+有效前沿+滚动重估）

app/                       Streamlit 界面层（不含核心计算逻辑）
├── Home.py                模块1：统一工作台
└── pages/
    ├── 1_📊_数据采集中心.py  模块2：数据采集中心
    ├── 2_🗂️_数据集管理.py    模块3：数据集管理与质量
    ├── 3_📈_自动分析.py      模块4：自动分析工作台
    ├── 4_📐_技术指标.py      阶段2-A：技术指标中心
    ├── 5_🤖_机器学习实验室.py 阶段2-B：机器学习实验室
    ├── 6_⚖️_策略回测.py      阶段2-C：策略回测
    ├── 7_📁_投资组合中心.py   阶段3-A：组合与再平衡
    ├── 8_🛡️_风险分析.py      阶段3-B：风险分析与压力测试
    ├── 9_🎛️_参数优化.py      阶段3-C：网格搜索与敏感性
    ├── 10_🔁_稳健性验证.py    阶段3-D：滚动样本外验证
    ├── 11_🆚_实验比较.py     阶段3-E：多实验比较与报告
    ├── 12_💼_模拟交易.py     阶段4-A：虚拟账户与模拟交易
    └── 13_⏰_任务调度.py     阶段4-B：任务调度中心

tests/                     pytest 测试（阶段 4 交付记录：181 个通过）
data/                      运行时数据（不入 Git）
├── raw/                   原始数据（只读）
├── clean/                 清洗后数据
├── exports/               导出结果
└── quantlab.db            SQLite 元数据库
docs/                      阶段交付文档
```

## 四、关键设计决策（教学说明）

| 决策 | 理由 |
|---|---|
| 适配器模式解耦数据源 | 分析/回测逻辑只依赖标准 OHLCV 契约，换数据源不改核心代码 |
| 原始与清洗数据物理分离 | `data/raw` 只读、`data/clean` 另存，任何清洗可回溯 |
| 质量检查只报告不修改 | 清洗是显式勾选操作，每项附依据并写入清洗日志 |
| 模拟数据处处标注 | `is_synthetic` 标志贯穿数据库、界面徽标、许可说明 |
| 任务状态存 SQLite | 程序崩溃重启后可恢复任务状态，失败可一键重试 |
| 核心计算不写界面层 | `quantlab` 包可被 pytest 直接测试，也可被未来回测引擎复用 |

## 五、已知限制

- yfinance 免费接口存在限流，且部分网络环境无法访问 Yahoo（此时用示例数据/CSV 导入）。
- Streamlit `use_container_width` 参数有弃用警告（2025-12 后需改 `width='stretch'`），不影响功能。
- 指数/停牌复权等深水区问题本阶段仅做检测提示，深入处理留到后续阶段。

## 六、后续改进方向

阶段 1—4 的功能与验收记录详见 `docs/`。后续可考虑完善模拟订单类型与部分成交、调度失败自动重试、集中化报告模板及高级统计检验；这些功能不应被误认为当前已实现。