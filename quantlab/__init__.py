"""QuantLab Student — 大学生量化研究与回测平台（后端核心包）。

分层结构：
- config.py      配置与路径
- db.py          SQLite 元数据（数据集/任务/日志）
- storage.py     行情文件读写（原始与清洗数据物理隔离）
- datasources/   可替换数据适配器（yfinance / CSV 导入 / 演示模拟）
- collector.py   采集任务编排（下载→校验→保存→注册，支持增量与取消）
- tasks.py       后台任务执行器（线程 + SQLite 状态）
- quality.py     数据质量检查与评分
- cleaning.py    显式清洗操作（留痕、可对比）
- analysis.py    EDA 计算层（统计/相关/回撤/滚动指标）
"""
__version__ = "0.1.0"

from . import db  # noqa: F401

db.init_db()
