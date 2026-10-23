"""QuantLab 全局配置：所有路径与常量集中管理，避免硬编码散落各处。"""
from pathlib import Path

# 项目根目录 = quantlab 包的上一级
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"        # 原始数据（只读，永不修改）
CLEAN_DIR = DATA_DIR / "clean"    # 清洗后数据（与原始数据分开保存）
EXPORT_DIR = DATA_DIR / "exports" # 导出的分析结果
DB_PATH = DATA_DIR / "quantlab.db"

# 确保目录存在
for d in (DATA_DIR, RAW_DIR, CLEAN_DIR, EXPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

APP_NAME = "QuantLab Student"
APP_VERSION = "0.1.0"

# 统一行情数据字段（所有适配器必须输出该结构）
# index: trade_date (DatetimeIndex, 已按时间升序)
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# 质量检查阈值（可解释的默认值，界面中可查看）
QUALITY_THRESHOLDS = {
    "extreme_daily_return": 0.30,   # 单日涨跌幅超过 ±30% 视为可疑（A股有涨跌停限制，美股无）
    "zero_volume_warn_ratio": 0.05, # 零成交天数占比超过 5% 提醒
    "max_missing_ratio": 0.10,      # 单字段缺失率超过 10% 视为严重问题
}
