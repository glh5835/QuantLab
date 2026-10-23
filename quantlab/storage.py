"""行情数据文件读写层：统一 CSV 存取，负责"原始数据只读、清洗数据另存"的物理隔离。"""
import pandas as pd

from . import config
from .config import OHLCV_COLUMNS


def save_raw(df: pd.DataFrame, name: str) -> str:
    """保存原始数据到 data/raw/<name>.csv，返回文件路径。"""
    path = config.RAW_DIR / f"{name}.csv"
    df.to_csv(path, index_label="trade_date")
    return str(path)


def save_clean(df: pd.DataFrame, name: str) -> str:
    """保存清洗后数据到 data/clean/<name>.csv（绝不覆盖 raw 目录）。"""
    path = config.CLEAN_DIR / f"{name}_clean.csv"
    df.to_csv(path, index_label="trade_date")
    return str(path)


def load_csv(path: str) -> pd.DataFrame:
    """读取行情 CSV：trade_date 解析为 DatetimeIndex 并升序排列。"""
    df = pd.read_csv(path, parse_dates=["trade_date"], index_col="trade_date")
    df = df.sort_index()
    df.index = df.index.tz_localize(None)  # 去掉时区信息，统一比较口径
    return df


def validate_ohlcv(df: pd.DataFrame) -> None:
    """校验 DataFrame 是否符合统一 OHLCV 契约，不符合则抛出带说明的异常。"""
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少必需字段: {missing}。统一字段为 {OHLCV_COLUMNS}（小写）。")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("数据索引必须是 DatetimeIndex（交易日期）。")
    if df.index.has_duplicates:
        raise ValueError("索引中存在重复交易日期。")
