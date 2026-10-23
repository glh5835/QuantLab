"""CSV 导入适配器：把用户上传的本地行情 CSV 规范化为统一 OHLCV 格式。

支持两种常见格式：
A. 中文表头：日期,开盘,最高,最低,收盘,成交量
B. 英文表头：date,open,high,low,close,volume（大小写不限）
其余格式会被拒绝并给出明确提示（不猜测、不静默修改）。
"""
import pandas as pd

from .base import BaseDataAdapter, DataFetchError, register_adapter
from ..storage import validate_ohlcv

_COL_MAP = {
    # 中文 / 常见别名 → 标准字段
    "日期": "trade_date", "date": "trade_date", "trade_date": "trade_date", "时间": "trade_date",
    "开盘": "open", "开盘价": "open", "open": "open",
    "最高": "high", "最高价": "high", "high": "high",
    "最低": "low", "最低价": "low", "low": "low",
    "收盘": "close", "收盘价": "close", "close": "close", "adj close": "close",
    "成交量": "volume", "volume": "volume", "vol": "volume", "qty": "volume",
}


@register_adapter
class CsvImportAdapter(BaseDataAdapter):
    source_id = "csv_import"
    display_name = "CSV 文件导入"
    license_note = "用户本地文件，数据来源与许可由用户自行负责。"

    def fetch(self, symbol: str, start=None, end=None, **_) -> pd.DataFrame:
        raise DataFetchError("CSV 适配器不支持在线下载，请使用 import_csv() 导入本地文件。")

    def import_csv(self, file_path: str) -> pd.DataFrame:
        """读取并规范化用户 CSV，返回标准 OHLCV DataFrame。"""
        try:
            df = pd.read_csv(file_path)
        except Exception as e:  # noqa: BLE001
            raise DataFetchError(f"无法读取文件 {file_path}：{e}") from e

        # 列名规范化（去空格、统一小写后查映射）
        renamed = {}
        for col in df.columns:
            key = str(col).strip().lower()
            std = _COL_MAP.get(key) or _COL_MAP.get(str(col).strip())
            if std:
                renamed[col] = std
        df = df.rename(columns=renamed)

        need = {"trade_date", "open", "high", "low", "close", "volume"}
        missing = need - set(df.columns)
        if missing:
            raise DataFetchError(
                f"CSV 缺少必需列 {sorted(missing)}。支持中文列（日期/开盘/最高/最低/收盘/成交量）"
                f"或英文列（date/open/high/low/close/volume）。")
        try:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
        except Exception as e:  # noqa: BLE001
            raise DataFetchError(f"日期列无法解析：{e}。请使用 YYYY-MM-DD 格式。") from e

        df = df.set_index("trade_date").sort_index()
        df.index.name = "trade_date"
        df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
        df = df[~df.index.duplicated(keep="first")]
        validate_ohlcv(df)
        return df
