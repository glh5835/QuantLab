"""本地文件导入适配器：支持 CSV 与 Excel（.xlsx/.xls），统一规范化为 OHLCV。

支持的表头（CSV 与 Excel 通用，大小写不限）：
A. 中文表头：日期,开盘,最高,最低,收盘,成交量
B. 英文表头：date,open,high,low,close,volume
其余格式会被拒绝并给出明确提示（不猜测、不静默修改）。

依赖：.xlsx 用 openpyxl；.xls 老格式用 xlrd（缺失时给明确的安装指引）。
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


def _read_table(path: str) -> pd.DataFrame:
    """按扩展名读取表格文件为 DataFrame（不规范化）。"""
    p = str(path).lower()
    try:
        if p.endswith(".csv"):
            return pd.read_csv(path)
        if p.endswith(".xlsx") or p.endswith(".xlsm"):
            return pd.read_excel(path, engine="openpyxl")
        if p.endswith(".xls"):
            try:
                return pd.read_excel(path, engine="xlrd")
            except ImportError as e:
                raise DataFetchError("读取 .xls 需要 xlrd 库。请运行: pip install xlrd，"
                                     "或将文件另存为 .xlsx/.csv 后重试。") from e
        raise DataFetchError(f"不支持的文件类型 {path}：请使用 .csv / .xlsx / .xls。")
    except DataFetchError:
        raise
    except Exception as e:  # noqa: BLE001
        raise DataFetchError(f"无法读取文件 {path}：{e}") from e


def _normalize(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """列名映射 → 类型转换 → 去重排序 → 契约校验。"""
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
            f"{source_name} 缺少必需列 {sorted(missing)}。支持中文列（日期/开盘/最高/最低/收盘/成交量）"
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


def import_file(path: str) -> pd.DataFrame:
    """统一导入入口：CSV / .xlsx / .xls → 标准 OHLCV DataFrame。"""
    return _normalize(_read_table(path), str(path))


# 向后兼容别名（阶段 1 起的调用方）
def import_csv(file_path: str) -> pd.DataFrame:
    return import_file(file_path)


@register_adapter
class CsvImportAdapter(BaseDataAdapter):
    source_id = "csv_import"
    display_name = "本地文件导入（CSV / Excel）"
    license_note = "用户本地文件（CSV/XLSX/XLS），数据来源与许可由用户自行负责。"

    def fetch(self, symbol: str, start=None, end=None, **_) -> pd.DataFrame:
        raise DataFetchError("文件导入适配器不支持在线下载，请使用 import_file() 导入本地文件。")

    def import_csv(self, file_path: str) -> pd.DataFrame:
        return import_file(file_path)
