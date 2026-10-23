"""数据源适配器基类与注册表。

设计要点（教学解释）：
"可替换的数据适配器"意思是：采集、分析、回测的核心逻辑只依赖 BaseDataAdapter
返回的标准 OHLCV DataFrame，不依赖任何具体供应商。以后想换数据源，只需新写一个
子类并在 ADAPTERS 里注册，其余代码一行不改。
"""
from abc import ABC, abstractmethod

import pandas as pd


class BaseDataAdapter(ABC):
    """所有数据源适配器的抽象基类。"""

    #: 适配器唯一标识（写入 datasets.source 字段）
    source_id: str = "base"
    #: 展示名称
    display_name: str = "基类"
    #: 数据许可 / 来源说明（必须如实填写，界面会展示）
    license_note: str = ""
    #: 字段说明
    field_desc: str = (
        "trade_date=交易日期; open=开盘价; high=最高价; low=最低价; close=收盘价; volume=成交量"
    )
    #: 是否模拟数据（模拟数据在所有界面必须显著标注）
    is_synthetic: bool = False

    @abstractmethod
    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """下载 [start, end] 区间日线行情。

        返回：DatetimeIndex（升序、无时区）+ open/high/low/close/volume 列。
        失败时必须抛出 DataFetchError，附带人类可读原因与建议。
        """
        raise NotImplementedError

    def check_availability(self) -> dict:
        """可用性检查：返回 {ok: bool, message: str}。默认实现为空操作。"""
        return {"ok": True, "message": "未实现检查，默认视为可用"}


class DataFetchError(RuntimeError):
    """数据获取失败统一异常：message 需包含原因与建议。"""


ADAPTERS: dict[str, BaseDataAdapter] = {}


def register_adapter(cls):
    """类装饰器：把适配器实例注册进全局表。"""
    instance = cls()
    ADAPTERS[instance.source_id] = instance
    return cls


def get_adapter(source_id: str) -> BaseDataAdapter:
    if source_id not in ADAPTERS:
        raise DataFetchError(f"未知数据源: {source_id}。已注册: {list(ADAPTERS)}")
    return ADAPTERS[source_id]
