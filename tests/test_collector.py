"""采集与存储链路测试：示例适配器 → 采集任务 → 数据集注册 → 增量更新 → CSV 导入适配器。"""
import threading

import numpy as np
import pandas as pd
import pytest

from quantlab import collector, db
from quantlab.datasources.csv_adapter import CsvImportAdapter
from quantlab.datasources.demo_adapter import DemoAdapter
from quantlab.datasources.base import DataFetchError
from quantlab.storage import load_csv, save_raw, save_clean, validate_ohlcv


def _cancel_event():
    e = threading.Event()
    return e


def test_demo_adapter_deterministic():
    a = DemoAdapter()
    df1 = a.fetch("DEMO_ETF", "2024-01-01", "2024-06-30")
    df2 = a.fetch("DEMO_ETF", "2024-01-01", "2024-06-30")
    pd.testing.assert_frame_equal(df1, df2)  # 固定种子 → 演示可复现
    assert list(df1.columns) == ["open", "high", "low", "close", "volume"]
    assert a.is_synthetic is True


def test_collect_task_registers_dataset(tmp_path):
    collector.run_collect_task(9999, _cancel_event(), "DEMO_TECH", "demo",
                               "2024-01-01", "2024-12-31", "stock", "T_demo1")
    ds = db.find_dataset("T_demo1")
    assert ds is not None
    assert ds["is_synthetic"] == 1
    assert ds["rows"] > 0
    assert ds["raw_path"].endswith("T_demo1.csv")
    df = load_csv(ds["raw_path"])
    assert str(df.index.min().date()) == ds["start_date"]


def test_incremental_update_extends_data():
    collector.run_collect_task(9998, _cancel_event(), "DEMO_BANK", "demo",
                               "2024-01-01", "2024-06-30", "stock", "T_demo2")
    ds1 = db.find_dataset("T_demo2")
    # 第二次采集：区间向后扩展 → 应拼接而不是覆盖前半段
    collector.run_collect_task(9997, _cancel_event(), "DEMO_BANK", "demo",
                               "2024-01-01", "2024-12-31", "stock", "T_demo2")
    ds2 = db.find_dataset("T_demo2")
    assert ds2["rows"] > ds1["rows"]
    assert ds2["version"] == ds1["version"] + 1  # 版本号递增
    assert ds2["start_date"] == ds1["start_date"]  # 历史区间保留


def test_cancel_before_download_stops_task():
    ev = threading.Event()
    ev.set()  # 预先取消
    collector.run_collect_task(9996, ev, "DEMO_ETF", "demo",
                               "2024-01-01", "2024-12-31", "stock", "T_demo3")
    assert db.find_dataset("T_demo3") is None  # 取消后不应注册数据集


def test_validate_ohlcv_rejects_bad_schema():
    df = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"], name="trade_date"))
    with pytest.raises(ValueError, match="缺少必需字段"):
        validate_ohlcv(df)


def test_raw_and_clean_are_separate_files():
    idx = pd.bdate_range("2024-01-01", periods=5)
    df = pd.DataFrame({"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 100.0}, index=idx)
    raw = save_raw(df, "T_sep")
    clean = save_clean(df, "T_sep")
    assert "\\raw\\" in raw.replace("/", "\\")
    assert "\\clean\\" in clean.replace("/", "\\")
    assert raw != clean


def test_csv_import_chinese_columns(tmp_path):
    f = tmp_path / "cn.csv"
    pd.DataFrame({
        "日期": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "开盘": [10.0, 10.5, 11.0], "最高": [10.8, 11.0, 11.5],
        "最低": [9.9, 10.2, 10.8], "收盘": [10.5, 11.0, 11.2],
        "成交量": [1000, 1200, 900],
    }).to_csv(f, index=False)
    df = CsvImportAdapter().import_csv(str(f))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing


def test_csv_import_missing_column_rejected(tmp_path):
    f = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2024-01-01"], "close": [10.0]}).to_csv(f, index=False)
    with pytest.raises(DataFetchError, match="缺少必需列"):
        CsvImportAdapter().import_csv(str(f))


def test_dataset_crud_and_search():
    i = db.register_dataset("T_crud", "XYZ", source="demo", is_synthetic=1)
    assert db.find_dataset("T_crud")["id"] == i
    db.update_dataset(i, rows=42)
    assert db.get_dataset(i)["rows"] == 42
    assert len(db.list_datasets(keyword="crud")) == 1
    res = db.delete_dataset(i)
    assert db.find_dataset("T_crud") is None
    assert res["name"] == "T_crud"
