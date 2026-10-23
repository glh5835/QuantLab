"""清洗模块测试：验证每个操作的效果与留痕。"""
import numpy as np
import pandas as pd
import pytest

from quantlab import cleaning, db
from quantlab.storage import save_raw, load_csv


@pytest.fixture
def dirty_dataset():
    """构造带缺失值、重复行、非正价格的脏数据集并注册。"""
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=10), name="trade_date")
    df = pd.DataFrame({
        "open": [10]*10, "high": [11]*10, "low": [9.5]*10,
        "close": [10.5]*10, "volume": [1000.0]*10,
    }, index=idx)
    df.iloc[2, df.columns.get_loc("close")] = np.nan   # 缺失
    df.iloc[3, df.columns.get_loc("open")] = -1        # 非正价格
    df.iloc[4] = df.iloc[1]                            # 完全重复行（索引不同则不算重复索引，值重复）
    path = save_raw(df, "T_dirty")
    ds_id = db.register_dataset("T_dirty", "DRT", source="demo")
    db.update_dataset(ds_id, raw_path=path, rows=len(df))
    return ds_id, df


def test_ffill_fixes_missing(dirty_dataset):
    ds_id, df = dirty_dataset
    rep = cleaning.run_cleaning(ds_id, ["ffill_missing"])
    assert rep["after_missing"] < rep["before_missing"]
    clean = load_csv(db.get_dataset(ds_id)["clean_path"])
    assert not clean["close"].isna().any()
    assert rep["changes"][0]["affected"] >= 1


def test_drop_nonpositive(dirty_dataset):
    ds_id, df = dirty_dataset
    rep = cleaning.run_cleaning(ds_id, ["drop_nonpositive"])
    clean = load_csv(db.get_dataset(ds_id)["clean_path"])
    # NaN 与 0 比较结果为 False，因此用 `<= 0` 判定：清洗后不应再存在非正价格
    assert not ((clean[["open", "high", "low", "close"]] <= 0).any().any())
    assert rep["after_rows"] == rep["before_rows"] - 1


def test_raw_preserved_after_cleaning(dirty_dataset):
    ds_id, df = dirty_dataset
    cleaning.run_cleaning(ds_id, ["drop_nonpositive", "ffill_missing"])
    raw = load_csv(db.get_dataset(ds_id)["raw_path"])
    assert raw.isna().sum().sum() > 0          # 原始数据的缺失仍在
    assert "\\raw\\" in db.get_dataset(ds_id)["raw_path"].replace("/", "\\")
    assert "\\clean\\" in db.get_dataset(ds_id)["clean_path"].replace("/", "\\")


def test_cleaning_log_recorded(dirty_dataset):
    ds_id, _ = dirty_dataset
    rep = cleaning.run_cleaning(ds_id, ["ffill_missing"])
    import json
    log = json.loads(db.get_dataset(ds_id)["cleaning_log"])
    assert log["changes"][0]["op"] == "ffill_missing"
    assert "reason" in log["changes"][0]  # 每个操作必须记录依据


def test_fix_ohlc():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=3), name="trade_date")
    df = pd.DataFrame({
        "open": [10, 10, 10], "high": [11, 5, 11],  # 第2行 high<open 矛盾
        "low": [9, 9.5, 9], "close": [10.5, 10.2, 10.5], "volume": [1e3]*3,
    }, index=idx)
    path = save_raw(df, "T_ohlc")
    ds_id = db.register_dataset("T_ohlc", "OHL", source="demo")
    db.update_dataset(ds_id, raw_path=path)
    cleaning.run_cleaning(ds_id, ["fix_ohlc"])
    clean = load_csv(db.get_dataset(ds_id)["clean_path"])
    assert (clean["high"] >= clean[["open", "close"]].max(axis=1)).all()
    assert (clean["low"] <= clean[["open", "close"]].min(axis=1)).all()
