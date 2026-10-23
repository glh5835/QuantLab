"""数据质量检查测试：用人工构造的脏数据验证每类问题都能被发现、评分正确。"""
import numpy as np
import pandas as pd

from quantlab.quality import check_quality


def _clean_df(n=60):
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = 100 + np.cumsum(np.random.default_rng(7).normal(0, 1, n))
    return pd.DataFrame({
        "open": close * 0.99, "high": close * 1.01, "low": close * 0.98,
        "close": close, "volume": 1_000_000.0,
    }, index=pd.DatetimeIndex(idx, name="trade_date"))


def test_clean_data_gets_full_score():
    rep = check_quality(_clean_df())
    assert rep["score"] == 100
    assert rep["issue_count"] == 0


def test_missing_values_detected():
    df = _clean_df()
    df.iloc[5, df.columns.get_loc("close")] = np.nan
    rep = check_quality(df)
    types = {i["type"] for i in rep["issues"]}
    assert "missing_values" in types
    iss = next(i for i in rep["issues"] if i["type"] == "missing_values")
    assert iss["count"] == 1
    assert rep["score"] < 100


def test_nonpositive_price_detected_as_severe():
    df = _clean_df()
    df.iloc[3, df.columns.get_loc("close")] = -5
    rep = check_quality(df)
    iss = next(i for i in rep["issues"] if i["type"] == "nonpositive_price")
    assert iss["severity"] == "严重"


def test_ohlc_inconsistency_detected():
    df = _clean_df()
    df.iloc[7, df.columns.get_loc("high")] = df.iloc[7]["close"] * 0.5  # high < close
    rep = check_quality(df)
    assert "ohlc_inconsistent" in {i["type"] for i in rep["issues"]}


def test_extreme_return_detected():
    df = _clean_df()
    df.iloc[10, df.columns.get_loc("close")] *= 2.0  # +100% 单日涨幅
    rep = check_quality(df)
    assert "extreme_returns" in {i["type"] for i in rep["issues"]}


def test_date_gap_detected():
    df = _clean_df()
    # 删掉 15 个交易日，制造 3 周（>10 自然日）的断档
    df = df.drop(df.index[20:35])
    rep = check_quality(df)
    assert "date_gap" in {i["type"] for i in rep["issues"]}


def test_zero_volume_ratio_warns():
    df = _clean_df(n=100)
    df.iloc[:20, df.columns.get_loc("volume")] = 0  # 20% 零成交
    rep = check_quality(df)
    assert "zero_volume" in {i["type"] for i in rep["issues"]}


def test_score_floor_at_zero():
    """所有问题叠加时分数不会为负。"""
    df = _clean_df(100)
    df.iloc[1, df.columns.get_loc("close")] = np.nan
    df.iloc[2, df.columns.get_loc("close")] = -1
    df.iloc[3, df.columns.get_loc("high")] = 1.0
    df.iloc[10, df.columns.get_loc("close")] *= 2
    df = df.drop(df.index[20:35])
    df.iloc[:25, df.columns.get_loc("volume")] = 0
    rep = check_quality(df)
    assert rep["score"] >= 0
