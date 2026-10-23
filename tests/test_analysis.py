"""分析模块测试：用已知数值人工构造行情，验证指标计算的正确性。"""
import numpy as np
import pandas as pd

from quantlab import analysis as A


def _df_from_prices(prices):
    p = np.asarray(prices, dtype=float)  # 用数组避免 Series 索引与 DatetimeIndex 对齐产生 NaN
    return pd.DataFrame({
        "open": p * 0.99, "high": p * 1.02, "low": p * 0.98,
        "close": p, "volume": 1e6,
    }, index=pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=len(p)), name="trade_date"))


def test_daily_returns_basic():
    df = _df_from_prices([100, 110, 99])
    r = A.daily_returns(df)
    assert np.isclose(r.iloc[0], 0.10)
    assert np.isclose(r.iloc[1], 99 / 110 - 1)


def test_max_drawdown_known_value():
    # 峰值 120，随后跌到 60 → 最大回撤 = 60/120 - 1 = -50%
    df = _df_from_prices([100, 110, 120, 90, 60, 80])
    mdd = A.max_drawdown(df)
    assert np.isclose(mdd["mdd"], -0.5)
    assert mdd["peak_date"] == str(df.index[2].date())
    assert mdd["trough_date"] == str(df.index[4].date())


def test_max_drawdown_monotonic_up_is_zero():
    df = _df_from_prices(range(100, 130))
    assert A.max_drawdown(df)["mdd"] == 0.0


def test_descriptive_stats_fields():
    df = _df_from_prices(100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 300)))
    stats = A.descriptive_stats(df)
    cols = set(stats["指标"])
    assert {"日均收益率", "中位数", "日收益率标准差", "偏度", "年化收益率(简单)", "年化波动率"} <= cols


def test_sharpe_insufficient_sample_returns_none_with_reason():
    df = _df_from_prices([100, 101])
    res = A.annualized_sharpe(df)
    assert res["sharpe"] is None
    assert "reason" in res  # 无法可靠计算时必须说明，而不是虚构数值


def test_correlation_identity_is_one():
    df = _df_from_prices(100 + np.cumsum(np.random.default_rng(3).normal(0, 1, 200)))
    corr = A.correlation_matrix({"a": df, "a_copy": df})
    assert np.isclose(corr.loc["a", "a_copy"], 1.0)


def test_rolling_metrics_shape():
    df = _df_from_prices(100 + np.cumsum(np.random.default_rng(5).normal(0, 1, 100)))
    roll = A.rolling_metrics(df, window=20)
    assert {"滚动收益率", "滚动年化波动率"} <= set(roll.columns)
    assert len(roll) == 100 - 20  # pct_change(20) 损失 20 期


def test_missing_overview_counts():
    df = _df_from_prices(np.full(50, 100.0))
    df.iloc[:5, df.columns.get_loc("volume")] = np.nan
    mo = A.missing_overview(df)
    assert mo.loc["volume", "缺失数量"] == 5
