"""技术指标测试：已知数值手算对照 + 因果性（截断等价）+ 边界与非法输入。"""
import numpy as np
import pandas as pd
import pytest

from quantlab import indicators as I
from quantlab.indicators import IndicatorError


def _px(prices):
    return pd.Series(np.asarray(prices, dtype=float),
                     index=pd.bdate_range("2024-01-01", periods=len(prices)), name="close")


# ---------------- 已知数值 ----------------

def test_sma_known_values():
    s = I.sma(_px([1, 2, 3, 4, 5]), 3)
    assert np.isnan(s.iloc[0]) and np.isnan(s.iloc[1])
    assert np.isclose(s.iloc[2], 2.0) and np.isclose(s.iloc[3], 3.0) and np.isclose(s.iloc[4], 4.0)


def test_ema_known_values():
    # α=2/(3+1)=0.5：EMA=[1, 1.5, 2.25, 3.125]
    s = I.ema(_px([1, 2, 3, 4]), 3)
    assert np.isclose(s.iloc[0], 1.0) and np.isclose(s.iloc[1], 1.5)
    assert np.isclose(s.iloc[2], 2.25) and np.isclose(s.iloc[3], 3.125)


def test_rsi_hand_computed():
    # alpha=1/2 递归平滑，手算见下（涨→100、平→50、跌→25）
    s = I.rsi(_px([100, 101, 102, 101, 100]), window=2)
    assert np.isnan(s.iloc[0])
    assert np.isclose(s.iloc[1], 100.0)   # ag=1, al=0
    assert np.isclose(s.iloc[2], 100.0)   # ag=1, al=0
    assert np.isclose(s.iloc[3], 50.0)    # ag=0.5, al=0.5
    assert np.isclose(s.iloc[4], 25.0)    # ag=0.25, al=0.75


def test_rsi_boundaries():
    up = I.rsi(_px([100, 101, 102, 103, 104]), 3)
    assert np.allclose(up.dropna(), 100.0)          # 只涨 → 100
    flat = I.rsi(_px([100] * 10), 3)
    assert np.allclose(flat.dropna().iloc[1:], 50.0)  # 完全无涨跌 → 明确规则 50


def test_macd_components_consistent():
    close = _px(list(100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 60))))
    m = I.macd(close, fast=3, slow=6, signal=3)
    dif_manual = I.ema(close, 3) - I.ema(close, 6)
    assert np.allclose(m["dif"], dif_manual, equal_nan=True)          # DIF = EMA快 - EMA慢
    assert np.allclose(m["dea"], dif_manual.ewm(span=3, adjust=False).mean(), equal_nan=True)
    assert np.allclose(m["hist"], 2 * (m["dif"] - m["dea"]), equal_nan=True)  # 国内口径 2×(DIF-DEA)


def test_bollinger_band_relations():
    close = _px(list(100 + np.random.default_rng(2).normal(0, 2, 50)))
    b = I.bollinger(close, 5, 2.0)
    valid = b.dropna()
    assert (valid["bb_upper"] >= valid["bb_mid"]).all()
    assert (valid["bb_mid"] >= valid["bb_lower"]).all()
    # 手算对照：总体标准差（ddof=0）
    seg = close.iloc[:5].to_numpy()
    mid, std = seg.mean(), seg.std(ddof=0)
    assert np.isclose(b["bb_mid"].iloc[4], mid)
    assert np.isclose(b["bb_upper"].iloc[4], mid + 2 * std)
    assert np.isclose(b["bb_lower"].iloc[4], mid - 2 * std)


def test_returns_family():
    s = _px([100, 110, 99])
    assert np.isclose(I.returns(s).iloc[1], 0.10)
    assert np.isclose(I.log_returns(s).iloc[1], np.log(1.1))
    assert np.isclose(I.cum_returns(s, 2).iloc[2], -0.01)


def test_rolling_vol_constant_price_is_zero():
    s = I.rolling_volatility(_px([100.0] * 30), 10)
    assert np.allclose(s.dropna(), 0.0)


def test_rolling_max_drawdown_window():
    # 窗口 [100,120,60]：回撤 = 60/120-1 = -50%；窗口 [120,60,130] 内仍含 120→60 的 -50%
    s = I.rolling_max_drawdown(_px([100, 120, 60, 130]), 3)
    assert np.isclose(s.iloc[2], -0.5)
    assert np.isclose(s.iloc[3], -0.5)
    # 窗口整体单调上涨时回撤为 0
    s2 = I.rolling_max_drawdown(_px([1, 2, 3, 4, 5]), 3)
    assert np.allclose(s2.dropna(), 0.0)


# ---------------- 因果性（防未来数据） ----------------

@pytest.mark.parametrize("name,params", [
    ("sma", (5,)), ("ema", (5,)), ("rsi", (5,)), ("macd", (3, 8, 3)),
    ("bollinger", (5, 2.0)), ("rolling_vol", (5,)), ("cum_returns", (3,)), ("rolling_mdd", (5,)),
])
def test_indicators_are_causal(name, params):
    """截断等价：只用前 k 行算出的最后一个值 == 用全量算出的第 k 行。"""
    rng = np.random.default_rng(42)
    close = pd.DataFrame({
        "open": 100 + rng.normal(0, 1, 80), "high": 101 + rng.normal(0, 1, 80),
        "low": 99 + rng.normal(0, 1, 80), "close": 100 + np.cumsum(rng.normal(0, 1, 80)),
        "volume": 1e6,
    }, index=pd.bdate_range("2024-01-01", periods=80))
    full = I.compute_indicator(close, name, params)
    k = 40
    part = I.compute_indicator(close.iloc[:k], name, params)
    f_row, p_row = full.iloc[k - 1], part.iloc[-1]
    for val_f, val_p in zip(np.atleast_1d(f_row), np.atleast_1d(p_row)):
        assert (np.isnan(val_f) and np.isnan(val_p)) or np.isclose(val_f, val_p)


# ---------------- 数据前提与非法输入 ----------------

def test_duplicate_dates_rejected():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02"])
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    with pytest.raises(IndicatorError, match="重复"):
        I.prepare_ohlcv(df)


def test_illegal_params():
    s = _px([1, 2, 3])
    with pytest.raises(IndicatorError):
        I.sma(s, 0)
    with pytest.raises(IndicatorError):
        I.ema(s, -2)
    with pytest.raises(IndicatorError, match="快速均线窗口"):
        I.macd(s, fast=26, slow=12)  # 快 >= 慢 必须拒绝
    with pytest.raises(IndicatorError):
        I.bollinger(s, 5, num_std=0)


def test_insufficient_window_returns_nan_not_zero():
    s = I.sma(_px([1, 2]), 5)
    assert s.isna().all()  # 数据不足窗口 → NaN，绝不补零


def test_input_not_modified():
    df = pd.DataFrame({"close": [3.0, 1.0, 2.0]},
                      index=pd.DatetimeIndex(["2024-01-03", "2024-01-01", "2024-01-02"]))
    before = df.copy()
    I.prepare_ohlcv(df)
    pd.testing.assert_frame_equal(df, before)  # 原始数据不被修改/排序


def test_repeatable_deterministic():
    close = _px(list(100 + np.cumsum(np.random.default_rng(9).normal(0, 1, 50))))
    r1 = I.rsi(close, 10).iloc[-1]
    r2 = I.rsi(close, 10).iloc[-1]
    assert np.isclose(r1, r2)
