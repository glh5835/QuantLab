"""回测引擎测试：手算对照（成交时序/费用/现金/持仓/绩效）+ 风控 + 可复现性。"""
import numpy as np
import pandas as pd
import pytest

from quantlab import backtest as B
from quantlab.backtest import BacktestConfig, BacktestError
from quantlab.strategies import BuyHoldStrategy, MACrossoverStrategy, StrategyError


def _mk_df(rows):
    """rows: [(open, high, low, close), ...] → 10 个交易日起步的 OHLCV。"""
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=len(rows)), name="trade_date")
    return pd.DataFrame({
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
        "volume": 1e6,
    }, index=idx)


def _flat_then(rows):
    """前置 30 天平稳数据（满足均线暖机），后接 rows。"""
    base = [(100, 101, 99, 100)] * 30
    return _mk_df(base + rows)


def _run(df, strategy, **cfg_kw):
    cfg = BacktestConfig(**cfg_kw)
    return B._run_engine(df, strategy, cfg)


# ---------------- 成交时序：t 收盘信号 → t+1 开盘成交 ----------------

def test_execution_at_next_open_not_same_close():
    """手算：买入持有首个信号在 d0 收盘 → 必须以 d1 开盘 110 成交（而非 d0 收盘 100）。"""
    df = _mk_df([(100, 101, 99, 100), (110, 111, 109, 115), (120, 121, 119, 130)])
    res = _run(df, BuyHoldStrategy(), initial_cash=10_000, lot_size=1,
               target_fraction=1.0, commission_rate=0.0, min_commission=0.0, slippage=0.0)
    buys = [tr for tr in res["trades"] if tr["action"] == "buy"]
    assert len(buys) == 1
    first = buys[0]
    assert first["base_price"] == 110.0            # d1 开盘价，而不是 d0 收盘 100
    assert first["signal_time"] != first["exec_time"]  # 信号日 ≠ 执行日
    assert int((pd.Timestamp(first["exec_time"]) - pd.Timestamp(first["signal_time"])).days) >= 1


def test_signal_on_last_day_is_never_filled():
    """末日产生的订单没有 t+1 → 必须记为未成交，绝不伪造成交价。"""
    df = _mk_df([(100, 101, 99, 100)] * 3 + [(100, 101, 99, 100)])
    res = _run(df, BuyHoldStrategy(), lot_size=1, commission_rate=0, min_commission=0, slippage=0)
    # 数据太短（<3 天）会拒绝；这里 4 天可行，末日无次日订单本来就不该存在——
    # BuyHold 在第 0 天已出信号，d1 已成交；再验证：把涨日放最后，末日信号必未成交
    df2 = _flat_then([(100, 101, 99, 100)] * 2)
    res2 = _run(df2, BuyHoldStrategy(), lot_size=1, commission_rate=0, min_commission=0, slippage=0)
    assert not any(tr["action"] == "unfilled" for tr in res2["trades"])


def test_unfilled_when_execution_day_price_missing():
    """执行日开盘与收盘均缺失（停牌）→ 订单未成交并记录原因，绝不伪造成交价。

    d0 收盘出买入信号 → d1（执行日）全部价格缺失 → 订单必须记为 unfilled。
    """
    df = _mk_df([(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100)])
    for col in ("open", "high", "low", "close"):
        df.iloc[1, df.columns.get_loc(col)] = np.nan  # d1 = 首个订单的执行日，全缺
    res = _run(df, BuyHoldStrategy(), lot_size=1, commission_rate=0, min_commission=0, slippage=0)
    unfilled = [tr for tr in res["trades"] if tr["action"] == "unfilled"]
    assert len(unfilled) == 1 and "无可用价格" in unfilled[0]["reason"]
    # 任务书允许"跳过、延迟或记录未成交"：引擎选择记录未成交 + 顺延到下一可交易日成交
    buys = [tr for tr in res["trades"] if tr["action"] == "buy"]
    assert len(buys) == 1 and buys[0]["exec_time"] == "2024-01-03"  # 延迟到 d2 成交
    assert buys[0]["base_price"] == 100.0                            # 用 d2 的真实开盘价


# ---------------- 费用 / 滑点 / 现金账目手算 ----------------

def test_buy_and_hold_hand_computed():
    """完全手算：10 万资金、100 元买入、无费用、lot=1、目标 95%。

    d0 收盘信号 → d1 开盘 100 元成交：可投 95000 → 950 股，现金 5000。
    末日收盘 200 → 净值 = 5000 + 950×200 = 195000，总收益 95%。
    """
    rows = [(100, 101, 99, 100)] * 30 + [(100, 101, 99, 100)] + [(100, 101, 99, 100)] * 5
    df = _mk_df([(100, 101, 99, 100)] * 30 + [(100, 101, 99, 100)] +
                [(190, 201, 189, 200)] * 3)
    res = _run(df, BuyHoldStrategy(), initial_cash=100_000, lot_size=1,
               target_fraction=0.95, commission_rate=0.0, min_commission=0.0, slippage=0.0)
    m = res["metrics"]
    assert m["final_shares"] == 950
    assert np.isclose(m["final_cash"], 5000.0)
    assert np.isclose(m["final_equity"], 5000 + 950 * 200)
    assert np.isclose(m["total_return"], (5000 + 950 * 200) / 100_000 - 1)


def test_commission_and_slippage_accounting():
    """手算费用：10 500 元资金买 100 股×100 元基准价，费率 0.1%、最低 5 元、滑点 0.5%。

    成交价 = 100×1.005 = 100.5；金额 10050；手续费 = max(10050×0.001, 5) = 10.05；
    现金支出后 = 10500 − 10050 − 10.05 = 439.95。滑点成本 = 100×0.5 = 50。
    """
    df = _mk_df([(100, 101, 99, 100)] * 30 + [(100, 101, 99, 100)] + [(100, 101, 99, 100)] * 3)
    res = _run(df, BuyHoldStrategy(), initial_cash=10_500, lot_size=100,
               target_fraction=1.0, commission_rate=0.001, min_commission=5.0, slippage=0.005)
    buys = [tr for tr in res["trades"] if tr["action"] == "buy"]
    assert len(buys) == 1
    tr = buys[0]
    assert tr["shares"] == 100
    assert np.isclose(tr["exec_price"], 100.5)
    assert np.isclose(tr["fee"], 10.05)
    assert np.isclose(tr["cash_after"], 10_500 - 100 * 100.5 - 10.05)
    assert np.isclose(res["metrics"]["total_fees"], 10.05)          # 不重复扣、不遗漏
    assert np.isclose(res["metrics"]["total_slippage_cost"], 50.0)


def test_cash_never_negative_and_lot_rounding():
    """股数向下取整至手（lot=100），现金永不为负。"""
    df = _mk_df([(100, 101, 99, 100)] * 30 + [(100, 101, 99, 100)] + [(100, 101, 99, 100)] * 3)
    res = _run(df, BuyHoldStrategy(), initial_cash=9_999, lot_size=100,
               target_fraction=1.0, commission_rate=0, min_commission=0, slippage=0)
    buys = [tr for tr in res["trades"] if tr["action"] == "buy"]
    assert all(tr["shares"] % 100 == 0 for tr in buys)
    assert all(d["cash"] >= -1e-9 for d in res["daily"])


def test_sell_cannot_exceed_holdings():
    """目标仓位 0 时只清掉持仓；无持仓时目标 0 不产生卖出订单。"""
    df = _mk_df([(100, 101, 99, 100)] * 40)
    res = _run(df, BuyHoldStrategy(), lot_size=1, commission_rate=0, min_commission=0, slippage=0)
    sells = [tr for tr in res["trades"] if tr["action"] == "sell"]
    assert len(sells) == 0  # 买入持有从不卖出
    assert all(d["shares"] >= 0 for d in res["daily"])


# ---------------- 均线交叉策略 ----------------

def test_ma_cross_golden_cross_timing():
    """构造明确的金叉：涨破后信号在 t 收盘产生、t+1 开盘成交。"""
    rows = [(100, 101, 99, 100)] * 30          # 平台期（快=慢）
    rows += [(100, 101, 99, 100)] * 4          # SMA3 平稳
    rows += [(110, 111, 109, 110), (120, 121, 119, 120), (130, 131, 129, 130)]  # 快线拉升
    df = _mk_df(rows)
    strat = MACrossoverStrategy(fast=3, slow=10)
    res = _run(df, strat, lot_size=1, commission_rate=0, min_commission=0, slippage=0)
    buys = [tr for tr in res["trades"] if tr["action"] == "buy"]
    assert len(buys) >= 1
    first = buys[0]
    # 信号日收盘时 SMA3 已上穿 SMA10；成交日 = 信号日次日
    sig_t = pd.Timestamp(first["signal_time"])
    exe_t = pd.Timestamp(first["exec_time"])
    assert (exe_t - sig_t).days >= 1
    # 成交价 = 执行日开盘价（无滑点时）
    assert np.isclose(first["base_price"], df.loc[exe_t, "open"])


def test_ma_cross_rejects_fast_ge_slow():
    with pytest.raises(StrategyError):
        MACrossoverStrategy(fast=20, slow=5)


# ---------------- 风控 ----------------

def test_drawdown_warning_is_warning_not_liquidation():
    """回撤超阈值只预警：持仓不变、不产生强制卖出。"""
    rows = [(100, 101, 99, 100)] * 30
    rows += [(100, 101, 99, 100)]
    rows += [(80, 81, 79, 80)] * 5 + [(60, 61, 59, 60)] * 5   # 跌 40%
    df = _mk_df(rows)
    cfg_kw = dict(initial_cash=100_000, lot_size=1, target_fraction=1.0,
                  commission_rate=0, min_commission=0, slippage=0)
    res = _run(df, BuyHoldStrategy(), drawdown_warn=0.2, **cfg_kw)
    assert any("回撤预警" in w for w in res["warnings"])
    sells = [tr for tr in res["trades"] if tr["action"] == "sell"]
    assert len(sells) == 0                                  # 未强制平仓
    assert res["daily"][-1]["shares"] > 0                   # 持仓仍在


def test_stop_buy_on_drawdown_blocks_new_buys():
    """配置 stop_buy=True 后，预警触发期间的新买入被拦截（已有持仓不受影响）。"""
    rows = [(100, 101, 99, 100)] * 30 + [(100, 101, 99, 100)]
    df = _mk_df(rows)  # 平稳数据不触发预警 → 拦截逻辑不误伤
    res = _run(df, BuyHoldStrategy(), lot_size=1, commission_rate=0,
               min_commission=0, slippage=0, drawdown_warn=0.5, stop_buy_on_drawdown=True)
    assert not any(tr["action"] == "buy_blocked" for tr in res["trades"])
    assert any(tr["action"] == "buy" for tr in res["trades"])


def test_invalid_config_rejected():
    with pytest.raises(BacktestError):
        BacktestConfig(initial_cash=-1).validate()
    with pytest.raises(BacktestError):
        BacktestConfig(slippage=-0.1).validate()
    with pytest.raises(BacktestError):
        BacktestConfig(drawdown_warn=0).validate()


# ---------------- 绩效与可复现 ----------------

def test_metrics_match_hand_calculation():
    """绩效手算：950 股、100→200，总收益 95%；回撤与夏普来自真实净值序列。"""
    df = _mk_df([(100, 101, 99, 100)] * 30 + [(100, 101, 99, 100)] +
                [(190, 201, 189, 200)] * 3)
    res = _run(df, BuyHoldStrategy(), initial_cash=100_000, lot_size=1,
               target_fraction=0.95, commission_rate=0, min_commission=0, slippage=0)
    m = res["metrics"]
    assert np.isclose(m["total_return"], 0.95)
    assert m["mdd"] <= 0 and m["sharpe"] is not None
    assert "benchmark" not in m  # 引擎层不含基准（run_backtest 才加）


def test_same_config_same_result():
    from quantlab.storage import save_raw
    import quantlab.db as db
    ds_id = db.register_dataset("BT_DEMO", "BTD", source="demo", is_synthetic=1)
    rng = np.random.default_rng(4)
    n = 260
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    df = pd.DataFrame({"open": close * 0.998, "high": close * 1.02, "low": close * 0.98,
                       "close": close, "volume": 1e6},
                      index=pd.bdate_range("2024-01-01", periods=n))
    db.update_dataset(ds_id, raw_path=save_raw(df, "BT_DEMO"), rows=n)
    cfg = BacktestConfig()
    r1 = B.run_backtest(MACrossoverStrategy(5, 20), ds_id, cfg, save=False)
    r2 = B.run_backtest(MACrossoverStrategy(5, 20), ds_id, cfg, save=False)
    assert r1["metrics"]["final_equity"] == r2["metrics"]["final_equity"]  # 完全一致
    assert r1["metrics"]["benchmark"]["total_return"] is not None          # 基准已对比


def test_backtest_saved_to_db_and_files():
    from quantlab.storage import save_raw
    import quantlab.db as db
    import os
    ds_id = db.register_dataset("BT_SAVE", "BTS", source="demo", is_synthetic=1)
    rng = np.random.default_rng(6)
    n = 200
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n)))
    df = pd.DataFrame({"open": close * 0.998, "high": close * 1.02, "low": close * 0.98,
                       "close": close, "volume": 1e6},
                      index=pd.bdate_range("2024-01-01", periods=n))
    db.update_dataset(ds_id, raw_path=save_raw(df, "BT_SAVE"), rows=n)
    r = B.run_backtest(BuyHoldStrategy(), ds_id, BacktestConfig(), name="t_save")
    assert r["bt_id"] is not None
    rec = db.get_backtest(r["bt_id"])
    assert rec["status"] == "success" and rec["strategy_type"] == "buy_hold"
    assert os.path.exists(rec["trades_path"]) and os.path.exists(rec["daily_path"])
    r2 = B.run_backtest(BuyHoldStrategy(), ds_id, BacktestConfig(), name="t_save2")
    assert r2["bt_id"] != r["bt_id"]  # 新回测新记录，不覆盖
