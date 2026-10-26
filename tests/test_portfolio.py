"""投资组合测试：权重方法 / 日历对齐 / 再平衡订单 / 费用 / 缺价处理 / 手算净值。"""
import numpy as np
import pandas as pd
import pytest

from quantlab import portfolio as P
from quantlab.portfolio import PortfolioConfig, PortfolioError


def _asset(name, closes, opens=None, start="2024-01-01"):
    idx = pd.DatetimeIndex(pd.bdate_range(start, periods=len(closes)), name="trade_date")
    closes = np.asarray(closes, dtype=float)
    opens = np.asarray(opens, dtype=float) if opens is not None else closes
    return name, pd.DataFrame({"open": opens, "high": closes * 1.01, "low": closes * 0.99,
                               "close": closes, "volume": 1e6}, index=idx)


def _cfg(**kw):
    d = dict(initial_cash=100_000, commission_rate=0, min_commission=0, slippage=0,
             lot_size=1, rebalance="none", cash_reserve=0.0, vol_window=20)
    d.update(kw)
    cfg = PortfolioConfig(**d)
    cfg.validate()  # 构造即校验（非法配置在测试助手处就应抛错）
    return cfg


# ---------------- 权重方法 ----------------

def test_normalize_weights_validation_and_normalize():
    with pytest.raises(PortfolioError):
        P.normalize_weights({"a": -0.5, "b": 1.0})          # 负权重拒绝
    with pytest.raises(PortfolioError):
        P.normalize_weights({"a": 0.0, "b": 0.0})           # 全零拒绝
    w = P.normalize_weights({"a": 2, "b": 6})                # 自动归一化
    assert np.isclose(w["a"], 0.25) and np.isclose(w["b"], 0.75)
    assert np.isclose(sum(w.values()), 1.0)


def test_max_weight_cap():
    w = P.normalize_weights({"a": 0.95, "b": 0.05}, max_weight=0.7)
    assert w["a"] <= 0.7 + 1e-6 and np.isclose(sum(w.values()), 1.0)


def test_equal_weights():
    w = P.equal_weights(["a", "b", "c"])
    assert all(np.isclose(v, 1 / 3) for v in w.values())
    with pytest.raises(PortfolioError):
        P.equal_weights(["a", "b"], max_weight=0.4)  # 2 资产无法都 ≤ 0.4


def test_inverse_vol_hand_computed():
    # A 恒定 100（零波动会触发等权回退）；改用两个都有波动、但波动不同的资产：
    # A 日波动 1%，B 日波动 2% → 权重比 = (1/1%):(1/2%) = 2:1
    idx = pd.bdate_range("2024-01-01", periods=30)
    rng = np.random.default_rng(1)
    a = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 30)))
    b = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 30)))
    closes = pd.DataFrame({"A": a, "B": b}, index=idx)
    w, warn = P.inverse_vol_weights(closes, window=20)
    assert warn is None
    assert np.isclose(w["A"], 2 / 3, atol=0.15)  # 样本波动率有噪声，容差校验 2:1 比例
    assert w["A"] > w["B"]                        # 低波动资产权重更高


def test_inverse_vol_insufficient_history_falls_back_to_equal():
    closes = pd.DataFrame({"A": np.linspace(100, 110, 10), "B": np.linspace(100, 105, 10)},
                          index=pd.bdate_range("2024-01-01", periods=10))
    w, warn = P.inverse_vol_weights(closes, window=20)  # 只有 10 天 < 21
    assert warn is not None and "回退为等权" in warn
    assert np.isclose(w["A"], 0.5) and np.isclose(w["B"], 0.5)


# ---------------- 日历对齐与缺价 ----------------

def test_calendar_alignment_union_and_stale_pricing():
    # B 缺少第 3、4 天行情 → 并集日历 5 天；B 缺价日按最后已知价估值并标 stale
    _, a = _asset("A", [100, 101, 102, 103, 104])
    _, b = _asset("B", [50, 51, 52, 53, 54])
    b = b.drop(b.index[2:4])  # B 缺第 3、4 天
    res = P.run_portfolio({"A": a, "B": b}, {"A": 0.5, "B": 0.5}, _cfg())
    assert len(res["daily"]) == 5              # 并集日历：保留 A 的独立交易日
    assert res["daily"][2]["stale_assets"] == ["B"]  # 第 3 天 B 缺价 → stale
    assert res["daily"][3]["stale_assets"] == ["B"]
    assert res["daily"][4]["stale_assets"] == []     # 第 5 天 B 恢复
    assert all(w["equity"] == w["equity"] for w in res["daily"])  # 净值永不为 NaN


def test_missing_price_asset_not_traded():
    # B 首日无开盘也无收盘 → 期初建仓时被跳过并告警，绝不伪造成交
    _, a = _asset("A", [100, 100, 100, 100, 100])
    _, b = _asset("B", [50, 50, 50, 50, 50])
    b.iloc[0, b.columns.get_loc("open")] = np.nan
    b.iloc[0, b.columns.get_loc("close")] = np.nan
    res = P.run_portfolio({"A": a, "B": b}, {"A": 0.5, "B": 0.5}, _cfg())
    assert any("B" in w and ("无可用价格" in w or "开盘与收盘均无价" in w) for w in res["warnings"])
    assert all(tr["asset"] != "B" or tr["exec_time"] != res["daily"][1]["date"]
               for tr in res["trades"])  # B 在缺价执行日没有成交


# ---------------- 建仓与再平衡订单 ----------------

def test_initial_allocation_at_next_open_hand_computed():
    # 手算：2 资产各 50%，d1 开盘 A=100、B=200，各投 5 万 → 500 股 A + 250 股 B
    _, a = _asset("A", [100, 100, 110, 110, 110], opens=[100, 100, 110, 110, 110])
    _, b = _asset("B", [200, 200, 200, 200, 200])
    res = P.run_portfolio({"A": a, "B": b}, {"A": 0.5, "B": 0.5}, _cfg())
    trades = [tr for tr in res["trades"] if tr["action"] == "buy"]
    assert {(t["asset"], t["shares"]) for t in trades} == {("A", 500), ("B", 250)}
    assert all(t["signal_time"] != t["exec_time"] for t in trades)  # t 收盘决策、t+1 成交
    m = res["metrics"]
    # 末日：A=110×500 + B=200×250 = 105000 → 总收益 5%
    assert np.isclose(m["final_positions"]["A"]["shares"], 500)
    assert np.isclose(m["final_positions"]["B"]["shares"], 250)
    assert np.isclose(m["total_return"], 0.05)


def test_monthly_rebalance_generates_orders_with_costs():
    # 跨 3 个月的数据 + 月度再平衡 → 至少 3 次再平衡事件、有真实订单和手续费
    idx = pd.bdate_range("2024-01-01", periods=70)
    rng = np.random.default_rng(3)
    a = pd.DataFrame({"open": np.linspace(100, 160, 70), "high": 0, "low": 0,
                      "close": np.linspace(100, 160, 70), "volume": 1e6}, index=idx)
    b = pd.DataFrame({"open": np.linspace(50, 40, 70), "high": 0, "low": 0,
                      "close": np.linspace(50, 40, 70), "volume": 1e6}, index=idx)
    res = P.run_portfolio({"A": a, "B": b}, {"A": 0.5, "B": 0.5},
                          _cfg(rebalance="monthly", commission_rate=0.001, min_commission=5))
    assert res["metrics"]["n_rebalances"] >= 3
    assert res["metrics"]["n_trades"] > 2
    assert res["metrics"]["total_fees"] > 0
    assert all(d["cash"] >= -1e-9 for d in res["daily"])  # 现金永不为负


def test_rebalance_sells_before_buys_and_cash_consistent():
    # A 在第 30 天后大涨超配 → 下一次月度再平衡（3 月首个交易日）应先卖 A 再买 B
    idx = pd.bdate_range("2024-01-01", periods=55)
    a_close = np.concatenate([np.full(30, 100.0), np.linspace(100, 300, 25)])
    a = pd.DataFrame({"open": a_close, "high": 0, "low": 0, "close": a_close, "volume": 1e6}, index=idx)
    b_close = np.full(55, 50.0)
    b = pd.DataFrame({"open": b_close, "high": 0, "low": 0, "close": b_close, "volume": 1e6}, index=idx)
    res = P.run_portfolio({"A": a, "B": b}, {"A": 0.5, "B": 0.5}, _cfg(rebalance="monthly"))
    mar_trades = [tr for tr in res["trades"] if tr["exec_time"].startswith("2024-03")]
    sells = [tr for tr in mar_trades if tr["action"] == "sell" and tr["asset"] == "A"]
    buys = [tr for tr in mar_trades if tr["action"] == "buy" and tr["asset"] == "B"]
    assert len(sells) >= 1 and len(buys) >= 1  # 卖超配 A、买低配 B
    sell_time = min(tr["exec_time"] for tr in sells)
    assert all(tr["exec_time"] >= sell_time for tr in buys)  # 先卖后买（同日订单内卖出先行）
    for tr in mar_trades:
        assert tr["cash_after"] >= -1e-9


def test_slippage_and_fees_applied_on_rebalance():
    _, a = _asset("A", [100, 100, 100, 100, 100, 100])
    _, b = _asset("B", [50, 50, 50, 50, 50, 50])
    res = P.run_portfolio({"A": a, "B": b}, {"A": 0.5, "B": 0.5},
                          _cfg(commission_rate=0.001, min_commission=5, slippage=0.005))
    buys = [tr for tr in res["trades"] if tr["action"] == "buy"]
    assert all(np.isclose(tr["exec_price"], tr["base_price"] * 1.005) for tr in buys)
    assert all(tr["fee"] >= 5.0 for tr in buys)
    assert res["metrics"]["total_fees"] == sum(tr["fee"] for tr in res["trades"])
    assert res["metrics"]["total_slippage_cost"] > 0


# ---------------- 与单资产一致性 ----------------

def test_single_asset_portfolio_matches_hand_calc():
    # 100% 单资产 = 阶段2买入持有的简化版：手算净值
    _, a = _asset("A", [100, 100, 150, 150, 150], opens=[100, 100, 150, 150, 150])
    res = P.run_portfolio({"A": a}, {"A": 1.0}, _cfg())
    # d1 开盘 100 买入 10 万 → 1000 股；末日 150 → 净值 150000
    assert np.isclose(res["metrics"]["final_positions"]["A"]["shares"], 1000)
    assert np.isclose(res["metrics"]["total_return"], 0.50)


def test_lot_rounding_and_cash_reserve():
    _, a = _asset("A", [333, 333, 333, 333, 333])
    _, b = _asset("B", [77, 77, 77, 77, 77])
    cfg = _cfg(lot_size=100, cash_reserve=0.05)
    res = P.run_portfolio({"A": a, "B": b}, {"A": 0.5, "B": 0.5}, cfg)
    assert all(tr["shares"] % 100 == 0 for tr in res["trades"])
    # 现金保留 5% → 期初可投 95000
    assert res["daily"][1]["cash"] >= 100_000 * 0.05 - 1e-6


def test_invalid_config_rejected():
    with pytest.raises(PortfolioError):
        _cfg(rebalance="weekly")
    with pytest.raises(PortfolioError):
        _cfg(cash_reserve=1.5)
    with pytest.raises(PortfolioError):
        PortfolioConfig(initial_cash=-1).validate()


def test_inverse_vol_uses_only_past_data():
    """防泄漏：再平衡日 t 的逆波动率权重只受 ≤t 数据影响（改动 t 之后的行情不变）。"""
    idx = pd.bdate_range("2024-01-01", periods=80)
    rng = np.random.default_rng(5)
    a = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 80)))
    b = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 80)))
    closes = pd.DataFrame({"A": a, "B": b}, index=idx)
    t = idx[40]
    w1, _ = P.inverse_vol_weights(closes.loc[:t], 30)
    # 篡改 t 之后（严格大于 t）的数据 → 权重必须完全不变
    closes2 = closes.copy()
    closes2.loc[idx[41]:, "A"] = 99999
    w2, _ = P.inverse_vol_weights(closes2.loc[:t], 30)
    assert w1 == w2
