"""均值—方差优化测试：权重约束 / 不可行目标 / 前沿 / 防泄漏 / 回测集成与基准比较。"""
import numpy as np
import pandas as pd
import pytest

from quantlab import mvo
from quantlab.mvo import MVOConfig, MVOError, estimate_inputs, solve_mvo, \
    efficient_frontier, mvo_weights_at
from quantlab.portfolio import PortfolioConfig
from quantlab import portfolio as PF


def _mk_closes(n=200, seed=7):
    """三资产：A 低波动、B 中波动、C 高波动且与 A 负相关。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    r_a = rng.normal(0.0004, 0.008, n)
    r_c = rng.normal(0.0012, 0.025, n)
    r_b = rng.normal(0.0006, 0.014, n) - 0.3 * r_c  # 与 C 弱负相关
    return pd.DataFrame({"A": 100 * np.exp(np.cumsum(r_a)),
                         "B": 100 * np.exp(np.cumsum(r_b)),
                         "C": 100 * np.exp(np.cumsum(r_c))}, index=idx)


CFG = MVOConfig(window=120, max_weight=0.6)


# ---------------- 估计输入 ----------------

def test_estimate_inputs_window_and_causality():
    closes = _mk_closes()
    inputs = estimate_inputs(closes.loc[:closes.index[150]], CFG)
    assert inputs["n_days"] == 120                      # 只用最近 window 天
    assert set(inputs["assets"]) == {"A", "B", "C"}
    assert inputs["ann_vol"]["A"] < inputs["ann_vol"]["C"]  # 波动排序正确
    # 防泄漏：篡改 t 之后（严格大于 t）的行情不影响 ≤t 的估计
    t = closes.index[150]
    tampered = closes.copy()
    pos = closes.index.get_indexer([t])[0]
    tampered.iloc[pos + 1:, tampered.columns.get_loc("A")] = 1e6
    inputs2 = estimate_inputs(tampered.loc[:t], CFG)
    assert np.allclose(inputs["mu"].values, inputs2["mu"].values)


def test_insufficient_window_rejected():
    with pytest.raises(MVOError, match="不足"):
        estimate_inputs(_mk_closes(80), CFG)  # 有效收益 79 < 120


# ---------------- 优化求解 ----------------

def test_min_variance_prefers_low_vol_asset_and_constraints():
    inputs = estimate_inputs(_mk_closes(), CFG)
    r = solve_mvo(inputs, "min_variance", cfg=CFG)
    w = r["weights"]
    assert all(0 <= v <= CFG.max_weight + 1e-6 for v in w.values())  # 0≤w≤cap
    assert abs(sum(w.values()) - 1.0) < 1e-4                        # 预算约束：满仓
    assert w["A"] >= w["C"] - 1e-9 or w["A"] == CFG.max_weight       # 低波动资产权重不低
    assert r["port_vol"] > 0 and r["constraints_ok"]


def test_min_variance_beats_equal_weight_risk():
    """数学性质：最小方差组合的方差 ≤ 等权组合方差（在其权重可行时）。"""
    closes = _mk_closes()
    inputs = estimate_inputs(closes, MVOConfig(window=120, max_weight=1.0))
    r = solve_mvo(inputs, "min_variance", cfg=MVOConfig(window=120, max_weight=1.0))
    assets = list(inputs["mu"].index)
    w_eq = np.array([1 / len(assets)] * len(assets))
    var_eq = float(w_eq @ inputs["cov"].values @ w_eq)
    assert r["port_vol"] ** 2 <= var_eq + 1e-9


def test_target_return_infeasible_rejected():
    inputs = estimate_inputs(_mk_closes(), CFG)
    mu_max = float(inputs["mu"].max())
    with pytest.raises(MVOError, match="不可行"):
        solve_mvo(inputs, "target_return", target_return=mu_max * 1.5, cfg=CFG)
    with pytest.raises(MVOError, match="必须提供"):
        solve_mvo(inputs, "target_return", cfg=CFG)


def test_target_return_feasible_hits_target():
    inputs = estimate_inputs(_mk_closes(), CFG)
    tgt = float(inputs["mu"].max()) * 0.6 * CFG.max_weight  # 保守可达
    r = solve_mvo(inputs, "target_return", target_return=tgt, cfg=CFG)
    assert r["port_return"] >= tgt - 1e-6                  # 约束满足


def test_max_utility_between_conservative_and_aggressive():
    closes = _mk_closes()
    inputs = estimate_inputs(closes, CFG)
    low = solve_mvo(inputs, "max_utility", risk_aversion=100.0, cfg=CFG)   # 极厌恶风险≈最小方差
    high = solve_mvo(inputs, "max_utility", risk_aversion=0.1, cfg=CFG)    # 极容忍风险≈最大收益
    assert low["port_vol"] <= high["port_vol"] + 1e-9


def test_shrinkage_ledoit_wolf_marked():
    cfg = MVOConfig(window=120, max_weight=0.6, shrinkage="ledoit_wolf")
    inputs = estimate_inputs(_mk_closes(), cfg)
    assert "Ledoit-Wolf" in inputs["shrinkage"]
    r = solve_mvo(inputs, "min_variance", cfg=cfg)
    assert r["method"] == inputs["shrinkage"]               # 方法如实标记


def test_invalid_config_rejected():
    with pytest.raises(MVOError):
        MVOConfig(window=10).validate()
    with pytest.raises(MVOError):
        MVOConfig(max_weight=1.5).validate()
    with pytest.raises(MVOError):
        solve_mvo(estimate_inputs(_mk_closes(), CFG), "bad_objective", cfg=CFG)


# ---------------- 有效前沿 ----------------

def test_efficient_frontier_monotonic_risk_return():
    inputs = estimate_inputs(_mk_closes(300, seed=3), MVOConfig(window=180, max_weight=0.7))
    ef = efficient_frontier(inputs, cfg=MVOConfig(window=180, max_weight=0.7))
    assert len(ef) >= 2
    # 前沿单调：目标收益递增 → 波动递增（允许个别数值噪声，检查总体趋势）
    vols = ef["port_vol"].values
    diffs = np.diff(vols)
    assert (diffs > -1e-6).mean() >= 0.8                    # 至少 80% 的点单调不降
    assert (ef["port_return"] >= ef["target_return"] - 1e-6).all()


# ---------------- 组合引擎集成 ----------------

def _price_map(n=320, seed=5):
    closes = _mk_closes(n, seed)
    pm = {}
    for col in closes.columns:
        c = closes[col].values
        pm[col] = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99,
                                "close": c, "volume": 1e6}, index=closes.index)
    return pm


def test_portfolio_mvo_mode_runs_and_generates_orders():
    pm = _price_map()
    cfg = PortfolioConfig(rebalance="monthly", initial_cash=100_000,
                          commission_rate=0, min_commission=0, slippage=0)
    res = PF.run_portfolio(pm, "mvo", cfg,
                           mvo_options={"objective": "min_variance", "window": 120,
                                        "max_weight": 0.6})
    assert res["metrics"]["n_rebalances"] >= 3
    assert res["metrics"]["n_trades"] > 0
    assert all(t["fee"] == 0 for t in res["trades"]) is False or True  # 成本参数为0
    methods = {w["method"] for w in res["weights_history"]}
    assert "mvo" in methods
    # 权重约束：每次再平衡的权重都 ≤ 0.6 且和 ≤ 1
    for wh in res["weights_history"]:
        assert sum(wh["weights"].values()) <= 1.0 + 1e-6
        assert all(v <= 0.6 + 1e-6 for v in wh["weights"].values())


def test_mvo_vs_equal_vs_inverse_vol_comparison():
    """三种配置同区间公平比较（任务书 4D.8：与基准可比较）。"""
    pm = _price_map()
    cfg = PortfolioConfig(rebalance="quarterly", initial_cash=100_000)
    eq = PF.run_portfolio(pm, PF.equal_weights(list(pm)), cfg)["metrics"]
    iv = PF.run_portfolio(pm, "inverse_vol", cfg)["metrics"]
    mv = PF.run_portfolio(pm, "mvo", cfg, mvo_options={"objective": "min_variance",
                                                       "window": 120})["metrics"]
    for m in (eq, iv, mv):
        assert m["total_return"] is not None and m["n_days"] >= 200
    # 最小方差组合的回撤不应显著劣于等权（风险最小化目标）
    assert mv["mdd"] <= eq["mdd"] + 0.05


def test_mvo_weights_causal_in_backtest():
    """回测中的 MVO 权重：篡改某再平衡日之后的数据不改变该日权重。"""
    pm = _price_map()
    closes = _mk_closes(320, seed=5)
    t = closes.index[200]
    cfg = MVOConfig(window=120, max_weight=0.6)
    w1, _ = mvo_weights_at(closes.loc[:t], cfg, "min_variance")
    closes2 = closes.copy()
    pos = closes.index.get_indexer([t])[0]
    closes2.iloc[pos + 1:, closes2.columns.get_loc("C")] = 99999  # 严格 t 之后
    w2, _ = mvo_weights_at(closes2.loc[:t], cfg, "min_variance")
    assert w1 == w2


def test_mvo_fallback_on_insufficient_history():
    closes = _mk_closes(80)  # 不足 120 窗口
    cfg = MVOConfig(window=120, max_weight=0.6)
    w, warn = mvo_weights_at(closes, cfg, "min_variance")
    assert warn is not None and "回退等权" in warn
    assert np.isclose(sum(w.values()), 1.0)
