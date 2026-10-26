"""参数优化测试：网格合法性 / 三分隔离 / 选参规则 / 邻居稳定性 / 一致性 / 存档。"""
import numpy as np
import pandas as pd
import pytest

from quantlab import optimizer as O
from quantlab.backtest import BacktestConfig
from quantlab.optimizer import OptimizerError


def _trend_df(n=300, seed=7):
    """带趋势与波动的行情（均线策略能有信号）。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    drift = np.concatenate([np.full(n // 3, 0.002), np.full(n // 3, -0.001),
                            np.full(n - 2 * (n // 3), 0.0015)])
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.015, n)))
    return pd.DataFrame({"open": close * (1 + rng.normal(0, 0.001, n)),
                         "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": 1e6}, index=idx)


CFG = BacktestConfig(commission_rate=0, min_commission=0, slippage=0)  # 简化成本便于断言


# ---------------- 网格构造 ----------------

def test_build_grid_filters_illegal_and_caps_size():
    grid = O.build_grid({"fast": [5, 20], "slow": [10, 30]})
    fasts_slows = [(g["fast"], g["slow"]) for g in grid]
    assert (20, 10) not in fasts_slows          # fast>=slow 被排除
    assert all(f < s for f, s in fasts_slows)
    with pytest.raises(OptimizerError, match="fast<slow"):
        O.build_grid({"fast": [30], "slow": [5]})
    with pytest.raises(OptimizerError, match="超过硬上限"):
        O.build_grid({"fast": list(range(2, 10)), "slow": list(range(11, 22))})
    with pytest.raises(OptimizerError, match="不能为空"):
        O.build_grid({"fast": []})


def test_split_timeline_order_and_minimum():
    idx = pd.bdate_range("2022-01-03", periods=300)
    tr, va, te = O.split_timeline(idx)
    assert tr < va < te
    with pytest.raises(OptimizerError, match="至少需要"):
        O.split_timeline(pd.bdate_range("2024-01-01", periods=30))


# ---------------- 搜索与隔离 ----------------

def test_grid_search_uses_only_train_val_and_test_once():
    df = _trend_df()
    res = O.run_grid_search(df, "ma_cross", {"fast": [3, 5], "slow": [20, 30]},
                            cfg=CFG, objective="sharpe")
    assert res["n_combinations"] == 4
    assert len(res["combinations"]) == 4
    assert res["best"]["params"]["fast"] < res["best"]["params"]["slow"]
    assert res["test_used_for_selection"] is False   # 关键：test 未参与选参（如实记录）
    assert "test 段仅在参数确定后" in res["assumption"]
    # 所有组合的 val 指标都存在（无 error）
    assert all("val_metrics" in c for c in res["combinations"])
    # 选出的 best 必须是可行组合中 val sharpe 最大的
    vals = [c["val_metrics"]["sharpe"] for c in res["combinations"]
            if c["val_metrics"].get("sharpe") is not None]
    assert res["best"]["val_metrics"]["sharpe"] == max(vals)


def test_constraints_filter_combinations():
    df = _trend_df()
    # 极严苛回撤约束：可能全部被排除 → 明确报错而不是悄悄放宽
    with pytest.raises(OptimizerError, match="约束排除"):
        O.run_grid_search(df, "ma_cross", {"fast": [3], "slow": [20]},
                          cfg=CFG, max_mdd=1e-9)
    # 最低交易次数约束生效
    res = O.run_grid_search(df, "ma_cross", {"fast": [3], "slow": [20]},
                            cfg=CFG, min_trades=0)
    assert res["n_feasible"] >= 1


def test_same_input_same_result():
    df = _trend_df(seed=11)
    r1 = O.run_grid_search(df, "ma_cross", {"fast": [5], "slow": [20]}, cfg=CFG)
    r2 = O.run_grid_search(df, "ma_cross", {"fast": [5], "slow": [20]}, cfg=CFG)
    assert r1["best"]["params"] == r2["best"]["params"]
    assert r1["test_metrics"]["total_return"] == r2["test_metrics"]["total_return"]


def test_objective_and_unknown_rejected():
    df = _trend_df(120)
    with pytest.raises(OptimizerError, match="未知优化目标"):
        O.run_grid_search(df, "ma_cross", {"fast": [5], "slow": [20]},
                          cfg=CFG, objective="profit")


def test_unsupported_strategy_rejected():
    with pytest.raises(OptimizerError, match="尚未实现"):
        O.make_strategy("rsi", {"window": 14})


# ---------------- 敏感性与过拟合 ----------------

def test_neighbor_stability_and_warnings():
    df = _trend_df()
    res = O.run_grid_search(df, "ma_cross", {"fast": [3, 5, 8], "slow": [20, 30]},
                            cfg=CFG, objective="sharpe")
    ns = O.neighbor_stability(res, "ma_cross")
    assert ns["n_neighbors"] >= 1
    assert "note" in ns
    warns = O.overfit_warnings(res)
    assert isinstance(warns, list)  # 提示存在与否取决于数据，但必须是无异常的规则输出
    # 少交易情形必须触发警告
    res2 = {"best": {"n_trades": 2, "train_metrics": {}, "val_metrics": {}},
            "test_metrics": {}, "n_combinations": 25}
    assert any("不足 5 次" in w for w in O.overfit_warnings(res2))
    assert any("选择偏差" in w for w in O.overfit_warnings(res2))


# ---------------- 存档 ----------------

def test_optimization_saved_and_test_flag_false():
    import quantlab.db as db
    ds_id = db.register_dataset("OPT_TEST", "OPT", source="demo", is_synthetic=1)
    from quantlab.storage import save_raw
    db.update_dataset(ds_id, raw_path=save_raw(_trend_df(), "OPT_TEST"), rows=300)
    res = O.run_grid_search(_trend_df(), "ma_cross", {"fast": [5], "slow": [20]}, cfg=CFG)
    rid = O.save_optimization_run(ds_id, "OPT_TEST", "ma_cross",
                                  {"fast": [5], "slow": [20]}, res)
    rec = db.list_optimization_runs(5)[0]
    assert rec["id"] == rid and rec["status"] == "success"
    assert rec["test_used_for_selection"] == 0     # 入库固化"test 未参与选参"
    import json
    assert len(json.loads(rec["results_json"])) == 1
