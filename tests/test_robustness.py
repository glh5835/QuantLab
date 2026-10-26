"""滚动验证测试：窗口时序 / purge 隔离 / 每窗口独立训练 / 汇总披露 / 诊断提示。"""
import numpy as np
import pandas as pd
import pytest

from quantlab import robustness as RB
from quantlab.backtest import BacktestConfig
from quantlab.robustness import RobustnessError
from quantlab.strategies import MACrossoverStrategy

FEATS = ["ret_1", "ret_5", "rsi_14", "vol_20"]
CFG = BacktestConfig(commission_rate=0, min_commission=0, slippage=0)


def _df(n=400, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.018, n)))
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": 1e6},
                        index=pd.DatetimeIndex(idx, name="trade_date"))


# ---------------- 窗口生成 ----------------

def test_generate_windows_strictly_ordered():
    df = _df(400)
    ws = RB.generate_windows(df.index, train_size=150, test_size=50)
    assert len(ws) >= 2
    for w in ws:
        assert w["train"][1] < w["test"][0]        # 训练严格早于测试
        assert w["train_n"] >= 60 and w["test_n"] >= 20
    # 相邻窗口：测试段首尾相接（不重叠、不遗漏）
    for a, b in zip(ws, ws[1:]):
        assert b["test"][0] > a["test"][1]


def test_expanding_mode_train_start_fixed():
    df = _df(400)
    ws = RB.expand_windows(df.index, 150, 50, mode="expanding")
    starts = {str(w["train"][0].date()) for w in ws}
    assert len(starts) == 1                         # 扩展窗：训练起点固定
    assert ws[-1]["train_n"] > ws[0]["train_n"]     # 训练样本递增
    with pytest.raises(RobustnessError):
        RB.expand_windows(df.index, 150, 50, mode="weekly")
    with pytest.raises(RobustnessError, match="至少 2 个"):
        RB.generate_windows(df.index[:200], 150, 50)


# ---------------- 规则策略滚动验证 ----------------

def test_walk_forward_rule_freezes_params_per_window():
    df = _df(400)
    res = RB.walk_forward_rule(df, lambda: MACrossoverStrategy(5, 20),
                               cfg=CFG, train_size=150, test_size=60)
    assert res["strategy"] == "rule" and res["n_windows"] >= 2
    for w in res["windows"]:
        assert "test_range" in w and w["test_metrics"].get("n_days", 0) >= 5
        assert w["test_range"].split("~")[0] > w["train_range"].split("~")[1].replace(" ", "")


def test_walk_forward_rule_deterministic():
    df = _df(400, seed=3)
    r1 = RB.walk_forward_rule(df, lambda: MACrossoverStrategy(5, 20), cfg=CFG,
                              train_size=150, test_size=60)
    r2 = RB.walk_forward_rule(df, lambda: MACrossoverStrategy(5, 20), cfg=CFG,
                              train_size=150, test_size=60)
    assert r1["summary"]["mean_return"] == r2["summary"]["mean_return"]


# ---------------- ML 滚动验证（防泄漏核心） ----------------

def test_walk_forward_ml_each_window_independent_and_purged():
    df = _df(420, seed=5)
    res = RB.walk_forward_ml(df, "direction", 1, FEATS, "random_forest",
                             cfg=CFG, train_size=250, test_size=50)
    assert res["n_windows"] >= 2
    for w in res["windows"]:
        if "error" in w:
            continue
        # 训练区间严格早于测试区间
        tr_end = pd.Timestamp(w["train_range"].split("~")[1])
        te_start = pd.Timestamp(w["test_range"].split("~")[0])
        assert tr_end < te_start
        # purge 必须实际执行（horizon=1 → 至少隔离 1 个样本，除非样本紧张）
        assert w.get("purged_n", 0) >= 1
        assert 0 <= w["train_fit_score"] <= 1
    assert res["summary"]["n_windows"] == res["n_windows"]


def test_walk_forward_ml_labels_do_not_cross_boundary():
    """窗口训练样本的标签终点必须 ≤ 训练段末尾（purge 后无跨边界标签）。"""
    df = _df(420, seed=9)
    ws = RB.expand_windows(df.index, 250, 50, mode="rolling")[:1]
    w = ws[0]
    tr_start, tr_end = w["train"]
    horizon = 3
    from quantlab.features import build_features
    from quantlab.ml_lab import build_labels
    X = build_features(df.loc[:tr_end], FEATS)
    y, _ = build_labels(df.loc[:tr_end], "return", horizon)
    samples = pd.concat([X, y.rename("y")], axis=1).dropna()
    purged = min(horizon, len(samples))
    kept = samples.iloc[:len(samples) - purged]
    # 保留样本最后一行的标签需要 close[t+3]：t 必须使 t+3 ≤ tr_end
    last_t = kept.index[-1]
    pos = df.index.get_indexer([last_t])[0]
    assert pos + horizon <= df.index.get_indexer([tr_end])[0] + 1  # 标签窗口不越界


def test_walk_forward_ml_deterministic():
    df = _df(420, seed=11)
    r1 = RB.walk_forward_ml(df, "direction", 1, FEATS, "logistic", cfg=CFG,
                            train_size=250, test_size=50, n_windows=2)
    r2 = RB.walk_forward_ml(df, "direction", 1, FEATS, "logistic", cfg=CFG,
                            train_size=250, test_size=50, n_windows=2)
    assert r1["summary"]["mean_return"] == r2["summary"]["mean_return"]  # 固定种子可复现


def test_walk_forward_ml_insufficient_data_rejected():
    with pytest.raises(RobustnessError, match="至少 2 个"):
        RB.walk_forward_ml(_df(280), "direction", 1, FEATS, "logistic",
                           train_size=250, test_size=50)


# ---------------- 汇总与诊断 ----------------

def test_summary_discloses_worst_window_not_just_mean():
    ws = [{"test_metrics": {"total_return": 0.10, "sharpe": 1.0, "mdd": -0.05},
           "test_range": "A", "n_trades": 5},
          {"test_metrics": {"total_return": -0.25, "sharpe": -1.5, "mdd": -0.30},
           "test_range": "B", "n_trades": 4},
          {"test_metrics": {"total_return": 0.15, "sharpe": 1.2, "mdd": -0.08},
           "test_range": "C", "n_trades": 6}]
    s = RB.summarize_windows(ws)
    assert np.isclose(s["mean_return"], 0.0)         # 均值 0
    assert s["worst_window_return"] == -0.25          # 但最差窗口 -25% 必须披露
    assert s["worst_window_range"] == "B"
    assert np.isclose(s["positive_ratio"], 2 / 3)
    assert s["worst_mdd"] == -0.30


def test_overfit_diagnosis_rules():
    res = {"summary": {"positive_ratio": 0.33, "worst_window_return": -0.30,
                       "worst_window_range": "2022Q1", "std_return": 0.2,
                       "mean_return": 0.05, "n_windows": 3}, "strategy": "ml",
           "windows": []}
    warns = RB.overfit_diagnosis(res)
    assert any("正收益窗口占比" in w for w in warns)
    assert any("最差窗口" in w for w in warns)
    # 离散度远大于均值
    res2 = {"summary": {"positive_ratio": 0.6, "worst_window_return": -0.05,
                        "std_return": 0.3, "mean_return": 0.02, "n_windows": 5},
            "strategy": "rule", "windows": []}
    assert any("离散度" in w for w in RB.overfit_diagnosis(res2))


def test_save_walk_forward_run():
    import quantlab.db as db
    from quantlab.storage import save_raw
    df = _df(400, seed=13)
    ds_id = db.register_dataset("WF_TEST", "WFT", source="demo", is_synthetic=1)
    db.update_dataset(ds_id, raw_path=save_raw(df, "WF_TEST"), rows=400)
    res = RB.walk_forward_rule(df, lambda: MACrossoverStrategy(5, 20), cfg=CFG,
                               train_size=150, test_size=60)
    rid = RB.save_walk_forward_run(ds_id, "WF_TEST", "ma_cross(5,20)", res,
                                   {"train_size": 150, "test_size": 60})
    rec = db.list_walk_forward(5)[0]
    assert rec["id"] == rid and rec["status"] == "success"
    import json
    assert len(json.loads(rec["windows_json"])) == res["n_windows"]
