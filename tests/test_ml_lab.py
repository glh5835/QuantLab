"""机器学习实验室测试：标签对齐 / 时间划分与 purge / 预处理隔离 / 实验存档。"""
import numpy as np
import pandas as pd
import pytest

from quantlab import ml_lab as M
from quantlab.features import build_features


def _synthetic_df(n=400, seed=7):
    """带可学结构的合成行情：未来收益与动量弱相关（保证两类标签都出现）。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-01", periods=n)
    ret = rng.normal(0, 0.015, n)
    ret[10:] += 0.04 * ret[:-10] * 0  # 占位：保持噪声为主
    mom = np.zeros(n)
    for i in range(5, n):
        mom[i] = np.sign(np.sum(ret[i - 5:i])) * 0.004  # 动量效应注入
    ret[5:] += mom[5:]
    close = 100 * np.exp(np.cumsum(ret))
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.002, n)),
        "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": rng.integers(1e6, 5e6, n).astype(float),
    }, index=pd.DatetimeIndex(idx, name="trade_date"))


FEATS = ["ret_1", "ret_5", "sma_dev", "rsi_14", "vol_20", "volume_chg"]


# ---------------- 标签与特征对齐 ----------------

def test_label_uses_only_future_prices():
    df = _synthetic_df(60)
    y, rule = M.build_labels(df, "return", 1)
    close = df["close"]
    # 手工对齐：t 行的标签 = close[t+1]/close[t]-1
    assert np.isclose(y.iloc[0], close.iloc[1] / close.iloc[0] - 1)
    assert np.isnan(y.iloc[-1])  # 最后 1 个样本没有未来价格 → NaN（会被剔除，绝不虚构）
    assert "收益率" in rule and "特征截止" in rule  # 目标定义与特征可用时间均已说明


def test_direction_label_zero_rule_documented():
    df = _synthetic_df(60)
    # 构造一个恰好收益为 0 的样本
    df.iloc[10, df.columns.get_loc("close")] = df.iloc[9]["close"]
    y, rule = M.build_labels(df, "direction", 1)
    assert y.iloc[9] == 0.0          # 零收益 → 明确归入 0 类
    assert "0（跌/平）" in rule       # 规则已记录


def test_features_are_causal_truncation_equivalence():
    """最强防泄漏断言：只用前 k 行构造的特征第 k 行 == 全量构造的第 k 行。"""
    df = _synthetic_df(200)
    full = build_features(df, FEATS)
    k = 120
    part = build_features(df.iloc[:k], FEATS)
    for col in full.columns:
        assert (np.isnan(full[col].iloc[k - 1]) and np.isnan(part[col].iloc[-1])) or \
            np.isclose(full[col].iloc[k - 1], part[col].iloc[-1]), f"特征 {col} 泄漏未来数据！"


# ---------------- 时间划分与 purge ----------------

def test_split_is_chronological():
    df = _synthetic_df(300)
    X = build_features(df, FEATS)
    y, _ = M.build_labels(df, "direction", 1)
    samples = pd.concat([X, y.rename("y")], axis=1).dropna()
    tr, va, te, purged = M.time_split(samples.index, 1)
    assert tr.max() < va.min() and va.max() < te.min()  # 训练 < 验证 < 测试
    assert purged >= 0


def test_purge_removes_overlap_samples():
    idx = pd.bdate_range("2022-01-01", periods=300)
    tr, va, te, purged = M.time_split(idx, horizon=5)
    # 训练段末尾 5 个样本被隔离 → purge 计数 ≥ 5，且训练段与验证段之间隔开
    assert purged >= 5
    assert (va.min() - tr.max()).days >= 5  # 至少隔出标签窗口长度


def test_insufficient_samples_rejected():
    with pytest.raises(M.MLError, match="不足以"):
        M.time_split(pd.bdate_range("2024-01-01", periods=30), 1)  # 每段 < 20


# ---------------- 完整实验 ----------------

def test_run_experiment_direction_and_records():
    import quantlab.db as db
    ds_id = db.register_dataset("ML_TEST", "MLT", source="demo", is_synthetic=1)
    from quantlab.storage import save_raw
    db.update_dataset(ds_id, raw_path=save_raw(_synthetic_df(420), "ML_TEST"), rows=420)
    res = M.run_experiment(ds_id, "direction", 1, FEATS, "random_forest", name="t_rf")
    m = res["metrics"]
    assert {"train", "val", "test"} <= set(m)
    assert 0 <= m["test"]["accuracy"] <= 1
    assert "confusion" in m["test"]
    exp = db.get_experiment(res["exp_id"])
    assert exp["status"] == "success"
    import os
    assert os.path.exists(exp["model_path"])              # 模型文件在磁盘
    assert os.path.exists(exp["predictions_path"])        # 预测文件在磁盘
    preds = pd.read_csv(exp["predictions_path"])
    assert {"feature_time", "target_time", "split", "y_true", "y_pred"} <= set(preds.columns)
    assert set(preds["split"]) == {"train", "val", "test"}


def test_retrain_creates_new_record_not_overwrite():
    import quantlab.db as db
    ds_id = db.register_dataset("ML_TEST2", "ML2", source="demo", is_synthetic=1)
    from quantlab.storage import save_raw
    db.update_dataset(ds_id, raw_path=save_raw(_synthetic_df(420, seed=3), "ML_TEST2"), rows=420)
    r1 = M.run_experiment(ds_id, "direction", 1, FEATS, "logistic")
    r2 = M.run_experiment(ds_id, "direction", 1, FEATS, "logistic")
    assert r1["exp_id"] != r2["exp_id"]  # 重训必建新记录


def test_single_class_label_rejected():
    import quantlab.db as db
    n = 300
    idx = pd.DatetimeIndex(pd.bdate_range("2022-01-01", periods=n), name="trade_date")
    close = pd.Series(np.linspace(100, 200, n), index=idx)  # 单调上涨 → 方向标签全 1
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": 1e6}, index=idx)
    from quantlab.storage import save_raw
    ds_id = db.register_dataset("ML_UP", "UP", source="demo", is_synthetic=1)
    db.update_dataset(ds_id, raw_path=save_raw(df, "ML_UP"), rows=n)
    with pytest.raises(M.MLError, match="一个类别"):
        M.run_experiment(ds_id, "direction", 1, FEATS, "logistic")


def test_model_task_mismatch_rejected():
    with pytest.raises(M.MLError, match="匹配"):
        M._make_estimator("logistic", "return", 42)   # 逻辑回归不能做回归
    with pytest.raises(M.MLError, match="匹配"):
        M._make_estimator("linear", "direction", 42)  # 线性回归不能做分类


def test_preprocessing_fitted_only_on_train():
    """验证 Pipeline 的标准化器均值 = 训练集均值（而非全量数据均值）。"""
    df = _synthetic_df(420, seed=11)
    X = build_features(df, FEATS)
    y, _ = M.build_labels(df, "return", 1)
    samples = pd.concat([X, y.rename("y")], axis=1).dropna()
    tr, va, te, _ = M.time_split(samples.index, 1)
    pipe = M._make_pipeline("linear", "return", 42)
    pipe.fit(samples.drop(columns="y").loc[tr], samples["y"].loc[tr])
    scaler_mean = pipe.named_steps["scaler"].mean_[0]
    col = samples.columns[0]
    assert np.isclose(scaler_mean, samples.loc[tr, col].mean())      # = 训练集均值
    assert not np.isclose(scaler_mean, samples[col].mean())          # ≠ 全量均值（差异即证明未泄漏拟合）


def test_regression_metrics_and_determinism():
    import quantlab.db as db
    ds_id = db.register_dataset("ML_REG", "REG", source="demo", is_synthetic=1)
    from quantlab.storage import save_raw
    df = _synthetic_df(420, seed=5)
    db.update_dataset(ds_id, raw_path=save_raw(df, "ML_REG"), rows=420)
    r1 = M.run_experiment(ds_id, "return", 5, FEATS, "random_forest")
    r2 = M.run_experiment(ds_id, "return", 5, FEATS, "random_forest")
    m1, m2 = r1["metrics"]["test"], r2["metrics"]["test"]
    assert {"mae", "rmse", "r2"} <= set(m1)
    assert np.isclose(m1["rmse"], m2["rmse"])  # 固定种子 → 结果可复现
