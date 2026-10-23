"""机器学习实验室核心（模块 B）：样本构造 → 时间划分(purge) → Pipeline 训练 → 存档。

防泄漏三板斧（与任务书第三节逐条对应）：
1. 特征/标签分离构造：特征只用 ≤t 数据（features.py），标签只用 (t, t+h] 的未来价格，
   两者通过显式索引对齐，绝不做列偏移或事后合并。
2. 时间顺序三分：train < val < test（严禁随机打乱）；训练集末尾标签区间与验证集
   重叠的样本用 purge 规则剔除（horizon 天隔离），val→test 边界同理。
3. 预处理只学训练集：sklearn Pipeline（中位数填补 + 标准化 + 模型）只在训练集 fit，
   val/test 仅 transform——杜绝"先全局标准化再划分"的经典泄漏。

模型与预测文件存磁盘（data/models），SQLite 只存路径与元数据；重训必建新记录，不覆盖。
"""
import json
import numpy as np
import pandas as pd
from datetime import datetime

from . import db
from . import __version__ as CODE_VERSION
from .config import MODELS_DIR
from .features import build_features, FeatureError
from .storage import load_csv
from .indicators import prepare_ohlcv

TRADING_DAYS = 252
MIN_SPLIT_SAMPLES = 20  # 每个数据段的最少样本数，低于此值拒绝训练


class MLError(ValueError):
    """机器学习实验错误：带人类可读原因。"""


# ---------------- 数据加载 ----------------

def load_dataset_frame(dataset_id: int) -> tuple[pd.DataFrame, "db.sqlite3.Row"]:
    """从数据库加载数据集行情（优先清洗版）。返回 (行情df, 数据集行)。"""
    ds = db.get_dataset(dataset_id)
    if not ds:
        raise MLError(f"数据集 #{dataset_id} 不存在。")
    path = ds["clean_path"] if (ds["cleaned"] and ds["clean_path"]) else ds["raw_path"]
    if not path:
        raise MLError("该数据集没有数据文件，请先采集。")
    return load_csv(path), ds


# ---------------- 标签构造（只用未来价格，与特征严格分离） ----------------

def build_labels(df: pd.DataFrame, task: str, horizon: int) -> tuple[pd.Series, str]:
    """构造目标标签。

    direction（分类）: 未来 h 期收益 > 0 → 1（涨），否则 0（跌或平——零收益归入 0，
                       规则明确记录，不随意丢弃样本）。
    return（回归）:   未来 h 期简单收益率原值。

    样本 t 的标签窗口 = (t, t+h]：只引用 t 之后的收盘价，永不进入特征。
    """
    if task not in ("direction", "return"):
        raise MLError(f"未知任务 {task}，可选 direction / return。")
    if not isinstance(horizon, int) or horizon < 1:
        raise MLError(f"预测期限必须为正整数，收到 {horizon}。")
    close = prepare_ohlcv(df)["close"]
    fwd_ret = close.shift(-horizon) / close - 1.0  # 未来收益：t+h 收盘 / t 收盘 - 1
    if task == "direction":
        y = (fwd_ret > 0).astype(float)
        y[fwd_ret.isna()] = np.nan
        rule = (f"预测未来 {horizon} 个交易日收益方向：收益>0 → 1（涨），"
                f"收益≤0（含恰好为 0）→ 0（跌/平）。特征截止 t 收盘，标签用 (t, t+{horizon}] 收盘价。")
    else:
        y = fwd_ret
        rule = (f"预测未来 {horizon} 个交易日的简单收益率 close_(t+{horizon})/close_t − 1。"
                f"特征截止 t 收盘。")
    return y, rule


# ---------------- 时间序列划分（含 purge） ----------------

def time_split(sample_index: pd.DatetimeIndex, horizon: int,
               ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)):
    """按时间顺序三分 + purge 隔离。

    返回 (train_idx, val_idx, test_idx, purged_n)。purged_n 是因标签区间
    与后一段重叠而被剔除的样本数（train 末尾 h 个若跨入 val 起点，剔除；val 同理）。
    """
    n = len(sample_index)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    if min(n_train, n_val, n - n_train - n_val) < MIN_SPLIT_SAMPLES:
        raise MLError(
            f"样本量不足以做可靠的时间序列三分（每段至少 {MIN_SPLIT_SAMPLES} 个）："
            f"当前总共 {n} 个 → train {n_train} / val {n_val} / test {n - n_train - n_val}。"
            f"建议：换更长的数据区间、缩短预测期限、或减少所需暖机期长的特征。")
    train, val, test = sample_index[:n_train], sample_index[n_train:n_train + n_val], \
        sample_index[n_train + n_val:]

    # purge：按"交易日位置"截掉段末 Horizon 个样本（其标签窗口伸入下一段时间范围），
    # 严格保证训练标签与验证/测试标签的信息区间不重叠；同时保留每段最少样本数。
    purged = 0
    n_purge_train = min(horizon, max(len(train) - MIN_SPLIT_SAMPLES, 0))
    if n_purge_train > 0 and len(train) - n_purge_train >= MIN_SPLIT_SAMPLES:
        purged += n_purge_train
    train = train[:len(train) - n_purge_train]
    n_purge_val = min(horizon, max(len(val) - MIN_SPLIT_SAMPLES, 0))
    if n_purge_val > 0 and len(val) - n_purge_val >= MIN_SPLIT_SAMPLES:
        purged += n_purge_val
    val = val[:len(val) - n_purge_val]
    return train, val, test, purged


# ---------------- 模型与 Pipeline ----------------

MODELS = {
    "baseline":   {"desc": "简单基准：分类取训练集多数类 / 回归取训练集均值",
                   "clf": "dummy", "reg": "dummy"},
    "logistic":   {"desc": "逻辑回归（仅分类任务）", "clf": "logistic", "reg": None},
    "linear":     {"desc": "线性回归（仅回归任务）", "clf": None, "reg": "linear"},
    "random_forest": {"desc": "随机森林（分类与回归均可用）", "clf": "rf_clf", "reg": "rf_reg"},
}


def _make_estimator(model_key: str, task: str, seed: int):
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    kind = MODELS[model_key]["clf" if task == "direction" else "reg"]
    if kind is None:
        raise MLError(f"模型 {model_key} 不支持任务 {task}（模型必须与任务类型匹配）。")
    if kind == "dummy":
        return (DummyClassifier(strategy="most_frequent") if task == "direction"
                else DummyRegressor(strategy="mean"))
    if kind == "logistic":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if kind == "linear":
        return LinearRegression()
    if kind == "rf_clf":
        return RandomForestClassifier(n_estimators=200, max_depth=6, random_state=seed)
    if kind == "rf_reg":
        return RandomForestRegressor(n_estimators=200, max_depth=6, random_state=seed)
    raise MLError(f"未知模型 {model_key}")


def _make_pipeline(model_key: str, task: str, seed: int):
    """Pipeline：中位数填补 → 标准化 → 模型。整个 Pipeline 只在训练集 fit。

    即使随机森林不需要标准化，也统一走 Pipeline，保证填补/缩放参数只来自训练集。
    """
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", _make_estimator(model_key, task, seed)),
    ])


def _metrics_classification(y_true, y_pred, y_prob=None) -> dict:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, roc_auc_score)
    out = {"accuracy": float(accuracy_score(y_true, y_pred)),
           "precision": float(precision_score(y_true, y_pred, zero_division=0)),
           "recall": float(recall_score(y_true, y_pred, zero_division=0)),
           "f1": float(f1_score(y_true, y_pred, zero_division=0))}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:  # noqa: BLE001 - AUC 不适用时不展示，绝不伪造
            pass
    return out


def _metrics_regression(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred))}


# ---------------- 实验主流程 ----------------

def run_experiment(dataset_id: int, task: str, horizon: int, feature_keys: list[str],
                   model_key: str, name: str = None, seed: int = 42,
                   ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> dict:
    """执行完整实验：特征 → 标签 → 划分 → 训练 → 评估 → 存档。返回实验摘要。

    任何一步不满足前提（单类别标签、样本不足、特征全空）都会以 MLError 拒绝，
    明确说明原因，而不是给出不可信的结果。
    """
    df, ds = load_dataset_frame(dataset_id)
    try:
        X_all = build_features(df, feature_keys)
    except FeatureError as e:
        raise MLError(str(e)) from e
    y_all, label_rule = build_labels(df, task, horizon)

    # 显式按索引对齐（特征行 t ↔ 标签 (t, t+h]），删除暖机期 NaN 与末尾无标签样本
    samples = pd.concat([X_all, y_all.rename("__y__")], axis=1).dropna()
    if len(samples) < 3 * MIN_SPLIT_SAMPLES:
        raise MLError(f"有效样本仅 {len(samples)} 个（特征暖机期与标签窗口会消耗两端样本），"
                      f"不足以训练。建议延长数据区间或减少长窗口特征。")
    if task == "direction" and samples["__y__"].nunique() < 2:
        raise MLError("标签只有一个类别（该区间内未来收益方向全部相同），"
                      "无法训练分类模型。建议更换数据区间或期限。")

    X, y = samples.drop(columns="__y__"), samples["__y__"]
    train_i, val_i, test_i, purged_n = time_split(samples.index, horizon, ratios)

    pipe = _make_pipeline(model_key, task, seed)
    pipe.fit(X.loc[train_i], y.loc[train_i])  # Pipeline 整体只在训练集 fit

    def _eval(idx: pd.DatetimeIndex) -> dict:
        pred = pipe.predict(X.loc[idx])
        prob = None
        if task == "direction" and hasattr(pipe, "predict_proba"):
            prob = pipe.predict_proba(X.loc[idx])[:, 1]
        m = _metrics_classification(y.loc[idx], pred, prob) if task == "direction" \
            else _metrics_regression(y.loc[idx], pred)
        m["n"] = int(len(idx))
        m["start"], m["end"] = str(idx.min().date()), str(idx.max().date())
        return m

    metrics = {"train": _eval(train_i), "val": _eval(val_i), "test": _eval(test_i)}

    # ---- 存档：先插库拿 id，再写文件，最后回填路径 ----
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with db.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO experiments (name, created_at, dataset_id, dataset_name, dataset_version, "
            "task, horizon, label_rule, features_json, model_type, model_params, seed, split_ratios, "
            "train_range, val_range, test_range, train_n, val_n, test_n, purged_n, "
            "status, code_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (name or f"{ds['name']}_{task}_{model_key}", ts, dataset_id, ds["name"], ds["version"],
             task, horizon, label_rule, json.dumps(feature_keys, ensure_ascii=False), model_key,
             json.dumps({"seed": seed}), seed, json.dumps(list(ratios)),
             f"{train_i.min().date()}~{train_i.max().date()}",
             f"{val_i.min().date()}~{val_i.max().date()}",
             f"{test_i.min().date()}~{test_i.max().date()}",
             len(train_i), len(val_i), len(test_i), purged_n, "success", CODE_VERSION))
        exp_id = cur.lastrowid

    import joblib
    model_path = MODELS_DIR / f"exp_{exp_id}.joblib"
    joblib.dump(pipe, model_path)

    # 预测存档：feature_time=特征截止日 t；target_time=标签终点 t+h；split 标注所属段
    rows = []
    for split, idx in (("train", train_i), ("val", val_i), ("test", test_i)):
        pred = pipe.predict(X.loc[idx])
        prob = pipe.predict_proba(X.loc[idx])[:, 1] if (
            task == "direction" and hasattr(pipe, "predict_proba")) else None
        y_true = y.loc[idx]
        for i, t in enumerate(idx):
            r = {"feature_time": str(t.date()),
                 "target_time": str((t + pd.Timedelta(days=horizon)).date()),
                 "split": split, "y_true": float(y_true.loc[t]), "y_pred": float(pred[i])}
            if prob is not None:
                r["y_prob"] = float(prob[i])
            rows.append(r)
    pred_df = pd.DataFrame(rows)
    predictions_path = MODELS_DIR / f"exp_{exp_id}_predictions.csv"
    pred_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    update_experiment(exp_id, metrics_json=json.dumps(metrics, ensure_ascii=False),
                      model_path=str(model_path), predictions_path=str(predictions_path))
    return {"exp_id": exp_id, "metrics": metrics, "label_rule": label_rule,
            "model_path": str(model_path), "predictions_path": str(predictions_path),
            "purged_n": purged_n, "feature_keys": list(X.columns)}


def update_experiment(exp_id: int, **fields) -> None:
    cols = ", ".join(f"{k}=?" for k in fields)
    with db.get_conn() as conn:
        conn.execute(f"UPDATE experiments SET {cols} WHERE id=?", (*fields.values(), exp_id))
