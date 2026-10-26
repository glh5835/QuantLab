"""滚动样本外验证与稳健性分析（阶段 3D）。

核心承诺（任务书 D1/D2/D3）：
- 按时间顺序逐窗口执行：窗口 i 的训练数据严格早于其测试区间；
  训练段末尾 horizon 个样本因标签窗口伸入测试段而被 purge（复用 ml_lab 的隔离原则）。
- 每个窗口独立拟合整个 Pipeline（预处理器只学该窗口的训练数据）。
- 规则策略（均线交叉）滚动验证：每窗口在 [窗口起点前史暖机, test_end] 回测，
  指标只取 test 段——策略在每个窗口都是同一组冻结参数，检验的是不同时段的稳定性。
- 汇总统计同时给出均值、中位数、离散度、正收益窗口占比与最差窗口，
  绝不用合并平均掩盖个别窗口的大幅亏损。
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import db, __version__ as CODE_VERSION
from .backtest import _run_engine, BacktestConfig
from .features import build_features
from .ml_lab import build_labels, _make_pipeline, MIN_SPLIT_SAMPLES
from .optimizer import _segment_metrics
from .strategies import MACrossoverStrategy, MLSignalStrategy

TRADING_DAYS = 252


class RobustnessError(ValueError):
    pass


def generate_windows(index: pd.DatetimeIndex, train_size: int, test_size: int,
                     n_windows: int = None) -> list[dict]:
    """生成滚动窗口序列：[{train:[start,end], test:[start,end]}...]。

    固定间隔 test_size 前进；train 起点按 mode（rolling/expanding）由调用方决定长度，
    这里只负责窗口边界：所有窗口 train.end < test.start（中间自然留出 horizon 缓冲，
    ML 用途下 purge 在样本层额外执行）。
    """
    if train_size < 60 or test_size < 20:
        raise RobustnessError("训练窗口至少 60 个交易日、测试窗口至少 20 个。")
    n = len(index)
    windows = []
    pos = train_size  # 首个测试段起点（索引）
    while pos + test_size <= n:
        train = (index[max(pos - train_size, 0)], index[pos - 1])
        test = (index[pos], index[min(pos + test_size, n) - 1])
        windows.append({"train": train, "test": test,
                        "train_n": pos - max(pos - train_size, 0),
                        "test_n": min(pos + test_size, n) - pos})
        pos += test_size
        if n_windows and len(windows) >= n_windows:
            break
    if len(windows) < 2:
        raise RobustnessError(
            f"数据长度只够 {len(windows)} 个滚动窗口（至少 2 个）。"
            f"需要 ≥ train+2×test 个交易日，当前 {n}。")
    return windows


def expand_windows(index, train_size, test_size, n_windows=None, mode="rolling"):
    """rolling=固定训练窗；expanding=训练起点固定在数据开头。"""
    base = generate_windows(index, train_size, test_size, n_windows)
    if mode == "expanding":
        for i, w in enumerate(base):
            start_i = index.get_indexer([w["train"][0]])[0]
            w["train"] = (index[0], w["train"][1])
            w["train_n"] = start_i + w["train_n"]
    elif mode != "rolling":
        raise RobustnessError(f"mode 必须是 rolling/expanding，收到 {mode}")
    return base


# ---------------- 规则策略滚动验证 ----------------

def walk_forward_rule(df: pd.DataFrame, strategy_factory, cfg: BacktestConfig = None,
                      train_size: int = 120, test_size: int = 40,
                      mode: str = "rolling", n_windows: int = None) -> dict:
    """规则策略（参数冻结）滚动窗口回测：每窗口同参数，检验跨时段稳定性。"""
    cfg = cfg or BacktestConfig()
    windows = expand_windows(df.index, train_size, test_size, n_windows, mode)
    results = []
    for w in windows:
        strat = strategy_factory()  # 每窗口全新实例（内部状态不跨窗口）
        seg = df.loc[:w["test"][1]]  # 保留前史供暖机（均线需要）
        run = _run_engine(seg, strat, cfg)
        test_m = _segment_metrics(run["daily"], w["test"][0], w["test"][1], cfg.initial_cash)
        results.append({
            "train_range": f"{w['train'][0].date()}~{w['train'][1].date()}",
            "test_range": f"{w['test'][0].date()}~{w['test'][1].date()}",
            "train_n": w["train_n"], "test_n": w["test_n"],
            "test_metrics": test_m, "n_trades": run["metrics"]["n_trades"],
            "label_coverage": f"{w['train'][0].date()}~{w['test'][1].date()}",
        })
    return {"windows": results, "n_windows": len(results), "mode": mode,
            "strategy": "rule", "summary": summarize_windows(results)}


# ---------------- ML 滚动验证（每窗口独立训练） ----------------

def walk_forward_ml(df: pd.DataFrame, task: str, horizon: int,
                    feature_keys: list[str], model_key: str, seed: int = 42,
                    cfg: BacktestConfig = None, train_size: int = 200,
                    test_size: int = 40, mode: str = "rolling",
                    n_windows: int = None,
                    buy_threshold: float = 0.0, exit_threshold: float = 0.0) -> dict:
    """ML 滚动验证：窗口 i 用 [train_start, train_end-purge] 样本训练全新 Pipeline，
    在 test 段生成预测信号并回测。每个窗口记录训练/测试区间、样本数与指标。
    """
    cfg = cfg or BacktestConfig()
    windows = expand_windows(df.index, train_size, test_size, n_windows, mode)
    results = []
    for w in windows:
        tr_start, tr_end = w["train"]
        te_start, te_end = w["test"]
        # 特征/标签构造只用 ≤ tr_end（未来标签仅用于训练样本）
        X_tr = build_features(df.loc[:tr_end], feature_keys)
        y_tr, rule = build_labels(df.loc[:tr_end], task, horizon)
        samples = pd.concat([X_tr, y_tr.rename("y")], axis=1).dropna()
        # purge：训练段末尾 horizon 个样本的标签窗口伸入测试时间范围 → 剔除
        purged = min(horizon, max(len(samples) - MIN_SPLIT_SAMPLES, 0))
        samples = samples.iloc[:len(samples) - purged]
        if len(samples) < MIN_SPLIT_SAMPLES * 2:
            results.append({"test_range": f"{te_start.date()}~{te_end.date()}",
                            "error": f"窗口训练样本不足（{len(samples)}），跳过"})
            continue
        if task == "direction" and samples["y"].nunique() < 2:
            results.append({"test_range": f"{te_start.date()}~{te_end.date()}",
                            "error": "训练标签只有单一类别，跳过"})
            continue
        pipe = _make_pipeline(model_key, task, seed)  # 每窗口独立 Pipeline（预处理器重学）
        pipe.fit(samples.drop(columns="y"), samples["y"])
        # 测试段特征：在全前史上滚动计算（因果性由截断等价保证），只取 test 段行
        X_te = build_features(df.loc[:te_end], feature_keys).loc[te_start:te_end].dropna()
        if len(X_te) < 5:
            results.append({"test_range": f"{te_start.date()}~{te_end.date()}",
                            "error": "测试段有效特征样本不足，跳过"})
            continue
        preds = pipe.predict(X_te)
        pred_df = pd.DataFrame({"feature_time": X_te.index, "y_pred": preds})
        try:
            strat = MLSignalStrategy(predictions=pred_df, task=task,
                                     buy_threshold=buy_threshold,
                                     exit_threshold=exit_threshold)
        except Exception as e:  # noqa: BLE001
            results.append({"test_range": f"{te_start.date()}~{te_end.date()}",
                            "error": f"策略构造失败：{e}"})
            continue
        run = _run_engine(df.loc[:te_end], strat, cfg)
        test_m = _segment_metrics(run["daily"], te_start, te_end, cfg.initial_cash)
        # 训练段拟合度（过拟合对照）：同一模型在训练样本上的准确率/R²
        if task == "direction":
            fit_score = float(pipe.score(samples.drop(columns="y"), samples["y"]))
        else:
            fit_score = float(pipe.score(samples.drop(columns="y"), samples["y"]))
        results.append({
            "train_range": f"{tr_start.date()}~{tr_end.date()}",
            "test_range": f"{te_start.date()}~{te_end.date()}",
            "train_n": int(len(samples)), "test_n": int(len(X_te)),
            "purged_n": int(purged),
            "label_rule": rule,
            "label_coverage": f"{tr_start.date()}~{te_end.date()}",
            "train_fit_score": fit_score,
            "test_metrics": test_m, "n_trades": run["metrics"]["n_trades"],
        })
    ok = [r for r in results if "error" not in r]
    if not ok:
        raise RobustnessError("所有滚动窗口都无法训练/评估（数据不足或标签单一）。")
    return {"windows": results, "n_windows": len(ok), "n_skipped": len(results) - len(ok),
            "mode": mode, "strategy": "ml", "summary": summarize_windows(ok),
            "data_version_note": "窗口训练只用该窗口当时可获得的数据；模型逐窗口重训。"}


# ---------------- 汇总与过拟合提示 ----------------

def summarize_windows(window_results: list[dict]) -> dict:
    """样本外汇总：均值/中位数/离散度/正收益占比/最差窗口（D3 全量披露）。"""
    rets, sharpes, mdds, trades = [], [], [], []
    for r in window_results:
        m = r.get("test_metrics", {})
        if m.get("total_return") is not None:
            rets.append(float(m["total_return"]))
        if m.get("sharpe") is not None:
            sharpes.append(float(m["sharpe"]))
        if m.get("mdd") is not None:
            mdds.append(float(m["mdd"]))
        if r.get("n_trades") is not None:
            trades.append(int(r["n_trades"]))
    if not rets:
        return {"note": "无有效窗口指标"}
    arr = np.array(rets)
    worst_idx = int(np.argmin(arr))
    return {
        "n_windows": len(rets),
        "mean_return": float(arr.mean()), "median_return": float(np.median(arr)),
        "std_return": float(arr.std(ddof=1)) if len(arr) > 1 else None,
        "positive_ratio": float((arr > 0).mean()),
        "worst_window_return": float(arr.min()),
        "worst_window_range": window_results[worst_idx].get("test_range"),
        "best_window_return": float(arr.max()),
        "mean_sharpe": float(np.mean(sharpes)) if sharpes else None,
        "worst_mdd": float(min(mdds)) if mdds else None,
        "total_trades": int(sum(trades)),
    }


def overfit_diagnosis(result: dict) -> list[str]:
    """滚动验证的过拟合/稳健性诊断提示（文字规则）。"""
    warns = []
    s = result.get("summary", {})
    if s.get("note"):
        return [s["note"]]
    if s.get("positive_ratio") is not None and s["positive_ratio"] < 0.5:
        warns.append(f"正收益窗口占比仅 {s['positive_ratio']:.0%}：多数样本外窗口亏损，"
                     f"策略稳健性存疑。")
    if s.get("worst_window_return") is not None and s["worst_window_return"] < -0.10:
        warns.append(f"最差窗口收益 {s['worst_window_return']:.1%}"
                     f"（{s.get('worst_window_range')}）：存在大幅亏损窗口，"
                     f"均值掩盖了尾部风险。")
    if s.get("std_return") is not None and abs(s.get("mean_return", 0)) > 0 \
            and s["std_return"] > 2 * abs(s["mean_return"]):
        warns.append("窗口收益离散度远大于均值：表现极不稳定，不应据均值下结论。")
    if result.get("strategy") == "ml":
        fits = [r.get("train_fit_score") for r in result.get("windows", [])
                if "error" not in r and r.get("train_fit_score") is not None]
        test_r = [r["test_metrics"].get("total_return") for r in result.get("windows", [])
                  if "error" not in r and r.get("test_metrics", {}).get("total_return") is not None]
        if fits and test_r:
            import numpy as _np
            if _np.mean(fits) > 0.7 and _np.mean(test_r) < 0:
                warns.append(f"各窗口训练拟合得分均值 {np.mean(fits):.2f} 偏高，而样本外平均收益为负："
                             f"模型反复记忆训练期、样本外失效（典型过拟合形态）。")
    return warns


# ---------------- 存档 ----------------

def save_walk_forward_run(dataset_id: int, dataset_name: str, strategy_desc: str,
                          result: dict, config: dict) -> int:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return db.save_walk_forward(
        created_at=ts, dataset_id=dataset_id, dataset_name=dataset_name,
        strategy_type=strategy_desc, mode=result.get("mode", "rolling"),
        config_json=json.dumps(config, ensure_ascii=False, default=str),
        windows_json=json.dumps(result["windows"], ensure_ascii=False, default=float),
        summary_json=json.dumps(result["summary"], ensure_ascii=False, default=float),
        n_windows=result.get("n_windows", 0), status="success", code_version=CODE_VERSION)
