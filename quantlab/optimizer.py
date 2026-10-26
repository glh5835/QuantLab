"""策略参数优化（阶段 3C）：有限网格搜索 + 严格的选参/评估数据隔离。

数据隔离规则（任务书 C3/D2，与 ml_lab 同一原则）：
- 数据按时间三分：train / val / test。
- 网格搜索只用 train+val 段回测，**以 val 段指标选参**（train 段提供暖机与前史）。
- 最优参数确定后，才在 test 段（更准确说：全区间回测中 test 段的净值）做一次
  最终评估——test 段从不参与参数选择，结果中 test_used_for_selection=False 如实记录。
- 记录尝试过的组合总数，供过拟合/筛选偏差提示（D4）使用。

回测口径：每组参数在 [数据起点, val_end] 上完整回测（含暖机），指标只从
daily 净值中 date ∈ val 段的子序列计算（连续截取口径，各组合完全一致，公平比较）。
"""
import itertools
import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import db, __version__ as CODE_VERSION
from .backtest import _run_engine, _performance_metrics, BacktestConfig
from .strategies import MACrossoverStrategy, MLSignalStrategy, StrategyError

MAX_COMBINATIONS = 60  # 硬上限：防止无上限搜索


class OptimizerError(ValueError):
    pass


# ---------------- 参数空间与合法性 ----------------

def build_grid(param_space: dict) -> list[dict]:
    """param_space: {"fast": [3,5,8], "slow": [10,20,30]} → 笛卡尔积，过滤非法组合。"""
    keys = list(param_space.keys())
    if not keys:
        raise OptimizerError("参数空间为空。")
    lists = []
    for k in keys:
        v = param_space[k]
        if not isinstance(v, (list, tuple)) or len(v) == 0:
            raise OptimizerError(f"参数 {k} 的取值列表不能为空。")
        lists.append(list(v))
    combos = []
    for values in itertools.product(*lists):
        params = dict(zip(keys, values))
        if "fast" in params and "slow" in params and params["fast"] >= params["slow"]:
            continue  # 快线必须小于慢线：非法组合直接排除
        combos.append(params)
    if not combos:
        raise OptimizerError("过滤非法组合后没有可运行的参数组合（检查 fast<slow）。")
    if len(combos) > MAX_COMBINATIONS:
        raise OptimizerError(
            f"参数组合数 {len(combos)} 超过硬上限 {MAX_COMBINATIONS}。"
            f"请缩小搜索范围（例如减少每个参数的取值个数）。")
    return combos


# ---------------- 时间三分 ----------------

def split_timeline(index: pd.DatetimeIndex, ratios=(0.6, 0.2, 0.2)):
    """时间三分点：返回 (train_end, val_end, test_end) 三个日期。train 段同时用于回测暖机。"""
    n = len(index)
    if n < 60:
        raise OptimizerError(f"数据仅 {n} 个交易日，参数优化至少需要 60 天（建议 ≥ 250）。")
    i_train = int(n * ratios[0])
    i_val = int(n * (ratios[0] + ratios[1]))
    if min(i_train, i_val - i_train, n - i_val) < 15:
        raise OptimizerError("三分后某段不足 15 个交易日，无法可靠评估。请延长数据或调整比例。")
    return index[i_train - 1], index[i_val - 1], index[n - 1]


def _segment_metrics(daily: list[dict], seg_start, seg_end, initial: float) -> dict:
    """从回测 daily 记录中截取 [seg_start, seg_end] 的净值序列计算指标。"""
    eq = pd.Series(
        {pd.Timestamp(d["date"]): d["equity"] for d in daily
         if seg_start <= pd.Timestamp(d["date"]) <= seg_end}).sort_index()
    if len(eq) < 5:
        return {"n_days": len(eq), "note": "区间样本不足，指标不可靠"}
    m = _performance_metrics(eq, initial=eq.iloc[0])  # 以区间起点净值为基数
    m["n_days"] = int(len(eq))
    return m


# ---------------- 网格搜索 ----------------

def make_strategy(strategy_type: str, params: dict):
    """按参数字典构造策略实例（非法参数由策略构造器抛 StrategyError）。"""
    if strategy_type == "ma_cross":
        return MACrossoverStrategy(fast=int(params["fast"]), slow=int(params["slow"]))
    if strategy_type == "ml_signal":
        return MLSignalStrategy(int(params["experiment_id"]),
                                buy_threshold=float(params.get("buy_threshold", 0.0)),
                                exit_threshold=float(params.get("exit_threshold", 0.0)))
    raise OptimizerError(f"暂不支持 {strategy_type} 的参数优化（RSI 策略尚未实现，不虚构支持）。")


def run_grid_search(df: pd.DataFrame, strategy_type: str, param_space: dict,
                    cfg: BacktestConfig = None, objective: str = "sharpe",
                    max_mdd: float = None, min_trades: int = 0,
                    progress_cb=None) -> dict:
    """执行网格搜索（同步；页面经任务系统在后台调用，progress_cb(i,total,params) 汇报进度）。

    返回：{combinations: [{params, val_metrics, train_metrics}], best, test_metrics,
           n_combinations, objective, constraints, test_used_for_selection: False}
    """
    combos = build_grid(param_space)
    cfg = cfg or BacktestConfig()
    train_end, val_end, test_end = split_timeline(df.index)
    df_to_val = df.loc[:val_end]  # 搜索只用 train+val（含前史暖机），绝不触碰 test

    results = []
    for i, params in enumerate(combos):
        if progress_cb:
            progress_cb(i + 1, len(combos), params)
        try:
            strat = make_strategy(strategy_type, params)
        except (StrategyError, KeyError) as e:
            results.append({"params": params, "error": f"非法参数: {e}"})
            continue
        run = _run_engine(df_to_val, strat, cfg)
        val_m = _segment_metrics(run["daily"], train_end, val_end, cfg.initial_cash)
        train_m = _segment_metrics(run["daily"], df.index[0], train_end, cfg.initial_cash)
        results.append({"params": params, "val_metrics": val_m, "train_metrics": train_m,
                        "n_trades": run["metrics"]["n_trades"],
                        "val_sharpe": val_m.get("sharpe"),
                        "val_total_return": val_m.get("total_return"),
                        "val_mdd": val_m.get("mdd")})

    # 约束过滤 + 目标最大化
    def _feasible(r):
        if "error" in r:
            return False
        if max_mdd is not None and r["val_metrics"].get("mdd") is not None \
                and r["val_metrics"]["mdd"] < -max_mdd:
            return False
        if min_trades and (r.get("n_trades") or 0) < min_trades:
            return False
        return True

    feasible = [r for r in results if _feasible(r)]
    if not feasible:
        raise OptimizerError(
            "所有参数组合都被约束排除（最大回撤上限/最低交易次数）。"
            "建议放宽约束或扩大搜索空间。")
    key = {"sharpe": lambda r: (r["val_metrics"].get("sharpe") is not None,
                                r["val_metrics"].get("sharpe") or -np.inf),
           "return": lambda r: (r["val_metrics"].get("total_return") is not None,
                                r["val_metrics"].get("total_return") or -np.inf),
           "calmar": lambda r: (r["val_metrics"].get("mdd", 0) not in (None, 0)
                                and r["val_metrics"].get("cagr") is not None,
                                (r["val_metrics"]["cagr"] / abs(r["val_metrics"]["mdd"]))
                                if r["val_metrics"].get("mdd") not in (None, 0)
                                and r["val_metrics"].get("cagr") is not None else -np.inf)}
    if objective not in key:
        raise OptimizerError(f"未知优化目标 {objective}，可选 {list(key)}")
    best = max(feasible, key=key[objective])

    # 最终评估：最优参数在全区间回测，指标取 test 段（test 从未参与选择）
    best_strat = make_strategy(strategy_type, best["params"])
    final_run = _run_engine(df, best_strat, cfg)
    test_m = _segment_metrics(final_run["daily"], val_end, test_end, cfg.initial_cash)

    return {"combinations": results, "best": best, "test_metrics": test_m,
            "n_combinations": len(combos), "n_feasible": len(feasible),
            "objective": objective,
            "constraints": {"max_mdd": max_mdd, "min_trades": min_trades},
            "segments": {"train_end": str(train_end.date()), "val_end": str(val_end.date()),
                         "test_end": str(test_end.date())},
            "test_used_for_selection": False,
            "assumption": "网格搜索仅用 train+val 段回测并以 val 段指标选参；"
                          "test 段仅在参数确定后评估一次，未参与选择。"}


# ---------------- 敏感性与过拟合提示 ----------------

def neighbor_stability(result: dict, strategy_type: str) -> dict:
    """最优参数的邻近组合稳定性：网格中与最优参数只差一格取值的组合。"""
    best_p = result["best"]["params"]
    metric_key = "val_sharpe" if result["objective"] == "sharpe" else "val_total_return"
    neighbors = []
    for r in result["combinations"]:
        if "error" in r:
            continue
        p = r["params"]
        diff = sum(1 for k in best_p if k in p and p[k] != best_p[k])
        if 0 < diff <= min(1, len(best_p)):  # 恰有一个参数取值不同
            neighbors.append({**p, metric_key: r.get(metric_key)})
    best_v = result["best"].get(metric_key)
    nb_vals = [n[metric_key] for n in neighbors if n[metric_key] is not None]
    if not nb_vals or best_v is None:
        return {"n_neighbors": len(neighbors), "stable": None,
                "note": "邻近组合无有效指标，无法评估稳定性"}
    mean_nb = float(np.mean(nb_vals))
    return {"n_neighbors": len(neighbors), "neighbor_mean": mean_nb,
            "best_value": float(best_v),
            "ratio": float(best_v / mean_nb) if mean_nb != 0 else None,
            "stable": bool(min(nb_vals) > 0 if best_v > 0 else max(nb_vals) < 0),
            "note": ("最优点的邻近组合均值明显更低（比值>1.5 时警惕孤立优点）" if
                     (mean_nb != 0 and abs(best_v / mean_nb) > 1.5 and best_v > 0)
                     else "邻近组合与最优点表现接近或同号")}


def overfit_warnings(result: dict) -> list[str]:
    """D4 过拟合与筛选偏差提示（文字规则，页面展示）。"""
    warns = []
    best = result["best"]
    tr, va, te = best.get("train_metrics", {}), best.get("val_metrics", {}), {}
    te = result.get("test_metrics", {})
    def _g(m, k):
        v = m.get(k)
        return v if isinstance(v, (int, float)) else None
    tr_r, va_r, te_r = _g(tr, "total_return"), _g(va, "total_return"), _g(te, "total_return")
    if tr_r is not None and va_r is not None and tr_r > 0 and va_r < tr_r * 0.5:
        warns.append(f"训练段收益 {tr_r:.1%} 明显高于验证段 {va_r:.1%}：可能过拟合训练期行情。")
    if va_r is not None and te_r is not None and va_r > 0 and te_r < 0:
        warns.append(f"验证段为正（{va_r:.1%}）但测试段为负（{te_r:.1%}）：参数可能不具样本外稳健性。")
    if result["n_combinations"] >= 20:
        warns.append(f"共尝试 {result['n_combinations']} 组参数：选择偏差风险随组合数上升，"
                     f"验证段最优可能是运气（多重检验问题），请结合邻近参数稳定性与滚动验证判断。")
    if (best.get("n_trades") or 0) < 5:
        warns.append("最优参数交易次数不足 5 次：样本过少，指标不具备统计意义。")
    return warns


# ---------------- 存档 ----------------

def save_optimization_run(dataset_id: int, dataset_name: str, strategy_type: str,
                          param_space: dict, result: dict) -> int:
    """优化记录入库（新表由 db.py 迁移创建；重跑生成新记录）。"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with db.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO optimization_runs (created_at, dataset_id, dataset_name, "
            "strategy_type, param_grid_json, objective, constraints_json, results_json, "
            "best_params_json, test_metrics_json, n_combinations, "
            "test_used_for_selection, status, code_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts, dataset_id, dataset_name, strategy_type,
             json.dumps(param_space, ensure_ascii=False), result["objective"],
             json.dumps(result["constraints"], ensure_ascii=False),
             json.dumps(result["combinations"], ensure_ascii=False, default=float),
             json.dumps(result["best"]["params"], ensure_ascii=False),
             json.dumps(result["test_metrics"], ensure_ascii=False, default=float),
             result["n_combinations"], 0, "success", CODE_VERSION))
        return cur.lastrowid
