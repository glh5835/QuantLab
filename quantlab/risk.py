"""高级风险分析（阶段 3B）：风险指标 / 回撤区间 / 历史分段 / 压力测试 / 风险预警。

设计原则：
- 每个指标附带公式与适用条件；样本不足、零波动、分母无效时返回
  {"value": None, "reason": "..."}，绝不产生虚假数值。
- 回撤区间（episodes）区分"已恢复"与"截至回测结束仍未恢复"（recovery=None），
  缺少完整恢复数据时不伪造恢复日期。
- 压力测试区分两类情景：真实参数变化（费用/滑点/延迟/仓位/频率/区间，重跑真实
  回测）与假设情景（价格冲击，人为设定，结果必须标记 hypothetical）。
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _nc(reason: str) -> dict:
    """统一的"不可计算"状态。"""
    return {"value": None, "reason": reason}


# ---------------- 风险指标（公式与适用条件见 docstring 与页面说明） ----------------

def risk_metrics(equity: pd.Series, initial: float,
                 benchmark: pd.Series = None,
                 total_fees: float = None, n_trades: int = None) -> dict:
    """从逐日净值序列计算完整风险指标集。

    公式（252 交易日年化，rf=0）：
    - 年化波动率 = 日收益标准差 × √252
    - 夏普 = 日均收益 / 日波动 × √252
    - 下行波动率 = 负日收益的下行偏差（MAR=0）× √252；
      索提诺 = 日均收益 / 下行偏差 × √252（无负收益时不可计算，如实说明）
    - 卡玛 = 年化收益 / |最大回撤|（无回撤时不可计算）
    - Beta = cov(策略, 基准) / var(基准)；超额收益 = 策略总收益 − 基准总收益
    - 成本占比 = 总费用 / 初始资金
    """
    out = {}
    if not isinstance(equity, pd.Series) or len(equity) < 2:
        return {"total_return": _nc("净值序列不足 2 个交易日")}
    ret = equity.pct_change(fill_method=None).dropna()
    n = len(ret)
    total = equity.iloc[-1] / initial - 1
    years = len(equity) / TRADING_DAYS
    out["total_return"] = {"value": float(total), "reason": ""}
    out["cagr"] = ({"value": float((1 + total) ** (1 / years) - 1), "reason": ""}
                   if years > 0 else _nc("时间跨度不足"))
    sd = float(ret.std()) if n >= 2 else None
    out["ann_vol"] = ({"value": float(sd * np.sqrt(TRADING_DAYS)), "reason": ""}
                      if sd is not None else _nc("样本不足"))
    if sd is not None and sd > 0:
        out["sharpe"] = {"value": float(ret.mean() / sd * np.sqrt(TRADING_DAYS)), "reason": ""}
    else:
        out["sharpe"] = _nc("收益无波动（标准差为 0）或样本不足，夏普比率不可计算")
    neg = ret[ret < 0]
    if len(neg) >= 2:
        downside = float(np.sqrt((neg ** 2).sum() / n))  # 全样本口径的下行偏差
        out["downside_vol"] = {"value": float(downside * np.sqrt(TRADING_DAYS)), "reason": ""}
        if downside > 0:
            out["sortino"] = {"value": float(ret.mean() / downside * np.sqrt(TRADING_DAYS)),
                              "reason": ""}
        else:
            out["sortino"] = _nc("无下行波动（亏损为零），索提诺比率不可计算")
    else:
        out["downside_vol"] = _nc("无负收益日，下行波动不可计算")
        out["sortino"] = _nc("无负收益日，索提诺比率不可计算")
    peak = equity.cummax()
    dd = equity / peak - 1
    mdd = float(dd.min())
    out["mdd"] = {"value": mdd, "reason": ""}
    if mdd < 0:
        out["calmar"] = ({"value": float(out["cagr"]["value"] / abs(mdd)), "reason": ""}
                         if out["cagr"]["value"] is not None else _nc("年化收益不可计算"))
    else:
        out["calmar"] = _nc("无回撤（净值单调上行），卡玛比率不可计算")
    if benchmark is not None and len(benchmark) >= 2:
        bench_ret = benchmark.pct_change(fill_method=None).dropna()
        joined = pd.concat([ret, bench_ret], axis=1, join="inner").dropna()
        if len(joined) >= 10 and joined.shape[1] == 2:
            cov = float(np.cov(joined.iloc[:, 0], joined.iloc[:, 1])[0][1])  # ddof=1
            var_b = float(np.var(joined.iloc[:, 1], ddof=1))  # 与 cov 同口径（样本方差）
            out["beta"] = ({"value": cov / var_b, "reason": ""} if var_b > 0
                           else _nc("基准收益无波动，Beta 不可计算"))
            out["excess_return"] = {"value": float(total - (benchmark.iloc[-1] /
                                                            benchmark.iloc[0] - 1)),
                                    "reason": ""}
        else:
            out["beta"] = _nc("策略与基准重叠样本不足 10 天")
            out["excess_return"] = _nc("重叠样本不足")
    else:
        out["beta"] = _nc("未提供基准")
        out["excess_return"] = _nc("未提供基准")
    if total_fees is not None and initial > 0:
        out["cost_ratio"] = {"value": float(total_fees / initial), "reason": ""}
    if n_trades is not None:
        out["n_trades"] = {"value": int(n_trades), "reason": ""}
    return out


# ---------------- 回撤区间分析 ----------------

def drawdown_analysis(equity: pd.Series) -> dict:
    """完整回撤区间：每个回撤事件的峰值日/谷底日/恢复日，区分已恢复与未恢复。"""
    if not isinstance(equity, pd.Series) or len(equity) < 2:
        return {"episodes": [], "max": None, "reason": "净值序列不足"}
    peak = equity.cummax()
    dd = equity / peak - 1
    episodes = []
    in_dd = False
    start = trough_i = None
    peak_val = None
    for t, v in dd.items():
        if not in_dd and v < -1e-12:
            in_dd, start, trough_i = True, t, t
            peak_val = peak.loc[t]
        elif in_dd:
            if v < dd.loc[trough_i]:
                trough_i = t
            if equity.loc[t] >= peak_val - 1e-12:  # 收复前高 → 回撤结束
                episodes.append({
                    "peak_date": None, "start_date": str(start.date()),
                    "trough_date": str(trough_i.date()),
                    "mdd": float(dd.loc[trough_i]),
                    "recovery_date": str(t.date()), "recovered": True})
                in_dd = False
    if in_dd:  # 截至序列结束仍未恢复 → recovery=None（不伪造）
        episodes.append({
            "peak_date": None, "start_date": str(start.date()),
            "trough_date": str(trough_i.date()),
            "mdd": float(dd.loc[trough_i]),
            "recovery_date": None, "recovered": False,
            "note": "截至数据结束仍未恢复，无恢复日期"})
    # 补每个事件的峰值日（起点前最后一个创新高日）
    for ep in episodes:
        s = pd.Timestamp(ep["start_date"])
        prior = equity.loc[:s]
        ep["peak_date"] = str(prior.idxmax().date()) if len(prior) else None
    mdd_idx = dd.idxmin()
    return {"episodes": episodes,
            "max": {"mdd": float(dd.min()),
                    "peak_date": str(equity.loc[:mdd_idx].idxmax().date()),
                    "trough_date": str(mdd_idx.date())},
            "drawdown_series": dd}


# ---------------- 历史区间分析 ----------------

def period_analysis(equity: pd.Series, freq: str = "Y") -> list[dict]:
    """按年(Y/YE)/季度(Q/QE)切分：各区间的收益、年化波动、最大回撤、样本天数。"""
    if freq not in ("Y", "YE", "Q", "QE"):
        raise ValueError("freq 必须是 'Y'/'YE' 或 'Q'/'QE'")
    # Pandas 3.0 移除了 Y、Q 两个旧偏移别名；对外兼容旧参数。
    pandas_freq = {"Y": "YE", "Q": "QE"}.get(freq, freq)
    groups = equity.groupby(pd.Grouper(freq=pandas_freq))
    out = []
    for period, seg in groups:
        if len(seg) < 2:
            continue
        ret = seg.pct_change(fill_method=None).dropna()
        sd = float(ret.std()) if len(ret) >= 2 else None
        mdd = float((seg / seg.cummax() - 1).min())
        out.append({
            "period": str(period.year if freq in ("Y", "YE") else f"{period.year}Q{(period.month-1)//3+1}"),
            "n_days": int(len(seg)),
            "return": float(seg.iloc[-1] / seg.iloc[0] - 1),
            "ann_vol": float(sd * np.sqrt(TRADING_DAYS)) if sd is not None else None,
            "mdd": mdd,
        })
    return out


# ---------------- 压力测试 ----------------

class StressScenario:
    def __init__(self, name, desc, cfg_overrides=None, df_transform=None,
                 date_range=None, hypothetical=False):
        self.name, self.desc = name, desc
        self.cfg_overrides = cfg_overrides or {}
        self.df_transform = df_transform      # df -> df（假设情景的价格冲击等）
        self.date_range = date_range          # (start, end) 独立区间回测
        self.hypothetical = hypothetical      # True=人为假设情景，输出必须标记


def default_scenarios() -> list[StressScenario]:
    """任务书 B4 的 1~4 号情景（真实参数变化，重跑真实回测）。"""
    return [
        StressScenario("手续费×3", "手续费率与最低费升至 3 倍",
                       cfg_overrides={"commission_rate": None}),  # 倍率在 runner 内计算
        StressScenario("滑点×3", "单边滑点升至 3 倍",
                       cfg_overrides={"slippage": None}),
        StressScenario("延迟成交 t+2", "信号延迟一个额外交易日执行",
                       cfg_overrides={"exec_delay": 2}),
        StressScenario("仓位减半", "目标仓位降至一半",
                       cfg_overrides={"target_fraction": None}),
    ]


def run_stress(runner, base_cfg_dict: dict, scenarios: list[StressScenario],
               price_shock: dict = None) -> list[dict]:
    """执行压力测试。

    runner(cfg_overrides: dict, df_transform=None, date_range=None) → metrics dict。
    price_shock: {"start": "YYYY-MM-DD", "end": ..., "shock": -0.2} → 人为假设情景。
    返回每情景：{name, hypothetical, base, stress, diff_total_return}。
    """
    results = []
    for sc in scenarios:
        ov = dict(sc.cfg_overrides)
        # 倍率情景：None 占位 → 用基准值×3 计算
        if "commission_rate" in ov and ov["commission_rate"] is None:
            ov["commission_rate"] = base_cfg_dict.get("commission_rate", 0) * 3
            if base_cfg_dict.get("min_commission"):
                ov["min_commission"] = base_cfg_dict["min_commission"] * 3
        if "slippage" in ov and ov["slippage"] is None:
            ov["slippage"] = base_cfg_dict.get("slippage", 0) * 3
        if "target_fraction" in ov and ov["target_fraction"] is None:
            ov["target_fraction"] = max(base_cfg_dict.get("target_fraction", 1.0) / 2, 0.05)
        m = runner(ov, sc.df_transform, sc.date_range)
        results.append({"name": sc.name, "desc": sc.desc,
                        "hypothetical": sc.hypothetical, "stress": m})
    if price_shock:
        def _shock(df):
            start, end = pd.Timestamp(price_shock["start"]), pd.Timestamp(price_shock["end"])
            k = 1.0 + float(price_shock["shock"])
            df = df.copy()
            mask = (df.index >= start) & (df.index <= end)
            for col in ("open", "high", "low", "close"):
                if col in df.columns:
                    df.loc[mask, col] = df.loc[mask, col] * k
            return df
        m = runner({}, _shock, None)
        results.append({"name": f"价格冲击 {price_shock['shock']:+.0%}"
                                 f"（{price_shock['start']}~{price_shock['end']}）",
                        "desc": "人为设定的假设情景：对区间内价格整体施加冲击，"
                                "并非真实历史事件", "hypothetical": True, "stress": m})
    return results


# ---------------- 风险预警（只预警，不改变交易行为） ----------------

def risk_warnings(equity: pd.Series, dd_threshold: float = 0.20,
                  vol_threshold: float = None, vol_window: int = 20) -> list[dict]:
    """逐日检查回撤/波动率是否越过阈值。输出 {date, metric, value, threshold}。

    本函数只生成预警记录，绝不修改策略行为（与强制平仓严格区分）。
    """
    out = []
    if not isinstance(equity, pd.Series) or len(equity) < 2:
        return out
    dd = equity / equity.cummax() - 1
    dd_warned = False
    for t, v in dd.items():
        if not dd_warned and v < -dd_threshold:
            out.append({"date": str(t.date()), "metric": "最大回撤",
                        "value": float(v), "threshold": -dd_threshold})
            dd_warned = True  # 每次跌破只报一次（恢复后重置见下）
        if dd_warned and v >= -1e-12:
            dd_warned = False  # 创新高后重置，允许再次预警
    if vol_threshold is not None:
        ret = equity.pct_change(fill_method=None)
        rv = ret.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
        v_warned = False
        for t, v in rv.items():
            if pd.isna(v):
                continue
            if not v_warned and v > vol_threshold:
                out.append({"date": str(t.date()), "metric": "滚动年化波动率",
                            "value": float(v), "threshold": vol_threshold})
                v_warned = True
            if v_warned and v <= vol_threshold * 0.8:  # 回落后重置
                v_warned = False
    return out
