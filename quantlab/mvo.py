"""均值—方差组合优化（阶段 4D）：估计输入 → SLSQP 优化 → 有效前沿 → 组合引擎集成。

数学口径（全部显式声明，供页面与 PDF 引用）：
- 收益率：日简单收益 close.pct_change()；估计窗口 = 最近 window 个交易日（只用 ≤t 数据）。
- 预期收益 μ：窗口内日均收益 × 252（年化，历史均值法——对估计误差敏感，页面必须提示）。
- 协方差 Σ：窗口内日收益样本协方差 × 252（ddof=1）；可选 Ledoit-Wolf 收缩
  （sklearn.covariance，方法在结果中标记）改善数值稳定性。
- 目标：min wᵀΣw（最小方差）/ min wᵀΣw s.t. μᵀw≥target（目标收益）/ max μᵀw−λwᵀΣw。
- 约束：0 ≤ w_i ≤ cap（不做空不加杠杆）、Σw ≤ 1（其余为现金）、可选 Σw ≥ 下限。
- 可行性预检：target 超过 cap 约束下可达最大收益 → 明确报"不可行"，绝不静默改参数。

防过拟合承诺：任何时点权重只用该时点之前的历史估计；历史回测按再平衡日滚动重估
（portfolio.run_portfolio 的 weights="mvo" 模式），不存在"用全历史定权重再回放全历史"。
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MIN_WINDOW = 60


class MVOError(ValueError):
    pass


@dataclass
class MVOConfig:
    window: int = 120              # 估计窗口（交易日）
    max_weight: float = 0.6        # 单资产权重上限
    min_total_weight: float = 0.0  # 投资比例下限（0=允许全现金）
    shrinkage: str = "none"        # none / ledoit_wolf（协方差收缩）
    risk_free: float = 0.0         # 无风险利率（年化，仅展示用）

    def validate(self):
        if self.window < MIN_WINDOW:
            raise MVOError(f"估计窗口至少 {MIN_WINDOW} 个交易日（当前 {self.window}）。")
        if not (0 < self.max_weight <= 1):
            raise MVOError("单资产权重上限必须在 (0,1]。")
        if self.min_total_weight < 0 or self.min_total_weight > 1:
            raise MVOError("投资比例下限必须在 [0,1]。")
        if self.shrinkage not in ("none", "ledoit_wolf"):
            raise MVOError("shrinkage 必须是 none 或 ledoit_wolf。")


def estimate_inputs(closes: pd.DataFrame, cfg: MVOConfig) -> dict:
    """从收盘价矩阵估计 μ 与 Σ（只用传入数据——调用方负责截取 ≤t）。

    返回 {mu, cov, corr, ann_ret, ann_vol, window, n_days, shrinkage, assets}。
    """
    cfg.validate()
    rets = closes.pct_change(fill_method=None).dropna(how="all")
    if len(rets) < cfg.window:
        raise MVOError(f"有效收益样本 {len(rets)} < 估计窗口 {cfg.window}："
                       f"历史数据不足，无法在该时点估计参数。")
    rets = rets.iloc[-cfg.window:]
    assets = list(closes.columns)
    mu_daily = rets.mean()
    if cfg.shrinkage == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(rets.values)
        cov_daily = pd.DataFrame(lw.covariance_, index=assets, columns=assets)
        method = f"Ledoit-Wolf 收缩（强度 {lw.shrinkage_:.3f}）"
    else:
        cov_daily = rets.cov(ddof=1)
        method = "样本协方差（ddof=1）"
    mu = mu_daily * TRADING_DAYS
    cov = cov_daily * TRADING_DAYS
    if not np.all(np.isfinite(cov.values)) or not np.all(np.isfinite(mu.values)):
        raise MVOError("估计出的 μ 或 Σ 含非有限值（检查数据是否含 NaN/极端值）。")
    # 数值稳定性：协方差对角非负
    if (np.diag(cov.values) < -1e-12).any():
        raise MVOError("协方差矩阵出现负方差：输入数据异常，拒绝优化。")
    return {"mu": mu, "cov": cov, "corr": rets.corr(),
            "ann_ret": mu, "ann_vol": pd.Series(np.sqrt(np.diag(cov.values)), index=assets),
            "window": cfg.window, "n_days": len(rets), "shrinkage": method,
            "assets": assets}


def _max_achievable_return(mu: pd.Series, cap: float) -> float:
    """cap 约束下可达的最大组合收益（贪婪全配最高收益资产；Σw≤1）。"""
    return float(mu.max()) * cap if cap < 1 else float(mu.max())


def solve_mvo(inputs: dict, objective: str = "min_variance",
              target_return: float = None, risk_aversion: float = 2.0,
              cfg: MVOConfig = None) -> dict:
    """求解均值—方差权重。

    返回 {weights, port_return, port_vol, sharpe, objective, method, elapsed_ms,
          constraints_ok, note}。优化失败/不可行 → MVOError（含原因），不静默返回。
    """
    cfg = cfg or MVOConfig()
    cfg.validate()
    from scipy.optimize import minimize
    mu, cov = inputs["mu"], inputs["cov"]
    assets = list(mu.index)
    n = len(assets)
    if n < 2:
        raise MVOError("均值—方差优化至少需要 2 个资产。")
    w0 = np.full(n, min(1.0 / n, cfg.max_weight))
    w0 = w0 / w0.sum()
    bounds = [(0.0, cfg.max_weight)] * n
    if cfg.max_weight * n < 1 - 1e-9:
        raise MVOError(f"不可行：{n} 个资产 × 权重上限 {cfg.max_weight:.0%} < 100%，"
                       f"无法满仓配置。请提高上限或减少资产数。")
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]   # 预算约束：满仓（现金由组合层 cash_reserve 控制）
    if objective == "min_variance":
        fun = lambda w: float(w @ cov.values @ w)
    elif objective == "target_return":
        if target_return is None:
            raise MVOError("target_return 目标必须提供目标收益。")
        max_ret = _max_achievable_return(mu, cfg.max_weight)
        if target_return > max_ret + 1e-9:
            raise MVOError(f"目标收益 {target_return:.2%} 不可行：权重上限 {cfg.max_weight:.0%}"
                           f"约束下最高可达约 {max_ret:.2%}（全配最优资产）。请降低目标或放宽上限。")
        fun = lambda w: float(w @ cov.values @ w)
        cons.append({"type": "ineq", "fun": lambda w: float(mu.values @ w) - target_return})
    elif objective == "max_utility":
        if risk_aversion <= 0:
            raise MVOError("风险厌恶系数必须为正。")
        fun = lambda w: -(float(mu.values @ w) - 0.5 * risk_aversion * float(w @ cov.values @ w))
    else:
        raise MVOError(f"未知目标 {objective}：min_variance / target_return / max_utility。")

    import time
    t0 = time.perf_counter()
    res = minimize(fun, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    elapsed = (time.perf_counter() - t0) * 1000
    if not res.success:
        raise MVOError(f"优化器未收敛：{res.message}。可尝试 Ledoit-Wolf 收缩或放宽约束。")
    w = np.asarray(res.x, dtype=float)
    if not np.all(np.isfinite(w)):
        raise MVOError("优化输出含非有限权重，拒绝返回。")
    # 结果约束校验（预算约束在容差内归一修正后重校验）
    tol = 1e-6
    if (w < -tol).any() or (w > cfg.max_weight + 1e-4).any():
        w = np.clip(w, 0, cfg.max_weight)
    if abs(w.sum() - 1.0) > 1e-4:
        w = w / w.sum()
        w = np.minimum(w, cfg.max_weight)
        w = w / w.sum() if w.sum() > 0 else w
    if (w < -tol).any() or (w > cfg.max_weight + 1e-3).any() or abs(w.sum() - 1) > 1e-3:
        raise MVOError(f"优化结果无法满足约束（Σw={w.sum():.4f}），拒绝返回。")
    pr = float(mu.values @ w)
    pv = math.sqrt(max(float(w @ cov.values @ w), 0.0))
    return {"weights": dict(zip(assets, [float(x) for x in w])),
            "cash_weight": 0.0,
            "port_return": pr, "port_vol": pv,
            "sharpe": (pr - cfg.risk_free) / pv if pv > 0 else None,
            "objective": objective, "method": inputs["shrinkage"],
            "elapsed_ms": round(elapsed, 1), "constraints_ok": True,
            "window": inputs["window"], "n_days": inputs["n_days"]}


def efficient_frontier(inputs: dict, cfg: MVOConfig = None, points: int = 12) -> pd.DataFrame:
    """有效前沿：目标收益从最小方差组合收益到可达最大收益之间取点求最小方差。"""
    cfg = cfg or MVOConfig()
    lo = solve_mvo(inputs, "min_variance", cfg=cfg)["port_return"]
    hi = _max_achievable_return(mu=inputs["mu"], cap=cfg.max_weight) * 0.98
    if hi <= lo:
        raise MVOError("可达收益区间为空（资产预期收益过于接近），无法构建前沿。")
    rows = []
    for tgt in np.linspace(lo, hi, points):
        try:
            r = solve_mvo(inputs, "target_return", target_return=float(tgt), cfg=cfg)
            rows.append({"target_return": float(tgt), "port_vol": r["port_vol"],
                         "port_return": r["port_return"], "sharpe": r["sharpe"]})
        except MVOError:
            continue  # 个别端点不可行就跳过（如实留空，不虚构）
    if len(rows) < 2:
        raise MVOError("有效前沿可解点不足 2 个。")
    return pd.DataFrame(rows)


# ---------------- 与组合回测引擎集成 ----------------

def mvo_weights_at(closes_up_to_t: pd.DataFrame, cfg: MVOConfig,
                   objective: str, target_return: float = None,
                   risk_aversion: float = 2.0, max_weight_cap: float = 1.0):
    """再平衡时点的权重函数（供 portfolio.run_portfolio 的 'mvo' 模式调用）。

    历史不足窗口 → 回退等权并返回警告（与逆波动率的回退策略一致）。
    """
    from .portfolio import equal_weights
    try:
        inputs = estimate_inputs(closes_up_to_t, cfg)
        cap = min(cfg.max_weight, max_weight_cap)
        if cap != cfg.max_weight:
            cfg = MVOConfig(**{**cfg.__dict__, "max_weight": cap})
        r = solve_mvo(inputs, objective, target_return=target_return,
                      risk_aversion=risk_aversion, cfg=cfg)
        w = {k: v for k, v in r["weights"].items() if v > 1e-6}
        if not w:
            return equal_weights(list(closes_up_to_t.columns), max_weight_cap), \
                "优化结果全为现金，本次回退等权"
        return w, None
    except MVOError as e:
        return equal_weights(list(closes_up_to_t.columns), max_weight_cap), \
            f"该时点优化失败（{e}），本次再平衡回退等权"
