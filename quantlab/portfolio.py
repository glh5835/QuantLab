"""多资产投资组合引擎（阶段 3A）：创建 → 配置权重 → 日历再平衡 → 多资产回测。

关键正确性约定（任务书 A2/A3）：
- 权重只用"当时已可获得的历史"计算：逆波动率权重在再平衡日 t 只用 ≤t 的数据
  （再平衡之前的完整历史属于"过去"，使用它不构成未来数据泄漏）。
- 再平衡产生真实订单（先卖超配、后买低配），按 t+1 开盘价±滑点成交、扣手续费、
  股数取整至手；绝不直接改写持仓比例而无视成本。
- 交易日不一致：以各资产共同交易日（内连接）为组合日历；某资产当日无价格 →
  该资产当日不交易、净值按最后已知价计（记录 stale 标记），绝不伪造成交价。
- 资产历史不足（逆波动率窗口不够）→ 该次再平衡回退为等权并记录警告。
"""
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd

from . import db, __version__ as CODE_VERSION
from .config import BACKTESTS_DIR
from .storage import load_csv
from .backtest import _commission, _performance_metrics

TRADING_DAYS = 252


class PortfolioError(ValueError):
    pass


@dataclass
class PortfolioConfig:
    initial_cash: float = 100_000.0
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    slippage: float = 0.001
    lot_size: int = 100
    rebalance: str = "none"          # none / monthly / quarterly
    cash_reserve: float = 0.02       # 最低现金保留比例（约束买入）
    max_weight: float = 1.0          # 单资产权重上限（1=不限制）
    vol_window: int = 60             # 逆波动率权重的回看窗口

    def validate(self):
        if self.initial_cash <= 0:
            raise PortfolioError("初始资金必须为正。")
        if self.rebalance not in ("none", "monthly", "quarterly"):
            raise PortfolioError(f"再平衡频率必须是 none/monthly/quarterly，收到 {self.rebalance}")
        if not (0 <= self.cash_reserve < 1):
            raise PortfolioError("现金保留比例必须在 [0,1)。")
        if not (0 < self.max_weight <= 1):
            raise PortfolioError("单资产权重上限必须在 (0,1]。")
        if self.commission_rate < 0 or self.slippage < 0:
            raise PortfolioError("手续费率与滑点不能为负。")
        if self.vol_window < 5:
            raise PortfolioError("逆波动率窗口至少 5 日。")


# ---------------- 权重方法 ----------------

def normalize_weights(weights: dict[str, float], max_weight: float = 1.0) -> dict[str, float]:
    """校验并归一化权重：非负、至少一个为正、按上限裁剪后重新归一化。"""
    if not weights:
        raise PortfolioError("权重不能为空。")
    w = {}
    for k, v in weights.items():
        if v is None or (isinstance(v, float) and math.isnan(v)) or v < 0:
            raise PortfolioError(f"资产 {k} 的权重非法（{v}）：权重必须非负。")
        w[k] = float(v)
    total = sum(w.values())
    if total <= 0:
        raise PortfolioError("权重总和必须为正。")
    w = {k: v / total for k, v in w.items()}
    if max_weight < 1.0:  # 裁剪超限权重并重新归一化（迭代收敛）
        for _ in range(10):
            over = {k: v for k, v in w.items() if v > max_weight + 1e-9}
            if not over:
                break
            excess = sum(v - max_weight for v in over.values())
            w = {k: (max_weight if v > max_weight else v) for k, v in w.items()}
            free = sum(v for k, v in w.items() if k not in over)
            if free <= 0:
                break
            w = {k: (v + excess * v / free) if k not in over else v for k, v in w.items()}
        total = sum(w.values())
        w = {k: v / total for k, v in w.items()}
    return w


def equal_weights(assets: list[str], max_weight: float = 1.0) -> dict[str, float]:
    if not assets:
        raise PortfolioError("资产列表不能为空。")
    if len(assets) * max_weight < 1 - 1e-9:  # n×上限 都到不了 100% → 无解
        raise PortfolioError(f"{len(assets)} 个资产无法在单资产上限 {max_weight:.0%} 内配置。")
    return normalize_weights({a: 1.0 for a in assets}, max_weight)


def inverse_vol_weights(closes_up_to_t: pd.DataFrame, window: int,
                        max_weight: float = 1.0) -> tuple[dict[str, float], str | None]:
    """逆波动率权重：只用 ≤t 数据。返回 (权重, 警告或None)。

    各资产用自己截至 t 的最近 window 日收益率标准差；历史不足的资产会导致
    整体回退为等权（返回警告说明）。
    """
    vols = {}
    for col in closes_up_to_t.columns:
        s = closes_up_to_t[col].dropna()
        if len(s) < window + 1:
            return (equal_weights(list(closes_up_to_t.columns), max_weight),
                    f"资产 {col} 历史不足 {window + 1} 日，本次再平衡回退为等权。")
        ret = s.pct_change(fill_method=None).dropna().iloc[-window:]
        sd = ret.std()
        if sd is None or sd == 0 or math.isnan(sd):
            return (equal_weights(list(closes_up_to_t.columns), max_weight),
                    f"资产 {col} 在窗口内无波动（可能长期停牌/一字），本次回退为等权。")
        vols[col] = float(sd)
    inv = {k: 1.0 / v for k, v in vols.items()}
    return normalize_weights(inv, max_weight), None


# ---------------- 组合回测引擎 ----------------

def _is_rebalance_day(prev_t, t, freq: str) -> bool:
    """日历再平衡：频率周期的第一个交易日（比较上一交易日与当前日的年月/季度）。

    freq='none' 表示只做期初建仓、此后不再平衡（prev_t is None 的首个交易日）。
    """
    if freq == "none":
        return prev_t is None
    if prev_t is None:
        return True  # 期初建仓视为一次"再平衡"
    if freq == "monthly":
        return (t.year, t.month) != (prev_t.year, prev_t.month)
    if freq == "quarterly":
        return (t.year, (t.month - 1) // 3) != (prev_t.year, (prev_t.month - 1) // 3)
    return False


def _align_prices(price_map: dict[str, pd.DataFrame], start, end) -> pd.DataFrame:
    """多资产收盘价对齐：并集日历（保留各资产自身交易日；缺失日为 NaN，由引擎按
    最后已知价估值并禁止交易）。各自升序去重后截取 [start,end]。"""
    closes = {}
    for name, df in price_map.items():
        s = df["close"].dropna().sort_index()
        s = s[~s.index.duplicated(keep="last")]
        closes[name] = s
    aligned = pd.DataFrame(closes).sort_index()  # DataFrame 对齐 = 索引并集
    return aligned.loc[start:end] if (start or end) else aligned


def run_portfolio(price_map: dict[str, pd.DataFrame], weights: dict[str, float] | str,
                  cfg: PortfolioConfig, start=None, end=None,
                  mvo_options: dict = None) -> dict:
    """执行多资产组合回测。

    weights 三种形式：
    - 显式权重字典（equal/user 方法）
    - "inverse_vol"：每次再平衡用 ≤t 历史重算逆波动率权重
    - "mvo"：每次再平衡用 ≤t 历史滚动重估均值—方差权重（阶段 4D）。
      mvo_options: {"objective","target_return","risk_aversion","window","max_weight",
                    "shrinkage"}（详见 quantlab.mvo）。
    price_map: {资产名: 单资产 OHLCV}（含回测区间前的历史，供估计窗口回看，均为过去数据）。
    返回：metrics / trades / rebalances / weights_history / daily / warnings。
    """
    cfg.validate()
    if not price_map or len(price_map) == 0:
        raise PortfolioError("至少需要一个资产。")
    assets = list(price_map.keys())
    if isinstance(weights, str) and weights not in ("inverse_vol", "mvo"):
        raise PortfolioError("权重方法必须是权重字典、'inverse_vol' 或 'mvo'。")
    mvo_state = {"options": dict(mvo_options or {}), "cache": {}} if weights == "mvo" else None
    fixed_weights = None if isinstance(weights, str) else normalize_weights(weights, cfg.max_weight)
    if fixed_weights is not None and set(fixed_weights) != set(assets):
        raise PortfolioError("权重字典的资产必须与资产列表一致。")

    closes = _align_prices(price_map, start, end)
    if len(closes) < 3:
        raise PortfolioError("共同交易日不足 3 天，无法回测（交易日历不一致或区间过短）。")
    opens = pd.DataFrame({k: df["open"].reindex(closes.index) for k, df in price_map.items()})

    cash = cfg.initial_cash
    shares = {a: 0 for a in assets}
    avg_cost = {a: 0.0 for a in assets}
    realized = {a: 0.0 for a in assets}
    total_fees = total_slip = traded_value = 0.0
    trades, rebalances, weights_history, daily, warnings = [], [], [], [], []
    pending = None            # 再平衡订单计划 {signal_time, target_weights}
    last_known_price = {a: float("nan") for a in assets}  # 未知价=NaN（不用未来数据初始化）

    def _val(a) -> float:
        """资产市值：未知价时用持仓成本估值（shares=0 则为 0），绝不返回 NaN。"""
        px = last_known_price[a]
        if px is None or math.isnan(px):
            return avg_cost[a] * shares[a] if shares[a] > 0 else 0.0
        return shares[a] * px

    dates = closes.index
    for i, t in enumerate(dates):
        row_c = closes.loc[t]
        row_o = opens.loc[t] if not opens.empty else row_c.copy()

        # ---------- 1) 执行昨日计划的再平衡订单（t 开盘价） ----------
        if pending is not None:
            tw = pending["target_weights"]
            # 1a) 估值（用信号日收盘价——决策时点已知信息）
            investable = (cash + sum(_val(a) for a in assets)) * (1 - cfg.cash_reserve)
            # 1b) 先卖超配
            sells, buys = [], []
            for a in assets:
                px_est = last_known_price[a]
                if px_est is None or math.isnan(px_est) or px_est <= 0:
                    warnings.append(f"{t.date()} 资产 {a} 无可用价格，本次再平衡跳过该资产（不伪造成交）。")
                    continue
                exec_px = row_o.get(a, row_c.get(a))
                if exec_px is None or (isinstance(exec_px, float) and math.isnan(exec_px)) or exec_px <= 0:
                    exec_px = row_c.get(a)
                if exec_px is None or (isinstance(exec_px, float) and math.isnan(exec_px)) or exec_px <= 0:
                    warnings.append(f"{t.date()} 资产 {a} 开盘与收盘均无价，跳过。")
                    continue
                target_val = tw.get(a, 0.0) * investable
                lot_val = exec_px * cfg.lot_size
                if lot_val <= 0:
                    continue
                tgt_sh = int(target_val / lot_val) * cfg.lot_size if cfg.lot_size > 1 \
                    else int(target_val / exec_px)
                delta = tgt_sh - shares[a]
                if delta <= -cfg.lot_size or (cfg.lot_size == 1 and delta <= -1):
                    sells.append((a, -delta, float(exec_px)))
                elif delta >= cfg.lot_size or (cfg.lot_size == 1 and delta >= 1):
                    buys.append((a, delta, float(exec_px)))
            # 卖出先行（回笼现金）
            for a, qty, base_px in sorted(sells, key=lambda x: -x[1]):
                px = base_px * (1 - cfg.slippage)
                value = qty * px
                fee = _commission(value, cfg)
                pnl = value - fee - avg_cost[a] * qty
                realized[a] += pnl
                cash += value - fee
                shares[a] -= qty
                total_fees += fee
                total_slip += qty * (base_px - px)
                traded_value += value
                trades.append({"signal_time": str(pending["signal_time"].date()),
                               "exec_time": str(t.date()), "asset": a, "action": "sell",
                               "shares": int(qty), "base_price": base_px, "exec_price": px,
                               "value": float(value), "fee": float(fee), "pnl": float(pnl),
                               "cash_after": float(cash), "shares_after": int(shares[a]),
                               "reason": pending["reason"]})
                if shares[a] == 0:
                    avg_cost[a] = 0.0
            # 买入（按目标权重从大到小，受现金约束）
            for a, qty, base_px in sorted(buys, key=lambda x: -x[1]):
                px = base_px * (1 + cfg.slippage)
                while qty >= cfg.lot_size or (cfg.lot_size == 1 and qty >= 1):
                    value = qty * px
                    fee = _commission(value, cfg)
                    if value + fee <= cash:
                        total_cost = avg_cost[a] * shares[a] + value
                        shares[a] += qty
                        avg_cost[a] = total_cost / shares[a]
                        cash -= value + fee
                        total_fees += fee
                        total_slip += qty * (px - base_px)
                        traded_value += value
                        trades.append({"signal_time": str(pending["signal_time"].date()),
                                       "exec_time": str(t.date()), "asset": a, "action": "buy",
                                       "shares": int(qty), "base_price": base_px, "exec_price": px,
                                       "value": float(value), "fee": float(fee), "pnl": None,
                                       "cash_after": float(cash), "shares_after": int(shares[a]),
                                       "reason": pending["reason"]})
                        break
                    qty -= cfg.lot_size if cfg.lot_size > 1 else 1  # 资金不足 → 减手重试
            rebalances.append({"date": str(t.date()), "weights": tw,
                               "reason": pending["reason"]})
            pending = None

        # ---------- 2) 收盘：更新最后已知价与净值 ----------
        for a in assets:
            c = row_c.get(a)
            if c is not None and not (isinstance(c, float) and math.isnan(c)):
                last_known_price[a] = float(c)
        equity = cash + sum(_val(a) for a in assets)
        stale = [a for a in assets if shares[a] > 0 and
                 (row_c.get(a) is None or (isinstance(row_c.get(a), float) and math.isnan(row_c.get(a))))]
        daily.append({"date": str(t.date()), "cash": float(cash),
                      "equity": float(equity),
                      "positions_value": float(sum(_val(a) for a in assets)),
                      "stale_assets": stale,
                      "weights_effective": {a: (_val(a) / equity
                                                if equity > 0 else 0.0) for a in assets}})

        # ---------- 3) 收盘后：判断是否需要再平衡（只用 ≤t 信息） ----------
        prev_t = dates[i - 1] if i > 0 else None
        if i + 1 < len(dates) and _is_rebalance_day(prev_t, t, cfg.rebalance):
            if fixed_weights is not None:
                tw, warn = dict(fixed_weights), None
            elif weights == "inverse_vol":
                hist = closes.loc[:t]
                tw, warn = inverse_vol_weights(hist, cfg.vol_window, cfg.max_weight)
            else:  # mvo：每次再平衡滚动重估（≤t 数据），权重在执行前计算完成
                from .mvo import mvo_weights_at, MVOConfig
                opts = mvo_state["options"]
                mcfg = MVOConfig(window=int(opts.get("window", 120)),
                                 max_weight=float(opts.get("max_weight", 0.6)),
                                 shrinkage=opts.get("shrinkage", "none"))
                hist = closes.loc[:t]
                tw, warn = mvo_weights_at(hist, mcfg,
                                          objective=opts.get("objective", "min_variance"),
                                          target_return=opts.get("target_return"),
                                          risk_aversion=float(opts.get("risk_aversion", 2.0)),
                                          max_weight_cap=cfg.max_weight)
            if warn:
                warnings.append(f"{t.date()} {warn}")
            weights_history.append({"signal_time": str(t.date()), "weights": tw,
                                    "method": ("fixed" if fixed_weights is not None
                                               else ("inverse_vol" if weights == "inverse_vol"
                                                     else "mvo"))})
            pending = {"signal_time": t, "target_weights": tw,
                       "reason": f"{'期初建仓' if i == 0 else cfg.rebalance + ' 再平衡'}（t 收盘决策，t+1 开盘成交）"}

    eq = pd.Series([d["equity"] for d in daily],
                   index=pd.to_datetime([d["date"] for d in daily]))
    metrics = _performance_metrics(eq, cfg.initial_cash)
    n_buys = sum(1 for tr in trades if tr["action"] == "buy")
    n_sells = sum(1 for tr in trades if tr["action"] == "sell")
    years = len(eq) / TRADING_DAYS
    final_px = {}
    for a in assets:
        px = last_known_price[a]
        final_px[a] = float(px) if (px is not None and not math.isnan(px)) else float(avg_cost[a] or 0.0)
    metrics.update({
        "n_rebalances": len(rebalances), "n_trades": len(trades),
        "n_buys": n_buys, "n_sells": n_sells,
        "total_fees": float(total_fees), "total_slippage_cost": float(total_slip),
        "traded_value": float(traded_value),
        "turnover_annual": float(traded_value / max(eq.mean(), 1e-9) / years) if years > 0 else None,
        "final_cash": float(cash),
        "final_positions": {a: {"shares": int(shares[a]), "avg_cost": float(avg_cost[a]),
                                "last_price": final_px[a],
                                "value": float(shares[a] * final_px[a]),
                                "realized_pnl": float(realized[a])} for a in assets},
        "contribution_pnl": {a: float(realized[a] + (shares[a] * (final_px[a] - avg_cost[a])
                                                     if shares[a] > 0 else 0.0)) for a in assets},
    })
    return {"metrics": metrics, "trades": trades, "rebalances": rebalances,
            "weights_history": weights_history, "daily": daily, "warnings": warnings}


# ---------------- 存档 ----------------

def save_portfolio_run(name: str, dataset_ids: dict[str, int], weights_desc: str,
                       cfg: PortfolioConfig, result: dict, start, end,
                       description: str = "") -> int:
    """组合回测存档：明细文件写 data/backtests/port_<id>/，元数据入 portfolios 表。"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bt_id = db.save_backtest_record(
        name=name, created_at=ts, strategy_type="portfolio", strategy_params=weights_desc,
        dataset_id=list(dataset_ids.values())[0],
        dataset_name="+".join(dataset_ids.keys()), dataset_version=1,
        experiment_id=None, start_date=str(start or ""), end_date=str(end or ""),
        config_json=json.dumps(asdict(cfg), ensure_ascii=False),
        metrics_json=json.dumps(result["metrics"], ensure_ascii=False),
        status="success", code_version=CODE_VERSION)
    pdir = BACKTESTS_DIR / f"port_{bt_id}"
    pdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["trades"]).to_csv(pdir / "trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(result["daily"]).drop(columns=["weights_effective", "stale_assets"], errors="ignore") \
        .to_csv(pdir / "daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(result["weights_history"]).to_csv(pdir / "weights.csv", index=False, encoding="utf-8-sig")
    db.update_backtest(bt_id, trades_path=str(pdir / "trades.csv"),
                       daily_path=str(pdir / "daily.csv"))
    return bt_id
