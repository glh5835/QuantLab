"""回测引擎：逐日步进、t 收盘出信号 / t+1 开盘成交、现金账户、费用滑点、风控与绩效。

与任务书交易时序约定的对应实现：
- 交易日 t 收盘后：引擎把"截至 t（含）"的行情交给策略 → 得到目标仓位（信号生成时间 = t）。
- 该订单最早在 t+1 执行：默认用 t+1 开盘价成交（基础价），叠加滑点得实际成交价；
  无开盘价时退化为 t+1 收盘成交并在订单中标记 fill_mode=close（简化假设，报告会注明）。
  严禁用 t 的收盘信号以 t 的收盘价成交。
- t+1 无可用价格（停牌/数据缺失/回测末日）→ 订单记为未成交，绝不伪造成交价。

资金规则：现金账户，无融资无做空；买入受现金、目标仓位、单笔上限与最低现金保留约束，
股数向下取整至最小交易单位；卖出不超过持仓；手续费从现金扣除；每笔成交后即时更新
持仓成本（移动平均成本法）、现金与净值。

风控规则：回撤超过阈值只"预警"（记录日期），不强制平仓；可选"触发后停止新买入"。
"""
import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
import pandas as pd

from . import db, __version__ as CODE_VERSION
from .config import BACKTESTS_DIR
from .indicators import prepare_ohlcv
from .strategies import BuyHoldStrategy, StrategyError
from .storage import load_csv

TRADING_DAYS = 252


class BacktestError(ValueError):
    pass


@dataclass
class BacktestConfig:
    initial_cash: float = 100_000.0
    commission_rate: float = 0.0003   # 按成交金额比例的手续费
    min_commission: float = 5.0       # 单笔最低手续费（元），0 表示不设
    slippage: float = 0.001           # 单边滑点比例（买卖都按不利方向调整）
    lot_size: int = 100               # 最小交易单位（股）
    target_fraction: float = 0.95     # 持仓状态下的目标仓位（占账户净值）
    max_position_fraction: float = 1.0  # 单资产最大持仓比例（上限约束）
    max_order_value: float = 0.0      # 单笔订单金额上限（0 = 不限制）
    min_cash_reserve: float = 0.0     # 最低现金保留比例（占总净值）
    drawdown_warn: float = 0.20       # 回撤预警阈值：超过只预警，不强制平仓
    stop_buy_on_drawdown: bool = False  # 预警触发后是否停止新买入（默认 False）

    def validate(self):
        if self.initial_cash <= 0:
            raise BacktestError("初始资金必须为正。")
        if self.commission_rate < 0 or self.slippage < 0:
            raise BacktestError("手续费率与滑点不能为负。")
        if self.lot_size < 1:
            raise BacktestError("最小交易单位必须 ≥ 1。")
        if not (0 < self.target_fraction <= 1) or not (0 < self.max_position_fraction <= 1):
            raise BacktestError("仓位比例必须在 (0, 1] 区间。")
        if self.drawdown_warn <= 0:
            raise BacktestError("回撤预警阈值必须为正。")
        if self.stop_buy_on_drawdown and self.drawdown_warn >= 1:
            raise BacktestError("回撤阈值应在 (0,1) 区间。")


def _commission(value: float, cfg: BacktestConfig) -> float:
    fee = value * cfg.commission_rate
    if cfg.min_commission > 0:
        fee = max(fee, cfg.min_commission)
    return fee


def _run_engine(df: pd.DataFrame, strategy, cfg: BacktestConfig) -> dict:
    """引擎主体（不落库，便于基准复用与单元测试）。"""
    data = prepare_ohlcv(df)
    if len(data) < 3:
        raise BacktestError("行情数据不足 3 个交易日，无法回测。")

    cash = cfg.initial_cash
    shares = 0
    avg_cost = 0.0            # 移动平均持仓成本
    realized_pnl = 0.0
    total_fees = 0.0
    total_slip_cost = 0.0
    traded_value = 0.0
    trades, daily, warnings = [], [], []
    round_trips = []          # 每次清仓/减仓实现的盈亏（胜率与盈亏比用）
    pending = None            # {signal_time, target, reason}
    equity_peak = -math.inf
    drawdown_warned = False
    last_target = None

    dates = data.index
    for i, t in enumerate(dates):
        row = data.loc[t]
        # ---------- 1) 开盘阶段：执行昨日信号 ----------
        if pending is not None:
            base = row.get("open")
            fill_mode = "open"
            if base is None or (isinstance(base, float) and math.isnan(base)) or base <= 0:
                base = row.get("close")
                fill_mode = "close"  # 简化假设：无开盘价时按当日收盘成交（订单中标记）
            if base is None or (isinstance(base, float) and math.isnan(base)) or base <= 0:
                trades.append({  # t+1 无任何可用价格 → 未成交，不伪造
                    "signal_time": str(pending["signal_time"].date()), "exec_time": str(t.date()),
                    "action": "unfilled", "shares": 0, "base_price": None, "exec_price": None,
                    "value": 0.0, "fee": 0.0, "slippage_cost": 0.0,
                    "cash_after": cash, "shares_after": shares,
                    "reason": f"订单未成交：{pending['reason']}；执行日无可用价格（停牌/数据缺失）"})
                warnings.append(f"{t.date()} 订单未成交（执行日无价格）")
                pending = None
            else:
                target = pending["target"]
                # 用"信号日收盘净值"估算目标股数（决策时点已知信息；NaN 时退化为成本法估值）
                sig_close = data.loc[pending["signal_time"], "close"]
                if isinstance(sig_close, float) and math.isnan(sig_close):
                    sig_close = avg_cost if avg_cost > 0 else 0.0
                sig_equity = cash + shares * sig_close
                cap_fraction = min(cfg.target_fraction, cfg.max_position_fraction)
                if target > 0 and (shares == 0 or shares * base < target * cap_fraction * sig_equity * 0.98):
                    # ---- 买入（建仓或补足）----
                    blocked = []
                    if drawdown_warned and cfg.stop_buy_on_drawdown:
                        blocked.append("回撤预警已触发且配置了停止新买入")
                    if blocked:
                        trades.append({"signal_time": str(pending["signal_time"].date()),
                                       "exec_time": str(t.date()), "action": "buy_blocked",
                                       "shares": 0, "base_price": float(base), "exec_price": None,
                                       "value": 0.0, "fee": 0.0, "slippage_cost": 0.0,
                                       "cash_after": cash, "shares_after": shares,
                                       "reason": "；".join(blocked)})
                        warnings.append(f"{t.date()} 买入被风控拦截（{'；'.join(blocked)}）")
                    else:
                        exec_price = base * (1 + cfg.slippage)
                        desired_value = cap_fraction * sig_equity
                        if cfg.max_order_value > 0:
                            desired_value = min(desired_value, cfg.max_order_value)
                        affordable = cash * (1 - cfg.min_cash_reserve) - _commission(
                            min(desired_value, cash), cfg)  # 手续费预留的保守估计
                        invest = max(min(desired_value, affordable), 0.0)
                        if cfg.lot_size > 1:
                            buy_shares = int(invest / (exec_price * cfg.lot_size)) * cfg.lot_size
                            too_small = buy_shares < cfg.lot_size
                        else:
                            buy_shares = int(invest / exec_price)
                            too_small = buy_shares < 1
                        if too_small:
                            trades.append({"signal_time": str(pending["signal_time"].date()),
                                           "exec_time": str(t.date()), "action": "buy_rejected",
                                           "shares": 0, "base_price": float(base),
                                           "exec_price": float(exec_price), "value": 0.0,
                                           "fee": 0.0, "slippage_cost": 0.0,
                                           "cash_after": cash, "shares_after": shares,
                                           "reason": f"可用资金不足一手（需 {exec_price*cfg.lot_size:.0f} 元，"
                                                     f"可投 {invest:.0f} 元）"})
                            warnings.append(f"{t.date()} 买单被拒：资金不足")
                        else:
                            value = buy_shares * exec_price
                            fee = _commission(value, cfg)
                            slip_cost = buy_shares * (exec_price - base)
                            cash -= value + fee
                            total_cost = avg_cost * shares + value  # 更新平均成本
                            shares_new = shares + buy_shares
                            avg_cost = total_cost / shares_new
                            shares = shares_new
                            total_fees += fee
                            total_slip_cost += slip_cost
                            traded_value += value
                            trades.append({"signal_time": str(pending["signal_time"].date()),
                                           "exec_time": str(t.date()), "action": "buy",
                                           "fill_mode": fill_mode,
                                           "shares": int(buy_shares), "base_price": float(base),
                                           "exec_price": float(exec_price), "value": float(value),
                                           "fee": float(fee), "slippage_cost": float(slip_cost),
                                           "cash_after": float(cash), "shares_after": int(shares),
                                           "reason": pending["reason"]})
                elif target <= 0 and shares > 0:
                    # ---- 清仓卖出 ----
                    exec_price = base * (1 - cfg.slippage)
                    value = shares * exec_price
                    fee = _commission(value, cfg)
                    slip_cost = shares * (base - exec_price)
                    pnl = value - fee - avg_cost * shares
                    realized_pnl += pnl
                    round_trips.append(pnl)
                    cash += value - fee
                    total_fees += fee
                    total_slip_cost += slip_cost
                    traded_value += value
                    trades.append({"signal_time": str(pending["signal_time"].date()),
                                   "exec_time": str(t.date()), "action": "sell",
                                   "fill_mode": fill_mode,
                                   "shares": int(shares), "base_price": float(base),
                                   "exec_price": float(exec_price), "value": float(value),
                                   "fee": float(fee), "slippage_cost": float(slip_cost),
                                   "pnl": float(pnl),
                                   "cash_after": float(cash), "shares_after": 0,
                                   "reason": pending["reason"]})
                    shares, avg_cost = 0, 0.0
                pending = None

        # ---------- 2) 收盘阶段：净值 / 回撤 / 预警 ----------
        close = row["close"]
        equity = cash + shares * (close if not (isinstance(close, float) and math.isnan(close)) else avg_cost * shares)
        if equity > equity_peak:
            equity_peak = equity
        dd = equity / equity_peak - 1 if equity_peak > 0 else 0.0
        if not drawdown_warned and dd < -cfg.drawdown_warn:
            drawdown_warned = True
            warnings.append(f"{t.date()} 回撤预警：当前回撤 {dd:.2%} 超过阈值 "
                            f"{cfg.drawdown_warn:.0%}（仅预警，不强制平仓）")
        daily.append({"date": str(t.date()), "cash": float(cash), "shares": int(shares),
                      "close": float(close) if not math.isnan(close) else None,
                      "avg_cost": float(avg_cost), "equity": float(equity),
                      "drawdown": float(dd), "warned": drawdown_warned})

        # ---------- 3) 收盘后：生成明日订单（策略只见 ≤ t 的数据） ----------
        if i + 1 < len(dates):
            if i >= strategy.warmup_days() - 1:
                try:
                    target = float(strategy.target_position(data.iloc[: i + 1]))
                except Exception as e:  # noqa: BLE001 - 策略异常不拖垮引擎
                    warnings.append(f"{t.date()} 策略信号计算异常：{e}，按空仓处理")
                    target = 0.0
                if math.isnan(close):
                    target = last_target if last_target is not None else 0.0  # 数据异常禁止新订单
                if target != last_target or (target > 0 and shares == 0):
                    reason = (f"目标仓位 {target:.0%}"
                              + ("（数据异常日，沿用此前目标）" if math.isnan(close) else ""))
                    pending = {"signal_time": t, "target": target, "reason": reason}
                    last_target = target

    # ---------- 绩效指标 ----------
    eq = pd.Series([d["equity"] for d in daily], index=pd.to_datetime([d["date"] for d in daily]))
    metrics = _performance_metrics(eq, initial=cfg.initial_cash)
    sells = [tr for tr in trades if tr["action"] == "sell"]
    buys = [tr for tr in trades if tr["action"] == "buy"]
    metrics.update({
        "n_trades": len(buys) + len(sells),
        "n_buys": len(buys), "n_sells": len(sells),
        "total_fees": float(total_fees),
        "total_slippage_cost": float(total_slip_cost),
        "realized_pnl": float(realized_pnl),
        "final_cash": float(cash), "final_shares": int(shares),
        "final_equity": float(eq.iloc[-1]),
    })
    if len(round_trips) >= 1:
        wins = [p for p in round_trips if p > 0]
        losses = [p for p in round_trips if p <= 0]
        metrics["win_rate"] = len(wins) / len(round_trips)
        metrics["avg_win"] = float(np.mean(wins)) if wins else 0.0
        metrics["avg_loss"] = float(np.mean(losses)) if losses else 0.0
        metrics["profit_loss_ratio"] = (abs(metrics["avg_win"] / metrics["avg_loss"])
                                        if metrics["avg_loss"] != 0 else None)
    years = len(eq) / TRADING_DAYS
    if years > 0:
        mean_eq = eq.mean()
        metrics["turnover_annual"] = float(traded_value / mean_eq / years)
    metrics["assumptions"] = ("成交模型：t 收盘信号 → t+1 开盘价±滑点成交；无开盘价时按 t+1 收盘成交"
                              "（订单 fill_mode 标记）；手续费=max(金额×费率, 最低费)；移动平均成本法；"
                              "回撤预警不触发强制平仓。")
    return {"metrics": metrics, "trades": trades, "daily": daily, "warnings": warnings}


def _performance_metrics(equity: pd.Series, initial: float) -> dict:
    """从逐日净值序列计算绩效（与 analysis.py 同口径：252 交易日年化）。"""
    ret = equity.pct_change(fill_method=None).dropna()
    total = equity.iloc[-1] / initial - 1
    years = len(equity) / TRADING_DAYS
    cagr = (equity.iloc[-1] / initial) ** (1 / years) - 1 if years > 0 else None
    vol = float(ret.std() * np.sqrt(TRADING_DAYS)) if len(ret) > 1 else None
    sharpe = float(ret.mean() / ret.std() * np.sqrt(TRADING_DAYS)) \
        if len(ret) > 1 and ret.std() > 0 else None
    peak = equity.cummax()
    dd = equity / peak - 1
    mdd = float(dd.min())
    trough_i = dd.idxmin()
    peak_i = equity.loc[:trough_i].idxmax()
    return {"total_return": float(total), "cagr": float(cagr) if cagr is not None else None,
            "ann_vol": vol, "sharpe": sharpe, "mdd": mdd,
            "mdd_peak": str(peak_i.date()), "mdd_trough": str(trough_i.date()),
            "n_days": int(len(equity))}


def run_backtest(strategy, dataset_id: int, cfg: BacktestConfig = None,
                 name: str = "", experiment_id: int = None,
                 save: bool = True) -> dict:
    """执行回测并（默认）落库存档：交易明细/逐日状态写 CSV 文件，库中存摘要。"""
    cfg = cfg or BacktestConfig()
    cfg.validate()
    ds = db.get_dataset(dataset_id)
    if not ds:
        raise BacktestError(f"数据集 #{dataset_id} 不存在。")
    path = ds["clean_path"] if (ds["cleaned"] and ds["clean_path"]) else ds["raw_path"]
    df = load_csv(path)

    result = _run_engine(df, strategy, cfg)
    metrics = result["metrics"]

    # 基准对比：同一数据、同一成本下的买入并持有
    bench = _run_engine(df, BuyHoldStrategy(), cfg)["metrics"]
    metrics["benchmark"] = {k: bench.get(k) for k in
                            ("total_return", "cagr", "ann_vol", "sharpe", "mdd", "n_trades",
                             "total_fees", "final_equity")}
    metrics["excess_return_vs_benchmark"] = (metrics["total_return"] - bench["total_return"])

    if not save:
        return {"strategy_name": strategy.name, "metrics": metrics,
                "trades": result["trades"], "daily": result["daily"],
                "warnings": result["warnings"], "bt_id": None}

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bt_id = db.save_backtest_record(
        name=name or f"{strategy.name}_{ds['name']}", created_at=ts,
        strategy_type=strategy.name, strategy_params=json.dumps(asdict(cfg), ensure_ascii=False),
        dataset_id=dataset_id, dataset_name=ds["name"], dataset_version=ds["version"],
        experiment_id=experiment_id,
        start_date=str(df.index.min().date()), end_date=str(df.index.max().date()),
        config_json=json.dumps(asdict(cfg), ensure_ascii=False),
        metrics_json=json.dumps(metrics, ensure_ascii=False),
        status="success", code_version=CODE_VERSION)
    bt_dir = BACKTESTS_DIR / f"bt_{bt_id}"
    bt_dir.mkdir(parents=True, exist_ok=True)
    trades_path = bt_dir / "trades.csv"
    daily_path = bt_dir / "daily.csv"
    pd.DataFrame(result["trades"]).to_csv(trades_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(result["daily"]).to_csv(daily_path, index=False, encoding="utf-8-sig")
    if result["warnings"]:
        (bt_dir / "warnings.txt").write_text("\n".join(result["warnings"]), encoding="utf-8")
    db.update_backtest(bt_id, trades_path=str(trades_path), daily_path=str(daily_path))
    return {"strategy_name": strategy.name, "metrics": metrics,
            "trades": result["trades"], "daily": result["daily"],
            "warnings": result["warnings"], "bt_id": bt_id,
            "trades_path": str(trades_path), "daily_path": str(daily_path)}
