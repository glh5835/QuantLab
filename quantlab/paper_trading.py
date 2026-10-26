"""虚拟账户与模拟交易（阶段 4A）：仅虚拟资金，永不连接真实券商。

成交模型（明确标记的简化假设，与任务书 4A.4 对齐）：
- 市价单按"该资产数据集中最后已知收盘价"作为参考价 ref_price 立即模拟成交，
  叠加单边滑点得 exec_price，扣手续费；fill_model='close_simplified'。
- 参考行情时间戳 ref_time 如实记录（数据集最后一根 K 线日期），界面明示
  "基于最新可用历史行情模拟，非实时行情"。
- 行情新鲜度检查：最后一根 K 线距今天超过 staleness_days（默认 14 天）→ 拒单。

风控（拒绝必须留痕）：负现金禁止、卖出≤持仓、单笔金额上限、单资产占比上限、
最低现金保留、单日累计成交额上限。默认无杠杆无做空。

策略驱动：绑定策略后手动/定时"执行信号"，signal_id 唯一约束（账户+策略+日期+方向）
防止同一信号因刷新/重试重复下单；暂停绑定后不再产生自动订单。
"""
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, date

import pandas as pd

from . import db
from .storage import load_csv

TRADING_DAYS = 252


class PaperTradingError(ValueError):
    pass


@dataclass
class RiskRules:
    max_order_value: float = 0.0        # 单笔订单金额上限（0=不限）
    max_asset_fraction: float = 1.0     # 单资产持仓占净值上限
    min_cash_fraction: float = 0.0      # 最低现金保留比例
    max_daily_traded_value: float = 0.0 # 单日累计成交额上限（0=不限）
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    slippage: float = 0.001
    lot_size: int = 100
    staleness_days: int = 14            # 参考行情最大允许陈旧天数


# ---------------- 账户 ----------------

def create_account(name: str, initial_cash: float, risk: RiskRules = None) -> int:
    if not name.strip():
        raise PaperTradingError("账户名称不能为空。")
    if initial_cash <= 0:
        raise PaperTradingError("初始虚拟资金必须为正。")
    ts = db.now_str()
    with db.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO paper_accounts (name, initial_cash, cash, risk_json, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (name.strip(), initial_cash, initial_cash,
             json.dumps(asdict(risk or RiskRules()), ensure_ascii=False), ts, ts))
        return cur.lastrowid


def get_account(account_id: int):
    with db.get_conn() as conn:
        return conn.execute("SELECT * FROM paper_accounts WHERE id=?", (account_id,)).fetchone()


def list_accounts():
    with db.get_conn() as conn:
        return conn.execute("SELECT * FROM paper_accounts ORDER BY id DESC").fetchall()


def delete_account(account_id: int, confirm_name: str) -> dict:
    """删除账户（有损操作）：必须传入与账户名一致的 confirm_name 才执行。"""
    acc = get_account(account_id)
    if not acc:
        raise PaperTradingError(f"账户 #{account_id} 不存在。")
    if confirm_name != acc["name"]:
        raise PaperTradingError("确认失败：输入的账户名与账户不一致，删除被拒绝"
                                "（防误删保护，删除是不可恢复操作）。")
    with db.get_conn() as conn:
        conn.execute("DELETE FROM paper_orders WHERE account_id=?", (account_id,))
        conn.execute("DELETE FROM paper_positions WHERE account_id=?", (account_id,))
        conn.execute("DELETE FROM paper_accounts WHERE id=?", (account_id,))
    return {"deleted": acc["name"], "orders_removed": True}


def positions(account_id: int) -> list:
    with db.get_conn() as conn:
        return conn.execute("SELECT * FROM paper_positions WHERE account_id=? AND shares > 0",
                            (account_id,)).fetchall()


def _latest_price(symbol_dataset_row) -> tuple[float, str]:
    """取数据集最后已知收盘价与其日期（参考行情，如实记录时间戳）。"""
    ds = symbol_dataset_row
    path = ds["clean_path"] if (ds["cleaned"] and ds["clean_path"]) else ds["raw_path"]
    if not path:
        raise PaperTradingError(f"数据集 {ds['name']} 没有数据文件。")
    df = load_csv(path)
    close = df["close"].dropna()
    if len(close) == 0:
        raise PaperTradingError(f"数据集 {ds['name']} 无有效收盘价。")
    return float(close.iloc[-1]), str(close.index[-1].date())


def account_snapshot(account_id: int, price_lookup) -> dict:
    """账户快照：现金/持仓/净值/已实现/未实现盈亏。price_lookup(dataset_id)->(价,日期)。"""
    acc = get_account(account_id)
    if not acc:
        raise PaperTradingError(f"账户 #{account_id} 不存在。")
    pos_rows = positions(account_id)
    market_value = 0.0
    unrealized = 0.0
    pos_detail = []
    for p in pos_rows:
        price, pdate = price_lookup(p["symbol"])
        mv = p["shares"] * price
        market_value += mv
        upnl = p["shares"] * (price - p["avg_cost"])
        unrealized += upnl
        pos_detail.append({"symbol": p["symbol"], "shares": p["shares"],
                           "avg_cost": p["avg_cost"], "price": price,
                           "price_date": pdate, "market_value": mv, "unrealized_pnl": upnl})
    equity = acc["cash"] + market_value
    return {"id": acc["id"], "name": acc["name"], "status": acc["status"],
            "initial_cash": acc["initial_cash"], "cash": acc["cash"],
            "market_value": market_value, "equity": equity,
            "realized_pnl": acc["realized_pnl"], "unrealized_pnl": unrealized,
            "total_fees": acc["total_fees"],
            "total_return": equity / acc["initial_cash"] - 1,
            "positions": pos_detail}


# ---------------- 订单与成交 ----------------

def place_order(account_id: int, dataset: "db.sqlite3.Row", side: str, qty: float,
                signal_id: str = "", risk: RiskRules = None,
                price_override: tuple[float, str] = None) -> dict:
    """提交市价模拟订单 → 立即按参考价模拟成交或拒绝（全部记录入库）。

    dataset: 目标资产的数据集行；price_override: (价,日期) 测试用注入。
    返回订单记录 dict。任何拒绝都以 status='rejected' + reason 入库（不静默丢弃）。
    """
    if side not in ("buy", "sell"):
        raise PaperTradingError("side 必须是 buy 或 sell。")
    if qty <= 0:
        raise PaperTradingError("数量必须为正。")
    acc = get_account(account_id)
    if not acc:
        raise PaperTradingError(f"账户 #{account_id} 不存在。")
    if acc["status"] != "active":
        return _record_order(account_id, dataset["name"], side, qty, "rejected",
                             reason=f"账户状态为 {acc['status']}，禁止新订单。")
    risk = risk or RiskRules(**json.loads(acc["risk_json"] or "{}"))

    # 同一信号防重复（幂等保护）
    if signal_id:
        with db.get_conn() as conn:
            dup = conn.execute(
                "SELECT id FROM paper_orders WHERE account_id=? AND signal_id=? "
                "AND status IN ('filled','unfilled')", (account_id, signal_id)).fetchone()
        if dup:
            return _record_order(account_id, dataset["name"], side, qty, "rejected",
                                 signal_id=signal_id, ref_price=None,
                                 reason=f"重复信号：信号 {signal_id} 已产生订单 #{dup['id']}，拒绝重复下单。")

    # 参考行情与新鲜度
    try:
        ref_price, ref_time = price_override or _latest_price(dataset)
    except PaperTradingError as e:
        return _record_order(account_id, dataset["name"], side, qty, "unfilled",
                             signal_id=signal_id, reason=f"无法取得参考行情：{e}")
    try:
        last_date = datetime.strptime(ref_time, "%Y-%m-%d").date()
    except ValueError:
        last_date = date.today()
    stale_days = (date.today() - last_date).days
    if stale_days > risk.staleness_days:
        return _record_order(account_id, dataset["name"], side, qty, "rejected",
                             signal_id=signal_id, ref_price=ref_price, ref_time=ref_time,
                             reason=f"参考行情过期：最后行情 {ref_time} 距今 {stale_days} 天"
                                    f"（允许 {risk.staleness_days} 天），拒绝成交以免失真。")

    pos = _get_pos(account_id, dataset["name"])
    held = pos["shares"] if pos else 0.0

    def _reject(reason):
        return _record_order(account_id, dataset["name"], side, qty, "rejected",
                             signal_id=signal_id, ref_price=ref_price, ref_time=ref_time,
                             reason=reason)

    if side == "buy":
        exec_price = ref_price * (1 + risk.slippage)
        value = qty * exec_price
        fee = max(value * risk.commission_rate, risk.min_commission)
        acc_cash = acc["cash"]
        # 风控：卖出在先回笼现金的场景本版不自动撮合（卖出订单独立提交）
        equity_est = acc_cash + held * ref_price
        if risk.max_order_value > 0 and value > risk.max_order_value:
            return _reject(f"单笔金额 {value:,.0f} 超上限 {risk.max_order_value:,.0f}。")
        if risk.max_asset_fraction < 1.0 and held == 0:
            if value > risk.max_asset_fraction * equity_est:
                return _reject(f"买入后单资产占比将超上限 {risk.max_asset_fraction:.0%}。")
        if risk.min_cash_fraction > 0 and acc_cash - value - fee < risk.min_cash_fraction * equity_est:
            return _reject(f"将违反最低现金保留 {risk.min_cash_fraction:.0%}"
                           f"（剩余 {acc_cash - value - fee:,.0f}）。")
        if acc_cash < value + fee:
            return _reject(f"现金不足：需 {value + fee:,.0f}（含费），可用 {acc_cash:,.0f}；"
                           f"本平台不允许融资。")
        if risk.max_daily_traded_value > 0:
            today_used = _daily_traded_value(account_id)
            if today_used + value > risk.max_daily_traded_value:
                return _reject(f"将超单日成交额上限 {risk.max_daily_traded_value:,.0f}"
                               f"（今日已成交 {today_used:,.0f}）。")
        # 成交：更新现金与持仓（移动平均成本）
        new_cash = acc_cash - value - fee
        total_cost = (pos["avg_cost"] * held if pos else 0) + value
        new_shares = held + qty
        _upsert_pos(account_id, dataset["name"], new_shares, total_cost / new_shares)
        _update_account(account_id, cash=new_cash, total_fees=acc["total_fees"] + fee)
        return _record_order(account_id, dataset["name"], side, qty, "filled",
                             signal_id=signal_id, fill_model="close_simplified",
                             ref_price=ref_price, ref_time=ref_time,
                             exec_price=exec_price, fee=fee,
                             slippage=qty * (exec_price - ref_price))
    else:  # sell
        if qty > held + 1e-9:
            return _reject(f"卖出数量 {qty} 超过持仓 {held}（不允许做空）。")
        exec_price = ref_price * (1 - risk.slippage)
        value = qty * exec_price
        fee = max(value * risk.commission_rate, risk.min_commission)
        avg = pos["avg_cost"]
        pnl = value - fee - avg * qty
        new_shares = held - qty
        _upsert_pos(account_id, dataset["name"], new_shares, avg if new_shares > 0 else 0)
        _update_account(account_id, cash=acc["cash"] + value - fee,
                        realized_pnl=acc["realized_pnl"] + pnl,
                        total_fees=acc["total_fees"] + fee)
        return _record_order(account_id, dataset["name"], side, qty, "filled",
                             signal_id=signal_id, fill_model="close_simplified",
                             ref_price=ref_price, ref_time=ref_time,
                             exec_price=exec_price, fee=fee,
                             slippage=qty * (ref_price - exec_price), pnl=pnl)


def cancel_order(order_id: int) -> dict:
    """撤销订单：只有未成交/排队状态的订单可撤（本版市价单即时成交，可撤的是 unfilled）。"""
    with db.get_conn() as conn:
        row = conn.execute("SELECT * FROM paper_orders WHERE id=?", (order_id,)).fetchone()
        if not row:
            raise PaperTradingError(f"订单 #{order_id} 不存在。")
        if row["status"] != "unfilled":
            raise PaperTradingError(f"订单 #{order_id} 状态为 {row['status']}，不可撤销"
                                    f"（仅未成交订单可撤）。")
        conn.execute("UPDATE paper_orders SET status='cancelled', exec_at=?, "
                     "reason=COALESCE(NULLIF(reason,''),'用户撤销') WHERE id=?",
                     (db.now_str(), order_id))
    return {"order_id": order_id, "status": "cancelled"}


def list_orders(account_id: int = None, limit: int = 100) -> list:
    with db.get_conn() as conn:
        if account_id:
            return conn.execute("SELECT * FROM paper_orders WHERE account_id=? "
                                "ORDER BY id DESC LIMIT ?", (account_id, limit)).fetchall()
        return conn.execute("SELECT * FROM paper_orders ORDER BY id DESC LIMIT ?",
                            (limit,)).fetchall()


# ---------------- 策略驱动 ----------------

def bind_strategy(account_id: int, binding: dict) -> None:
    """绑定/更新策略：{strategy, dataset_id, params, target_fraction, paused}。"""
    required = {"strategy", "dataset_id"}
    if not required <= set(binding):
        raise PaperTradingError(f"绑定配置缺少 {required - set(binding)}。")
    _update_account(account_id, binding_json=json.dumps(binding, ensure_ascii=False))


def run_strategy_signal(account_id: int, price_override: tuple[float, str] = None) -> dict:
    """对绑定策略执行一次信号评估：t=数据集最后一根 K 线收盘后计算目标仓位。

    信号变化且非暂停时按目标仓位差额生成一单（买卖数量=目标股数-当前股数）。
    signal_id 含策略+日期+目标 → 同一信号日重复调用不会重复下单。
    """
    acc = get_account(account_id)
    if not acc:
        raise PaperTradingError(f"账户 #{account_id} 不存在。")
    binding = json.loads(acc["binding_json"] or "{}")
    if not binding:
        raise PaperTradingError("该账户未绑定策略。")
    if binding.get("paused"):
        return {"action": "skipped", "reason": "策略绑定已暂停，不生成新订单。",
                "signal_id": None}
    from .strategies import BuyHoldStrategy, MACrossoverStrategy, MLSignalStrategy
    ds = db.get_dataset(int(binding["dataset_id"]))
    if not ds:
        raise PaperTradingError("绑定的数据集不存在。")
    path = ds["clean_path"] if (ds["cleaned"] and ds["clean_path"]) else ds["raw_path"]
    df = load_csv(path)
    st = binding["strategy"]
    params = binding.get("params", {})
    if st == "buy_hold":
        strat = BuyHoldStrategy()
    elif st == "ma_cross":
        strat = MACrossoverStrategy(int(params.get("fast", 5)), int(params.get("slow", 20)))
    elif st == "ml_signal":
        strat = MLSignalStrategy(int(params["experiment_id"]))
    else:
        raise PaperTradingError(f"不支持的策略 {st}。")
    target = float(strat.target_position(df))
    last_date = str(df.index[-1].date())
    signal_id = f"{st}|{ds['name']}|{last_date}|{target:.0%}"
    ref = price_override or _latest_price(ds)
    # 目标股数（按目标仓位×净值/参考价，取整至手）
    risk = RiskRules(**json.loads(acc["risk_json"] or "{}"))
    snap = account_snapshot(account_id, lambda _: ref)
    equity = snap["equity"]
    lot = max(int(risk.lot_size), 1)
    target_shares = int(binding.get("target_fraction", 0.95) * target * equity
                       / max(ref[0], 1e-9) / lot) * lot
    held = 0.0
    pos = _get_pos(account_id, ds["name"])
    if pos:
        held = pos["shares"]
    delta = target_shares - held
    if abs(delta) < lot:
        return {"action": "hold", "reason": f"目标仓位 {target:.0%} 与当前持仓 {held:.0f} 股"
                f"差额不足一手，不下单。", "signal_id": signal_id}
    side = "buy" if delta > 0 else "sell"
    order = place_order(account_id, ds, side, abs(delta), signal_id=signal_id,
                        price_override=price_override)
    return {"action": f"order_{order['status']}", "signal_id": signal_id,
            "side": side, "qty": abs(delta), "order": order,
            "signal_target": target, "signal_date": last_date}


# ---------------- 内部工具 ----------------

def _get_pos(account_id, symbol):
    with db.get_conn() as conn:
        return conn.execute("SELECT * FROM paper_positions WHERE account_id=? AND symbol=?",
                            (account_id, symbol)).fetchone()


def _upsert_pos(account_id, symbol, shares, avg_cost):
    with db.get_conn() as conn:
        conn.execute("INSERT INTO paper_positions (account_id, symbol, shares, avg_cost) "
                     "VALUES (?,?,?,?) ON CONFLICT(account_id, symbol) "
                     "DO UPDATE SET shares=excluded.shares, avg_cost=excluded.avg_cost",
                     (account_id, symbol, shares, avg_cost))


def _update_account(account_id, **fields):
    cols = ", ".join(f"{k}=?" for k in fields)
    with db.get_conn() as conn:
        conn.execute(f"UPDATE paper_accounts SET {cols}, updated_at=? WHERE id=?",
                     (*fields.values(), db.now_str(), account_id))


def _daily_traded_value(account_id) -> float:
    today = date.today().isoformat()
    with db.get_conn() as conn:
        row = conn.execute("SELECT COALESCE(SUM(qty*exec_price),0) v FROM paper_orders "
                           "WHERE account_id=? AND status='filled' AND exec_at LIKE ?",
                           (account_id, f"{today}%")).fetchone()
    return float(row["v"])


def _record_order(account_id, symbol, side, qty, status, signal_id="",
                  fill_model="", ref_price=None, ref_time="", exec_price=None,
                  fee=0.0, slippage=0.0, reason="", pnl=None) -> dict:
    ts = db.now_str()
    with db.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO paper_orders (account_id, symbol, side, qty, status, fill_model, "
            "signal_id, ref_price, ref_time, exec_price, fee, slippage, reason, created_at, exec_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (account_id, symbol, side, qty, status, fill_model, signal_id,
             ref_price, ref_time, exec_price, fee, slippage, reason, ts,
             ts if status in ("filled", "rejected", "cancelled", "unfilled") else ""))
        oid = cur.lastrowid
    out = {"order_id": oid, "status": status, "reason": reason, "symbol": symbol,
           "side": side, "qty": qty, "fee": fee, "exec_price": exec_price,
           "ref_price": ref_price, "ref_time": ref_time}
    if pnl is not None:
        out["pnl"] = pnl
    return out
