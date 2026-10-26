"""模拟交易测试：账户/订单/成交/风控/持久化/防重复/策略驱动。"""
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from quantlab import paper_trading as PT
from quantlab.paper_trading import RiskRules, PaperTradingError
from quantlab.storage import save_raw


def _mk_ds(name="PT_ASSET", seed=1, n=100, end=None):
    import quantlab.db as db
    rng = np.random.default_rng(seed)
    end = end or str(date.today())
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=n)
    close = np.linspace(100, 120, n)
    df = pd.DataFrame({"open": close * 0.998, "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": 1e6}, index=idx)
    ds_id = db.register_dataset(name, name[:6], source="demo", is_synthetic=1)
    db.update_dataset(ds_id, raw_path=save_raw(df, name), rows=n,
                      start_date=str(idx[0].date()), end_date=str(idx[-1].date()))
    return db.get_dataset(ds_id)


PRICE = (100.0, str(date.today()))  # 注入的参考价（当日，保证新鲜）


def _account(initial=100_000, risk=None):
    return PT.create_account(f"测试{PT.list_accounts() and len(PT.list_accounts())+1 or 1}",
                             initial, risk)


# ---------------- 账户 ----------------

def test_account_create_persist_snapshot():
    aid = _account(50_000)
    acc = PT.get_account(aid)
    assert acc["cash"] == 50_000
    snap = PT.account_snapshot(aid, lambda ds: PRICE)
    assert np.isclose(snap["equity"], 50_000) and snap["positions"] == []
    # 持久化：重新查询仍在（同连接进程重启模拟由 SQLite 保证，这里验证记录落库）
    import quantlab.db as db
    with db.get_conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM paper_accounts WHERE id=?",
                            (aid,)).fetchone()[0] == 1


def test_delete_requires_name_confirmation():
    aid = PT.create_account("保护账户", 10_000)
    with pytest.raises(PaperTradingError, match="确认失败"):
        PT.delete_account(aid, "错误名字")     # 名字不一致 → 拒绝
    res = PT.delete_account(aid, "保护账户")   # 一致 → 删除
    assert res["deleted"] == "保护账户"
    assert PT.get_account(aid) is None


# ---------------- 买卖与账目 ----------------

def test_buy_hand_computed():
    ds = _mk_ds()
    aid = _account(100_000, RiskRules(commission_rate=0.001, min_commission=5,
                                      slippage=0.005, lot_size=100))
    r = PT.place_order(aid, ds, "buy", 900, price_override=PRICE)
    assert r["status"] == "filled"
    assert np.isclose(r["exec_price"], 100.5)            # 100×(1+0.005)
    assert np.isclose(r["fee"], max(900 * 100.5 * 0.001, 5))
    acc = PT.get_account(aid)
    assert np.isclose(acc["cash"], 100_000 - 900 * 100.5 - r["fee"])
    snap = PT.account_snapshot(aid, lambda d: PRICE)
    assert snap["positions"][0]["shares"] == 900
    assert np.isclose(snap["equity"], acc["cash"] + 900 * 100)  # 市值按参考价


def test_sell_updates_realized_pnl():
    ds = _mk_ds()
    aid = _account(100_000, RiskRules(lot_size=1, commission_rate=0, min_commission=0,
                                      slippage=0))
    PT.place_order(aid, ds, "buy", 1000, price_override=PRICE)
    up = (110.0, str(date.today()))
    r = PT.place_order(aid, ds, "sell", 600, price_override=up)
    assert r["status"] == "filled" and np.isclose(r["pnl"], 600 * 10)  # 每股赚 10
    acc = PT.get_account(aid)
    assert np.isclose(acc["realized_pnl"], 6000)
    snap = PT.account_snapshot(aid, lambda d: up)
    assert np.isclose(snap["positions"][0]["shares"], 400)
    # 未实现盈亏 = 400×(110-100)
    assert np.isclose(snap["unrealized_pnl"], 4000)
    # 账户净值一致性：现金+市值
    assert np.isclose(snap["equity"], acc["cash"] + 400 * 110)


def test_cash_never_negative_and_no_short():
    ds = _mk_ds()
    aid = _account(10_000, RiskRules(lot_size=1, commission_rate=0, min_commission=0))
    r = PT.place_order(aid, ds, "buy", 500)     # 需要 ~50000 > 10000
    assert r["status"] == "rejected" and "现金不足" in r["reason"]
    r2 = PT.place_order(aid, ds, "sell", 1)     # 无持仓卖出
    assert r2["status"] == "rejected" and "做空" in r2["reason"]
    assert PT.get_account(aid)["cash"] == 10_000  # 拒单不改变现金


def test_risk_limits_rejected_with_reason():
    ds = _mk_ds()
    aid = _account(100_000, RiskRules(lot_size=1, max_order_value=5_000))
    r = PT.place_order(aid, ds, "buy", 100, price_override=PRICE)
    assert r["status"] == "rejected" and "单笔金额" in r["reason"]
    aid2 = _account(100_000, RiskRules(lot_size=1, min_cash_fraction=0.5))
    r2 = PT.place_order(aid2, ds, "buy", 900, price_override=PRICE)
    assert r2["status"] == "rejected" and "最低现金" in r2["reason"]
    aid3 = _account(100_000, RiskRules(lot_size=1, max_daily_traded_value=10_000))
    PT.place_order(aid3, ds, "buy", 50, price_override=PRICE)   # 5000 成交
    r3 = PT.place_order(aid3, ds, "buy", 100, price_override=PRICE)  # 再 10000 超 10000
    assert r3["status"] == "rejected" and "单日成交额" in r3["reason"]


def test_stale_price_rejected():
    ds = _mk_ds()
    aid = _account(100_000)
    old = (95.0, str(date.today() - timedelta(days=30)))
    r = PT.place_order(aid, ds, "buy", 100, price_override=old)
    assert r["status"] == "rejected" and "过期" in r["reason"]


def test_stale_from_dataset_end_detected():
    ds = _mk_ds("PT_OLD", end="2024-01-05")   # 数据集最后一根 K 线很旧
    aid = _account(100_000)
    r = PT.place_order(aid, ds, "buy", 100)   # 不注入价格 → 用数据集最后收盘
    assert r["status"] == "rejected" and "过期" in r["reason"]


def test_cancel_unfilled_only():
    ds = _mk_ds()
    aid = _account()
    # 制造一个 unfilled：数据集无有效收盘价难造，直接检查状态机约束
    r = PT.place_order(aid, ds, "buy", 100, price_override=PRICE)
    assert r["status"] == "filled"
    with pytest.raises(PaperTradingError, match="不可撤销"):
        PT.cancel_order(r["order_id"])          # 已成交订单不可撤


# ---------------- 策略驱动与防重复 ----------------

def test_strategy_signal_and_duplicate_protection():
    import quantlab.db as db
    ds = _mk_ds("PT_STRAT", seed=2)
    aid = _account(100_000, RiskRules(lot_size=100))
    PT.bind_strategy(aid, {"strategy": "buy_hold", "dataset_id": ds["id"],
                           "target_fraction": 0.5})
    r1 = PT.run_strategy_signal(aid, price_override=PRICE)
    assert r1["action"] == "order_filled"       # 空仓 → 买入
    r2 = PT.run_strategy_signal(aid, price_override=PRICE)  # 同一信号日重复执行
    assert r2["action"] == "order_rejected" or r2["action"] == "hold"
    if r2["action"] == "order_rejected":
        assert "重复信号" in r2["order"]["reason"]
    # 订单可追踪：signal_id 已入库
    orders = PT.list_orders(aid)
    assert any(o["signal_id"] and "buy_hold" in o["signal_id"] for o in orders)


def test_strategy_pause_blocks_orders():
    ds = _mk_ds("PT_PAUSE", seed=3)
    aid = _account(100_000)
    PT.bind_strategy(aid, {"strategy": "buy_hold", "dataset_id": ds["id"], "paused": True})
    r = PT.run_strategy_signal(aid, price_override=PRICE)
    assert r["action"] == "skipped" and "暂停" in r["reason"]
    assert len(PT.list_orders(aid)) == 0         # 暂停后零订单


def test_ma_cross_strategy_signal_direction():
    import quantlab.db as db
    # 构造上涨尾段 → SMA5 > SMA20 → 目标 1 → 买入
    rng = np.random.default_rng(9)
    n = 60
    idx = pd.bdate_range(end=pd.Timestamp(str(date.today())), periods=n)
    close = np.concatenate([np.full(40, 100.0), np.linspace(100, 140, 20)])
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                       "volume": 1e6}, index=idx)
    ds_id = db.register_dataset("PT_MA", "PTMA", source="demo", is_synthetic=1)
    db.update_dataset(ds_id, raw_path=save_raw(df, "PT_MA"), rows=n)
    ds = db.get_dataset(ds_id)
    aid = _account(100_000, RiskRules(lot_size=100))
    PT.bind_strategy(aid, {"strategy": "ma_cross", "dataset_id": ds["id"],
                           "params": {"fast": 5, "slow": 20}, "target_fraction": 0.5})
    r = PT.run_strategy_signal(aid, price_override=PRICE)
    assert r["action"] == "order_filled" and r["side"] == "buy"


def test_orders_persist_across_reload():
    ds = _mk_ds("PT_PERSIST", seed=4)
    aid = _account(100_000, RiskRules(lot_size=1))
    PT.place_order(aid, ds, "buy", 100, price_override=PRICE)
    # 重新打开数据库连接（模拟重启后的读取路径）
    from quantlab import db as db2
    db2.init_db()
    orders = PT.list_orders(aid)
    assert len(orders) == 1 and orders[0]["status"] == "filled"
    acc = PT.get_account(aid)
    assert acc["cash"] < 100_000                     # 现金变化也持久


def test_equity_consistency_after_round_trip():
    ds = _mk_ds("PT_RT", seed=5)
    aid = _account(100_000, RiskRules(lot_size=1, commission_rate=0, min_commission=0,
                                      slippage=0))
    PT.place_order(aid, ds, "buy", 500, price_override=PRICE)
    PT.place_order(aid, ds, "sell", 500, price_override=PRICE)  # 同价平仓
    snap = PT.account_snapshot(aid, lambda d: PRICE)
    assert np.isclose(snap["equity"], 100_000)        # 无费用同价往返 → 净值不变
    assert np.isclose(snap["realized_pnl"], 0)
