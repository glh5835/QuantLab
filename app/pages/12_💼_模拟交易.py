"""阶段 4A 页面：模拟交易中心（仅虚拟资金）。"""
import json

import pandas as pd
import streamlit as st

from ui_common import init_page, empty_state, synthetic_badge

from quantlab import db
from quantlab import paper_trading as PT
from quantlab.paper_trading import RiskRules, PaperTradingError

init_page("💼 模拟交易", "虚拟账户 · 模拟订单与成交 · 风控 · 策略信号绑定（非实时、非真实交易）")

st.warning("本中心**只使用虚拟资金**模拟交易：成交基于所连数据集的最新可用历史行情"
           "（收盘价简化模型，订单中如实记录行情时间戳），不是实时行情，也永不连接真实券商。")

accounts = PT.list_accounts()

# ---------------- 账户区 ----------------
st.subheader("① 虚拟账户")
c1, c2 = st.columns([3, 2])
with c1:
    with st.expander("➕ 创建账户", expanded=not accounts):
        n1, n2 = st.columns(2)
        acc_name = n1.text_input("账户名称", f"模拟账户{len(accounts)+1}")
        init_cash = float(n2.number_input("初始虚拟资金", 1_000, 10_000_000, 100_000, 10_000))
        r1, r2, r3, r4 = st.columns(4)
        rr = RiskRules(max_order_value=float(r1.number_input("单笔上限(0=不限)", 0.0, 1e7, 0.0, 1e4)),
                       min_cash_fraction=float(r2.number_input("最低现金比例", 0.0, 0.9, 0.0, 0.05)),
                       max_daily_traded_value=float(r3.number_input("单日成交上限(0=不限)", 0.0, 1e8, 0.0, 1e4)),
                       max_asset_fraction=float(r4.number_input("单资产上限", 0.1, 1.0, 1.0, 0.1)))
        if st.button("创建", use_container_width=True):
            try:
                aid = PT.create_account(acc_name, init_cash, rr)
                st.success(f"账户 #{aid} 已创建。")
                st.rerun()
            except PaperTradingError as e:
                st.error(str(e))
with c2:
    if accounts:
        acc_opts = {f"#{a['id']} {a['name']}（{a['status']}）": a["id"] for a in accounts}
        aid = acc_opts[st.selectbox("当前账户", list(acc_opts.keys()))]
    else:
        empty_state("还没有虚拟账户。", "在左侧创建第一个账户。")
        st.stop()

# ---------------- 账户总览 ----------------
acc = PT.get_account(aid)
datasets = db.list_datasets()
if not datasets:
    st.error("没有可交易的资产数据集。")
    st.stop()
ds_opts = {f"#{d['id']} {d['name']}（行情至 {d['end_date']}）": d for d in datasets}
price_cache = {}
def _price_of(sym):
    for d in datasets:
        if d["name"] == sym:
            if sym not in price_cache:
                price_cache[sym] = PT._latest_price(d)
            return price_cache[sym]
    return (float("nan"), "")

snap = PT.account_snapshot(aid, _price_of)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("账户净值", f"{snap['equity']:,.0f}", f"{snap['total_return']:+.2%}")
k2.metric("现金", f"{snap['cash']:,.0f}")
k3.metric("持仓市值", f"{snap['market_value']:,.0f}")
k4.metric("已实现盈亏", f"{snap['realized_pnl']:+,.0f}")
k5.metric("未实现盈亏", f"{snap['unrealized_pnl']:+,.0f}")

t1, t2, t3, t4 = st.tabs(["📦 持仓", "📝 手动下单", "🤖 策略绑定", "📋 订单记录"])
with t1:
    if snap["positions"]:
        pos_df = pd.DataFrame(snap["positions"])
        st.dataframe(pos_df, use_container_width=True, hide_index=True)
        st.caption("price_date 为参考行情的 K 线日期（非实时）。")
    else:
        st.info("空仓。")
with t2:
    sel_ds = ds_opts[st.selectbox("资产", list(ds_opts.keys()))]
    synthetic_badge(sel_ds)
    side = st.selectbox("方向", ["buy 买入", "sell 卖出"]).split()[0]
    qty = float(st.number_input("数量（股）", 1, 10_000_000, 100, 100))
    if st.button("提交模拟订单（市价·收盘价简化模型）", type="primary"):
        r = PT.place_order(aid, sel_ds, side, qty)
        if r["status"] == "filled":
            st.success(f"✅ 成交：{r['qty']:.0f} 股 × {r['exec_price']:.3f}"
                       f"（参考价 {r['ref_price']}，行情 {r['ref_time']}，费 {r['fee']:.2f}）")
        else:
            st.error(f"订单 #{r['order_id']} 未成交（{r['status']}）：{r['reason']}")
        st.rerun()
with t3:
    binding = json.loads(acc["binding_json"] or "{}")
    st.markdown(f"当前绑定：`{binding or '无'}`")
    b1, b2, b3 = st.columns(3)
    b_strategy = b1.selectbox("策略", ["buy_hold", "ma_cross", "ml_signal"])
    b_ds = ds_opts[b2.selectbox("策略盯的资产", list(ds_opts.keys()))]
    params = {}
    if b_strategy == "ma_cross":
        f1, f2 = st.columns(2)
        params = {"fast": int(f1.number_input("快线", 2, 100, 5)),
                  "slow": int(f2.number_input("慢线", 3, 300, 20))}
    elif b_strategy == "ml_signal":
        exps = db.list_experiments(30)
        if not exps:
            st.info("无 ML 实验，请先训练。")
        else:
            params = {"experiment_id": st.selectbox("实验",
                     [e["id"] for e in exps],
                     format_func=lambda i: f"#{i}")}
    paused = st.checkbox("暂停策略（暂停后不再生成自动订单）",
                         value=bool(binding.get("paused")))
    bb1, bb2 = st.columns(2)
    if bb1.button("保存绑定"):
        PT.bind_strategy(aid, {"strategy": b_strategy, "dataset_id": b_ds["id"],
                               "params": params, "paused": paused,
                               "target_fraction": 0.95})
        st.success("绑定已保存。") ; st.rerun()
    if bb2.button("▶️ 立即执行一次信号评估"):
        try:
            r = PT.run_strategy_signal(aid)
            st.info(f"信号结果：**{r['action']}** — {r.get('reason', '')}"
                    f"（signal_id: {r.get('signal_id')}）")
        except PaperTradingError as e:
            st.error(str(e))
with t4:
    orders = PT.list_orders(aid, 100)
    if orders:
        odf = pd.DataFrame([dict(o) for o in orders])
        show = [c for c in ["id", "symbol", "side", "qty", "status", "fill_model",
                            "ref_price", "ref_time", "exec_price", "fee", "signal_id",
                            "reason", "created_at"] if c in odf.columns]
        st.dataframe(odf[show], use_container_width=True, hide_index=True)
        st.download_button("⬇️ 导出交易历史 CSV",
                           odf.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"paper_orders_{aid}.csv")
        cid = st.number_input("撤销订单 ID（仅未成交可撤）", 1, 10**9, 1)
        if st.button("撤销该订单"):
            try:
                st.success(PT.cancel_order(int(cid)))
                st.rerun()
            except PaperTradingError as e:
                st.error(str(e))
    else:
        st.info("暂无订单。")

# ---------------- 危险区 ----------------
with st.expander("🗑️ 删除账户（不可恢复，需输入账户名确认）"):
    confirm = st.text_input("输入账户名以确认删除")
    if st.button("删除账户") and confirm:
        try:
            res = PT.delete_account(aid, confirm)
            st.warning(f"已删除账户：{res['deleted']}")
            st.rerun()
        except PaperTradingError as e:
            st.error(str(e))
