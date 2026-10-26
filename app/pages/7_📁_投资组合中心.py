"""阶段 3A 页面：投资组合中心——创建组合、配置权重、再平衡回测与组合比较。"""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui_common import init_page, empty_state, synthetic_badge

from quantlab import db
from quantlab import portfolio as PF
from quantlab.portfolio import PortfolioConfig, PortfolioError
from quantlab.storage import load_csv

init_page("📁 投资组合中心", "等权/自定义/逆波动率 · 月度·季度再平衡 · 真实订单与成本 · 组合比较")

datasets = db.list_datasets()
if len(datasets) < 2:
    empty_state("投资组合至少需要 2 个数据集。", "先到「数据采集中心」采集多个资产。")
    st.stop()

st.subheader("① 创建组合")
names = {f"#{d['id']} {d['name']}（{d['rows']}行 {d['start_date']}~{d['end_date']}）": d
         for d in datasets}
picked = st.multiselect("选择资产（2~6 个）", list(names.keys()),
                        default=[list(names)[i] for i in range(min(3, len(names)))])
if len(picked) < 2:
    st.info("请至少选择 2 个资产。")
    st.stop()
sel_ds = [names[k] for k in picked]
for d in sel_ds:
    synthetic_badge(d)

method = st.radio("权重方法", ["等权重", "自定义权重", "逆波动率（每次再平衡用当时历史重算）"],
                  horizontal=True)
weights = None
if method == "自定义权重":
    cols = st.columns(len(sel_ds))
    w = {}
    for c, d in zip(cols, sel_ds):
        w[d["name"]] = float(c.number_input(d["name"], 0.0, 100.0, 100.0 / len(sel_ds), 1.0))
    weights = w

c1, c2, c3, c4 = st.columns(4)
rebalance = c1.selectbox("再平衡频率", ["none（仅期初建仓）", "monthly", "quarterly"]).split("（")[0]
cash_reserve = float(c2.number_input("现金保留比例", 0.0, 0.5, 0.02, 0.01))
commission = float(c3.number_input("手续费率", 0.0, 0.01, 0.0003, 0.0001, format="%.4f"))
slippage = float(c4.number_input("滑点", 0.0, 0.02, 0.001, 0.0005, format="%.4f"))
c5, c6 = st.columns(2)
initial = float(c5.number_input("初始资金", 10_000, 10_000_000, 100_000, 10_000))
max_weight = float(c6.number_input("单资产权重上限", 0.1, 1.0, 1.0, 0.05))

if st.button("🚀 运行组合回测", type="primary", use_container_width=True):
    try:
        price_map = {}
        for d in sel_ds:
            path = d["clean_path"] if (d["cleaned"] and d["clean_path"]) else d["raw_path"]
            price_map[d["name"]] = load_csv(path)
        cfg = PortfolioConfig(initial_cash=initial, commission_rate=commission,
                              slippage=slippage, rebalance=rebalance,
                              cash_reserve=cash_reserve, max_weight=max_weight)
        w_arg = "inverse_vol" if method.startswith("逆波动率") else \
            (weights if weights else "equal")
        if w_arg == "equal":
            w_arg = PF.equal_weights([d["name"] for d in sel_ds], max_weight)
        res = PF.run_portfolio(price_map, w_arg, cfg)
        bt_id = PF.save_portfolio_run(f"组合·{'+'.join(d['name'] for d in sel_ds[:3])}",
                                      {d["name"]: d["id"] for d in sel_ds},
                                      method, cfg, res,
                                      start=res["daily"][0]["date"], end=res["daily"][-1]["date"])
        st.session_state["pf_res"] = (res, bt_id, method)
    except (PortfolioError, Exception) as e:  # noqa: BLE001
        st.error(f"组合回测失败：{e}")

saved = st.session_state.get("pf_res")
if not saved:
    st.stop()
res, bt_id, method_used = saved
m = res["metrics"]

st.subheader(f"② 结果（存档 #{bt_id}）")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("总收益", f"{m['total_return']:.2%}")
k2.metric("年化", f"{m['cagr']:.2%}" if m["cagr"] is not None else "—")
k3.metric("夏普", f"{m['sharpe']:.2f}" if m["sharpe"] is not None else "—")
k4.metric("最大回撤", f"{m['mdd']:.2%}")
k5.metric("再平衡次数", m["n_rebalances"], f"{m['n_trades']} 笔订单 / 费用 {m['total_fees']:.0f}",
          delta_color="off")

tab1, tab2, tab3, tab4 = st.tabs(["净值曲线", "权重历史", "收益贡献", "交易明细"])
with tab1:
    daily = pd.DataFrame(res["daily"])
    daily["date"] = pd.to_datetime(daily["date"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["equity"] / m.get("final_cash", 1) /
                             daily["equity"].iloc[0], name="组合净值（归一化）"))
    fig.update_layout(height=340, yaxis_title="净值（期初=1）", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(daily[["date", "cash", "positions_value", "equity"]].tail(100),
                 use_container_width=True, hide_index=True)
with tab2:
    wh = pd.DataFrame(res["weights_history"])
    st.dataframe(wh, use_container_width=True, hide_index=True)
    st.caption("逆波动率权重每次再平衡只用当时已知的历史波动率（防未来数据）。")
with tab3:
    contrib = pd.DataFrame([{"资产": a, "盈亏贡献": v} for a, v in m["contribution_pnl"].items()])
    fig = go.Figure(go.Bar(x=contrib["资产"], y=contrib["盈亏贡献"],
                           marker_color=["green" if v >= 0 else "red" for v in contrib["盈亏贡献"]]))
    fig.update_layout(height=300, title="各资产盈亏贡献（元）", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame(m["final_positions"]).T, use_container_width=True)
with tab4:
    trades = pd.DataFrame(res["trades"])
    if len(trades):
        st.dataframe(trades, use_container_width=True, hide_index=True)
        st.download_button("⬇️ 导出交易明细", trades.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"portfolio_{bt_id}_trades.csv")
if res["warnings"]:
    with st.expander(f"⚠️ 运行提示（{len(res['warnings'])} 条）"):
        for w in res["warnings"][:30]:
            st.markdown(f"- {w}")

# ---------------- 组合比较 ----------------
st.divider()
st.subheader("③ 同条件组合比较")
if st.button("🔄 比较：各单资产 / 等权 / 逆波动率（同区间同成本）"):
    try:
        price_map = {}
        for d in sel_ds:
            path = d["clean_path"] if (d["cleaned"] and d["clean_path"]) else d["raw_path"]
            price_map[d["name"]] = load_csv(path)
        cfg = PortfolioConfig(initial_cash=initial, commission_rate=commission,
                              slippage=slippage, rebalance=rebalance,
                              cash_reserve=cash_reserve, max_weight=max_weight)
        rows = []
        for name in price_map:  # 单资产
            r = PF.run_portfolio({name: price_map[name]}, {name: 1.0}, cfg)
            mm = r["metrics"]
            rows.append({"配置": f"单资产 {name}", "总收益": f"{mm['total_return']:.2%}",
                         "最大回撤": f"{mm['mdd']:.2%}",
                         "夏普": f"{mm['sharpe']:.2f}" if mm["sharpe"] is not None else "—",
                         "费用": round(mm["total_fees"], 0)})
        eq = PF.equal_weights(list(price_map), max_weight)
        r = PF.run_portfolio(price_map, eq, cfg)
        rows.append({"配置": "等权组合", "总收益": f"{r['metrics']['total_return']:.2%}",
                     "最大回撤": f"{r['metrics']['mdd']:.2%}",
                     "夏普": f"{r['metrics']['sharpe']:.2f}",
                     "费用": round(r["metrics"]["total_fees"], 0)})
        r = PF.run_portfolio(price_map, "inverse_vol", cfg)
        rows.append({"配置": "逆波动率组合", "总收益": f"{r['metrics']['total_return']:.2%}",
                     "最大回撤": f"{r['metrics']['mdd']:.2%}",
                     "夏普": f"{r['metrics']['sharpe']:.2f}",
                     "费用": round(r["metrics"]["total_fees"], 0)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("比较基于相同区间、初始资金与成本假设；模拟数据集结果不代表真实市场。")
    except Exception as e:  # noqa: BLE001
        st.error(f"比较失败：{e}")
