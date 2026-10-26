"""模块 7（阶段2-C）：策略回测。

策略 A 买入持有 / B 均线交叉 / C 机器学习信号 → 逐日回测（t 收盘信号、t+1 开盘成交）→
绩效与基准对比 → 净值/回撤/成交点图表 → 交易明细 → 存档与历史。
"""
import json
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui_common import init_page, empty_state, synthetic_badge

from quantlab import db
from quantlab import backtest as BT
from quantlab.backtest import BacktestConfig, BacktestError
from quantlab.strategies import BuyHoldStrategy, MACrossoverStrategy, MLSignalStrategy, StrategyError

init_page("⚖️ 策略回测", "t 收盘信号 → t+1 开盘成交 · 现金账户 · 手续费/滑点 · 风控 · 基准对比")

datasets = db.list_datasets()
if not datasets:
    empty_state("没有可用的数据集。", "先到「数据采集中心」采集数据。")
    st.stop()

st.subheader("① 配置回测")
c1, c2 = st.columns(2)
with c1:
    strategy_type = st.selectbox("策略", [
        "A：买入并持有（基准）", "B：均线交叉", "C：机器学习信号（需先在实验室训练）"])
    ds_opts = {f"#{d['id']} {d['name']}（{d['rows']} 行）": d["id"] for d in datasets}
    dataset_id = ds_opts[st.selectbox("数据集", list(ds_opts.keys()))]
    synthetic_badge(db.get_dataset(dataset_id))

strategy = None
experiment_id = None
with c2:
    if strategy_type.startswith("B"):
        f_col, s_col = st.columns(2)
        fast = int(f_col.number_input("快线窗口", 2, 200, 5))
        slow = int(s_col.number_input("慢线窗口", 3, 500, 20))
        st.caption("快线需 < 慢线；金叉买入、死叉退出。")
    elif strategy_type.startswith("C"):
        exps = db.list_experiments()
        if not exps:
            st.warning("还没有机器学习实验。请先到「🤖 机器学习实验室」训练模型。")
        else:
            e_opts = {f"#{e['id']} {e['name']}（{e['task']}/h={e['horizon']}, test {e['test_range'].split('~')[0]} 起）": e["id"]
                      for e in exps}
            experiment_id = e_opts[st.selectbox("选择实验（默认只用 test 段样本外预测）",
                                                list(e_opts.keys()))]
            exp = db.get_experiment(experiment_id)
            if exp["task"] == "return":
                t1, t2 = st.columns(2)
                buy_th = float(t1.number_input("买入阈值（预测收益 > ）", -0.05, 0.05, 0.001, 0.001, format="%.3f"))
                exit_th = float(t2.number_input("退出阈值（预测收益 < ）", -0.05, 0.05, 0.0, 0.001, format="%.3f"))
            else:
                buy_th = exit_th = 0.5

with st.expander("💰 资金、费用与风控参数（全部可配置）"):
    m1, m2, m3, m4 = st.columns(4)
    initial_cash = float(m1.number_input("初始资金", 1_000, 100_000_000, 100_000, 10_000))
    commission_rate = float(m2.number_input("手续费率（按金额）", 0.0, 0.01, 0.0003, 0.0001, format="%.4f"))
    min_commission = float(m3.number_input("最低手续费（元，0=不设）", 0.0, 100.0, 5.0, 1.0))
    slippage = float(m4.number_input("单边滑点比例", 0.0, 0.02, 0.001, 0.0005, format="%.4f"))
    r1, r2, r3, r4 = st.columns(4)
    lot_size = int(r1.number_input("最小交易单位（股）", 1, 1000, 100))
    target_fraction = float(r2.number_input("目标仓位（占净值）", 0.1, 1.0, 0.95, 0.05))
    drawdown_warn = float(r3.number_input("回撤预警阈值", 0.05, 0.9, 0.20, 0.05))
    stop_buy = r4.checkbox("预警后停止新买入", value=False)
    st.caption("回撤预警只提示不平仓；现金账户无融资无做空；买入受现金与仓位上限约束。")

bt_name = st.text_input("回测名称（留空自动生成）", "")

if st.button("🚀 运行回测", type="primary", use_container_width=True):
    try:
        if strategy_type.startswith("A"):
            strategy = BuyHoldStrategy()
        elif strategy_type.startswith("B"):
            strategy = MACrossoverStrategy(fast, slow)
        else:
            strategy = MLSignalStrategy(experiment_id, buy_threshold=buy_th,
                                        exit_threshold=exit_th, split="test")
        cfg = BacktestConfig(initial_cash=initial_cash, commission_rate=commission_rate,
                             min_commission=min_commission, slippage=slippage,
                             lot_size=lot_size, target_fraction=target_fraction,
                             drawdown_warn=drawdown_warn, stop_buy_on_drawdown=stop_buy)
        with st.spinner("逐日步进回测中（信号→次日成交→更新净值）…"):
            res = BT.run_backtest(strategy, dataset_id, cfg, name=bt_name or None,
                                  experiment_id=experiment_id)
        st.session_state["last_bt"] = res
    except (BacktestError, StrategyError) as e:
        st.error(f"回测被拒绝：{e}")
    except Exception as e:  # noqa: BLE001
        st.error(f"回测失败：{type(e).__name__}: {e}")

# ---------------- 结果展示 ----------------
res = st.session_state.get("last_bt")
# 最近存档回退：只取策略回测（组合存档属于投资组合中心，数据结构不同）
bts = [b for b in db.list_backtests() if b["status"] == "success"
       and b["strategy_type"] != "portfolio"]
if res is None and bts:
    st.info(f"显示最近一次存档回测 #{bts[0]['id']}。")
    rec = bts[0]
    res = {"bt_id": rec["id"], "strategy_name": rec["strategy_type"],
           "metrics": json.loads(rec["metrics_json"]),
           "trades": pd.read_csv(rec["trades_path"]).to_dict("records"),
           "daily": pd.read_csv(rec["daily_path"], parse_dates=["date"]).to_dict("records"),
           "warnings": []}
if res is None:
    st.stop()

st.subheader(f"② 回测结果（#{res.get('bt_id', '未存档')} · {res['strategy_name']}）")
m = res["metrics"]
bench = m.get("benchmark", {})

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("总收益率", f"{m['total_return']:.2%}",
          f"{m['excess_return_vs_benchmark']:+.2%} vs 基准" if "excess_return_vs_benchmark" in m else None,
          delta_color="normal")
k2.metric("年化收益率", f"{m['cagr']:.2%}" if m["cagr"] is not None else "—")
k3.metric("年化波动率", f"{m['ann_vol']:.2%}" if m["ann_vol"] is not None else "—")
k4.metric("夏普比率", f"{m['sharpe']:.2f}" if m["sharpe"] is not None else "—")
k5.metric("最大回撤", f"{m['mdd']:.2%}",
          f"{m['mdd_peak']} → {m['mdd_trough']}" if m.get("mdd_peak") else None, delta_color="off")
k6.metric("交易次数", f"{m['n_buys']}买/{m['n_sells']}卖",
          f"费用 {m['total_fees']:.0f} + 滑点 {m['total_slippage_cost']:.0f}", delta_color="off")

if "win_rate" in m:
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("胜率（清仓交易）", f"{m['win_rate']:.1%}")
    w2.metric("平均盈利", f"{m['avg_win']:.0f}")
    w3.metric("平均亏损", f"{m['avg_loss']:.0f}")
    w4.metric("盈亏比", f"{m['profit_loss_ratio']:.2f}" if m.get("profit_loss_ratio") else "—")

b1, b2, b3 = st.columns(3)
b1.metric("基准·买入持有", f"{bench.get('total_return', float('nan')):.2%}")
b2.metric("基准·最大回撤", f"{bench.get('mdd', float('nan')):.2%}")
b3.metric("基准·夏普", f"{bench['sharpe']:.2f}" if bench.get("sharpe") is not None else "—")

st.caption(f"成交模型假设：{m.get('assumptions', '')}")

# ---------------- 图表 ----------------
daily = pd.DataFrame(res["daily"])
daily["date"] = pd.to_datetime(daily["date"])
eq = daily.set_index("date")["equity"] / m.get("final_equity", daily["equity"].iloc[-1]) \
    * (1 + 1)  # 归一化占位——直接用净值/初始资金
norm = daily["equity"] / daily["equity"].iloc[0]

f1 = go.Figure()
f1.add_trace(go.Scatter(x=daily["date"], y=norm, name="策略净值", line=dict(color="#1f77b4", width=2)))
buys = [t for t in res["trades"] if t.get("action") == "buy"]
sells = [t for t in res["trades"] if t.get("action") == "sell"]
bd = pd.to_datetime([t["exec_time"] for t in buys])
sd = pd.to_datetime([t["exec_time"] for t in sells])
if len(bd):
    f1.add_trace(go.Scatter(x=bd, y=[norm[daily["date"].tolist().index(d)] for d in bd],
                            mode="markers", name="买入", marker=dict(symbol="triangle-up", size=11, color="red")))
if len(sd):
    f1.add_trace(go.Scatter(x=sd, y=[norm[daily["date"].tolist().index(d)] for d in sd],
                            mode="markers", name="卖出", marker=dict(symbol="triangle-down", size=11, color="green")))
f1.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), title="策略净值与买卖点",
                 legend=dict(orientation="h"))
st.plotly_chart(f1, use_container_width=True)

f2 = go.Figure()
f2.add_trace(go.Scatter(x=daily["date"], y=daily["drawdown"], fill="tozeroy",
                        name="回撤", line=dict(color="#9467bd")))
f2.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10), title="回撤曲线",
                 yaxis_tickformat=".0%")
st.plotly_chart(f2, use_container_width=True)

if res.get("warnings"):
    with st.expander(f"⚠️ 回测预警与订单记录（{len(res['warnings'])} 条）"):
        for w in res["warnings"][:50]:
            st.markdown(f"- {w}")

st.subheader("③ 交易明细")
trades_df = pd.DataFrame(res["trades"])
if len(trades_df):
    show_cols = [c for c in ["signal_time", "exec_time", "action", "fill_mode", "shares",
                             "base_price", "exec_price", "value", "fee", "pnl",
                             "cash_after", "shares_after", "reason"] if c in trades_df.columns]
    st.dataframe(trades_df[show_cols], use_container_width=True, hide_index=True)
    if res.get("bt_id"):
        st.download_button("⬇️ 导出交易明细 CSV",
                           trades_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"trades_bt{res['bt_id']}_{datetime.now():%Y%m%d_%H%M}.csv")

st.subheader("④ 回测历史（可复现：同数据版本+同配置 → 同结果）")
if bts:
    hist = pd.DataFrame([{
        "ID": b["id"], "名称": b["name"], "策略": b["strategy_type"], "数据集": b["dataset_name"],
        "区间": f"{b['start_date']}~{b['end_date']}",
        "总收益": (f"{json.loads(b['metrics_json']).get('total_return', float('nan')):.2%}"
                  if b["metrics_json"] not in ("", "{}") else "—"),
        "状态": b["status"], "创建": b["created_at"],
    } for b in bts])
    st.dataframe(hist, use_container_width=True, hide_index=True)

# ---------------- PDF 导出（阶段4C） ----------------
pc1, pc2 = st.columns(2)
with pc1:
    if st.button("📄 导出本回测 PDF 报告") and res.get("bt_id"):
        try:
            from quantlab.pdf_report import pdf_backtest_report, PDFFontError, PDFBuildError
            p = pdf_backtest_report(res["bt_id"])
            st.success(f"已生成：`{p.name}`")
            with open(p, "rb") as f:
                st.download_button("⬇️ 下载 PDF", f.read(), file_name=p.name,
                                   mime="application/pdf")
        except PDFFontError as e:
            st.error(f"PDF 中文字体不可用，已拒绝生成（{e}）。请改用 HTML/CSV 导出。")
        except PDFBuildError as e:
            st.error(str(e))
with pc2:
    if st.button("📄 导出 HTML 报告") and res.get("bt_id"):
        from quantlab.reports import backtest_report
        _, p = backtest_report(res["bt_id"])
        with open(p, "rb") as f:
            st.download_button("⬇️ 下载 HTML", f.read(), file_name=p.name, mime="text/html")
