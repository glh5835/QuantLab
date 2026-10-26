"""阶段 3B 页面：风险分析中心——风险指标 / 回撤区间 / 历史分段 / 压力测试 / 预警。"""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui_common import init_page, empty_state

from quantlab import db
from quantlab import risk as RK
from quantlab import backtest as BT
from quantlab.backtest import BacktestConfig
from quantlab.strategies import BuyHoldStrategy, MACrossoverStrategy, MLSignalStrategy
from quantlab.storage import load_csv
from quantlab.reports import save_report, build_html_report

init_page("🛡️ 风险分析中心", "风险指标（含不可计算状态）· 回撤区间 · 历史分段 · 压力测试 · 预警")

bts = [b for b in db.list_backtests(100) if b["status"] == "success"]
if not bts:
    empty_state("还没有回测记录。", "先到「⚖️ 策略回测」运行一次回测。")
    st.stop()

sel = st.selectbox("选择回测存档",
                   [f"#{b['id']} {b['name']}（{b['strategy_type']}，{b['dataset_name']}）" for b in bts])
rec = db.get_backtest(int(sel.split("#")[1].split(" ")[0]))
m = json.loads(rec["metrics_json"])
cfg_dict = json.loads(rec["config_json"] or "{}")
ds = db.get_dataset(rec["dataset_id"])
path = ds["clean_path"] if (ds["cleaned"] and ds["clean_path"]) else ds["raw_path"]
df = load_csv(path)
daily = pd.read_csv(rec["daily_path"], parse_dates=["date"])
equity = daily.set_index("date")["equity"]
if ds["is_synthetic"]:
    st.warning("🎭 该数据集为**模拟数据**：以下分析用于演示方法论，不代表真实市场事件。")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 风险指标", "📉 回撤区间", "🗓️ 历史分段", "🔥 压力测试", "🔔 风险预警"])

with tab1:
    bench_daily = BT._run_engine(df, BuyHoldStrategy(),
                                 BacktestConfig(**{k: v for k, v in cfg_dict.items()
                                                   if k in BacktestConfig.__dataclass_fields__}))
    bench_eq = pd.Series({pd.Timestamp(d["date"]): d["equity"] for d in bench_daily["daily"]})
    rm = RK.risk_metrics(equity, initial=cfg_dict.get("initial_cash", 100_000),
                         benchmark=bench_eq, total_fees=m.get("total_fees"),
                         n_trades=m.get("n_trades"))
    rows = []
    formulas = {
        "ann_vol": "日收益标准差 × √252", "sharpe": "日均收益/日波动 × √252（rf=0）",
        "downside_vol": "负收益平方均值开方 × √252", "sortino": "日均收益/下行偏差 × √252",
        "calmar": "年化收益 ÷ |最大回撤|", "beta": "cov(策略,基准)/var(基准)",
        "excess_return": "策略总收益 − 基准总收益", "cost_ratio": "总费用 ÷ 初始资金",
    }
    for k, v in rm.items():
        val = v.get("value") if isinstance(v, dict) else v
        reason = v.get("reason", "") if isinstance(v, dict) else ""
        shown = ("—" if val is None else
                 f"{val:.2%}" if k in ("total_return", "cagr", "ann_vol", "mdd",
                                       "excess_return", "cost_ratio", "downside_vol")
                 else f"{val:.3f}")
        rows.append({"指标": k, "数值": shown, "公式/适用条件": formulas.get(k, ""),
                     "状态": "不可计算：" + reason if val is None and reason else "可计算"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("分母无效（零波动/无回撤/无负收益）时明确返回不可计算状态，绝不输出虚假数值。")

with tab2:
    da = RK.drawdown_analysis(equity)
    fig = go.Figure()
    dd = da["drawdown_series"]
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", name="回撤"))
    mx = da["max"]
    fig.add_trace(go.Scatter(x=[pd.Timestamp(mx["trough_date"])], y=[mx["mdd"]],
                             mode="markers", marker=dict(size=12, color="red"),
                             name=f"最大回撤 {mx['mdd']:.1%}"))
    fig.update_layout(height=300, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    eps = pd.DataFrame(da["episodes"])
    if len(eps):
        eps_show = eps.rename(columns={"start_date": "开始", "trough_date": "谷底",
                                       "recovery_date": "恢复", "mdd": "回撤深度",
                                       "recovered": "已恢复"})
        st.dataframe(eps_show, use_container_width=True, hide_index=True)
        n_ongoing = int((~eps["recovered"]).sum())
        if n_ongoing:
            st.warning(f"{n_ongoing} 个回撤在回测结束时仍未恢复（恢复日期留空，不伪造）。")
    else:
        st.success("全程无回撤。")

with tab3:
    freq = st.radio("分段方式", ["按年", "按季度"], horizontal=True)
    periods = RK.period_analysis(equity, freq="Y" if freq == "按年" else "Q")
    pdf = pd.DataFrame(periods)
    if len(pdf):
        pdf["return"] = pdf["return"].map("{:.2%}".format)
        pdf["mdd"] = pdf["mdd"].map("{:.2%}".format)
        pdf["ann_vol"] = pdf["ann_vol"].map(lambda v: f"{v:.1%}" if v is not None and v == v else "—")
        st.dataframe(pdf.rename(columns={"period": "区间", "n_days": "交易日数",
                                         "return": "区间收益", "ann_vol": "年化波动",
                                         "mdd": "区间最大回撤"}),
                     use_container_width=True, hide_index=True)
        best = max(periods, key=lambda p: p["return"])
        worst = min(periods, key=lambda p: p["return"])
        st.info(f"表现差异：最好区间 {best['period']}（{best['return']:+.1%}）vs "
                f"最差 {worst['period']}（{worst['return']:+.1%}）——策略收益是否依赖特殊区间一目了然。")

with tab4:
    st.markdown("**真实参数情景**（重跑真实回测）与**假设情景**（价格冲击，人为设定）分开标注。")
    if st.button("🔥 运行压力测试", type="primary"):
        strat_type = rec["strategy_type"]
        exp_id = rec["experiment_id"]
        if strat_type == "ma_cross":
            sp = json.loads(rec["strategy_params"] or "{}")
            base_strat = lambda: MACrossoverStrategy(int(sp.get("fast", 5)), int(sp.get("slow", 20)))
        elif strat_type == "ml_signal" and exp_id:
            base_strat = lambda: MLSignalStrategy(int(exp_id))
        else:
            base_strat = lambda: BuyHoldStrategy()

        def runner(ov, df_transform=None, date_range=None):
            cfg_kw = {k: v for k, v in cfg_dict.items()
                      if k in BacktestConfig.__dataclass_fields__}
            cfg_kw.update({k: v for k, v in ov.items() if v is not None})
            run = BT._run_engine(df_transform(df) if df_transform else df,
                                 base_strat(), BacktestConfig(**cfg_kw))
            return run["metrics"]

        with st.spinner("重跑各压力情景…"):
            results = RK.run_stress(runner, cfg_dict, RK.default_scenarios(),
                                    price_shock={"start": str(df.index[len(df)//2].date()),
                                                 "end": str(df.index[len(df)//2+5].date()),
                                                 "shock": -0.2})
        st.session_state["stress"] = results
    results = st.session_state.get("stress")
    if results:
        rows = []
        base_ret = m.get("total_return")
        for r in results:
            s = r["stress"]
            tag = "🎭 假设情景（人为设定）" if r["hypothetical"] else "真实参数变化"
            rows.append({"情景": r["name"], "类型": tag,
                         "总收益": f"{s.get('total_return', float('nan')):.2%}",
                         "vs 基准情景": f"{s.get('total_return', float('nan')) - base_ret:+.2%}"
                         if base_ret is not None else "—",
                         "最大回撤": f"{s.get('mdd', float('nan')):.2%}",
                         "夏普": f"{s.get('sharpe', float('nan')):.2f}"
                         if s.get("sharpe") is not None else "—"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"基准情景（原回测）总收益 {base_ret:.2%}。差异为负说明策略对该压力敏感。")

with tab5:
    c1, c2 = st.columns(2)
    dd_th = float(c1.number_input("最大回撤预警阈值", 0.05, 0.9, 0.20, 0.05))
    vol_th = float(c2.number_input("滚动波动率预警阈值（0=不启用）", 0.0, 2.0, 0.0, 0.1))
    warns = RK.risk_warnings(equity, dd_threshold=dd_th,
                             vol_threshold=vol_th if vol_th > 0 else None)
    if warns:
        st.dataframe(pd.DataFrame(warns), use_container_width=True, hide_index=True)
        st.warning("以上为**预警记录**：只提示风险，不改变策略交易行为（与强制平仓严格区分）。")
    else:
        st.success("选定阈值下无预警触发。")

# ---------------- 导出 ----------------
st.divider()
if st.button("📤 导出风险分析 HTML 报告"):
    rm_rows = pd.DataFrame([{"指标": k, "数值": (v.get("value") if isinstance(v, dict) else v)}
                            for k, v in rm.items()])
    dd_rows = pd.DataFrame(da["episodes"]) if da["episodes"] else pd.DataFrame()
    content = build_html_report(
        f"风险分析报告 · {rec['name']}",
        {"回测ID": f"#{rec['id']}", "数据集": rec["dataset_name"],
         "区间": f"{rec['start_date']}~{rec['end_date']}",
         "数据性质": "模拟" if ds["is_synthetic"] else "真实/导入"},
        [("风险指标", rm_rows), ("回撤区间", dd_rows),
         ("历史分段（年度）", pd.DataFrame(RK.period_analysis(equity, "Y")))])
    p = save_report(content, f"risk_report_bt{rec['id']}")
    st.success(f"已导出：`{p}`")
    with open(p, "rb") as f:
        st.download_button("⬇️ 下载报告", f.read(), file_name=p.name,
                           mime="text/html")
