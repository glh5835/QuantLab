"""模块 4：自动数据分析工作台（一键 EDA）。

包含：价格/成交量/收益率时间序列 · 描述性统计 · 相关性热力图（多资产）·
滚动波动率与滚动收益率 · 最大回撤曲线 · 缺失值可视化 · 结果导出 CSV。
所有图表由真实数据实时计算生成（plotly）。
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui_common import init_page, empty_state, synthetic_badge

from quantlab import db
from quantlab import analysis as A
from quantlab.storage import load_csv
from quantlab.config import EXPORT_DIR

init_page("📈 自动分析", "一键探索性数据分析（EDA）· 统计 · 相关性 · 回撤 · 滚动指标 · 导出")

datasets = db.list_datasets()
if not datasets:
    empty_state("没有可分析的数据集。", "先到「数据采集中心」采集或导入数据。")
    st.stop()

st.subheader("① 选择分析对象")
names = {d["name"]: d for d in datasets}
default = [n for n in names if "demo" not in names[n]["source"].lower()][:1] or list(names)[:1]
picked = st.multiselect("选择一个或多个数据集（多选可计算跨资产相关性）",
                        list(names.keys()), default=default[:2])
if not picked:
    empty_state("请至少选择一个数据集。")
    st.stop()

use_clean = st.checkbox("优先使用清洗后数据（若已生成）", value=True,
                        help="未清洗的数据集自动回退到原始数据。")
date_range = st.date_input("分析日期区间",
                           value=[],
                           help="留空则分析全部数据。")

# 载入数据（应用日期过滤）
dfs = {}
for n in picked:
    d = names[n]
    path = (d["clean_path"] if (use_clean and d["cleaned"] and d["clean_path"]) else d["raw_path"]) or d["raw_path"]
    df = load_csv(path)
    if len(date_range) == 2:
        df = df.loc[date_range[0]:date_range[1]]
    if len(df) >= 2:
        dfs[n] = df
if not dfs:
    st.error("所选区间内数据不足（每个数据集至少需要 2 个交易日）。")
    st.stop()

first_name = picked[0]
synthetic_badge(names[first_name])

if st.button("🔬 一键运行分析", type="primary", use_container_width=True):
    st.session_state["eda_done"] = True
if not st.session_state.get("eda_done"):
    st.info("点击「一键运行分析」生成完整 EDA 报告。")
    st.stop()

st.success(f"分析完成：{len(dfs)} 个数据集，共 {sum(len(d) for d in dfs.values())} 行。"
           f"（图表由所选区间内的真实数据实时计算）")

# ---------------- 1. 价格与成交量时间序列 ----------------
st.subheader("② 价格与成交量")
for name, df in dfs.items():
    with st.expander(f"{name}（{df.index.min().date()} ~ {df.index.max().date()}，{len(df)} 行）", expanded=(name == first_name)):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="收盘价", line=dict(color="#1f77b4")))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                          yaxis_title="价格", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        fig_v = px.bar(df, x=df.index, y="volume", title=None)
        fig_v.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="成交量")
        st.plotly_chart(fig_v, use_container_width=True)

# ---------------- 2. 收益率与描述性统计 ----------------
st.subheader("③ 日收益率与描述性统计")
tabs = st.tabs(list(dfs.keys()))
for tab, (name, df) in zip(tabs, dfs.items()):
    r = A.daily_returns(df)
    with tab:
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = px.histogram(x=r.values, nbins=60, title="日收益率分布（直方图）")
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="日收益率")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            stats = A.descriptive_stats(df)
            fmt = stats.copy()
            pct_cols = ["日均收益率", "日收益率标准差", "最小值", "25%分位", "中位数", "75%分位",
                        "最大值", "年化收益率(简单)", "年化波动率", "正收益天数占比"]
            fmt["数值"] = fmt.apply(lambda x: f"{x['数值']:.4%}" if x["指标"] in pct_cols else f"{x['数值']:.4f}", axis=1)
            st.dataframe(fmt, hide_index=True, use_container_width=True)
            st.caption("年化假设：252 个交易日。偏度>0 右偏（大涨尾部），超额峰度>0 尖峰厚尾（极端行情多于正态）。")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=r.index, y=r.values, name="日收益率", line=dict(width=1)))
        fig2.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="收益率")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------- 3. 相关性热力图 ----------------
if len(dfs) >= 2:
    st.subheader("④ 多资产相关性热力图")
    corr = A.correlation_matrix(dfs)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="日收益率相关性矩阵")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("接近 +1 = 两资产同涨同跌（分散化效果差）；接近 0 = 涨跌无关（组合分散化的来源）。")
    st.dataframe(corr.round(3), use_container_width=True)

# ---------------- 4. 滚动指标 ----------------
st.subheader("⑤ 滚动波动率与滚动收益率（窗口 20 日）")
for name, df in dfs.items():
    roll = A.rolling_metrics(df, window=20)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roll.index, y=roll["滚动年化波动率"], name="滚动年化波动率",
                             line=dict(color="#d62728")))
    fig.add_trace(go.Scatter(x=roll.index, y=roll["滚动收益率"], name="滚动20日收益率",
                             line=dict(color="#2ca02c")))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                      title=name, yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- 5. 最大回撤 ----------------
st.subheader("⑥ 历史最大回撤")
for name, df in dfs.items():
    mdd = A.max_drawdown(df)
    if mdd["mdd"] is None:
        st.warning(f"{name}：{mdd['reason']}")
        continue
    col1, col2 = st.columns([3, 2])
    with col1:
        dd = mdd["drawdown_series"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", name="回撤",
                                 line=dict(color="#9467bd")))
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10),
                          title=f"{name} 回撤曲线", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("最大回撤", f"{mdd['mdd']:.2%}")
        st.markdown(f"- 峰值日期：{mdd['peak_date']}")
        st.markdown(f"- 谷底日期：{mdd['trough_date']}")
        st.markdown(f"- 截至数据末尾{'已创新高 ✅' if mdd['recovered'] else '尚未收复失地 ⚠️'}")
        sh = A.annualized_sharpe(df)
        st.metric("年化夏普比率（rf=0）",
                  f"{sh['sharpe']:.2f}" if sh["sharpe"] is not None else "无法计算",
                  help=sh.get("reason", "（日均超额收益 / 日波动率）× √252"))
        st.caption("最大回撤 = 从历史最高点到之后最低点的最大跌幅，衡量最痛苦的亏损深度。")

# ---------------- 6. 缺失值可视化 ----------------
st.subheader("⑦ 数据质量可视化（缺失值）")
mo = {name: A.missing_overview(df) for name, df in dfs.items()}
mo_df = pd.concat(mo, names=["数据集", "字段"]).reset_index()
fig = px.bar(mo_df, x="level_1" if "level_1" in mo_df.columns else "字段", y="缺失占比",
             color="数据集", barmode="group", title="各字段缺失占比")
fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), yaxis_tickformat=".1%")
st.plotly_chart(fig, use_container_width=True)

# ---------------- 7. 导出 ----------------
st.subheader("⑧ 导出分析结果")
ec1, ec2 = st.columns(2)
with ec1:
    if st.button("📤 导出统计表 CSV"):
        out = EXPORT_DIR / f"eda_stats_{datetime.now():%Y%m%d_%H%M%S}.csv"
        pd.concat({n: A.descriptive_stats(df) for n, df in dfs.items()}).to_csv(
            out, index=False, encoding="utf-8-sig")
        st.success(f"已导出：`{out}`")
        with open(out, "rb") as f:
            st.download_button("⬇️ 下载", f.read(), file_name=out.name, mime="text/csv")
with ec2:
    if len(dfs) >= 2 and st.button("📤 导出相关性矩阵 CSV"):
        out = EXPORT_DIR / f"eda_corr_{datetime.now():%Y%m%d_%H%M%S}.csv"
        A.correlation_matrix(dfs).to_csv(out, encoding="utf-8-sig")
        st.success(f"已导出：`{out}`")
        with open(out, "rb") as f:
            st.download_button("⬇️ 下载 ", f.read(), file_name=out.name, mime="text/csv")

st.caption("⚠️ 以上全部结果由所选数据实时计算。模拟数据集的结果不代表真实市场。")
