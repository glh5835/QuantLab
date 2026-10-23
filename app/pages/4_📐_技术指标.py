"""模块 5（阶段2-A）：技术指标计算中心。

选数据集 → 选指标与参数 → 计算 → 价格叠加图 / 独立指标图 / 结果表 / CSV 导出。
所有计算走 quantlab/indicators.py 统一接口（页面不含计算逻辑）。
"""
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui_common import init_page, empty_state, synthetic_badge

from quantlab import db
from quantlab import indicators as IND
from quantlab.indicators import IndicatorError
from quantlab.storage import load_csv
from quantlab.config import EXPORT_DIR

init_page("📐 技术指标", "统一指标接口 · 参数可调 · 价格叠加与独立指标图 · CSV 导出")

datasets = db.list_datasets()
if not datasets:
    empty_state("没有可分析的数据集。", "先到「数据采集中心」采集或导入数据。")
    st.stop()

# ---------------- 选择区 ----------------
c1, c2 = st.columns([2, 1])
opts = {f"#{d['id']} {d['name']}（{d['symbol']}）": d["id"] for d in datasets}
ds = db.get_dataset(opts[c1.selectbox("选择数据集", list(opts.keys()))])
synthetic_badge(ds)
path = ds["clean_path"] if (ds["cleaned"] and ds["clean_path"]) else ds["raw_path"]
df = load_csv(path)
date_range = c2.date_input("日期区间", [], help="留空=全部")

if len(date_range) == 2:
    df = df.loc[date_range[0]:date_range[1]]
if len(df) < 5:
    st.error("所选区间数据不足 5 个交易日，无法计算指标。")
    st.stop()
st.caption(f"当前数据：{len(df)} 行 · {df.index.min().date()} ~ {df.index.max().date()} · `{path}`")

# ---------------- 指标选择与参数 ----------------
st.subheader("选择指标与参数")
cols = st.columns(4)
sel = {}
sel["sma"] = cols[0].checkbox("SMA 简单均线", value=True)
sel["ema"] = cols[1].checkbox("EMA 指数均线")
sel["bollinger"] = cols[2].checkbox("布林带")
sel["rsi"] = cols[3].checkbox("RSI")
sel["macd"] = cols[0].checkbox("MACD", value=True)
sel["rolling_vol"] = cols[1].checkbox("滚动年化波动率")
sel["cum_returns"] = cols[2].checkbox("多期累计收益率")
sel["rolling_mdd"] = cols[3].checkbox("滚动最大回撤")

pc1, pc2, pc3, pc4, pc5 = st.columns(5)
p_sma = pc1.number_input("SMA 窗口", 2, 500, 20)
p_ema = pc2.number_input("EMA 窗口", 2, 500, 20)
p_rsi = pc3.number_input("RSI 窗口", 2, 100, 14)
p_bb_w = pc4.number_input("布林带窗口", 2, 500, 20)
p_bb_k = pc5.number_input("布林带带宽 k", 0.1, 5.0, 2.0, 0.1)
p_macd_f = pc1.number_input("MACD 快线", 2, 100, 12)
p_macd_s = pc2.number_input("MACD 慢线", 3, 200, 26)
p_macd_g = pc3.number_input("MACD 信号", 2, 100, 9)
p_cum = pc4.number_input("累计收益率期数", 1, 250, 5)
p_mdd_w = pc5.number_input("回撤窗口", 2, 500, 60)

# ---------------- 计算 ----------------
st.subheader("计算结果")
results = {}
errors = []
if sel["sma"]:   results["SMA"] = IND.compute_indicator(df, "sma", (int(p_sma),))
if sel["ema"]:   results["EMA"] = IND.compute_indicator(df, "ema", (int(p_ema),))
if sel["bollinger"]: results["布林带"] = IND.compute_indicator(df, "bollinger", (int(p_bb_w), float(p_bb_k)))
if sel["rsi"]:   results["RSI"] = IND.compute_indicator(df, "rsi", (int(p_rsi),))
if sel["macd"]:  results["MACD"] = IND.compute_indicator(df, "macd", (int(p_macd_f), int(p_macd_s), int(p_macd_g)))
if sel["rolling_vol"]: results["滚动波动率"] = IND.compute_indicator(df, "rolling_vol", (20,))
if sel["cum_returns"]: results["累计收益率"] = IND.compute_indicator(df, "cum_returns", (int(p_cum),))
if sel["rolling_mdd"]: results["滚动回撤"] = IND.compute_indicator(df, "rolling_mdd", (int(p_mdd_w),))

# 图 1：价格 + 叠加指标
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="收盘价", line=dict(color="#1f77b4", width=2)))
overlay_count = 0
for name, res in results.items():
    if isinstance(res, pd.Series) and name in ("SMA", "EMA"):
        valid = res.dropna()
        if len(valid) == 0:
            errors.append(f"{name}：窗口 ≥ 数据长度，无法计算（保留 NaN，不补零）。")
            continue
        fig.add_trace(go.Scatter(x=valid.index, y=valid.values, name=name,
                                 line=dict(width=1.4, dash="dot" if name == "EMA" else "solid")))
        overlay_count += 1
    elif isinstance(res, pd.DataFrame) and name == "布林带":
        for col, dash in (("bb_mid", "solid"), ("bb_upper", "dash"), ("bb_lower", "dash")):
            v = res[col].dropna()
            if len(v):
                fig.add_trace(go.Scatter(x=v.index, y=v.values, name=f"布林{col}",
                                         line=dict(width=1, dash=dash)))
        overlay_count += 1
fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="价格",
                  legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# 图 2：独立指标图
def _sub_chart(title, series_dict, height=240, yformat=None):
    f = go.Figure()
    for nm, s in series_dict.items():
        v = s.dropna() if isinstance(s, pd.Series) else s
        f.add_trace(go.Scatter(x=v.index if isinstance(v, pd.Series) else v.index,
                               y=v.values if isinstance(v, pd.Series) else v[s.name],
                               name=nm, line=dict(width=1.4)))
    f.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), title=title,
                    legend=dict(orientation="h"), yaxis_tickformat=yformat)
    st.plotly_chart(f, use_container_width=True)

if sel["rsi"]:
    rsi = results["RSI"].dropna()
    if len(rsi) == 0:
        errors.append("RSI：窗口不足，前若干日按定义为 NaN。")
    else:
        f = go.Figure()
        f.add_trace(go.Scatter(x=rsi.index, y=rsi.values, name="RSI", line=dict(width=1.4)))
        f.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="70 超买")
        f.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="30 超卖")
        f.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10), title="RSI 相对强弱")
        st.plotly_chart(f, use_container_width=True)

if sel["macd"]:
    m = results["MACD"]
    has_data = m["dif"].notna().any()
    if not has_data:
        errors.append("MACD：数据长度不足慢线窗口，无法计算。")
    else:
        colors = ["red" if v >= 0 else "green" for v in m["hist"].dropna()]
        f = go.Figure()
        f.add_trace(go.Bar(x=m.index, y=m["hist"], name="MACD 柱（2×(DIF−DEA)）",
                           marker_color=["red" if (v == v and v >= 0) else "green" for v in m["hist"]]))
        f.add_trace(go.Scatter(x=m.index, y=m["dif"], name="DIF", line=dict(width=1.3)))
        f.add_trace(go.Scatter(x=m.index, y=m["dea"], name="DEA", line=dict(width=1.3)))
        f.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10), title="MACD",
                        legend=dict(orientation="h"))
        st.plotly_chart(f, use_container_width=True)

if sel["rolling_vol"]:
    v = results["滚动波动率"].dropna()
    if len(v) == 0:
        errors.append("滚动波动率：窗口不足。")
    else:
        _sub_chart("滚动年化波动率（20 日）", {"波动率": v}, yformat=".0%")

if sel["cum_returns"]:
    v = results["累计收益率"].dropna()
    if len(v):
        _sub_chart(f"{int(p_cum)} 期累计收益率", {"累计收益": v}, yformat=".0%")

if sel["rolling_mdd"]:
    v = results["滚动回撤"].dropna()
    if len(v):
        _sub_chart(f"滚动最大回撤（{int(p_mdd_w)} 日窗口）", {"回撤": v}, yformat=".0%")

# ---------------- 提醒 / 表格 / 导出 ----------------
st.subheader("数据质量与计算说明")
st.markdown(f"""
- **指标口径**：{IND.INDICATORS['rsi']['desc']}；MACD 柱 = 2×(DIF−DEA)（国内常用口径）；
  布林带用滚动**总体**标准差（ddof=0）。
- **因果性**：所有指标只使用当前及之前的数据（截断等价性有自动化测试保证），不存在未来数据。
- **暖机期**：滚动窗口初期按指标定义保留 NaN（不补零），图表自动跳过这些日期。
""")
if errors:
    for e in errors:
        st.warning(e)
if "SMA" in results or "EMA" in results:
    st.info("💡 均线交叉策略可直接在「⚖️ 策略回测」页使用，成交时序为 t 收盘信号 → t+1 开盘成交。")

table_cols = {}
for name, res in results.items():
    if isinstance(res, pd.Series):
        table_cols[name] = res
    else:
        for c in res.columns:
            table_cols[f"{name}.{c}"] = res[c]
if table_cols:
    tbl = pd.concat([df[["close", "volume"]].rename(columns={"close": "收盘", "volume": "成交量"}),
                     pd.DataFrame(table_cols)], axis=1)
    st.dataframe(tbl.tail(300), use_container_width=True)
    st.download_button("⬇️ 导出指标数据 CSV",
                       tbl.to_csv(index_label="trade_date").encode("utf-8-sig"),
                       file_name=f"indicators_{ds['name']}_{datetime.now():%Y%m%d_%H%M}.csv",
                       mime="text/csv")
