"""阶段 3D 页面：稳健性验证中心——滚动窗口 / 样本外汇总 / 过拟合诊断。"""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui_common import init_page, empty_state

from quantlab import db
from quantlab import robustness as RB
from quantlab.backtest import BacktestConfig
from quantlab.robustness import RobustnessError
from quantlab.features import FEATURES
from quantlab.strategies import MACrossoverStrategy
from quantlab.storage import load_csv
from quantlab.reports import save_report, build_html_report

init_page("🔁 稳健性验证中心", "滚动/扩展窗口 · 每窗口独立训练 · 样本外汇总与过拟合诊断")

datasets = db.list_datasets()
if not datasets:
    empty_state("没有可用的数据集。")
    st.stop()

with st.expander("📖 滚动验证的样本外承诺", expanded=False):
    st.markdown("""
- 窗口按时间顺序执行：**训练区间严格早于测试区间**（有自动化测试保证）。
- ML 模式下**每个窗口独立重新训练**整个 Pipeline（预处理器只学该窗口的训练数据），
  并对训练段末尾 horizon 个样本做 purge 隔离，防止未来标签跨边界泄漏。
- 汇总同时披露：均值、中位数、离散度、**正收益窗口占比**与**最差窗口**——
  绝不用合并平均掩盖个别窗口的大幅亏损。
""")

c1, c2 = st.columns(2)
ds_opts = {f"#{d['id']} {d['name']}（{d['rows']}行）": d["id"] for d in datasets}
dataset_id = ds_opts[c1.selectbox("数据集", list(ds_opts.keys()))]
kind = c2.selectbox("验证对象", ["rule：规则策略（参数冻结，检验跨时段稳定性）",
                                "ml：机器学习信号（每窗口独立重训）"]).split("：")[0]

c3, c4, c5 = st.columns(3)
train_size = int(c3.number_input("训练窗口（交易日）", 60, 500, 150 if kind == "rule" else 250))
test_size = int(c4.number_input("测试窗口（交易日）", 20, 200, 60 if kind == "rule" else 40))
mode = c5.selectbox("窗口模式", ["rolling：固定训练窗", "expanding：扩展训练窗"]).split("：")[0]

strat_factory = None
feats, model_key, horizon, task = None, None, 1, "direction"
if kind == "rule":
    f1, f2 = st.columns(2)
    fast = int(f1.number_input("均线快线", 2, 100, 5))
    slow = int(f2.number_input("均线慢线", 3, 300, 20))
    strat_factory = lambda: MACrossoverStrategy(fast, slow)
else:
    m1, m2, m3 = st.columns(3)
    task = m1.selectbox("任务", ["direction", "return"])
    horizon = int(m2.number_input("预测期限 h", 1, 20, 1))
    model_key = m3.selectbox("模型", ["random_forest", "logistic", "linear"])
    feats = st.multiselect("特征", list(FEATURES.keys()),
                           default=["ret_1", "ret_5", "sma_dev", "rsi_14", "vol_20"])

if st.button("🚀 运行滚动验证", type="primary", use_container_width=True):
    ds = db.get_dataset(dataset_id)
    path = ds["clean_path"] if (ds["cleaned"] and ds["clean_path"]) else ds["raw_path"]
    df = load_csv(path)
    cfg = BacktestConfig()
    try:
        if kind == "rule":
            res = RB.walk_forward_rule(df, strat_factory, cfg=cfg,
                                       train_size=train_size, test_size=test_size, mode=mode)
            RB.save_walk_forward_run(dataset_id, ds["name"], f"ma_cross_rule", res,
                                     {"train": train_size, "test": test_size, "mode": mode})
        else:
            res = RB.walk_forward_ml(df, task, horizon, feats, model_key, cfg=cfg,
                                     train_size=train_size, test_size=test_size, mode=mode)
            RB.save_walk_forward_run(dataset_id, ds["name"], f"ml_{model_key}", res,
                                     {"train": train_size, "test": test_size, "mode": mode,
                                      "task": task, "horizon": horizon})
        st.session_state["wf_res"] = res
    except RobustnessError as e:
        st.error(f"滚动验证被拒绝：{e}")
    except Exception as e:  # noqa: BLE001
        st.error(f"运行失败：{type(e).__name__}: {e}")

res = st.session_state.get("wf_res") or None
if res is None:
    runs = db.list_walk_forward(10)
    if runs:
        st.info(f"显示最近一次存档（#{runs[0]['id']}）。")
        res = {"windows": json.loads(runs[0]["windows_json"]),
               "summary": json.loads(runs[0]["summary_json"]),
               "mode": runs[0]["mode"], "strategy": "archived",
               "n_windows": runs[0]["n_windows"], "n_skipped": 0}
if res is None:
    st.info("尚无滚动验证结果。配置上方参数并运行。")
    st.stop()

s = res["summary"]
st.subheader(f"① 样本外汇总（{res['n_windows']} 个窗口"
             + (f"，{res.get('n_skipped', 0)} 个跳过）" if res.get("n_skipped") else "") + "）")
k1, k2, k3, k4, k5 = st.columns(5)
if s.get("note"):
    st.warning(s["note"])
else:
    k1.metric("平均窗口收益", f"{s['mean_return']:.2%}")
    k2.metric("中位数", f"{s['median_return']:.2%}")
    k3.metric("正收益窗口占比", f"{s['positive_ratio']:.0%}")
    k4.metric("最差窗口", f"{s['worst_window_return']:.2%}",
              s.get("worst_window_range", ""), delta_color="off")
    k5.metric("收益标准差", f"{s['std_return']:.2%}" if s["std_return"] is not None else "—")

rows = []
for w in res["windows"]:
    if "error" in w:
        rows.append({"训练区间": "—", "测试区间": w["test_range"], "测试收益": "跳过",
                     "夏普": "—", "回撤": "—", "交易数": "—", "备注": w["error"]})
        continue
    m = w["test_metrics"]
    rows.append({"训练区间": w.get("train_range", "—"), "测试区间": w["test_range"],
                 "测试收益": f"{m.get('total_return', float('nan')):.2%}",
                 "夏普": f"{m.get('sharpe', float('nan')):.2f}"
                 if m.get("sharpe") is not None else "—",
                 "回撤": f"{m.get('mdd', float('nan')):.2%}",
                 "交易数": w.get("n_trades", "—"),
                 "备注": f"purge {w.get('purged_n', '—')} / 训练拟合 {w['train_fit_score']:.2f}"
                 if "train_fit_score" in w else ""})
wdf = pd.DataFrame(rows)
st.subheader("② 各窗口明细")
st.dataframe(wdf, use_container_width=True, hide_index=True)

valid = [w for w in res["windows"] if "error" not in w and
         w["test_metrics"].get("total_return") is not None]
if valid:
    fig = go.Figure(go.Bar(
        x=[w["test_range"] for w in valid],
        y=[w["test_metrics"]["total_return"] for w in valid],
        marker_color=["green" if w["test_metrics"]["total_return"] >= 0 else "red"
                      for w in valid]))
    fig.update_layout(height=320, title="各窗口样本外收益（红=亏损窗口必须可见）",
                      yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("③ 过拟合与稳健性诊断")
warns = RB.overfit_diagnosis(res)
if warns:
    for w in warns:
        st.warning(f"⚠️ {w}")
else:
    st.success("未触发过拟合诊断规则（窗口表现一致为正且离散度可控）。")
st.caption("提示：滚动样本外结果是稳健性证据，不是收益保证；若此前已反复查看同一测试段"
           "并据此调参，其独立性已受损，应视为已参与筛选。")

if st.button("📤 导出稳健性报告（HTML）"):
    content = build_html_report(
        "滚动验证稳健性报告", {"模式": res.get("mode", ""), "窗口数": res["n_windows"],
                            "数据集": db.get_dataset(dataset_id)["name"]},
        [("样本外汇总", pd.DataFrame([s])), ("窗口明细", wdf)],
        warnings=warns or None)
    p = save_report(content, "walk_forward_report")
    st.success(f"已导出：`{p}`")
    with open(p, "rb") as f:
        st.download_button("⬇️ 下载", f.read(), file_name=p.name, mime="text/html")
