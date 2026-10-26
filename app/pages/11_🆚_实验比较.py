"""阶段 3E 页面：实验比较中心——多回测横向对比 / 条件差异披露 / 报告导出。"""
import json

import pandas as pd
import streamlit as st

from ui_common import init_page, empty_state

from quantlab import db
from quantlab.reports import compare_backtests, backtest_report, save_report, build_html_report
from quantlab import reports as RP

init_page("🆚 实验比较中心", "多回测横向对比 · 条件一致性披露 · HTML/CSV 导出")

bts = [b for b in db.list_backtests(100) if b["status"] == "success"]
if len(bts) < 2:
    empty_state("比较至少需要 2 条成功回测记录。", "先在「⚖️ 策略回测」或「📁 投资组合中心」各跑一次。")
    st.stop()

opts = {f"#{b['id']} {b['name']}（{b['strategy_type']}，{b['dataset_name']}，"
        f"{b['start_date']}~{b['end_date']}）": b["id"] for b in bts}
picked = st.multiselect("选择要比较的回测（≥2）", list(opts.keys()),
                        default=list(opts.keys())[:2])
if len(picked) < 2:
    st.info("请至少选择 2 条。")
    st.stop()
ids = [opts[k] for k in picked]

try:
    cmp_df, diffs = compare_backtests(ids)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.subheader("① 指标对比")
st.dataframe(cmp_df, use_container_width=True, hide_index=True)

st.subheader("② 比较条件检查")
if diffs:
    for d in diffs:
        st.warning(f"⚠️ {d}")
    st.caption("条件不一致时，上表排名不能当作公平比较——请先统一数据集/区间/成本假设。")
else:
    st.success("数据集、回测区间与成本假设一致，可公平比较。")

st.subheader("③ 导出")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("📤 对比表 CSV"):
        p = save_report(cmp_df.to_csv(index=False).encode("utf-8-sig").decode("utf-8"),
                        "experiment_compare", suffix=".csv")
        with open(p, "rb") as f:
            st.download_button("⬇️ 下载 CSV", f.read(), file_name=p.name, mime="text/csv")
        st.success(f"已生成：`{p}`")
with c2:
    if st.button("📤 对比 HTML 报告"):
        content = build_html_report(
            "多实验对比报告",
            {"参与比较": f"#{', #'.join(str(i) for i in ids)}",
             "数据性质": RP.dataset_nature_badge(
                 [b["dataset_name"] for b in bts if b["id"] in ids])},
            [("指标对比", cmp_df),
             ("条件差异", pd.DataFrame({"差异": diffs}) if diffs else
              pd.DataFrame([{"说明": "条件一致，可公平比较"}]))],
            warnings=diffs or None)
        p = save_report(content, "experiment_compare")
        with open(p, "rb") as f:
            st.download_button("⬇️ 下载 HTML", f.read(), file_name=p.name, mime="text/html")
        st.success(f"已生成：`{p}`")
with c3:
    first_id = ids[0]
    if st.button(f"📤 首条（#{first_id}）完整研究报告"):
        _, p = backtest_report(first_id)
        with open(p, "rb") as f:
            st.download_button("⬇️ 下载报告", f.read(), file_name=p.name, mime="text/html")
        st.success(f"已生成：`{p}`")

# ---------------- 全局实验档案索引 ----------------
st.divider()
st.subheader("④ 实验档案索引（全部可追溯）")
rows = []
for b in bts[:30]:
    m = json.loads(b["metrics_json"] or "{}")
    rows.append({"类型": "回测", "ID": b["id"], "名称": b["name"],
                 "策略": b["strategy_type"], "数据集": b["dataset_name"],
                 "总收益": f"{m.get('total_return', float('nan')):.2%}"
                 if m.get("total_return") is not None else "—",
                 "创建": b["created_at"]})
for e in db.list_experiments(30):
    m = json.loads(e["metrics_json"] or "{}")
    rows.append({"类型": "ML实验", "ID": e["id"], "名称": e["name"],
                 "策略": f"{e['task']}/{e['model_type']}", "数据集": e["dataset_name"],
                 "总收益": "—", "创建": e["created_at"]})
for r in db.list_optimization_runs(10):
    rows.append({"类型": "参数优化", "ID": r["id"], "名称": r["strategy_type"],
                 "策略": r["objective"], "数据集": r["dataset_name"],
                 "总收益": "—", "创建": r["created_at"]})
for w in db.list_walk_forward(10):
    rows.append({"类型": "滚动验证", "ID": w["id"], "名称": w["strategy_type"],
                 "策略": w["mode"], "数据集": w["dataset_name"],
                 "总收益": "—", "创建": w["created_at"]})
st.dataframe(pd.DataFrame(rows).sort_values("创建", ascending=False),
             use_container_width=True, hide_index=True)
st.caption("所有记录只增不改；修改参数重跑会生成新记录，历史结果永不覆盖。")
