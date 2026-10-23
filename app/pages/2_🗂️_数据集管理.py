"""模块 3：数据集管理与数据质量。

功能：列表/搜索/筛选 · 详情与版本记录 · 质量检查与评分 · 显式清洗（留痕）·
清洗前后对比 · 原始/清洗数据分开保存展示 · 删除。
"""
import json

import pandas as pd
import streamlit as st

from ui_common import init_page, empty_state, synthetic_badge

from quantlab import db
from quantlab.quality import check_quality, quality_summary_json
from quantlab.cleaning import run_cleaning, OPERATIONS
from quantlab.storage import load_csv

init_page("🗂️ 数据集管理", "数据集列表 · 质量检查与评分 · 显式清洗与对比 · 版本记录")

datasets = db.list_datasets()

# ---------------- 搜索与筛选 ----------------
c1, c2 = st.columns([2, 1])
kw = c1.text_input("🔍 搜索（名称/代码）", "")
src = c2.selectbox("按来源筛选", ["全部", "yfinance", "csv_import", "demo"])
rows = db.list_datasets(source=None if src == "全部" else src, keyword=kw or None)

if not rows:
    empty_state("没有匹配的数据集。", "先到「数据采集中心」创建数据集，或调整搜索条件。")
    st.stop()

# ---------------- 列表 ----------------
st.subheader(f"数据集列表（{len(rows)} 个）")
df = pd.DataFrame([{
    "ID": d["id"], "名称": d["name"], "代码": d["symbol"],
    "性质": "🎭 模拟" if d["is_synthetic"] else "真实/导入",
    "行数": d["rows"], "时间范围": f"{d['start_date']} ~ {d['end_date']}",
    "质量评分": f"{d['quality_score']:.0f}" if d["quality_score"] is not None else "未检查",
    "清洗": "✅ 已清洗" if d["cleaned"] else "—", "版本": f"v{d['version']}",
} for d in rows])
st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------- 详情与操作 ----------------
st.divider()
st.subheader("数据集详情与操作")
opts = {f"#{d['id']} {d['name']}（v{d['version']}）": d["id"] for d in rows}
sel = st.selectbox("选择数据集", list(opts.keys()))
ds = db.get_dataset(opts[sel])
synthetic_badge(ds)

tc1, tc2, tc3, tc4 = st.columns(4)
tc1.metric("数据行数", ds["rows"])
tc2.metric("时间范围", f"{ds['start_date']}\n~ {ds['end_date']}" if ds["rows"] else "—")
tc3.metric("质量评分", f"{ds['quality_score']:.0f}" if ds["quality_score"] is not None else "未检查")
tc4.metric("当前版本", f"v{ds['version']}（更新于 {ds['updated_at']}）")

st.markdown(f"**来源与许可**：{ds['license_note'] or '—'}")
st.markdown(f"**字段说明**：{ds['field_desc'] or '—'}")
st.markdown(f"**原始数据文件**：`{ds['raw_path'] or '无'}`")
st.markdown(f"**清洗数据文件**：`{ds['clean_path'] or '尚未生成（先执行质量检查，再选择清洗操作）'}`")

tab_data, tab_quality, tab_clean, tab_ver = st.tabs(["👀 数据预览", "🧪 质量检查", "🧹 数据清洗", "📜 版本与记录"])

# ---------------- 数据预览 ----------------
with tab_data:
    if ds["raw_path"]:
        use_clean = False
        if ds["cleaned"] and ds["clean_path"]:
            use_clean = st.checkbox("预览清洗后数据（取消勾选查看原始数据）", value=True)
        path = ds["clean_path"] if (use_clean and ds["clean_path"]) else ds["raw_path"]
        dfx = load_csv(path)
        st.caption(f"正在显示：`{path}`")
        st.dataframe(dfx.head(100), use_container_width=True)
        st.download_button("⬇️ 下载当前视图 CSV",
                           dfx.to_csv(index_label="trade_date").encode("utf-8-sig"),
                           file_name=f"{ds['name']}.csv", mime="text/csv")
    else:
        st.warning("该数据集还没有数据文件。")

# ---------------- 质量检查 ----------------
with tab_quality:
    col_btn, col_info = st.columns([1, 2])
    if col_btn.button("🧪 运行质量检查", use_container_width=True):
        if not ds["raw_path"]:
            st.error("没有原始数据文件，无法检查。")
        else:
            with st.spinner("正在逐项检查…"):
                report = check_quality(load_csv(ds["raw_path"]))
                db.update_dataset(ds["id"], quality_score=report["score"],
                                  quality_report=quality_summary_json(report))
            st.success(f"检查完成：评分 **{report['score']:.0f} / 100**，发现 {report['issue_count']} 类问题。")
            st.rerun()
    col_info.caption("检查项：缺失值 · 重复记录 · 价格≤0 · OHLC 逻辑矛盾 · 极端涨跌幅 · 日期间隔 · 零成交占比。"
                     "检查只报告问题，不修改任何数据。")

    ds = db.get_dataset(opts[sel])  # 重新读取（可能刚更新）
    if ds["quality_report"]:
        report = json.loads(ds["quality_report"])
        sc = report["score"]
        (st.success if sc >= 90 else st.warning if sc >= 70 else st.error)(
            f"## 质量评分：{sc:.0f} / 100\n\n"
            f"扣分规则：{ {k: -v for k, v in report['weights'].items()} }（同类型只扣一次，明细见下）")
        st.markdown(f"检查范围：{report['rows']} 行 · {report['date_range'][0]} ~ {report['date_range'][1]}")
        if report["issues"]:
            st.markdown("### 问题清单")
            for i, iss in enumerate(report["issues"], 1):
                sev_icon = {"严重": "🔴", "一般": "🟡", "提醒": "🔵"}.get(iss["severity"], "⚪")
                with st.expander(f"{sev_icon} 问题{i}：{iss['detail']}（{iss['count']} 处）"):
                    st.markdown(f"**严重级别**：{iss['severity']}")
                    st.markdown(f"**示例日期**：{', '.join(iss['example_dates'])}")
                    st.markdown(f"**建议处理**：{iss['suggestion']}")
        else:
            st.success("未发现数据质量问题。")
    else:
        empty_state("该数据集尚未运行质量检查。", "点击上方「运行质量检查」按钮。")

# ---------------- 数据清洗 ----------------
with tab_clean:
    st.markdown("**原则**：清洗是显式操作——你勾选每一项并看到依据；原始数据永远保留，清洗结果另存；"
                "所有实际执行的操作都会记录在案。")
    if not ds["quality_report"]:
        st.info("建议先运行质量检查，再决定勾选哪些清洗操作。")
    ops = st.multiselect("选择要执行的清洗操作（可多选）",
                         list(OPERATIONS.keys()),
                         format_func=lambda k: f"{OPERATIONS[k][0]} —— {OPERATIONS[k][1]}")
    if st.button("🧹 执行清洗（另存为新文件，不修改原始数据）", disabled=not ops):
        with st.spinner("正在执行清洗…"):
            try:
                rep = run_cleaning(ds["id"], ops)
                st.success(f"清洗完成并另存为：`{rep['clean_path']}`")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"清洗失败：{e}")

    ds = db.get_dataset(opts[sel])
    if ds["cleaned"] and ds["cleaning_log"]:
        log = json.loads(ds["cleaning_log"])
        st.markdown("### 清洗报告（最近一次）")
        c1, c2, c3 = st.columns(3)
        c1.metric("清洗前行数", log["before_rows"])
        c2.metric("清洗后行数", log["after_rows"])
        c3.metric("缺失值变化", f"{log['before_missing']} → {log['after_missing']}")
        st.markdown(f"执行时间：{log['cleaned_at']}")
        for ch in log["changes"]:
            st.markdown(f"- ✅ **{ch['title']}**：影响 {ch['affected']} 行。{ch['reason']}")
        st.markdown("### 清洗前后差异对比")
        if ds["raw_path"] and ds["clean_path"]:
            raw, clean = load_csv(ds["raw_path"]), load_csv(ds["clean_path"])
            diff = pd.DataFrame({
                "指标": ["行数", "缺失值总数", "重复行", "价格≤0 行数", "OHLC 矛盾行数"],
                "原始数据": [len(raw), int(raw.isna().sum().sum()), int(raw.duplicated().sum()),
                             int(((raw[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum()),
                             int(((raw["high"] < raw[["open", "close", "low"]].max(axis=1)) |
                                  (raw["low"] > raw[["open", "close", "high"]].min(axis=1))).sum())],
                "清洗后": [len(clean), int(clean.isna().sum().sum()), int(clean.duplicated().sum()),
                           int(((clean[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum()),
                           int(((clean["high"] < clean[["open", "close", "low"]].max(axis=1)) |
                                (clean["low"] > clean[["open", "close", "high"]].min(axis=1))).sum())],
            })
            st.dataframe(diff, use_container_width=True, hide_index=True)

# ---------------- 版本与删除 ----------------
with tab_ver:
    st.markdown(f"- 数据集 **{ds['name']}** 当前版本 v{ds['version']}，创建于 {ds['created_at']}，"
                f"最近更新 {ds['updated_at']}。")
    st.markdown("- 每次重新采集/导入同名数据集，版本号 +1（覆盖数据文件但保留版本计数与时间戳）。")
    st.markdown(f"- 历史采集任务可在「数据采集中心 → 任务面板」查看日志。")
    st.divider()
    if st.button("🗑️ 删除此数据集（含数据文件，不可恢复）"):
        res = db.delete_dataset(ds["id"])
        st.warning(f"已删除数据集 “{res['name']}” 与文件：{res['deleted_files']}")
        st.rerun()
