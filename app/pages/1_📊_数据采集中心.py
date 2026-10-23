"""模块 2：数据源与自动采集中心。

功能：数据源可用性检查 / 历史下载与增量更新 / 任务开始·取消·重试·日志 /
进度与耗时 / 来源与许可信息 / CSV 导入导出。
"""
import time
from datetime import date

import pandas as pd
import streamlit as st

from ui_common import init_page, status_label, empty_state

from quantlab import db, collector
from quantlab.datasources import base as ds_base
from quantlab.datasources.csv_adapter import CsvImportAdapter
from quantlab.storage import save_raw

init_page("📊 数据采集中心", "数据源配置 · 历史下载与增量更新 · 任务管理 · CSV 导入导出")

st.subheader("① 发起采集")

tab_yf, tab_demo, tab_csv = st.tabs(["🌐 在线数据源（Yahoo Finance）", "🎭 示例数据（随机模拟）", "📁 CSV 文件导入"])

# ---------------- 在线数据源 ----------------
with tab_yf:
    adapters = ds_base.ADAPTERS
    yf_adapter = adapters.get("yfinance")
    if yf_adapter is None:
        st.error("yfinance 未安装。请在终端运行：`pip install yfinance` 后重启应用。"
                 "在此之前可使用「示例数据」或「CSV 导入」。")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔌 检查数据源可用性"):
                with st.spinner("正在向 Yahoo Finance 发送测试请求…"):
                    res = yf_adapter.check_availability()
                (st.success if res["ok"] else st.error)(res["message"])
        with col2:
            st.caption(f"**许可说明**：{yf_adapter.license_note}")
        with st.form("collect_yf"):
            symbol = st.text_input("资产代码",
                                   help="美股如 AAPL/MSFT；上证如 600519.SS；港股如 00700.HK；ETF 如 SPY；指数如 ^GSPC")
            c1, c2, c3 = st.columns(3)
            start = c1.text_input("开始日期", value="2024-01-01")
            end = c2.text_input("结束日期", value=date.today().isoformat())
            asset_type = c3.selectbox("资产类型", ["stock", "etf", "index", "other"])
            if st.form_submit_button("🚀 开始采集（后台任务）", use_container_width=True):
                if not symbol.strip():
                    st.error("请输入资产代码。")
                else:
                    task_id = collector.start_collection(symbol.strip(), "yfinance", start, end, asset_type)
                    st.success(f"任务 #{task_id} 已创建，正在后台下载… 请在下方任务面板查看进度。")
                    st.rerun()

# ---------------- 示例数据 ----------------
with tab_demo:
    st.warning("以下数据由内置随机数生成器**模拟生成**，仅用于无网络时演示完整流程，不是真实行情。")
    demo = ds_base.ADAPTERS["demo"]
    with st.form("collect_demo"):
        symbol = st.selectbox("选择演示资产", list(demo.DEMO_ASSETS),
                              format_func=lambda s: f"{s}（年化收益≈{demo.DEMO_ASSETS[s][0]:.0%}，波动≈{demo.DEMO_ASSETS[s][1]:.0%}）")
        c1, c2 = st.columns(2)
        start = c1.text_input("开始日期 ", value="2023-01-01")
        end = c2.text_input("结束日期 ", value="2025-12-31")
        if st.form_submit_button("🚀 生成并采集示例数据", use_container_width=True):
            task_id = collector.start_collection(symbol, "demo", start, end, "stock")
            st.success(f"任务 #{task_id} 已创建。")
            st.rerun()

# ---------------- CSV 导入 ----------------
with tab_csv:
    st.markdown("上传本地行情 CSV（支持中文列：日期/开盘/最高/最低/收盘/成交量，或英文 date/open/high/low/close/volume）。"
                "导入的数据会注册为数据集，来源标记为 `csv_import`。")
    up = st.file_uploader("选择 CSV 文件", type=["csv"])
    if up is not None and st.button("📥 导入并注册数据集"):
        try:
            tmp = pd.read_csv(up)  # 先读出内容（上传对象只能读一次）
            csv_name = up.name.rsplit(".", 1)[0]
            import tempfile, os
            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="") as f:
                tmp.to_csv(f, index=False)
                tmp_path = f.name
            df = CsvImportAdapter().import_csv(tmp_path)
            os.unlink(tmp_path)
            ds_id = db.register_dataset(f"{csv_name}_daily", csv_name, source="csv_import",
                                        asset_type="other", is_synthetic=0,
                                        license_note="用户本地导入文件，来源由用户负责。",
                                        field_desc=CsvImportAdapter.field_desc)
            path = save_raw(df, f"{csv_name}_daily")
            db.update_dataset(ds_id, raw_path=path,
                              start_date=str(df.index.min().date()), end_date=str(df.index.max().date()),
                              rows=len(df))
            st.success(f"导入成功：{len(df)} 行（{df.index.min().date()} ~ {df.index.max().date()}），"
                       f"已注册数据集 “{csv_name}_daily”。")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"导入失败：{e}")

st.divider()

# ---------------- 任务面板 ----------------
st.subheader("② 采集任务（进度 · 日志 · 取消 · 重试）")
from quantlab import tasks as task_sys

tasks = db.list_tasks(limit=10, task_type="collect")
if tasks:
    for t in tasks:
        with st.expander(f"任务 #{t['id']} · {t['target']} · {status_label(t['status'])}"
                         + (f" · 进度 {t['progress']:.0%}" if t["status"] == "running" else "")):
            st.markdown(f"- **参数**：`{t['params']}`")
            st.markdown(f"- **最新消息**：{t['message'] or '—'}")
            if t["status"] == "running":
                # 耗时
                import datetime as dt
                try:
                    s = dt.datetime.strptime(t["started_at"], "%Y-%m-%d %H:%M:%S")
                    st.markdown(f"- **已耗时**：{(dt.datetime.now()-s).total_seconds():.0f} 秒")
                except Exception:  # noqa: BLE001
                    pass
                col1, col2 = st.columns(2)
                if col1.button("🚫 取消任务", key=f"cancel_{t['id']}"):
                    task_sys.cancel_task(t["id"])
                    st.rerun()
                col2.progress(min(t["progress"], 1.0))
            elif t["status"] == "failed":
                st.error(f"**失败原因**：\n\n```\n{(t['error'] or '无详情')[:800]}\n```")
                if st.button("🔁 重试任务", key=f"retry_{t['id']}"):
                    import json as _json
                    p = _json.loads(t["params"] or "{}")
                    new_id = collector.start_collection(p.get("symbol", t["target"]), p.get("source", "yfinance"),
                                                        p.get("start", "2024-01-01"), p.get("end"), p.get("asset_type", "stock"))
                    st.success(f"已创建重试任务 #{new_id}")
                    st.rerun()
            if t["log"]:
                st.text_area("运行日志（只读）", t["log"], height=180, disabled=True, key=f"log_{t['id']}")
    # 有任务在运行时自动刷新
    if any(t["status"] == "running" for t in tasks):
        time.sleep(2)
        st.rerun()
else:
    empty_state("暂无采集任务。", "在上方选择数据源发起第一次采集。")

# ---------------- 数据导出 ----------------
st.divider()
st.subheader("③ 数据导出")
from quantlab.config import EXPORT_DIR
from quantlab.storage import load_csv

opts = {d["name"]: d["id"] for d in db.list_datasets()}
if opts:
    sel = st.selectbox("选择数据集", list(opts.keys()))
    if st.button("📤 导出为 CSV"):
        out = EXPORT_DIR / f"{sel}_export.csv"
        try:
            collector.export_dataset_csv(opts[sel], str(out))
            st.success(f"已导出：`{out}`")
            with open(out, "rb") as f:
                st.download_button("⬇️ 下载导出文件", f.read(), file_name=out.name, mime="text/csv")
        except Exception as e:  # noqa: BLE001
            st.error(f"导出失败：{e}")
else:
    st.caption("尚无数据集可导出。")
