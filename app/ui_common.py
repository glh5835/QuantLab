"""Streamlit 页面共享工具：统一初始化、样式与空状态组件。"""
import sys
from pathlib import Path

# 让 app/ 目录下的脚本可以直接 import quantlab
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from quantlab import db  # noqa: E402
from quantlab.config import APP_NAME, APP_VERSION  # noqa: E402


def init_page(title: str, subtitle: str = ""):
    """每个页面开头调用：恢复中断任务 + 页标题。"""
    db.recover_stale_tasks()  # 应用重启后把遗留 running 任务标记为失败（可重试）
    st.set_page_config(page_title=f"{title} | {APP_NAME}", page_icon="📊",
                       layout="wide", initial_sidebar_state="expanded")
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.sidebar.markdown(f"**{APP_NAME}** v{APP_VERSION}")


def dataset_options() -> dict[str, int]:
    """{展示名: 数据集id}，供下拉框使用。"""
    rows = db.list_datasets()
    return {f"{r['name']}（{r['symbol']}）": r["id"] for r in rows}


def synthetic_badge(row) -> None:
    """模拟数据必须显著标注。"""
    if row and row["is_synthetic"]:
        st.warning("⚠️ 此数据集为**随机模拟数据**（由示例数据适配器生成），不是真实行情，仅用于功能演示。")


def empty_state(message: str, hint: str = ""):
    """统一空状态：没有数据时明确提示，绝不伪造。"""
    st.info(f"📭 {message}" + (f"\n\n👉 {hint}" if hint else ""))


def status_label(status: str) -> str:
    return {"queued": "⏳ 排队中", "running": "🔄 运行中", "success": "✅ 成功",
            "failed": "❌ 失败", "cancelled": "🚫 已取消"}.get(status, status)
