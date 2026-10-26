"""模块 1：统一工作台（项目首页）。

所有指标均来自 SQLite 中的真实数据集/任务记录；无数据时显示明确空状态。
"""
from ui_common import init_page, status_label, empty_state

import streamlit as st
import pandas as pd

from quantlab import db
from quantlab.config import APP_NAME, APP_VERSION

init_page("🏠 统一工作台", "项目总览 · 数据状态 · 最近任务 · 快速开始")


# ---------------- 调度器生命周期（cache_resource 单例：页面刷新/切换不重启） ----------------
@st.cache_resource(show_spinner=False)
def _ensure_scheduler():
    from quantlab import scheduler as SCH
    thread, owner = SCH.get_scheduler()
    return {"owner": owner, "started_at": db.now_str()}


_sched_info = _ensure_scheduler()

# ---------- 恢复中断任务后统计 ----------
datasets = db.list_datasets()
tasks = db.list_tasks(limit=50)
failed_tasks = [t for t in tasks if t["status"] == "failed"]
collect_tasks = [t for t in tasks if t["task_type"] == "collect"]
last_collect = collect_tasks[0] if collect_tasks else None

# ---------- 顶部指标卡 ----------
n_experiments = len(db.list_experiments(limit=1000))
n_backtests = len(db.list_backtests(limit=1000))
c1, c2, c3, c4 = st.columns(4)
c1.metric("📚 数据集数量", len(datasets), help="已注册并保存了数据文件的数据集")
c2.metric("🧪 ML 实验 / 回测", f"{n_experiments} / {n_backtests}",
          help="机器学习实验与策略回测存档（阶段2起记录）")
from quantlab.db import list_optimization_runs, list_walk_forward, list_schedules
c3.metric("🎛️ 优化 / 滚动验证", f"{len(list_optimization_runs(1000))} / {len(list_walk_forward(1000))}",
          help="参数优化与滚动样本外验证存档（阶段3起记录）")
try:
    from quantlab.paper_trading import list_accounts
    n_acc = len(list_accounts())
except Exception:  # noqa: BLE001
    n_acc = 0
n_sched_enabled = len(list_schedules("enabled"))
c4.metric("💼 虚拟账户 / 活跃调度", f"{n_acc} / {n_sched_enabled}",
          help="模拟交易账户数与启用中的调度数（阶段4起记录）")

st.divider()

# ---------- 新手快速开始 ----------
with st.expander("🚀 新手快速开始指南（第一次使用请展开）", expanded=not datasets):
    st.markdown("""
**本阶段（数据阶段）演示路线，5 分钟走完：**

1. 进入 **📊 数据采集中心** → 选择来源"示例数据（随机模拟）"→ 选择资产 → 点击开始采集。
   （有网络时可切换到 Yahoo Finance 下载真实行情，如 `AAPL`、`600519.SS`）
2. 进入 **🗂️ 数据集管理** → 选择数据集 → 运行**质量检查** → 查看评分与问题清单 → 勾选清洗操作并执行。
3. 进入 **📈 自动分析** → 一键运行 EDA → 查看统计表、相关性热力图、回撤曲线 → 导出结果 CSV。
4. 回到本页查看刚刚产生的真实指标变化。

> ⚠️ 合规提示：平台只使用公开接口与本地导入，不绕过任何登录/反爬措施；
> 模拟数据在任何页面都会明确标注，绝不冒充真实行情。
""")

# ---------- 数据异常与失败任务提示 ----------
issues = []
for d in datasets:
    if d["quality_score"] is not None and d["quality_score"] < 90:
        issues.append(f"数据集 **{d['name']}** 质量评分 {d['quality_score']:.0f} 分（低于 90，建议在数据集管理页查看问题清单）")
for t in failed_tasks[:3]:
    issues.append(f"任务 **#{t['id']} {t['task_type']}·{t['target']}** 失败：{(t['error'] or '')[:120]}…")

left, right = st.columns([1.4, 1])
with left:
    st.subheader("🧾 最近任务")
    if tasks:
        df = pd.DataFrame([{
            "ID": t["id"], "类型": {"collect": "采集", "clean": "清洗", "analyze": "分析"}.get(t["task_type"], t["task_type"]),
            "对象": t["target"], "状态": status_label(t["status"]),
            "进度": f"{t['progress']:.0%}" if t["status"] == "running" else "—",
            "开始时间": t["started_at"] or "—",
        } for t in tasks[:8]])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        empty_state("还没有任何任务记录。", "先到「数据采集中心」发起第一次采集。")

with right:
    st.subheader("⚠️ 需要关注")
    if issues:
        for msg in issues:
            st.markdown(f"- {msg}")
    else:
        st.success("暂无数据异常、失败任务或配置错误。")

st.divider()

# ---------- 数据集概览（含数据更新状态）----------
st.subheader("📚 数据集概览")
if datasets:
    df = pd.DataFrame([{
        "名称": d["name"], "代码": d["symbol"], "类型": d["asset_type"],
        "来源": {"yfinance": "Yahoo Finance", "csv_import": "CSV 导入", "demo": "示例模拟"}.get(d["source"], d["source"]),
        "数据性质": "🎭 模拟数据" if d["is_synthetic"] else "真实/导入数据",
        "行数": d["rows"], "时间范围": f"{d['start_date']} ~ {d['end_date']}",
        "质量评分": f"{d['quality_score']:.0f}" if d["quality_score"] is not None else "未检查",
        "已清洗": "是" if d["cleaned"] else "否", "版本": f"v{d['version']}",
    } for d in datasets])
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    empty_state("还没有任何数据集。", "到「数据采集中心」用示例数据或真实接口创建第一个数据集。")

# ---------- 模块导航 ----------
st.subheader("🧭 功能导航")
n1, n2, n3, n4 = st.columns(4)
n1.page_link("pages/1_📊_数据采集中心.py", label="📊 数据采集中心", icon="📡",
             help="模块2：数据源配置、下载、增量更新、任务管理（已验收）")
n2.page_link("pages/2_🗂️_数据集管理.py", label="🗂️ 数据集管理", icon="🗃️",
             help="模块3：质量检查、清洗、版本记录（已验收）")
n3.page_link("pages/3_📈_自动分析.py", label="📈 自动分析", icon="🔬",
             help="模块4：一键 EDA（已验收）")
n4.page_link("pages/4_📐_技术指标.py", label="📐 技术指标", icon="📐",
             help="阶段2-A：SMA/EMA/RSI/MACD/布林带统一指标中心")
m1, m2, m3, m4 = st.columns(4)
m1.page_link("pages/5_🤖_机器学习实验室.py", label="🤖 机器学习实验室", icon="🤖",
             help="阶段2-B：时间序列三分防泄漏训练与实验存档")
m2.page_link("pages/6_⚖️_策略回测.py", label="⚖️ 策略回测", icon="⚖️",
             help="阶段2-C：买入持有/均线交叉/ML 信号回测（t+1 成交）")
m3.page_link("pages/7_📁_投资组合中心.py", label="📁 投资组合中心", icon="📁",
             help="阶段3-A：多资产组合、权重配置与再平衡")
m4.page_link("pages/8_🛡️_风险分析.py", label="🛡️ 风险分析", icon="🛡️",
             help="阶段3-B：风险指标/回撤区间/分段/压力测试")
p1, p2, p3, p4 = st.columns(4)
p1.page_link("pages/9_🎛️_参数优化.py", label="🎛️ 参数优化", icon="🎛️",
             help="阶段3-C：网格搜索与敏感性分析（验证段选参）")
p2.page_link("pages/10_🔁_稳健性验证.py", label="🔁 稳健性验证", icon="🔁",
             help="阶段3-D：滚动样本外验证与过拟合诊断")
p3.page_link("pages/11_🆚_实验比较.py", label="🆚 实验比较", icon="🆚",
             help="阶段3-E：多实验对比、条件检查与报告导出")
p4.page_link("pages/12_💼_模拟交易.py", label="💼 模拟交易", icon="💼",
             help="阶段4-A：虚拟账户与模拟订单（仅虚拟资金）")
q1, q2, q3, q4 = st.columns(4)
q1.page_link("pages/13_⏰_任务调度.py", label="⏰ 任务调度", icon="⏰",
             help="阶段4-B：定时任务调度与统一状态管理")
q2.markdown("📄 PDF 报告：回测/ML/组合页内一键导出（阶段4-C）")
q3.markdown("📊 均值-方差优化：「📁 投资组合中心」内（阶段4-D）")
q4.markdown("🔌 真实交易接入\n\n🚫 **永不实现**（本项目定位）")

st.caption(f"{APP_NAME} v{APP_VERSION} · 本页所有指标来自本地 SQLite 真实记录 · 生成时间见任务表")
