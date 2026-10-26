"""阶段 4B 页面：任务调度中心——调度配置 / 统一状态 / 执行历史 / 调度器健康。"""
import json
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from ui_common import init_page, status_label, empty_state

from quantlab import db
from quantlab import scheduler as SCH
from quantlab.scheduler import compute_next_run

init_page("⏰ 任务调度", "调度配置与执行分离 · 统一任务状态 · 持久化恢复 · 单实例保护")

# ---------------- 调度器生命周期（st.cache_resource 单例 + DB 租约跨进程唯一） ----------------
@st.cache_resource(show_spinner=False)
def _start_scheduler():
    thread, owner = SCH.get_scheduler()
    return {"owner": owner, "started_at": db.now_str()}

info = _start_scheduler()
health = SCH.scheduler_status()

st.subheader("① 调度器状态")
h1, h2, h3 = st.columns(3)
h1.metric("运行状态", "🟢 运行中" if health["running"] else "⚪ 未运行",
          help="单实例保护：进程内单例线程 + SQLite 租约（心跳 90 秒超时可接管）")
h2.metric("心跳时间", health.get("heartbeat_at") or "—")
h3.metric("租约持有者", health.get("owner") or "—", delta_color="off")
st.caption(f"调度循环每 {SCH.TICK_SECONDS} 秒扫描一次到期配置；页面刷新/切换不会重启线程；"
           f"错过计划超过 {SCH.MISFIRE_GRACE_MINUTES} 分钟按各调度的策略跳过或补跑一次。")

# ---------------- 新建调度 ----------------
st.subheader("② 调度配置")
datasets = db.list_datasets()
accounts_ids = [a["id"] for a in __import__("quantlab.paper_trading", fromlist=["list_accounts"]).list_accounts()]
with st.expander("➕ 新建调度"):
    c1, c2 = st.columns(2)
    s_name = c1.text_input("名称", "每日数据更新")
    s_type = c2.selectbox("任务类型", ["collect：数据采集", "backtest：策略回测",
                                      "walk_forward：滚动验证", "paper_signal：模拟账户信号",
                                      "report：生成回测报告"])
    task_type = s_type.split("：")[0]
    params = {}
    if task_type == "collect":
        datasets_all = datasets
        p1, p2 = st.columns(2)
        params["symbol"] = p1.text_input("资产代码", "AAPL")
        params["source"] = p2.selectbox("数据源", ["yfinance", "demo"])
        params["start"] = "2024-01-01"
    elif task_type == "backtest":
        if not datasets:
            st.info("无数据集。")
        ds_opts = {f"#{d['id']} {d['name']}": d["id"] for d in datasets}
        params["dataset_id"] = ds_opts[st.selectbox("数据集", list(ds_opts.keys()))]
        params["strategy"] = "buy_hold"
    elif task_type == "walk_forward":
        ds_opts = {f"#{d['id']} {d['name']}": d["id"] for d in datasets}
        params["dataset_id"] = ds_opts[st.selectbox("数据集 ", list(ds_opts.keys()))]
    elif task_type == "paper_signal":
        if not accounts_ids:
            st.info("先到「💼 模拟交易」创建账户并绑定策略。")
        params["account_id"] = st.selectbox("账户", accounts_ids)
    elif task_type == "report":
        bts = db.list_backtests(20)
        params["backtest_id"] = st.selectbox("回测记录", [b["id"] for b in bts]) if bts else None

    f1, f2, f3 = st.columns(3)
    freq = f1.selectbox("频率", ["daily：每天", "weekly：每周", "interval：间隔", "once：单次"]).split("：")[0]
    tod = f2.text_input("每日时刻 HH:MM", "09:00")
    wd = int(f3.number_input("周几（1=周一，weekly 用）", 1, 7, 1))
    iv = int(f3.number_input("间隔分钟（interval 用）", 1, 10080, 60))
    misfire = f1.selectbox("错过策略", ["skip：跳过并推进", "catchup_once：补跑一次"]).split("：")[0]

    if st.button("创建调度", type="primary"):
        try:
            now = datetime.now()
            nxt = compute_next_run(freq, now, tod, wd, iv)
            sid = db.save_schedule(s_name, task_type, params, freq,
                                   nxt.strftime("%Y-%m-%d %H:%M:%S"), time_of_day=tod,
                                   weekday=wd, interval_minutes=iv, misfire_policy=misfire)
            st.success(f"调度 #{sid} 已创建，下次计划：{nxt:%Y-%m-%d %H:%M}。")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"创建失败：{e}")

# ---------------- 调度列表 ----------------
schedules = db.list_schedules()
st.subheader(f"③ 调度列表（{len(schedules)}）")
if not schedules:
    empty_state("尚无调度配置。")
    st.stop()
rows = []
for s in schedules:
    runs = db.schedule_runs(s["id"], 100)
    n_failed = sum(1 for r in runs if db.normalize_status(r["status"]) == "failed")
    rows.append({"ID": s["id"], "名称": s["name"], "类型": s["task_type"],
                 "频率": s["freq"], "状态": {"enabled": "🟢 enabled", "paused": "⏸️ paused",
                                             "disabled": "⚪ disabled", "error": "🔴 error"}[s["status"]],
                 "下次计划": s["next_run_at"], "上次执行": s["last_run_at"] or "—",
                 "历史执行": len(runs), "失败次数": n_failed})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

sel_sid = st.number_input("操作调度 ID", 1, 10**6, schedules[0]["id"])
sch = db.get_schedule(int(sel_sid))
if sch:
    a1, a2, a3, a4 = st.columns(4)
    if a1.button("⏸️ 暂停") and sch["status"] == "enabled":
        db.update_schedule(sch["id"], status="paused"); st.rerun()
    if a2.button("▶️ 恢复") and sch["status"] == "paused":
        nxt = compute_next_run(sch["freq"], datetime.now(), sch["time_of_day"],
                               sch["weekday"], sch["interval_minutes"])
        db.update_schedule(sch["id"], status="enabled",
                           next_run_at=nxt.strftime("%Y-%m-%d %H:%M:%S")); st.rerun()
    if a3.button("🚫 停用"):
        db.update_schedule(sch["id"], status="disabled"); st.rerun()
    if a4.button("🏃 立即触发一次"):
        db.update_schedule(sch["id"], next_run_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        fired = SCH.tick_once(datetime.now() + timedelta(seconds=1))
        st.info(f"触发结果：{fired}")

    st.markdown(f"**该调度的执行历史**（调度配置状态与单次执行状态相互独立）")
    runs = db.schedule_runs(sch["id"], 30)
    if runs:
        rdf = pd.DataFrame([{"执行ID": r["id"], "计划时间": r["planned_at"] or "手动",
                             "状态": status_label(r["status"]),
                             "开始": r["started_at"] or "—", "结束": r["finished_at"] or "—",
                             "消息": (r["message"] or r["error"] or "")[:60]}
                            for r in runs])
        st.dataframe(rdf, use_container_width=True, hide_index=True)

# ---------------- 统一任务状态规范 ----------------
st.divider()
st.subheader("④ 统一任务状态规范")
st.markdown("""
| 调度配置状态 | 含义 | 单次执行状态 | 含义 |
|---|---|---|---|
| `enabled` | 按规则执行 | `queued → running` | 排队/执行中 |
| `paused` | 暂停触发 | `running → succeeded / failed` | 正常结束 |
| `disabled` | 停用 | `running → cancel_requested → cancelled` | 取消流程 |
| `error` | 配置错误 | `skipped / missed` | 按策略跳过/待处理 |

历史记录中的 `success` 为阶段 1—3 旧值，读取时自动映射为 `succeeded`（原记录不改写）。
状态转换由 `db.transition_task` 原子校验，非法跳转直接拒绝（有自动化测试覆盖）。
""")
all_tasks = db.list_tasks(30)
st.dataframe(pd.DataFrame([{"ID": t["id"], "类型": t["task_type"], "对象": t["target"],
                            "状态": status_label(t["status"]),
                            "调度ID": t["schedule_id"] or "—",
                            "计划": t["planned_at"] or "—",
                            "重试": t["retry_count"]} for t in all_tasks]),
             use_container_width=True, hide_index=True)
