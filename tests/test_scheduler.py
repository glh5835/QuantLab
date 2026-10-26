"""调度器与统一任务状态测试：状态机/时间计算/幂等执行/misfire/租约单例。"""
import json
import threading
import time
from datetime import datetime, timedelta

import pytest

from quantlab import db, scheduler as SCH
from quantlab.scheduler import (SchedulerError, compute_next_run, tick_once,
                                acquire_lease, release_lease, scheduler_status)


# ---------------- 下次运行时间 ----------------

def test_next_run_daily():
    now = datetime(2026, 1, 15, 10, 0)
    assert compute_next_run("daily", now, "09:00") == datetime(2026, 1, 16, 9, 0)
    assert compute_next_run("daily", now, "11:00") == datetime(2026, 1, 15, 11, 0)


def test_next_run_weekly():
    # 2026-01-15 是周四；下周一 = 01-19
    now = datetime(2026, 1, 15, 10, 0)
    assert compute_next_run("weekly", now, "09:00", weekday=1) == datetime(2026, 1, 19, 9, 0)
    with pytest.raises(SchedulerError):
        compute_next_run("weekly", now, "09:00", weekday=8)


def test_next_run_interval_and_once():
    now = datetime(2026, 1, 15, 10, 0)
    assert compute_next_run("interval", now, interval_minutes=30) == datetime(2026, 1, 15, 10, 30)
    assert compute_next_run("once", now) == now
    with pytest.raises(SchedulerError):
        compute_next_run("interval", now, interval_minutes=0)


# ---------------- 统一状态机 ----------------

def test_valid_transitions_and_atomicity():
    tid = db.create_task("demo", "x")
    assert db.transition_task(tid, "running")
    assert db.transition_task(tid, "succeeded", message="done")
    with pytest.raises(ValueError, match="非法状态转换"):
        db.transition_task(tid, "running")      # 终态不可再变
    with pytest.raises(ValueError, match="非法状态转换"):
        db.transition_task(tid, "succeeded")    # 幂等也不允许重复终态转换


def test_cancel_requested_flow():
    tid = db.create_task("demo", "x")
    db.transition_task(tid, "running")
    db.transition_task(tid, "cancel_requested", message="用户取消")
    assert db.transition_task(tid, "cancelled", message="已响应")   # 请求→确认
    # 取消请求后任务实际跑完：如实记录 succeeded 也合法（转换表允许）
    tid2 = db.create_task("demo", "y")
    db.transition_task(tid2, "running")
    db.transition_task(tid2, "cancel_requested")
    assert db.transition_task(tid2, "succeeded", message="取消前来得及完成")


def test_legacy_success_alias_readonly():
    """旧记录的 success 值不改库，读取时规范化为 succeeded。"""
    tid = db.create_task("demo", "legacy")
    with db.get_conn() as conn:
        conn.execute("UPDATE tasks SET status='success' WHERE id=?", (tid,))
    assert db.normalize_status("success") == "succeeded"
    assert db.get_task_safe(tid)["status"] == "succeeded"
    with db.get_conn() as conn:
        raw = conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()
    assert raw["status"] == "success"          # 历史记录未被改写


def test_finished_task_not_revived():
    """已完成的执行不会被改写成运行状态（页面刷新/重复请求防护）。"""
    tid = db.create_task("demo", "z")
    db.transition_task(tid, "running")
    db.transition_task(tid, "succeeded")
    with pytest.raises(ValueError):
        db.transition_task(tid, "running")


# ---------------- tick 调度逻辑 ----------------

def _mk_sched(**kw):
    d = dict(name="测试调度", task_type="paper_signal", params={}, freq="once",
             status="enabled", misfire_policy="skip",
             next_run_at=(datetime.now() - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"))
    d.update(kw)
    return db.save_schedule(d["name"], d["task_type"], d["params"], d["freq"],
                            d["next_run_at"], misfire_policy=d["misfire_policy"],
                            status=d["status"])


def test_tick_fires_due_schedule_once_only():
    sid = _mk_sched()
    fired = tick_once(datetime.now())
    assert any(t["schedule"] == sid and t["action"] == "launched" for t in fired)
    # 幂等：立即再 tick 不应为同一计划实例创建第二次执行（once 已 disabled）
    fired2 = tick_once(datetime.now())
    assert not any(t["schedule"] == sid and t["action"] == "launched" for t in fired2)
    runs = db.schedule_runs(sid)
    assert len(runs) == 1


def test_tick_respects_paused_schedule():
    sid = _mk_sched(status="paused")
    fired = tick_once(datetime.now())
    assert not any(t["schedule"] == sid for t in fired)
    assert db.schedule_runs(sid) == []


def test_misfire_skip_policy():
    # 计划时间落后 2 小时（> 30 分钟宽限）→ skip + 推进
    old = (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    sid = db.save_schedule("错过调度", "paper_signal", {}, "interval", old,
                           interval_minutes=60, misfire_policy="skip")
    fired = tick_once(datetime.now())
    assert any(t["schedule"] == sid and t["action"] == "skipped" for t in fired)
    runs = db.schedule_runs(sid)
    assert db.normalize_status(runs[0]["status"]) == "skipped"
    # next_run 已推进到现在之后
    sch = db.get_schedule(sid)
    assert datetime.strptime(sch["next_run_at"], "%Y-%m-%d %H:%M:%S") > datetime.now() - timedelta(minutes=1)


def test_misfire_catchup_runs():
    old = (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    sid = db.save_schedule("补跑调度", "paper_signal", {}, "once", old,
                           misfire_policy="catchup_once")
    fired = tick_once(datetime.now())
    assert any(t["schedule"] == sid and t["action"] == "launched" for t in fired)


def test_planned_at_dedup_prevents_double_run():
    """同一 schedule+planned_at 只产生一条执行记录（防重复执行核心断言）。"""
    sid = _mk_sched(freq="interval")
    tick_once(datetime.now())   # 触发一次
    # 手工把 next_run_at 改回与首次执行完全相同的计划时间 → tick 不得重复创建
    first_planned = db.schedule_runs(sid)[0]["planned_at"]
    with db.get_conn() as conn:
        conn.execute("UPDATE schedules SET next_run_at=?, status='enabled' WHERE id=?",
                     (first_planned, sid))
    tick_once(datetime.now())
    assert len(db.schedule_runs(sid)) == 1


def test_handler_executes_and_records():
    """注册的处理器真实执行：paper_signal 任务对无绑定账户会失败并留痕。"""
    from quantlab import paper_trading as PT
    aid = PT.create_account("调度测试账户", 10_000)
    sid = db.save_schedule("信号调度", "paper_signal", {"account_id": aid}, "once",
                           datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    fired = tick_once(datetime.now())
    assert any(t["schedule"] == sid for t in fired)
    # 后台线程执行：等它完成（未绑定策略 → 任务 failed 且有原因）
    for _ in range(50):
        runs = db.schedule_runs(sid)
        if runs and db.normalize_status(runs[0]["status"]) in ("succeeded", "failed"):
            break
        time.sleep(0.2)
    st = db.normalize_status(runs[0]["status"])
    assert st in ("succeeded", "failed")   # 有真实执行结果
    if st == "failed":
        assert runs[0]["error"]             # 失败原因已保存


# ---------------- 租约单例 ----------------

def test_lease_single_owner_and_takeover():
    assert acquire_lease("ownerA", pid=1)
    assert not acquire_lease("ownerB")                      # 租约被占 → 拒绝
    assert acquire_lease("ownerA")                          # 本人续约 OK
    # 心跳过期 → 可被接管
    with db.get_conn() as conn:
        conn.execute("UPDATE scheduler_lease SET heartbeat_at=? WHERE id=1",
                     ((datetime.now() - timedelta(seconds=SCH.LEASE_TTL_SECONDS * 2))
                      .strftime("%Y-%m-%d %H:%M:%S"),))
    assert acquire_lease("ownerB", pid=2)
    assert scheduler_status()["running"] is True
    release_lease("ownerB")
    assert scheduler_status()["running"] is False


def test_scheduler_thread_singleton():
    t1, o1 = SCH.get_scheduler()
    t2, o2 = SCH.get_scheduler()
    assert t1 is t2 and o1 == o2            # 进程内单例
    t1.stop()
    SCH._scheduler_instance = None          # 复位供后续测试
