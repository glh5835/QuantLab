"""自动化任务调度（阶段 4B）：调度配置与单次执行分离 + 单实例租约调度线程。

设计（与任务书 4B 对应）：
- schedules 表存调度配置（enabled/paused/disabled/error），tasks 表存每次执行
  （含 schedule_id + planned_at，唯一组合防重复执行同一计划实例）。
- tick_once() 是纯同步逻辑：扫描到期调度 → misfire 判定（skip/catchup_once）→
  幂等创建执行记录 → 复用 tasks.run_in_background 运行注册的处理器。
  调度线程只是定时调用 tick_once（每 30 秒），页面刷新不会重启线程
  （st.cache_resource 单例 + SQLite 租约跨进程保护）。
- 时区：统一使用本地时区的 naive datetime（与用户配置"每天 09:00"的直觉一致），
  数据库中所有时间字符串都来自同一时钟源 db.now_str()。
"""
import json
import threading
import time
from datetime import datetime, timedelta

from . import db, tasks
from .config import DB_PATH

TICK_SECONDS = 30
LEASE_TTL_SECONDS = 90  # 租约超时：心跳停止超过此时间可被其他实例抢占
MISFIRE_GRACE_MINUTES = 30  # 计划时间落后超过此值视为"错过"


class SchedulerError(ValueError):
    pass


# ---------------- 下次运行时间计算 ----------------

def compute_next_run(freq: str, after: datetime, time_of_day: str = "09:00",
                     weekday: int = 1, interval_minutes: int = 60) -> datetime:
    """计算下一次计划运行时间（本地时区；freq=once 时返回 after 本身）。"""
    if freq == "once":
        return after
    hh, mm = (int(x) for x in time_of_day.split(":")[:2])
    if freq == "daily":
        candidate = after.replace(hour=hh, minute=mm, second=0, microsecond=0)
        return candidate if candidate > after else candidate + timedelta(days=1)
    if freq == "weekly":
        if not 1 <= weekday <= 7:
            raise SchedulerError("weekday 必须在 1~7（周一=1）。")
        candidate = after.replace(hour=hh, minute=mm, second=0, microsecond=0)
        days_ahead = (weekday - 1 - candidate.weekday()) % 7
        candidate += timedelta(days=days_ahead)
        if candidate <= after:
            candidate += timedelta(weeks=1)
        return candidate
    if freq == "interval":
        if interval_minutes < 1:
            raise SchedulerError("interval_minutes 必须 ≥ 1。")
        return after + timedelta(minutes=interval_minutes)
    raise SchedulerError(f"未知 freq {freq}")


# ---------------- 任务处理器注册表 ----------------
# 处理器签名 func(task_id, cancel, params: dict)；新任务类型在此注册即可被调度。

TASK_HANDLERS = {}


def register_handler(task_type: str):
    def deco(func):
        TASK_HANDLERS[task_type] = func
        return func
    return deco


def _default_collect(task_id, cancel, params):
    from .collector import run_collect_task
    run_collect_task(task_id, cancel, params["symbol"], params.get("source", "yfinance"),
                     params.get("start", "2024-01-01"), params.get("end"), 
                     params.get("asset_type", "stock"), params.get("name"))


def _default_backtest(task_id, cancel, params):
    from .backtest import run_backtest, BacktestConfig
    from .strategies import BuyHoldStrategy, MACrossoverStrategy, MLSignalStrategy
    st = params.get("strategy", "buy_hold")
    if st == "ma_cross":
        strat = MACrossoverStrategy(int(params["fast"]), int(params["slow"]))
    elif st == "ml_signal":
        strat = MLSignalStrategy(int(params["experiment_id"]))
    else:
        strat = BuyHoldStrategy()
    run_backtest(strat, int(params["dataset_id"]), BacktestConfig(),
                 name=params.get("name", f"sched_{st}"))


def _default_walk_forward(task_id, cancel, params):
    from .robustness import walk_forward_rule, save_walk_forward_run
    from .strategies import MACrossoverStrategy
    from .ml_lab import load_dataset_frame
    df, ds = load_dataset_frame(int(params["dataset_id"]))
    f, s = int(params.get("fast", 5)), int(params.get("slow", 20))
    res = walk_forward_rule(df, lambda: MACrossoverStrategy(f, s),
                            train_size=int(params.get("train_size", 200)),
                            test_size=int(params.get("test_size", 60)))
    save_walk_forward_run(int(params["dataset_id"]), ds["name"], f"ma_cross({f},{s})",
                          res, params)


def _default_paper_signal(task_id, cancel, params):
    from .paper_trading import run_strategy_signal
    db.append_task_log(task_id, f"执行账户 #{params['account_id']} 的策略信号评估")
    r = run_strategy_signal(int(params["account_id"]))
    db.append_task_log(task_id, f"信号结果：{r.get('action')}（{r.get('reason', '')}）")


for _t, _f in [("collect", _default_collect), ("backtest", _default_backtest),
               ("walk_forward", _default_walk_forward), ("paper_signal", _default_paper_signal)]:
    TASK_HANDLERS.setdefault(_t, _f)


def _default_report(task_id, cancel, params):
    from .reports import backtest_report
    _, p = backtest_report(int(params["backtest_id"]))
    db.append_task_log(task_id, f"报告已生成：{p}")


TASK_HANDLERS.setdefault("report", _default_report)


# ---------------- 核心调度循环（同步可测） ----------------

def tick_once(now: datetime = None) -> list[dict]:
    """扫描一次到期调度并触发执行。返回本次触发的记录摘要（测试/页面可见）。"""
    now = now or datetime.now()
    triggered = []
    for sch in db.list_schedules(status="enabled"):
        try:
            planned = datetime.strptime(sch["next_run_at"], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            db.update_schedule(sch["id"], status="error")
            triggered.append({"schedule": sch["id"], "action": "config_error"})
            continue
        if planned > now:
            continue
        late_minutes = (now - planned).total_seconds() / 60
        if late_minutes > MISFIRE_GRACE_MINUTES and sch["misfire_policy"] == "skip":
            # 错过且策略为跳过：记 skipped 执行 + 推进下一次
            tid = db.create_task(sch["task_type"], sch["name"],
                                 {"schedule_id": sch["id"]})
            with db.get_conn() as conn:
                conn.execute("UPDATE tasks SET schedule_id=?, planned_at=? WHERE id=?",
                             (sch["id"], sch["next_run_at"], tid))
            db.transition_task(tid, "skipped",
                               message=f"错过计划时间 {sch['next_run_at']}（离线超 "
                                       f"{MISFIRE_GRACE_MINUTES} 分钟），按策略跳过",
                               finished_at=db.now_str())
            _advance(sch, now)
            triggered.append({"schedule": sch["id"], "action": "skipped", "task": tid})
            continue
        # 到期（或 catchup_once 补跑）：幂等创建执行
        planned_str = sch["next_run_at"]
        if db.run_exists_for_planned(sch["id"], planned_str):
            _advance(sch, now)  # 已有该计划实例（可能正运行）→ 只推进时间防重复触发
            continue
        if sch["task_type"] not in TASK_HANDLERS:
            db.update_schedule(sch["id"], status="error")
            triggered.append({"schedule": sch["id"], "action": "no_handler"})
            continue
        tid = db.create_task(sch["task_type"], sch["name"],
                             json.loads(sch["params_json"] or "{}"))
        # 关联调度与计划时间（防重复执行的依据）
        with db.get_conn() as conn:
            conn.execute("UPDATE tasks SET schedule_id=?, planned_at=? WHERE id=?",
                         (sch["id"], planned_str, tid))
        params = json.loads(sch["params_json"] or "{}")
        tasks.run_in_background(tid, TASK_HANDLERS[sch["task_type"]], params)
        db.update_schedule(sch["id"], last_run_at=db.now_str())
        _advance(sch, now)
        triggered.append({"schedule": sch["id"], "action": "launched", "task": tid,
                          "planned": planned_str})
    return triggered


def _advance(sch, now: datetime) -> None:
    """推进调度的 next_run_at（once 型完成后置 disabled）。"""
    if sch["freq"] == "once":
        db.update_schedule(sch["id"], status="disabled",
                           next_run_at=sch["next_run_at"])
        return
    nxt = compute_next_run(sch["freq"], now, sch["time_of_day"], sch["weekday"],
                           sch["interval_minutes"])
    db.update_schedule(sch["id"], next_run_at=nxt.strftime("%Y-%m-%d %H:%M:%S"))


# ---------------- 调度器线程 + 单实例租约 ----------------

def acquire_lease(owner: str, pid: int = 0) -> bool:
    """抢占调度器租约：空闲或租约过期（心跳超时）时获得，否则失败。"""
    now = datetime.now()
    with db.get_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO scheduler_lease (id) VALUES (1)")
        row = conn.execute("SELECT owner, heartbeat_at FROM scheduler_lease WHERE id=1").fetchone()
        if row["heartbeat_at"]:
            try:
                hb = datetime.strptime(row["heartbeat_at"], "%Y-%m-%d %H:%M:%S")
                alive = (now - hb).total_seconds() < LEASE_TTL_SECONDS
            except ValueError:
                alive = False
            if alive and row["owner"] != owner:
                return False
        conn.execute("UPDATE scheduler_lease SET owner=?, heartbeat_at=?, pid=? WHERE id=1",
                     (owner, db.now_str(), pid))
    return True


def heartbeat(owner: str) -> None:
    with db.get_conn() as conn:
        conn.execute("UPDATE scheduler_lease SET heartbeat_at=? WHERE id=1 AND owner=?",
                     (db.now_str(), owner))


def release_lease(owner: str) -> None:
    with db.get_conn() as conn:
        conn.execute("UPDATE scheduler_lease SET owner='', heartbeat_at='' "
                     "WHERE id=1 AND owner=?", (owner,))


class SchedulerThread:
    """定时调用 tick_once 的守护线程；由 get_scheduler() 单例管理。"""

    def __init__(self, owner: str):
        self.owner = owner
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="quantlab-scheduler")

    def _loop(self):
        while not self._stop.is_set():
            try:
                if acquire_lease(self.owner):
                    tick_once()
                    heartbeat(self.owner)
            except Exception:  # noqa: BLE001 - 调度循环永不因单次异常退出
                import traceback
                with open(DB_PATH.parent / "scheduler_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{db.now_str()} {traceback.format_exc()}\n")
            self._stop.wait(TICK_SECONDS)
        release_lease(self.owner)

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        self._stop.set()


_scheduler_instance = None
_scheduler_lock = threading.Lock()


def get_scheduler() -> tuple[SchedulerThread | None, str]:
    """进程内单例（Streamlit 页面用 st.cache_resource 包裹调用；线程内租约保证跨进程唯一）。

    返回 (线程或None, 说明)。未获得租约时线程仍会运行并周期性重试抢租约
    （心跳过期的持有者会被自动接管），保证任一时刻全局只有一个实例在 tick。
    """
    global _scheduler_instance
    with _scheduler_lock:
        if _scheduler_instance is None:
            owner = f"{__import__('os').getpid()}-{threading.get_ident()}"
            _scheduler_instance = SchedulerThread(owner)
            _scheduler_instance.start()
        return _scheduler_instance, _scheduler_instance.owner


def scheduler_status() -> dict:
    """健康检查：租约持有者、心跳时间、是否存活。"""
    with db.get_conn() as conn:
        row = conn.execute("SELECT * FROM scheduler_lease WHERE id=1").fetchone()
    if not row or not row["heartbeat_at"]:
        return {"running": False, "owner": None, "heartbeat_at": None,
                "note": "调度器未启动或从未运行"}
    hb = datetime.strptime(row["heartbeat_at"], "%Y-%m-%d %H:%M:%S")
    alive = (datetime.now() - hb).total_seconds() < LEASE_TTL_SECONDS + TICK_SECONDS
    return {"running": alive, "owner": row["owner"], "heartbeat_at": row["heartbeat_at"],
            "pid": row["pid"]}
