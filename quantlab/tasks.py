"""后台任务系统：基于 threading 的轻量任务执行器（满足"界面可查状态/进度/日志"要求）。

教学解释：Streamlit 每次交互都会重跑脚本，任务不能放在脚本里直接执行，
否则界面会一直阻塞。我们把任务交给后台线程，状态实时写入 SQLite，
界面每次刷新时从数据库读取最新状态——这就是"异步执行、避免阻塞前端"的最简实现。
程序重启时调用 recover_stale_tasks() 把中断任务标记为失败，支持一键重试。
"""
import threading
import traceback

from . import db

# 全局任务注册表（进程内）：用于取消正在运行的任务
_running: dict[int, threading.Event] = {}
_cancel_flags: dict[int, threading.Event] = {}


def cancel_task(task_id: int) -> bool:
    """请求取消任务。任务函数内部会在关键点检查标志并退出。"""
    if task_id in _running:
        _cancel_flags[task_id].set()
        db.update_task(task_id, status="cancelled", message="已请求取消",
                       finished_at=db.now_str())
        return True
    db.update_task(task_id, status="cancelled", message="任务已不在运行（可能已结束）",
                   finished_at=db.now_str())
    return False


def run_in_background(task_id: int, func, *args, **kwargs) -> None:
    """在守护线程中执行 func(task_id, cancel_event, *args)。

    约定：func 第一个参数是 task_id（用于写日志/进度），第二个是取消标志 Event。
    """
    cancel = threading.Event()
    _cancel_flags[task_id] = cancel

    def _worker():
        _running[task_id] = threading.current_thread()
        db.update_task(task_id, status="running", started_at=db.now_str(), progress=0)
        try:
            func(task_id, cancel, *args, **kwargs)
            final = db.get_task(task_id)
            if final and final["status"] not in ("cancelled", "failed"):
                db.update_task(task_id, status="success", progress=1.0,
                               message="任务完成", finished_at=db.now_str())
        except Exception as e:  # noqa: BLE001 - 任务系统必须捕获所有异常并入库
            db.update_task(task_id, status="failed",
                           error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
                           finished_at=db.now_str())
            db.append_task_log(task_id, f"[失败] {type(e).__name__}: {e}")
        finally:
            _running.pop(task_id, None)
            _cancel_flags.pop(task_id, None)

    t = threading.Thread(target=_worker, daemon=True, name=f"task-{task_id}")
    t.start()
