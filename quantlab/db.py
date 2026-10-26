"""SQLite 元数据库层：数据集注册表、任务表、采集日志表。

设计说明（教学要点）：
- 使用 Python 标准库 sqlite3，单文件数据库，零配置，适合个人电脑。
- 所有写操作立即 commit，保证程序意外退出后状态不丢失。
- 这个文件只负责"元数据"，行情数据本体存在 data/raw 与 data/clean 的 CSV 文件中。
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from . import config
from .config import DB_PATH  # 默认路径；测试可通过覆盖 config.DB_PATH 重定向

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,          -- 数据集名称，如 "AAPL_daily"
    symbol TEXT NOT NULL,               -- 资产代码
    asset_type TEXT DEFAULT 'stock',    -- stock / etf / index / other
    source TEXT NOT NULL,               -- yfinance / csv_import / demo
    is_synthetic INTEGER DEFAULT 0,     -- 1 = 模拟数据（必须在界面显著标注）
    license_note TEXT DEFAULT '',       -- 数据许可/来源说明
    field_desc TEXT DEFAULT '',         -- 字段说明
    raw_path TEXT DEFAULT '',           -- 原始数据文件路径
    clean_path TEXT DEFAULT '',         -- 清洗后数据文件路径
    start_date TEXT DEFAULT '',         -- 数据时间范围（来自真实文件统计）
    end_date TEXT DEFAULT '',
    rows INTEGER DEFAULT 0,
    quality_score REAL,                 -- 最近一次质量评分（NULL = 未检查）
    quality_report TEXT DEFAULT '',     -- 最近一次质量报告 JSON
    cleaned INTEGER DEFAULT 0,          -- 是否已生成清洗版本
    cleaning_log TEXT DEFAULT '',       -- 实际执行的清洗操作 JSON 记录
    version INTEGER DEFAULT 1,          -- 数据集版本（重新采集/导入时 +1）
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type TEXT NOT NULL,            -- collect / clean / analyze
    target TEXT NOT NULL,               -- 操作对象（如数据集名/股票代码）
    params TEXT DEFAULT '{}',           -- 任务参数 JSON
    status TEXT NOT NULL,               -- queued / running / success / failed / cancelled
    progress REAL DEFAULT 0,            -- 0~1
    message TEXT DEFAULT '',            -- 最近一条状态消息
    error TEXT DEFAULT '',              -- 失败原因（含解决建议）
    log TEXT DEFAULT '',                -- 逐条运行日志（换行分隔）
    started_at TEXT DEFAULT '',
    finished_at TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

-- 阶段2：机器学习实验存档（模型与预测文件存磁盘，库中只存路径引用与元数据）
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    dataset_id INTEGER NOT NULL,
    dataset_name TEXT DEFAULT '',
    dataset_version INTEGER DEFAULT 1,
    task TEXT NOT NULL,                 -- direction（分类）/ return（回归）
    horizon INTEGER NOT NULL,           -- 预测期限（未来 h 期）
    label_rule TEXT DEFAULT '',         -- 标签定义说明（含零收益规则）
    features_json TEXT DEFAULT '[]',    -- 特征列表
    model_type TEXT NOT NULL,
    model_params TEXT DEFAULT '{}',
    seed INTEGER DEFAULT 42,
    split_ratios TEXT DEFAULT '[0.7,0.15,0.15]',
    train_range TEXT DEFAULT '', val_range TEXT DEFAULT '', test_range TEXT DEFAULT '',
    train_n INTEGER DEFAULT 0, val_n INTEGER DEFAULT 0, test_n INTEGER DEFAULT 0,
    purged_n INTEGER DEFAULT 0,         -- 因标签区间重叠被隔离（purge）的样本数
    metrics_json TEXT DEFAULT '{}',     -- train/val/test 三段指标
    model_path TEXT DEFAULT '',
    predictions_path TEXT DEFAULT '',
    status TEXT DEFAULT 'success',      -- success / failed
    error TEXT DEFAULT '',
    code_version TEXT DEFAULT ''
);

-- 阶段2：回测存档（交易明细与逐日状态存磁盘 CSV，库中只存路径与摘要）
CREATE TABLE IF NOT EXISTS backtests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    strategy_type TEXT NOT NULL,        -- buy_hold / ma_cross / ml_signal
    strategy_params TEXT DEFAULT '{}',
    dataset_id INTEGER NOT NULL,
    dataset_name TEXT DEFAULT '',
    dataset_version INTEGER DEFAULT 1,
    experiment_id INTEGER,              -- 仅 ml_signal 策略引用
    start_date TEXT DEFAULT '', end_date TEXT DEFAULT '',
    config_json TEXT DEFAULT '{}',      -- 初始资金/手续费/滑点/风控参数
    metrics_json TEXT DEFAULT '{}',
    trades_path TEXT DEFAULT '',
    daily_path TEXT DEFAULT '',
    status TEXT DEFAULT 'success',
    error TEXT DEFAULT '',
    code_version TEXT DEFAULT ''
);

-- 阶段3：参数优化存档（网格搜索记录；test 段不参与选参的事实由字段固化）
CREATE TABLE IF NOT EXISTS optimization_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    dataset_id INTEGER NOT NULL,
    dataset_name TEXT DEFAULT '',
    strategy_type TEXT NOT NULL,
    param_grid_json TEXT DEFAULT '{}',
    objective TEXT DEFAULT 'sharpe',
    constraints_json TEXT DEFAULT '{}',
    results_json TEXT DEFAULT '[]',     -- 每组参数的 val/train 指标
    best_params_json TEXT DEFAULT '{}',
    test_metrics_json TEXT DEFAULT '{}',
    n_combinations INTEGER DEFAULT 0,
    test_used_for_selection INTEGER DEFAULT 0,  -- 恒 0：test 从不参与选参
    status TEXT DEFAULT 'success',
    error TEXT DEFAULT '',
    code_version TEXT DEFAULT ''
);

-- 阶段3：滚动验证存档（每窗口记录训练/测试区间与样本外指标）
CREATE TABLE IF NOT EXISTS walk_forward_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    dataset_id INTEGER NOT NULL,
    dataset_name TEXT DEFAULT '',
    strategy_type TEXT NOT NULL,
    mode TEXT DEFAULT 'rolling',        -- rolling(固定窗) / expanding(扩展窗)
    config_json TEXT DEFAULT '{}',
    windows_json TEXT DEFAULT '[]',     -- 每窗口 train/test 区间+指标+模型参数
    summary_json TEXT DEFAULT '{}',     -- 样本外汇总统计与过拟合提示
    n_windows INTEGER DEFAULT 0,
    status TEXT DEFAULT 'success',
    error TEXT DEFAULT '',
    code_version TEXT DEFAULT ''
);

-- 阶段4A：虚拟账户（模拟交易，仅虚拟资金）
CREATE TABLE IF NOT EXISTS paper_accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    initial_cash REAL NOT NULL,
    cash REAL NOT NULL,
    realized_pnl REAL DEFAULT 0,
    total_fees REAL DEFAULT 0,
    status TEXT DEFAULT 'active',        -- active / paused / closed
    risk_json TEXT DEFAULT '{}',         -- 风控参数（单笔上限/仓位上限/最低现金/单日限额）
    binding_json TEXT DEFAULT '{}',      -- 策略绑定（strategy/dataset/params/paused）
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    shares REAL DEFAULT 0,
    avg_cost REAL DEFAULT 0,
    UNIQUE(account_id, symbol)
);

CREATE TABLE IF NOT EXISTS paper_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,                  -- buy / sell
    qty REAL NOT NULL,
    status TEXT NOT NULL,                -- filled / rejected / cancelled / unfilled
    fill_model TEXT DEFAULT '',          -- close_simplified 等成交模型标记
    signal_id TEXT DEFAULT '',           -- 策略信号标识（防同一信号重复下单）
    ref_price REAL,                      -- 参考价（成交依据）
    ref_time TEXT DEFAULT '',            -- 参考行情的时间戳（如实标注数据时间）
    exec_price REAL,
    fee REAL DEFAULT 0,
    slippage REAL DEFAULT 0,
    reason TEXT DEFAULT '',              -- 拒绝/未成交原因
    created_at TEXT NOT NULL,
    exec_at TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_orders_signal ON paper_orders(account_id, signal_id);

-- 阶段4B：调度配置（与单次执行 tasks 分离；tasks 表新增列由下方迁移补齐）
CREATE TABLE IF NOT EXISTS schedules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    task_type TEXT NOT NULL,             -- collect / backtest / walk_forward / report ...
    params_json TEXT DEFAULT '{}',
    freq TEXT NOT NULL,                  -- daily / weekly / interval
    time_of_day TEXT DEFAULT '09:00',    -- HH:MM（本地时区）
    weekday INTEGER DEFAULT 1,           -- weekly 用：1=周一…7=周日
    interval_minutes INTEGER DEFAULT 60, -- interval 用
    status TEXT DEFAULT 'enabled',       -- enabled / paused / disabled / error
    misfire_policy TEXT DEFAULT 'skip',  -- skip / catchup_once
    last_run_at TEXT DEFAULT '',
    next_run_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- 调度器单例租约（心跳+持有者，防多进程/多会话重复启动）
CREATE TABLE IF NOT EXISTS scheduler_lease (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    owner TEXT DEFAULT '',
    heartbeat_at TEXT DEFAULT '',
    pid INTEGER DEFAULT 0
);
"""

# 阶段4B：tasks 表增量迁移（幂等：先查后加，兼容既有数据库）
_TASK_MIGRATIONS = [
    ("schedule_id", "ALTER TABLE tasks ADD COLUMN schedule_id INTEGER"),
    ("planned_at", "ALTER TABLE tasks ADD COLUMN planned_at TEXT DEFAULT ''"),
    ("retry_count", "ALTER TABLE tasks ADD COLUMN retry_count INTEGER DEFAULT 0"),
    ("result_ref", "ALTER TABLE tasks ADD COLUMN result_ref TEXT DEFAULT ''"),
]


def _migrate_tasks_table(conn) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
    for col, ddl in _TASK_MIGRATIONS:
        if col not in existing:
            conn.execute(ddl)


@contextmanager
def get_conn():
    """获取数据库连接的上下文管理器，自动关闭。运行时读取 config.DB_PATH 以支持测试重定向。"""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """初始化所有表。应用每次启动时调用（CREATE IF NOT EXISTS 与增量迁移都是幂等的）。"""
    with get_conn() as conn:
        conn.executescript(_SCHEMA)
        _migrate_tasks_table(conn)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------- 数据集 ----------------

def register_dataset(name, symbol, source, asset_type="stock", is_synthetic=0,
                     license_note="", field_desc="") -> int:
    """注册新数据集；若同名已存在则版本 +1 并返回原 id（不删除历史记录）。
    license_note: 数据许可说明（yfinance → Yahoo Finance 公开接口，仅供学习研究）"""
    with get_conn() as conn:
        row = conn.execute("SELECT id, version FROM datasets WHERE name=?", (name,)).fetchone()
        ts = now_str()
        if row:
            conn.execute(
                "UPDATE datasets SET symbol=?, source=?, asset_type=?, is_synthetic=?, "
                "license_note=?, field_desc=?, version=version+1, updated_at=? WHERE id=?",
                (symbol, source, asset_type, is_synthetic, license_note, field_desc, ts, row["id"]))
            return row["id"]
        cur = conn.execute(
            "INSERT INTO datasets (name, symbol, asset_type, source, is_synthetic, "
            "license_note, field_desc, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (name, symbol, asset_type, source, is_synthetic, license_note, field_desc, ts, ts))
        return cur.lastrowid


def update_dataset(dataset_id: int, **fields) -> None:
    """更新数据集的任意字段（如 raw_path / quality_score 等）。"""
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    with get_conn() as conn:
        conn.execute(f"UPDATE datasets SET {cols}, updated_at=? WHERE id=?",
                     (*fields.values(), now_str(), dataset_id))


def get_dataset(dataset_id: int):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()


def find_dataset(name: str):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM datasets WHERE name=?", (name,)).fetchone()


def list_datasets(source: str = None, keyword: str = None):
    """数据集列表，支持按来源筛选与关键字搜索（名称/代码）。"""
    sql, args = "SELECT * FROM datasets WHERE 1=1", []
    if source:
        sql += " AND source=?"; args.append(source)
    if keyword:
        sql += " AND (name LIKE ? OR symbol LIKE ?)"; args.extend([f"%{keyword}%", f"%{keyword}%"])
    sql += " ORDER BY updated_at DESC"
    with get_conn() as conn:
        return conn.execute(sql, args).fetchall()


def delete_dataset(dataset_id: int) -> dict:
    """删除数据集注册记录及其数据文件。返回被删除的文件列表（供界面确认）。"""
    ds = get_dataset(dataset_id)
    if not ds:
        return {"deleted_files": [], "name": None}
    from pathlib import Path
    removed = []
    for key in ("raw_path", "clean_path"):
        p = Path(ds[key]) if ds[key] else None
        if p and p.exists():
            p.unlink()
            removed.append(str(p))
    with get_conn() as conn:
        conn.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
    return {"deleted_files": removed, "name": ds["name"]}


# ---------------- 任务 ----------------

def create_task(task_type: str, target: str, params: dict = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO tasks (task_type, target, params, status, created_at) VALUES (?,?,?,?,?)",
            (task_type, target, str(params or {}), "queued", now_str()))
        return cur.lastrowid


def update_task(task_id: int, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    with get_conn() as conn:
        conn.execute(f"UPDATE tasks SET {cols} WHERE id=?", (*fields.values(), task_id))


def append_task_log(task_id: int, line: str) -> None:
    """向任务追加一条日志（带时间戳）。这是"日志必须能在界面查看"的数据来源。"""
    ds = get_task(task_id)
    if ds is None:  # 任务行不存在（如直接调用采集函数）时安全跳过
        return
    old = ds["log"] if ds["log"] else ""
    new_log = f"{old}[{now_str()}] {line}\n"
    with get_conn() as conn:
        conn.execute("UPDATE tasks SET log=? WHERE id=?", (new_log, task_id))


def get_task(task_id: int):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()


def list_tasks(limit: int = 20, task_type: str = None):
    sql, args = "SELECT * FROM tasks WHERE 1=1", []
    if task_type:
        sql += " AND task_type=?"; args.append(task_type)
    sql += " ORDER BY id DESC LIMIT ?"; args.append(limit)
    with get_conn() as conn:
        return conn.execute(sql, args).fetchall()


def recover_stale_tasks() -> int:
    """程序意外退出后，把仍在 running/queued/cancel_requested 的任务标记为 failed
    （状态恢复机制；规范状态机下这些是非终态，重启后不可能还在执行）。"""
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE tasks SET status='failed', error='程序中断：任务未正常结束（可能因应用重启）。可点击重试。', "
            "finished_at=? WHERE status IN ('running','queued','cancel_requested')",
            (now_str(),))
        return cur.rowcount


# ---------------- 机器学习实验（阶段2） ----------------

def get_experiment(exp_id: int):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()


def list_experiments(limit: int = 50):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


# ---------------- 统一任务状态机（阶段4B） ----------------
# 规范状态：queued/running/cancel_requested/succeeded/failed/cancelled/skipped/missed
# 历史旧值 success 通过 TASK_STATUS_ALIASES 只读映射为 succeeded（不改旧记录）。

TASK_STATUS_ALIASES = {"success": "succeeded"}

VALID_TASK_TRANSITIONS = {
    "queued": {"running", "cancelled", "missed", "skipped"},
    "running": {"cancel_requested", "succeeded", "failed"},
    "cancel_requested": {"cancelled", "succeeded", "failed"},  # 取消请求后仍如实记录最终态
    "succeeded": set(), "failed": set(), "cancelled": set(),
    "skipped": set(), "missed": set(),
}


def normalize_status(status: str) -> str:
    """旧状态值 → 规范状态值（只读映射，不改库）。"""
    return TASK_STATUS_ALIASES.get(status, status)


def transition_task(task_id: int, new_status: str, **extra) -> bool:
    """原子状态转换：仅当当前状态允许转换到 new_status 时生效。

    返回是否成功；非法转换抛 ValueError（供测试与页面提示）。
    extra 中的 finished_at/error/message 等字段一并写入。
    """
    new_status = normalize_status(new_status)
    with get_conn() as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise ValueError(f"任务 #{task_id} 不存在。")
        cur_status = normalize_status(row["status"])
        if new_status not in VALID_TASK_TRANSITIONS.get(cur_status, set()):
            raise ValueError(f"非法状态转换：{cur_status} → {new_status}"
                             f"（合法目标：{sorted(VALID_TASK_TRANSITIONS.get(cur_status, set())) or '无，终态'}）")
        cols = ["status=?"]
        vals = [new_status]
        for k, v in extra.items():
            cols.append(f"{k}=?")
            vals.append(v)
        vals.append(task_id)
        cur = conn.execute(f"UPDATE tasks SET {', '.join(cols)} WHERE id=?", vals)
        return cur.rowcount == 1


def get_task_safe(task_id: int):
    """读取任务并把状态规范化显示（历史 success → succeeded）。"""
    t = get_task(task_id)
    if t is None:
        return None
    d = dict(t)
    d["status"] = normalize_status(t["status"])
    return d


# ---------------- 调度配置（阶段4B） ----------------

def save_schedule(name, task_type, params, freq, next_run_at, time_of_day="09:00",
                  weekday=1, interval_minutes=60, misfire_policy="skip",
                  status="enabled") -> int:
    if freq not in ("daily", "weekly", "interval", "once"):
        raise ValueError(f"freq 必须是 daily/weekly/interval/once，收到 {freq}")
    ts = now_str()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO schedules (name, task_type, params_json, freq, time_of_day, "
            "weekday, interval_minutes, status, misfire_policy, next_run_at, "
            "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (name, task_type, json.dumps(params, ensure_ascii=False, default=str), freq,
             time_of_day, weekday, interval_minutes,
             status, misfire_policy, next_run_at, ts, ts))
        return cur.lastrowid


def update_schedule(schedule_id: int, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    with get_conn() as conn:
        conn.execute(f"UPDATE schedules SET {cols}, updated_at=? WHERE id=?",
                     (*fields.values(), now_str(), schedule_id))


def get_schedule(schedule_id: int):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM schedules WHERE id=?", (schedule_id,)).fetchone()


def list_schedules(status: str = None):
    sql, args = "SELECT * FROM schedules WHERE 1=1", []
    if status:
        sql += " AND status=?"; args.append(status)
    sql += " ORDER BY id DESC"
    with get_conn() as conn:
        return conn.execute(sql, args).fetchall()


def schedule_runs(schedule_id: int, limit: int = 20):
    """某调度配置的历史执行（tasks 表按 schedule_id 关联）。"""
    with get_conn() as conn:
        return conn.execute("SELECT * FROM tasks WHERE schedule_id=? ORDER BY id DESC LIMIT ?",
                            (schedule_id, limit)).fetchall()


def run_exists_for_planned(schedule_id: int, planned_at: str) -> bool:
    """防重复执行：同一调度+同一计划时间只允许产生一次执行记录。"""
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM tasks WHERE schedule_id=? AND planned_at=?",
                           (schedule_id, planned_at)).fetchone()
        return row is not None


# ---------------- 回测存档（阶段2） ----------------

def save_backtest_record(**fields) -> int:
    cols = ", ".join(fields.keys())
    ph = ", ".join("?" for _ in fields)
    with get_conn() as conn:
        cur = conn.execute(f"INSERT INTO backtests ({cols}) VALUES ({ph})", tuple(fields.values()))
        return cur.lastrowid


def update_backtest(bt_id: int, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    with get_conn() as conn:
        conn.execute(f"UPDATE backtests SET {cols} WHERE id=?", (*fields.values(), bt_id))


def get_backtest(bt_id: int):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM backtests WHERE id=?", (bt_id,)).fetchone()


def list_backtests(limit: int = 50):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM backtests ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


# ---------------- 参数优化 / 滚动验证存档（阶段3） ----------------

def list_optimization_runs(limit: int = 50):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM optimization_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def get_optimization_run(run_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM optimization_runs WHERE id=?", (run_id,)).fetchone()


def save_walk_forward(**fields) -> int:
    cols = ", ".join(fields.keys())
    ph = ", ".join("?" for _ in fields)
    with get_conn() as conn:
        cur = conn.execute(f"INSERT INTO walk_forward_runs ({cols}) VALUES ({ph})",
                           tuple(fields.values()))
        return cur.lastrowid


def list_walk_forward(limit: int = 50):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM walk_forward_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
