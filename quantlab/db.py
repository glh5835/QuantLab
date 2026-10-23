"""SQLite 元数据库层：数据集注册表、任务表、采集日志表。

设计说明（教学要点）：
- 使用 Python 标准库 sqlite3，单文件数据库，零配置，适合个人电脑。
- 所有写操作立即 commit，保证程序意外退出后状态不丢失。
- 这个文件只负责"元数据"，行情数据本体存在 data/raw 与 data/clean 的 CSV 文件中。
"""
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
"""


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
    """初始化所有表。应用每次启动时调用（CREATE IF NOT EXISTS 是幂等的）。"""
    with get_conn() as conn:
        conn.executescript(_SCHEMA)


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
    """程序意外退出后，把仍在 running/queued 的任务标记为 failed（状态恢复机制）。"""
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE tasks SET status='failed', error='程序中断：任务未正常结束（可能因应用重启）。可点击重试。', "
            "finished_at=? WHERE status IN ('running','queued')", (now_str(),))
        return cur.rowcount


# ---------------- 机器学习实验（阶段2） ----------------

def get_experiment(exp_id: int):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()


def list_experiments(limit: int = 50):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


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
