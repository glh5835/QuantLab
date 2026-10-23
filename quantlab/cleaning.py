"""数据清洗：显式选择清洗操作，原始数据保留，清洗结果另存，操作全程留痕。

教学要点：清洗不是"一键全修"。每个操作都要学生勾选并看到依据，
执行的操作逐条记录进 datasets.cleaning_log，清洗前后差异用对比报告呈现。
"""
import json
from datetime import datetime

import pandas as pd

from . import db
from .storage import load_csv, save_clean

# 可用清洗操作注册表：key -> (函数, 说明/依据)
OPERATIONS = {
    "drop_duplicates": (
        "删除完全重复的行",
        "依据：同一交易日出现完全相同的记录属于数据源冗余，删除不损失任何信息。"),
    "drop_nonpositive": (
        "剔除价格 ≤ 0 的记录",
        "依据：价格为负或 0 在现实中不可能，且会导致对数收益率计算失败。属于必须剔除的脏数据。"),
    "ffill_missing": (
        "缺失值用前值填充（仅价格字段）",
        "依据：停牌期间价格未变，用前一交易日价格填充符合价格连续性假设；成交量缺失填 0 而非前值。"),
    "fix_ohlc": (
        "修正 OHLC 逻辑矛盾（以 open/close 为准收紧 high/low）",
        "依据：high<low 等矛盾来自数据源错误，按开收盘价修正最保守，不引入外部信息。"),
}


def _apply(df: pd.DataFrame, op: str) -> tuple[pd.DataFrame, int, str]:
    """执行单个清洗操作，返回 (新df, 影响行数, 描述)。"""
    before = len(df)
    if op == "drop_duplicates":
        df = df[~df.duplicated(keep="first")]
        df = df[~df.index.duplicated(keep="first")]
        return df, before - len(df), "删除重复行"
    if op == "drop_nonpositive":
        mask = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        return df[~mask], int(mask.sum()), "剔除价格≤0 的记录"
    if op == "ffill_missing":
        n_missing = int(df.isna().sum().sum())
        price_cols = ["open", "high", "low", "close"]
        df[price_cols] = df[price_cols].ffill()
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0)
        return df, n_missing, "前值填充价格缺失 / 成交量填 0"
    if op == "fix_ohlc":
        upper = df[["open", "close"]].max(axis=1)
        lower = df[["open", "close"]].min(axis=1)
        n_fix = int(((df["high"] < upper) | (df["low"] > lower)).sum())
        df["high"] = df["high"].clip(lower=upper)
        df["low"] = df["low"].clip(upper=lower)
        return df, n_fix, "修正 OHLC 逻辑矛盾"
    raise ValueError(f"未知清洗操作: {op}")


def run_cleaning(dataset_id: int, operations: list[str], task_id: int = None) -> dict:
    """对数据集执行勾选的清洗操作，生成对比报告并另存清洗版本。

    返回：{before_rows, after_rows, changes: [{op, affected, note, reason}], clean_path}
    """
    ds = db.get_dataset(dataset_id)
    if not ds or not ds["raw_path"]:
        raise FileNotFoundError("数据集不存在或没有原始数据文件。")
    raw = load_csv(ds["raw_path"])

    df = raw.copy()
    changes = []
    for op in operations:
        if op not in OPERATIONS:
            continue
        df, affected, note = _apply(df, op)
        title, reason = OPERATIONS[op]
        changes.append({"op": op, "title": title, "affected": affected, "reason": reason})
        if task_id:
            db.append_task_log(task_id, f"执行 {title}：影响 {affected} 行")

    clean_path = save_clean(df, ds["name"])

    report = {
        "cleaned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "before_rows": int(len(raw)), "after_rows": int(len(df)),
        "before_missing": int(raw.isna().sum().sum()),
        "after_missing": int(df.isna().sum().sum()),
        "changes": changes,
    }
    db.update_dataset(dataset_id, cleaned=1, clean_path=clean_path,
                      cleaning_log=json.dumps(report, ensure_ascii=False))
    report["clean_path"] = clean_path
    return report
