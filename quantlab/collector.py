"""采集任务：把"下载→校验→保存原始数据→注册数据集"编排为一个可取消的后台任务。

增量更新逻辑：若数据集已存在本地数据，只下载 [本地最新日期+1, 今天] 的缺失区间，
拼接后整体覆盖保存为新版本（版本号 +1，历史文件被替换但版本号留在数据库中）。
"""
from datetime import date, timedelta

import pandas as pd

from . import db, tasks
from .datasources.base import get_adapter
from .storage import load_csv, save_raw, validate_ohlcv


def _log(task_id, line):
    db.append_task_log(task_id, line)


def run_collect_task(task_id: int, cancel, symbol: str, source: str,
                     start: str, end: str, asset_type: str, name: str):
    """后台采集函数（由 tasks.run_in_background 调度）。

    逐阶段写进度与日志，任何失败抛异常由任务系统记录。
    """
    adapter = get_adapter(source)
    _log(task_id, f"开始采集 {symbol}（来源 {adapter.display_name}，区间 {start} ~ {end}）")
    db.update_task(task_id, progress=0.05, message=f"正在下载 {symbol} 行情…")

    dataset_name = name or f"{symbol}_daily"
    old_df = None
    old = db.find_dataset(dataset_name)
    if old and old["raw_path"]:
        try:
            old_df = load_csv(old["raw_path"])
            # 增量更新：起点改为本地最新日期的下一天
            next_day = (old_df.index.max() + timedelta(days=1)).date().isoformat()
            if next_day > end:
                _log(task_id, f"本地数据已覆盖至 {end}，无需更新。")
                db.update_task(task_id, message="数据已是最新", progress=1.0)
                return
            _log(task_id, f"检测到本地数据（{len(old_df)} 行，最新 {old_df.index.max().date()}），"
                          f"改为增量更新：从 {next_day} 开始下载")
            start = next_day
        except Exception as e:  # noqa: BLE001 - 旧文件损坏时退回全量下载
            _log(task_id, f"读取本地旧数据失败（{e}），退回全量下载")
            old_df = None

    if cancel.is_set():
        _log(task_id, "任务已被用户取消（下载前）。")
        return

    df_new = adapter.fetch(symbol, start=start, end=end)  # 失败会抛 DataFetchError
    _log(task_id, f"下载完成：{len(df_new)} 行（{df_new.index.min().date()} ~ {df_new.index.max().date()}）")
    db.update_task(task_id, progress=0.6, message="下载完成，正在校验数据…")

    if old_df is not None:
        df = pd.concat([old_df, df_new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        _log(task_id, f"与本地数据拼接后共 {len(df)} 行")
    else:
        df = df_new.sort_index()

    validate_ohlcv(df)  # 契约校验：字段、索引、重复
    _log(task_id, "数据契约校验通过（OHLCV 字段完整、日期升序、无重复）")

    if cancel.is_set():
        _log(task_id, "任务已被用户取消（保存前）。")
        return

    db.update_task(task_id, progress=0.85, message="正在保存原始数据…")
    dataset_id = db.register_dataset(
        dataset_name, symbol, source=source, asset_type=asset_type,
        is_synthetic=int(adapter.is_synthetic),
        license_note=adapter.license_note, field_desc=adapter.field_desc)
    raw_path = save_raw(df, dataset_name)
    db.update_dataset(
        dataset_id, raw_path=raw_path,
        start_date=str(df.index.min().date()), end_date=str(df.index.max().date()),
        rows=len(df), quality_score=None, quality_report="", cleaned=0, cleaning_log="",
        clean_path="")
    _log(task_id, f"原始数据已保存：{raw_path}")
    _log(task_id, f"数据集 “{dataset_name}” 已注册/更新（当前版本 v{db.get_dataset(dataset_id)['version']}）")


def start_collection(symbol: str, source: str, start: str, end: str = None,
                     asset_type: str = "stock", name: str = None) -> int:
    """创建采集任务并在后台启动，返回 task_id 供界面轮询。"""
    end = end or date.today().isoformat()
    task_id = db.create_task("collect", symbol or name,
                             {"symbol": symbol, "source": source, "start": start,
                              "end": end, "asset_type": asset_type, "name": name})
    tasks.run_in_background(task_id, run_collect_task, symbol, source, start, end,
                            asset_type, name)
    return task_id


def export_dataset_csv(dataset_id: int, export_path: str) -> str:
    """导出数据集为 CSV（模块 2 的导出功能）。"""
    ds = db.get_dataset(dataset_id)
    if not ds or not ds["raw_path"]:
        raise FileNotFoundError("数据集不存在或尚无数据文件。")
    df = load_csv(ds["raw_path"])
    df.to_csv(export_path, index_label="trade_date", encoding="utf-8-sig")  # Excel 友好编码
    return export_path
