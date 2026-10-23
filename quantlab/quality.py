"""数据质量检查：逐项检测问题并给出可解释的问题清单与 0~100 评分。

设计原则（来自项目总指令）：
- 检查只"发现并报告"问题，绝不擅自修改数据（清洗是另一个显式步骤）。
- 每个问题都带：问题描述、数量、示例日期、建议处理方式与依据。
- 评分是加权扣分制，界面同时展示分数与明细，让学生理解"为什么是这个分"。
"""
import json

import numpy as np
import pandas as pd

from .config import QUALITY_THRESHOLDS as TH

# 各问题扣分权重（总分 100，扣完为止）
_WEIGHTS = {
    "missing_values": 25,     # 缺失值
    "duplicates": 15,         # 重复记录
    "nonpositive_price": 20,  # 价格 ≤ 0
    "ohlc_inconsistent": 15,  # high < low 等逻辑矛盾
    "extreme_returns": 15,    # 单日极端涨跌幅
    "date_gap": 10,           # 日期间隔异常（超过 10 个自然日）
    "zero_volume": 5,         # 长期零成交
}


def check_quality(df: pd.DataFrame) -> dict:
    """对标准 OHLCV DataFrame 执行全套质量检查，返回报告字典。"""
    issues = []

    # 1. 缺失值
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if len(miss):
        for col, n in miss.items():
            severity = "严重" if n / len(df) > TH["max_missing_ratio"] else "一般"
            issues.append({
                "type": "missing_values", "severity": severity,
                "count": int(n),
                "detail": f"字段 {col} 有 {n} 个缺失值（占 {n/len(df):.1%}）",
                "example_dates": [str(d.date()) for d in df[df[col].isna()].index[:5]],
                "suggestion": "可用前值填充（停牌常导致缺失，前值填充符合价格连续性假设）；"
                              "缺失率过高时建议重新采集或换数据源。依据：不填补直接删除会破坏交易日连续性。",
            })

    # 2. 重复交易日期（storage 已保证无重复索引，但值层面重复行仍值得报告）
    dup_rows = int(df.duplicated().sum())
    if dup_rows:
        issues.append({
            "type": "duplicates", "severity": "一般", "count": dup_rows,
            "detail": f"存在 {dup_rows} 行完全相同的记录",
            "example_dates": [str(d.date()) for d in df[df.duplicated()].index[:5]],
            "suggestion": "建议删除重复行（同一交易日同一价格属于冗余记录，删除不损失信息）。",
        })

    # 3. 价格非正数
    bad_price = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bad_price.any():
        issues.append({
            "type": "nonpositive_price", "severity": "严重", "count": int(bad_price.sum()),
            "detail": f"{int(bad_price.sum())} 个交易日的价格 ≤ 0（不可能出现，属于脏数据）",
            "example_dates": [str(d.date()) for d in df[bad_price].index[:5]],
            "suggestion": "必须处理：剔除这些日期或整列重新采集。价格非正会使对数收益率无法计算。",
        })

    # 4. OHLC 逻辑一致性：high 应 ≥ max(open,close,low)，low 应 ≤ min(open,close,high)
    ohlc_bad = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | \
               (df["low"] > df[["open", "close", "high"]].min(axis=1))
    if ohlc_bad.any():
        issues.append({
            "type": "ohlc_inconsistent", "severity": "一般", "count": int(ohlc_bad.sum()),
            "detail": f"{int(ohlc_bad.sum())} 个交易日最高/最低价与开收盘价矛盾",
            "example_dates": [str(d.date()) for d in df[ohlc_bad].index[:5]],
            "suggestion": "多为数据源错误。可保守处理：以 open/close 为准修正 high/low，或标记后剔除。",
        })

    # 5. 极端单日收益率
    ret = df["close"].pct_change(fill_method=None)
    extreme = ret.abs() > TH["extreme_daily_return"]
    extreme = extreme & ret.notna()
    if extreme.any():
        issues.append({
            "type": "extreme_returns", "severity": "提醒", "count": int(extreme.sum()),
            "detail": f"{int(extreme.sum())} 个交易日涨跌幅超过 ±{TH['extreme_daily_return']:.0%}"
                      f"（可能是拆股/分红未复权，或真实暴跌）",
            "example_dates": [str(d.date()) for d in df[extreme].index[:5]],
            "suggestion": "先核对公司公告：若是拆股应使用复权价；若真实极端行情则保留（删除会美化风险）。",
        })

    # 6. 日期间隔异常（超过 10 个自然日无数据，提示漏采或长期停牌）
    gaps = df.index.to_series().diff().dt.days
    gap_days = gaps[gaps > 10]
    if len(gap_days):
        issues.append({
            "type": "date_gap", "severity": "一般", "count": int(len(gap_days)),
            "detail": f"{len(gap_days)} 处日期间隔超过 10 天（正常节假日最长约 8 天）",
            "example_dates": [f"{(d - pd.Timedelta(days=int(v))).date()} → {d.date()}（间隔 {int(v)} 天）"
                              for d, v in gap_days.head(5).items()],
            "suggestion": "检查是否漏采：可尝试重新采集该区间；若为长期停牌则属正常，保留即可。",
        })

    # 7. 零成交占比
    zero_vol = (df["volume"] == 0) | df["volume"].isna()
    ratio = zero_vol.mean() if len(df) else 0
    if ratio > TH["zero_volume_warn_ratio"]:
        issues.append({
            "type": "zero_volume", "severity": "提醒", "count": int(zero_vol.sum()),
            "detail": f"零成交/无成交记录占比 {ratio:.1%}（阈值 {TH['zero_volume_warn_ratio']:.0%}）",
            "example_dates": [str(d.date()) for d in df[zero_vol].index[:5]],
            "suggestion": "常见于停牌日或指数数据（指数无成交量概念）。指数类数据可忽略此提醒。",
        })

    # 评分：按问题类型加权扣分（同类型只扣一次，与数量无关——数量在明细里体现）
    score = 100
    applied = set()
    for iss in issues:
        if iss["type"] not in applied:
            score -= _WEIGHTS[iss["type"]]
            applied.add(iss["type"])
    score = max(0, score)

    return {
        "score": score,
        "rows": int(len(df)),
        "date_range": [str(df.index.min().date()), str(df.index.max().date())],
        "issue_count": len(issues),
        "issues": issues,
        "thresholds": TH,
        "weights": _WEIGHTS,
    }


def quality_summary_json(report: dict) -> str:
    """报告序列化为 JSON 存库（quality_report 字段）。"""
    return json.dumps(report, ensure_ascii=False)
