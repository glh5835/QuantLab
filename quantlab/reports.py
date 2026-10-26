"""统一实验比较与研究报告导出（阶段 3E）。

- compare_backtests：多回测记录横向对比 + 条件一致性检查（数据集/区间/成本假设
  不同时明确展示差异，不冒充公平比较）。
- build_html_report / export_report：把页面同源数据组装成单文件 HTML 报告
  （内嵌样式与表格，无外部依赖），CSV 报告直接落盘。
- 报告必须包含：实验 ID、数据版本、时间区间、主要假设与数据性质标注（模拟/真实）。
"""
import html as _html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import db
from .config import EXPORT_DIR


# ---------------- 实验比较 ----------------

_COMPARE_KEYS = [
    ("total_return", "总收益率"), ("cagr", "年化收益率"), ("ann_vol", "年化波动率"),
    ("sharpe", "夏普比率"), ("mdd", "最大回撤"), ("n_trades", "交易次数"),
]


def compare_backtests(bt_ids: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """横向对比多条回测存档。返回 (对比表, 条件差异提示列表)。"""
    rows, diffs = [], []
    recs = []
    for bid in bt_ids:
        rec = db.get_backtest(int(bid))
        if not rec:
            raise ValueError(f"回测 #{bid} 不存在。")
        recs.append(rec)
    if len(recs) < 2:
        raise ValueError("比较至少需要选择 2 条回测记录。")

    # 条件一致性：数据集 / 区间 / 成本配置
    def _cfg(rec):
        try:
            return json.loads(rec["config_json"] or "{}")
        except Exception:  # noqa: BLE001
            return {}
    ref = recs[0]
    for rec in recs[1:]:
        if rec["dataset_name"] != ref["dataset_name"]:
            diffs.append(f"数据集不同：#{rec['id']} 用 {rec['dataset_name']}，"
                         f"#{ref['id']} 用 {ref['dataset_name']}——收益不可直接比较。")
        if (rec["start_date"][:7], rec["end_date"][:7]) != \
           (ref["start_date"][:7], ref["end_date"][:7]):
            diffs.append(f"回测区间不同：#{rec['id']} {rec['start_date']}~{rec['end_date']}"
                         f" vs #{ref['id']} {ref['start_date']}~{ref['end_date']}。")
        for k in ("commission_rate", "slippage"):
            a, b = _cfg(rec).get(k), _cfg(ref).get(k)
            if a != b:
                diffs.append(f"成本假设不同（{k}）：#{rec['id']}={a} vs #{ref['id']}={b}。")

    for rec in recs:
        m = json.loads(rec["metrics_json"] or "{}")
        row = {"ID": rec["id"], "名称": rec["name"], "策略": rec["strategy_type"],
               "数据集": rec["dataset_name"]}
        for key, label in _COMPARE_KEYS:
            v = m.get(key)
            row[label] = (f"{v:.2%}" if key in ("total_return", "cagr", "ann_vol", "mdd")
                          and v is not None else (f"{v:.2f}" if isinstance(v, float) else v))
        rows.append(row)
    return pd.DataFrame(rows), diffs


# ---------------- HTML 报告 ----------------

_CSS = """
body{font-family:'Microsoft YaHei',system-ui,sans-serif;max-width:960px;margin:24px auto;
     padding:0 16px;color:#1f2937;line-height:1.6}
h1{border-bottom:3px solid #2563eb;padding-bottom:8px} h2{margin-top:28px;color:#1d4ed8}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}
th,td{border:1px solid #d1d5db;padding:6px 10px;text-align:left}
th{background:#eff6ff} tr:nth-child(even){background:#f9fafb}
.meta{background:#f3f4f6;border-radius:8px;padding:10px 14px;font-size:13px;color:#374151}
.warn{background:#fef3c7;border-left:4px solid #f59e0b;padding:8px 12px;margin:8px 0}
.note{color:#6b7280;font-size:13px} footer{margin-top:32px;border-top:1px solid #e5e7eb;
 padding-top:8px;font-size:12px;color:#9ca3af}
"""


def _df_to_html(df: pd.DataFrame) -> str:
    # escape 默认 True：单元格内容一律转义，杜绝 HTML 注入
    return df.to_html(index=False, border=0, na_rep="—")


def build_html_report(title: str, meta: dict, sections: list[tuple[str, object]],
                      warnings: list[str] = None) -> str:
    """sections: [(标题, DataFrame 或 str)]。meta 至少含 数据性质/区间/ID。"""
    parts = [f"<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
             f"<title>{_html.escape(title)}</title><style>{_CSS}</style></head><body>",
             f"<h1>{_html.escape(title)}</h1>"]
    meta_lines = " ｜ ".join(f"{k}：{_html.escape(str(v))}" for k, v in meta.items())
    parts.append(f"<div class='meta'>{meta_lines}<br>生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}</div>")
    for w in (warnings or []):
        parts.append(f"<div class='warn'>⚠️ {_html.escape(w)}</div>")
    for heading, content in sections:
        parts.append(f"<h2>{_html.escape(heading)}</h2>")
        if isinstance(content, pd.DataFrame):
            if len(content) == 0:
                parts.append("<p class='note'>（无数据）</p>")
            else:
                parts.append(_df_to_html(content))
        else:
            parts.append(f"<div>{content}</div>")  # 调用方自行转义富文本
    parts.append("<footer>QuantLab Student 自动生成 · 图表与指标均来自真实计算记录 · "
                 "历史回测与滚动验证结果不构成未来收益保证</footer></body></html>")
    return "\n".join(parts)


def save_report(content: str, stem: str, suffix: str = ".html") -> Path:
    """落盘到 data/exports/，返回路径。stem 自动带时间戳防覆盖。"""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPORT_DIR / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}{suffix}"
    out.write_text(content, encoding="utf-8")
    return out


def dataset_nature_badge(dataset_names: list[str]) -> str:
    """报告的数据性质标注：任一为模拟 → 整体标注模拟。"""
    synthetic = []
    for name in dataset_names:
        for d in db.list_datasets():
            if d["name"] == name and d["is_synthetic"]:
                synthetic.append(name)
    if synthetic:
        return f"🎭 含模拟数据（{', '.join(sorted(set(synthetic)))}）——结果不代表真实市场表现"
    return "真实/导入数据"


# ---------------- 回测存档的完整研究报告 ----------------

def backtest_report(bt_id: int) -> tuple[str, Path]:
    """生成单条回测的完整 HTML 研究报告（与页面同源数据）。返回 (html, 路径)。"""
    rec = db.get_backtest(bt_id)
    if not rec:
        raise ValueError(f"回测 #{bt_id} 不存在。")
    m = json.loads(rec["metrics_json"] or "{}")
    bench = m.get("benchmark", {})

    kpis = pd.DataFrame([{
        "指标": "总收益率", "数值": f"{m.get('total_return', float('nan')):.2%}",
        "基准（买入持有）": f"{bench.get('total_return', float('nan')):.2%}"},
        {"指标": "年化收益率", "数值": f"{m.get('cagr', float('nan')):.2%}",
         "基准（买入持有）": f"{bench.get('cagr', float('nan')):.2%}"},
        {"指标": "夏普比率", "数值": f"{m.get('sharpe', float('nan')):.2f}",
         "基准（买入持有）": f"{bench.get('sharpe', float('nan')):.2f}"},
        {"指标": "最大回撤", "数值": f"{m.get('mdd', float('nan')):.2%}",
         "基准（买入持有）": f"{bench.get('mdd', float('nan')):.2%}"},
        {"指标": "交易次数", "数值": m.get("n_trades"),
         "基准（买入持有）": bench.get("n_trades")},
        {"指标": "费用+滑点成本", "数值": round(m.get("total_fees", 0) + m.get("total_slippage_cost", 0), 2),
         "基准（买入持有）": round(bench.get("total_fees", 0), 2)}])
    cfg = json.loads(rec["config_json"] or "{}")
    trades = pd.read_csv(rec["trades_path"]) if rec["trades_path"] and \
        Path(rec["trades_path"]).exists() else pd.DataFrame()

    meta = {"回测ID": f"#{rec['id']} {rec['name']}", "策略": rec["strategy_type"],
            "数据集": f"{rec['dataset_name']}",
            "区间": f"{rec['start_date']}~{rec['end_date']}",
            "数据性质": dataset_nature_badge([rec["dataset_name"]]),
            "成本假设": f"费率{cfg.get('commission_rate')}/滑点{cfg.get('slippage')}"}
    content = build_html_report(
        f"回测研究报告 · {rec['name']}", meta,
        [("关键指标（vs 基准）", kpis),
         ("成交模型与假设", f"<p class='note'>{_html.escape(m.get('assumptions', ''))}</p>"),
         ("交易明细（前 50 笔）", trades.head(50))])
    path = save_report(content, f"backtest_report_bt{bt_id}")
    return content, path
