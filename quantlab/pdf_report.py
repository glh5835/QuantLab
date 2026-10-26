"""PDF 研究报告生成（阶段 4C）：中文三级字体回退 + matplotlib 图表 + 分页表格。

字体回退链（任务书 4C.3/E.3 强制）：
  1) 项目字体：data/fonts/*.ttf（用户自行放置，授权由用户确认）
  2) 系统已验证中文字体：Windows 黑体 simhei.ttf（纯 TTF，ReportLab 可注册），
     其次楷体 simkai.ttf / 仿宋 simfang.ttf
  3) ReportLab 内置 CJK CID 字体 STSong-Light（无需字体文件）
全部失败 → 抛 PDFFontError，绝不生成乱码 PDF；页面降级提示并保留 HTML/CSV 出口。

图表：matplotlib（Agg 后端渲染 PNG，中文字体跟随回退结果）→ 嵌入报告。
表格：LongTable 自动分页；页脚含页码/生成时间/实验 ID。
失败降级：图表数据缺失 → 该节明确标注"数据缺失"，不用虚构数字补位。
"""
import os
from datetime import datetime
from pathlib import Path

from . import db
from .config import EXPORT_DIR, DATA_DIR

FONT_CANDIDATES = [
    # (来源说明, 路径或 None)
    ("项目字体目录", None),  # data/fonts/*.ttf 动态扫描
    ("Windows 黑体", "C:/Windows/Fonts/simhei.ttf"),
    ("Windows 楷体", "C:/Windows/Fonts/simkai.ttf"),
    ("Windows 仿宋", "C:/Windows/Fonts/simfang.ttf"),
]


class PDFFontError(RuntimeError):
    """中文显示无法保障：PDF 生成必须失败并给出修复指引。"""


class PDFBuildError(RuntimeError):
    pass


# ---------------- 字体解析（可测、可注入） ----------------

def resolve_font(reportlab_fonts=None, candidates=None) -> dict:
    """解析可用中文字体方案。返回 {method, font_name, path, fallback_used, tested}。

    candidates 可注入用于测试（(来源, 路径) 列表，路径 None 表示项目字体目录）。
    reportlab_fonts 可注入注册表（默认延迟导入 reportlab）。
    """
    rl_fonts = reportlab_fonts
    if rl_fonts is None:
        from reportlab.pdfbase import pdfmetrics
        rl_fonts = pdfmetrics
    cands = candidates if candidates is not None else FONT_CANDIDATES
    # 优先级 1&2：注册 TTF
    for source, path in cands:
        try_paths = []
        if path is None:  # 项目字体目录
            fdir = DATA_DIR / "fonts"
            if fdir.exists():
                try_paths = [(source, p) for p in sorted(fdir.glob("*.ttf"))]
        else:
            try_paths = [(source, Path(path))]
        for src, p in try_paths:
            try:
                if not p.exists() or p.stat().st_size < 100_000:  # 过小的多半是坏文件
                    continue
                from reportlab.pdfbase.ttfonts import TTFont
                name = f"QuantLabCN_{p.stem}"
                rl_fonts.registerFont(TTFont(name, str(p)))
                # 注册成功 ≠ 中文可用：注册后立即可用于绘制（ReportLab TTF 子集化支持 CJK）
                return {"method": f"ttf:{src}", "font_name": name, "path": str(p),
                        "fallback_used": src != cands[0][0]}
            except Exception:  # noqa: BLE001 - 单个字体失败继续回退
                continue
    # 优先级 3：内置 CID 字体
    try:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        rl_fonts.registerFont(UnicodeCIDFont("STSong-Light"))
        return {"method": "cid:STSong-Light", "font_name": "STSong-Light", "path": None,
                "fallback_used": True}
    except Exception as e:  # noqa: BLE001
        raise PDFFontError(
            "无法获得可用的中文字体方案：项目字体目录为空、系统字体不可读、"
            f"内置 CID 字体注册失败（{e}）。请将授权允许的 .ttf 中文字体放入 "
            f"{DATA_DIR / 'fonts'} 后重试；或使用 HTML/CSV 报告作为替代。") from e


# ---------------- 报告构建 ----------------

def _setup_mpl_cn(font_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mpl_font = "SimHei" if "QuantLabCN" in font_name else font_name
    if "QuantLabCN" in font_name:
        # 直接把注册的 ttf 文件喂给 matplotlib（它不认识 reportlab 注册名）
        import glob
        cand = ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simkai.ttf"]
        from matplotlib import font_manager
        added = False
        for c in cand:
            if os.path.exists(c):
                font_manager.fontManager.addfont(c)
                mpl_font = font_manager.FontProperties(fname=c).get_name()
                added = True
                break
        if not added:
            mpl_font = "sans-serif"
    plt.rcParams["font.family"] = [mpl_font, "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _equity_png(dates, values, title, font_name, tmp_dir: Path) -> str | None:
    """净值/回撤曲线 → PNG 路径；数据缺失返回 None（调用方标注，不虚构）。"""
    if dates is None or values is None or len(values) < 2:
        return None
    plt = _setup_mpl_cn(font_name)
    fig, ax = plt.subplots(figsize=(7.2, 2.8), dpi=150)
    ax.plot(dates, values, linewidth=1.2, color="#1f77b4")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = tmp_dir / f"chart_{abs(hash(title)) % 10**8}.png"
    fig.savefig(p)
    plt.close(fig)
    return str(p)


def build_pdf(title: str, meta: dict, sections: list, out_path: Path,
              font: dict = None, warnings: list[str] = None) -> Path:
    """生成 PDF。sections: [(标题, {'table': DataFrame} | {'image': path} | {'text': str})]。

    生成失败（字体/数据/IO）抛 PDFBuildError / PDFFontError，不产出半成品文件。
    """
    font = font or resolve_font()
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Image, LongTable, PageBreak)

    fn = font["font_name"]
    h1 = ParagraphStyle("h1", fontName=fn, fontSize=16, leading=22,
                        textColor=colors.HexColor("#1d4ed8"), spaceAfter=8)
    h2 = ParagraphStyle("h2", fontName=fn, fontSize=12, leading=16,
                        textColor=colors.HexColor("#1d4ed8"), spaceBefore=10)
    body = ParagraphStyle("body", fontName=fn, fontSize=9, leading=13)
    meta_style = ParagraphStyle("meta", fontName=fn, fontSize=8, leading=11,
                                textColor=colors.HexColor("#374151"))

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(fn, 7.5)
        canvas.drawString(1.5 * cm, 1 * cm,
                          f"{meta.get('ID', '')} · 生成于 {datetime.now():%Y-%m-%d %H:%M} · QuantLab")
        canvas.drawRightString(A4[0] - 1.5 * cm, 1 * cm, f"第 {doc.page} 页")
        canvas.restoreState()

    story = [Paragraph(title, h1),
             Paragraph(" ｜ ".join(f"{k}：{v}" for k, v in meta.items()), meta_style),
             Spacer(1, 6)]
    for w in (warnings or []):
        story.append(Paragraph(f"⚠️ {w}", ParagraphStyle(
            "warn", parent=body, textColor=colors.HexColor("#b45309"))))
    for heading, content in sections:
        story.append(Paragraph(heading, h2))
        if "table" in content:
            df = content["table"]
            if df is None or len(df) == 0:
                story.append(Paragraph("（本节数据缺失：无可用记录，不以虚构数据填充）", body))
                continue
            data = [[str(c) for c in df.columns]] + df.astype(str).values.tolist()
            # 宽表缩字号；LongTable 跨页自动续
            t = LongTable(data, repeatRows=1, colWidths=[None] * len(df.columns))
            t.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), fn),
                ("FONTSIZE", (0, 0), (-1, -1), 7 if len(df.columns) > 5 else 8),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eff6ff")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#f9fafb")]),
            ]))
            story.append(t)
        elif "image" in content:
            p = content["image"]
            if p is None:
                story.append(Paragraph("（图表数据缺失，跳过渲染）", body))
                continue
            story.append(Image(p, width=17 * cm, height=6.6 * cm))
        elif "text" in content:
            story.append(Paragraph(str(content["text"]), body))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "免责声明：本报告由 QuantLab 自动生成，全部数字来自真实计算记录；"
        "历史回测与模拟交易结果不构成未来收益保证；含模拟数据的报告仅用于方法论演示。",
        ParagraphStyle("disc", parent=body, textColor=colors.HexColor("#6b7280"), fontSize=7.5)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        SimpleDocTemplate(str(out_path), pagesize=A4,
                          leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                          topMargin=1.8 * cm, bottomMargin=1.6 * cm) \
            .build(story, onFirstPage=_footer, onLaterPages=_footer)
    except PDFFontError:
        raise
    except Exception as e:  # noqa: BLE001
        if out_path.exists():
            out_path.unlink()  # 失败不留半成品
        raise PDFBuildError(f"PDF 构建失败：{type(e).__name__}: {e}") from e
    if out_path.exists() and out_path.stat().st_size < 1024:
        out_path.unlink()
        raise PDFBuildError("PDF 生成异常（文件过小，疑似空报告），已删除。")
    return out_path


# ---------------- 三类报告封装（与页面同源数据） ----------------

def pdf_backtest_report(bt_id: int) -> Path:
    """报告 A：策略回测 PDF。数字与页面/HTML 报告同源（backtests 表+明细 CSV）。"""
    import json
    import pandas as pd
    from .reports import dataset_nature_badge
    rec = db.get_backtest(bt_id)
    if not rec:
        raise PDFBuildError(f"回测 #{bt_id} 不存在。")
    m = json.loads(rec["metrics_json"] or "{}")
    bench = m.get("benchmark", {})
    cfg = json.loads(rec["config_json"] or "{}")
    kpi = pd.DataFrame([
        {"指标": "总收益率", "本策略": f"{m.get('total_return', float('nan')):.2%}",
         "基准(买入持有)": f"{bench.get('total_return', float('nan')):.2%}"},
        {"指标": "年化收益率", "本策略": f"{m.get('cagr', float('nan')):.2%}",
         "基准(买入持有)": f"{bench.get('cagr', float('nan')):.2%}"},
        {"指标": "夏普比率", "本策略": f"{m.get('sharpe', float('nan')):.2f}",
         "基准(买入持有)": f"{bench.get('sharpe', float('nan')):.2f}"},
        {"指标": "最大回撤", "本策略": f"{m.get('mdd', float('nan')):.2%}",
         "基准(买入持有)": f"{bench.get('mdd', float('nan')):.2%}"},
        {"指标": "交易次数", "本策略": m.get("n_trades"),
         "基准(买入持有)": bench.get("n_trades")},
    ])
    daily = pd.read_csv(rec["daily_path"], parse_dates=["date"]) \
        if rec["daily_path"] and Path(rec["daily_path"]).exists() else None
    eq_png = dd_png = None
    if daily is not None:
        tmp = EXPORT_DIR / "_chart_cache"
        tmp.mkdir(parents=True, exist_ok=True)
        font = resolve_font()
        eq_png = _equity_png(daily["date"], daily["equity"] / daily["equity"].iloc[0],
                             "策略净值曲线（期初=1）", font["font_name"], tmp)
        dd = daily["equity"] / daily["equity"].cummax() - 1
        dd_png = _equity_png(daily["date"], dd, "回撤曲线", font["font_name"], tmp)
    trades = pd.read_csv(rec["trades_path"]).head(80) \
        if rec["trades_path"] and Path(rec["trades_path"]).exists() else None

    out = EXPORT_DIR / f"pdf_backtest_bt{bt_id}_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    return build_pdf(
        f"策略回测研究报告 · {rec['name']}",
        {"ID": f"#{rec['id']}", "策略": rec["strategy_type"], "数据集": rec["dataset_name"],
         "区间": f"{rec['start_date']}~{rec['end_date']}",
         "数据性质": dataset_nature_badge([rec["dataset_name"]]),
         "假设": f"初始资金{cfg.get('initial_cash')}，费率{cfg.get('commission_rate')}，"
                 f"滑点{cfg.get('slippage')}，t收盘信号→t+1开盘成交"},
        [("关键指标（对比基准）", {"table": kpi}),
         ("净值与回撤", {"image": eq_png}),
         ("回撤曲线", {"image": dd_png}),
         ("交易明细（前 80 笔）", {"table": trades}),
         ("已知限制", {"text": "单资产现货多头现金账户；未含停牌撮合与涨跌停限制；"
                             "手续费为比例+最低费简化模型。"})],
        out)


def pdf_ml_report(exp_id: int) -> Path:
    """报告 B：机器学习实验 PDF。"""
    import json
    import pandas as pd
    exp = db.get_experiment(exp_id)
    if not exp:
        raise PDFBuildError(f"实验 #{exp_id} 不存在。")
    m = json.loads(exp["metrics_json"] or "{}")
    rows = []
    for seg, label in (("train", "训练"), ("val", "验证"), ("test", "测试(样本外)")):
        s = m.get(seg, {})
        if exp["task"] == "direction":
            rows.append({"数据段": label, "样本": s.get("n"), "Accuracy": f"{s.get('accuracy', float('nan')):.4f}",
                         "F1": f"{s.get('f1', float('nan')):.4f}",
                         "ROC-AUC": (f"{s['roc_auc']:.4f}" if "roc_auc" in s else "不适用")})
        else:
            rows.append({"数据段": label, "样本": s.get("n"), "MAE": f"{s.get('mae', float('nan')):.5f}",
                         "RMSE": f"{s.get('rmse', float('nan')):.5f}",
                         "R²": f"{s.get('r2', float('nan')):.4f}"})
    out = EXPORT_DIR / f"pdf_ml_exp{exp_id}_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    return build_pdf(
        f"机器学习实验报告 · {exp['name']}",
        {"ID": f"#{exp['id']}", "数据集": f"{exp['dataset_name']} v{exp['dataset_version']}",
         "任务": f"{exp['task']}(h={exp['horizon']})", "模型": f"{exp['model_type']}(seed={exp['seed']})",
         "数据性质": dataset_nature([exp["dataset_name"]])},
        [("标签定义", {"text": exp["label_rule"]}),
         ("数据划分", {"table": pd.DataFrame([{
             "段": "训练/验证/测试", "训练区间": exp["train_range"],
             "验证区间": exp["val_range"], "测试区间": exp["test_range"],
             "purge 隔离样本": exp["purged_n"]}])}),
         ("三段评估指标", {"table": pd.DataFrame(rows)}),
         ("解读提醒", {"text": "预测指标（Accuracy/RMSE）衡量模型拟合方向，"
                             "不等于交易盈利能力；训练与样本外差距大即过拟合信号；"
                             "本实验结果不构成投资建议。"})],
        out)


def dataset_nature(names):
    from .reports import dataset_nature_badge
    return dataset_nature_badge(names)


def pdf_portfolio_report(bt_id: int) -> Path:
    """报告 C：投资组合分析 PDF（组合存档亦存于 backtests 表，strategy_type=portfolio）。"""
    import json
    import pandas as pd
    from .reports import dataset_nature_badge
    rec = db.get_backtest(bt_id)
    if not rec:
        raise PDFBuildError(f"组合存档 #{bt_id} 不存在。")
    m = json.loads(rec["metrics_json"] or "{}")
    assets = list((m.get("final_positions") or {}).keys())
    weights_tbl = pd.DataFrame([
        {"资产": a,
         "期末持仓(股)": m["final_positions"][a]["shares"],
         "均价": round(m["final_positions"][a]["avg_cost"], 3),
         "市值": round(m["final_positions"][a]["value"], 0),
         "盈亏贡献(元)": round(m.get("contribution_pnl", {}).get(a, 0), 0)}
        for a in assets]) if assets else None
    kpi = pd.DataFrame([
        {"指标": "总收益率", "数值": f"{m.get('total_return', float('nan')):.2%}"},
        {"指标": "年化波动率", "数值": f"{m.get('ann_vol', float('nan')):.2%}"},
        {"指标": "最大回撤", "数值": f"{m.get('mdd', float('nan')):.2%}"},
        {"指标": "再平衡次数", "数值": m.get("n_rebalances")},
        {"指标": "交易成本(费用+滑点)", "数值": round(m.get("total_fees", 0) + m.get("total_slippage_cost", 0), 0)},
    ])
    daily = pd.read_csv(rec["daily_path"], parse_dates=["date"]) \
        if rec["daily_path"] and Path(rec["daily_path"]).exists() else None
    eq_png = None
    if daily is not None:
        tmp = EXPORT_DIR / "_chart_cache"
        tmp.mkdir(parents=True, exist_ok=True)
        font = resolve_font()
        eq_png = _equity_png(daily["date"], daily["equity"] / daily["equity"].iloc[0],
                             "组合净值曲线（期初=1）", font["font_name"], tmp)
    out = EXPORT_DIR / f"pdf_portfolio_bt{bt_id}_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    return build_pdf(
        f"投资组合分析报告 · {rec['name']}",
        {"ID": f"#{rec['id']}", "资产": rec["dataset_name"],
         "区间": f"{rec['start_date']}~{rec['end_date']}",
         "数据性质": dataset_nature_badge(rec["dataset_name"].split("+"))},
        [("组合关键指标", {"table": kpi}),
         ("净值曲线", {"image": eq_png}),
         ("期末持仓与贡献", {"table": weights_tbl}),
         ("适用条件与限制", {"text": "现金账户、无杠杆无做空；再平衡按日历频率以 t+1 开盘价"
                                  "±滑点成交；历史权重表现不代表未来；若为模拟数据仅演示方法论。"})],
        out)
