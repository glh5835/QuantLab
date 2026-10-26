"""PDF 报告测试：字体回退 / 中文可读性(pypdf 提取) / 三类报告 / 分页 / 失败降级。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab import pdf_report as PR
from quantlab.pdf_report import PDFFontError, PDFBuildError, resolve_font, build_pdf
from quantlab.config import EXPORT_DIR


def _read_pdf_text(path) -> str:
    from pypdf import PdfReader
    return "".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def _pdf_pages(path) -> int:
    from pypdf import PdfReader
    return len(PdfReader(str(path)).pages)


# ---------------- 字体回退 ----------------

def test_resolve_font_prefers_project_font(tmp_path, monkeypatch):
    fdir = tmp_path / "fonts"
    fdir.mkdir()
    # 复制系统黑体作为"项目字体"（授权：随 Windows 系统分发的界面字体，仅本机使用）
    src = Path("C:/Windows/Fonts/simhei.ttf")
    if not src.exists():
        pytest.skip("本机无 simhei.ttf")
    (fdir / "myfont.ttf").write_bytes(src.read_bytes())
    monkeypatch.setattr(PR, "DATA_DIR", tmp_path)  # PR 模块内已导入的 DATA_DIR 引用
    font = PR.resolve_font(candidates=[("项目字体目录", None)])
    assert font["method"].startswith("ttf:项目字体")
    assert not font["fallback_used"]


def test_resolve_font_falls_back_to_system_then_cid(monkeypatch):
    # 候选全为不存在的路径 + 项目目录为空 → 回退 CID
    import quantlab.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", Path("Z:/不存在的目录"), raising=False)
    font = PR.resolve_font(candidates=[("项目字体目录", None),
                                       ("坏字体", "Z:/nope.ttf")])
    assert font["method"] == "cid:STSong-Light" and font["fallback_used"]


def test_resolve_font_all_fail_raises_with_guidance(monkeypatch):
    import quantlab.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", Path("Z:/不存在"), raising=False)
    with pytest.raises(PDFFontError, match="字体"):
        PR.resolve_font(candidates=[("项目字体目录", None), ("坏", "Z:/x.ttf")],
                        reportlab_fonts=_BrokenRegistry())
    # 提示必须包含修复指引（HTML 替代）
    try:
        PR.resolve_font(candidates=[("项目字体目录", None), ("坏", "Z:/x.ttf")],
                        reportlab_fonts=_BrokenRegistry())
    except PDFFontError as e:
        assert "HTML" in str(e) or "fonts" in str(e)


class _BrokenRegistry:
    """模拟注册全部失败的 pdfmetrics 替身。"""
    def registerFont(self, *a, **k):
        raise RuntimeError("注册失败（测试注入）")


# ---------------- PDF 构建与中文可读性 ----------------

def test_build_pdf_chinese_readable_and_pagenum(tmp_path):
    font = resolve_font()
    long_tbl = pd.DataFrame({"中文列": [f"第{i}行数据" for i in range(200)]})
    out = tmp_path / "t.pdf"
    build_pdf("中文标题回测报告", {"ID": "#42", "数据性质": "模拟"},
              [("中文正文段落", {"text": "策略在样本外的表现为收益与回撤的权衡。"}),
               ("长表格（触发分页）", {"table": long_tbl}),
               ("缺失图表", {"image": None})],
              out, font=font)
    assert out.exists() and out.stat().st_size > 10_000
    text = _read_pdf_text(out)
    assert "中文标题回测报告" in text            # 中文标题可提取（非方框乱码）
    assert "第0行数据" in text and "第199行数据" in text
    assert "第 1 页" in text or "第 2 页" in text  # 页码存在
    assert _pdf_pages(out) >= 2                  # 长表分页
    assert "图表数据缺失" in text                 # 缺数据如实标注


def test_build_pdf_failure_leaves_no_file(tmp_path):
    font = resolve_font()
    out = tmp_path / "bad.pdf"
    with pytest.raises(Exception):
        build_pdf("t", {}, [("崩", {"table": _BoomDf({"a": [1, 2, 3]})})], out, font=font)
    assert not out.exists()                      # 失败不留半成品


class _BoomDf(pd.DataFrame):
    def astype(self, *a, **k):
        raise RuntimeError("数据炸了（测试注入）")


# ---------------- 三类真实报告 ----------------

def _mk_bt():
    from quantlab import backtest as BT
    from quantlab.backtest import BacktestConfig
    from quantlab.strategies import BuyHoldStrategy
    from quantlab.storage import save_raw
    import quantlab.db as db
    rng = np.random.default_rng(1)
    n = 260
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n)))
    df = pd.DataFrame({"open": close * 0.998, "high": close * 1.02, "low": close * 0.98,
                       "close": close, "volume": 1e6},
                      index=pd.bdate_range("2024-01-01", periods=n))
    ds_id = db.register_dataset("PDF_BT", "PDFB", source="demo", is_synthetic=1)
    db.update_dataset(ds_id, raw_path=save_raw(df, "PDF_BT"), rows=n)
    r = BT.run_backtest(BuyHoldStrategy(), ds_id, BacktestConfig(), name="PDF测试回测")
    return r


def test_backtest_pdf_numbers_match_record():
    r = _mk_bt()
    p = PR.pdf_backtest_report(r["bt_id"])
    assert p.exists() and p.suffix == ".pdf"
    text = _read_pdf_text(p)
    m = r["metrics"]
    assert f"#{r['bt_id']}" in text
    assert f"{m['total_return']:.2%}" in text       # PDF 数字与回测记录一致
    assert "含模拟数据" in text                       # 数据性质标注
    assert "免责声明" in text
    # 图表页存在（页数 ≥2：KPI+图表+明细）
    assert _pdf_pages(p) >= 2


def test_ml_pdf_report():
    import quantlab.db as db
    from quantlab import ml_lab
    from quantlab.storage import save_raw
    from tests.test_ml_lab import _synthetic_df, FEATS
    ds_id = db.register_dataset("PDF_ML", "PDFM", source="demo", is_synthetic=1)
    db.update_dataset(ds_id, raw_path=save_raw(_synthetic_df(420), "PDF_ML"), rows=420)
    res = ml_lab.run_experiment(ds_id, "direction", 1, FEATS, "logistic", name="PDF实验")
    p = PR.pdf_ml_report(res["exp_id"])
    text = _read_pdf_text(p)
    assert "机器学习实验报告" in text and f"#{res['exp_id']}" in text
    assert "测试(样本外)" in text
    assert "不等于交易盈利能力" in text               # 预测≠盈利的明确区分


def test_portfolio_pdf_report():
    import quantlab.db as db
    from quantlab import portfolio as PF
    from quantlab.portfolio import PortfolioConfig
    from quantlab.storage import save_raw
    rng = np.random.default_rng(2)
    n = 300
    price_map = {}
    for name in ("PA", "PB"):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.015, n)))
        df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                           "close": close, "volume": 1e6},
                          index=pd.bdate_range("2024-01-01", periods=n))
        ds_id = db.register_dataset(f"PDF_PF_{name}", name, source="demo", is_synthetic=1)
        db.update_dataset(ds_id, raw_path=save_raw(df, f"PDF_PF_{name}"), rows=n)
        price_map[name] = df
    res = PF.run_portfolio(price_map, {"PA": 0.5, "PB": 0.5},
                           PortfolioConfig(rebalance="monthly"))
    bt_id = PF.save_portfolio_run("PDF组合测试", {"PA": 1, "PB": 2}, "equal",
                                  PortfolioConfig(rebalance="monthly"), res,
                                  res["daily"][0]["date"], res["daily"][-1]["date"])
    p = PR.pdf_portfolio_report(bt_id)
    text = _read_pdf_text(p)
    assert "投资组合分析报告" in text
    assert "盈亏贡献" in text and "无杠杆无做空" in text
