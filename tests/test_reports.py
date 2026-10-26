"""报告与比较测试：条件差异披露 / HTML 结构与转义 / 报告落盘 / 数据性质标注。"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab import reports as RP
from quantlab import backtest as BT
from quantlab.backtest import BacktestConfig
from quantlab.strategies import BuyHoldStrategy, MACrossoverStrategy
from quantlab.storage import save_raw


def _mk_ds(name, seed=1, n=260):
    import quantlab.db as db
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n)))
    df = pd.DataFrame({"open": close * 0.998, "high": close * 1.02, "low": close * 0.98,
                       "close": close, "volume": 1e6},
                      index=pd.bdate_range("2024-01-01", periods=n))
    ds_id = db.register_dataset(name, name[:6], source="demo", is_synthetic=1)
    db.update_dataset(ds_id, raw_path=save_raw(df, name), rows=n)
    return ds_id


def test_compare_backtests_flags_condition_differences():
    ds = _mk_ds("RP_DS1")
    r1 = BT.run_backtest(BuyHoldStrategy(), ds, BacktestConfig(commission_rate=0.001))
    r2 = BT.run_backtest(MACrossoverStrategy(5, 20), ds, BacktestConfig(commission_rate=0.003))
    df, diffs = RP.compare_backtests([r1["bt_id"], r2["bt_id"]])
    assert {"总收益率", "夏普比率", "最大回撤"} <= set(df.columns)
    assert any("成本假设不同" in d for d in diffs)   # 费率差异必须披露
    ds2 = _mk_ds("RP_DS2", seed=2)
    r3 = BT.run_backtest(BuyHoldStrategy(), ds2, BacktestConfig(commission_rate=0.001))
    _, diffs2 = RP.compare_backtests([r1["bt_id"], r3["bt_id"]])
    assert any("数据集不同" in d for d in diffs2)


def test_compare_requires_two():
    ds = _mk_ds("RP_DS3")
    r = BT.run_backtest(BuyHoldStrategy(), ds, BacktestConfig())
    with pytest.raises(ValueError, match="至少"):
        RP.compare_backtests([r["bt_id"]])


def test_html_report_structure_and_escaping():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "<script>"]})
    content = RP.build_html_report("测试<b>报告</b>", {"ID": "#1", "数据性质": "模拟"},
                                   [("表格", df), ("说明", "纯文本")],
                                   warnings=["回撤超限<script>"])
    assert "<table" in content and "测试&lt;b&gt;报告&lt;/b&gt;" in content  # 标题转义
    assert "<script>" not in content        # 内容中的脚本被转义，不会注入
    assert "模拟" in content and "⚠️ 回撤超限" in content
    assert "QuantLab Student 自动生成" in content  # 页脚免责


def test_backtest_report_written_to_disk():
    ds = _mk_ds("RP_DS4", seed=4)
    r = BT.run_backtest(MACrossoverStrategy(5, 20), ds, BacktestConfig())
    content, path = RP.backtest_report(r["bt_id"])
    assert path.exists() and path.suffix == ".html"
    text = path.read_text(encoding="utf-8")
    assert "关键指标（vs 基准）" in text
    assert "含模拟数据" in text                       # 模拟数据性质必须标注
    assert f"#{r['bt_id']}" in text                   # 实验 ID 在报告中


def test_nature_badge_mixed():
    _mk_ds("RP_REAL_LIKE", seed=5)
    import quantlab.db as db
    db.register_dataset("RP_FAKE", "RP_F", source="demo", is_synthetic=1)
    assert "模拟" in RP.dataset_nature_badge(["RP_REAL_LIKE", "RP_FAKE"])
