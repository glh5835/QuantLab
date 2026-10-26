"""文件导入测试：CSV 与 Excel（.xlsx/.xls）统一规范化、错误提示、兼容性。"""
import pandas as pd
import pytest

from quantlab.datasources.csv_adapter import import_file, import_csv, CsvImportAdapter
from quantlab.datasources.base import DataFetchError


def _sample_cn():
    return pd.DataFrame({
        "日期": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "开盘": [10.0, 10.5, 11.0], "最高": [10.8, 11.0, 11.5],
        "最低": [9.9, 10.2, 10.8], "收盘": [10.5, 11.0, 11.2],
        "成交量": [1000, 1200, 900],
    })


def _sample_en():
    return pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02"], "Open": [5, 6], "High": [6, 7],
        "Low": [4.5, 5.5], "Close": [5.5, 6.5], "Volume": [100, 200],
    })


def test_import_csv_chinese_columns(tmp_path):
    f = tmp_path / "cn.csv"
    _sample_cn().to_csv(f, index=False)
    df = import_file(str(f))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing and len(df) == 3


def test_import_xlsx_chinese_columns(tmp_path):
    f = tmp_path / "cn.xlsx"
    _sample_cn().to_excel(f, index=False)
    df = import_file(str(f))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].iloc[0] == 10.5
    assert str(df.index[0].date()) == "2024-01-01"


def test_import_xlsx_english_columns_mixed_case(tmp_path):
    f = tmp_path / "en.xlsx"
    _sample_en().to_excel(f, index=False)
    df = import_file(str(f))
    assert len(df) == 2 and df["volume"].iloc[1] == 200


def test_import_xls_legacy_format(tmp_path):
    # pandas 2.x 无法生成真正的 .xls（老 BIFF 格式需已停维护的 xlwt），
    # 此处仅验证 xlrd 引擎可用时分派逻辑可被触达；无样本则跳过，不虚构测试。
    pytest.importorskip("xlrd")
    import struct
    f = tmp_path / "legacy.xls"
    # xlrd 无法读取伪 BIFF 头，跳过生成步骤——本环境无法构造合法 .xls 样本
    pytest.skip("当前环境无法生成合法 .xls 样本（pandas 仅支持写 xlsx）；"
                "xls 分派逻辑与 xlsx 用例共用同一规范化路径")


def test_import_rejects_unsupported_type(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("日期,收盘\n2024-01-01,10", encoding="utf-8")
    with pytest.raises(DataFetchError, match="不支持的文件类型"):
        import_file(str(f))


def test_import_xlsx_missing_column_rejected(tmp_path):
    f = tmp_path / "bad.xlsx"
    pd.DataFrame({"日期": ["2024-01-01"], "收盘": [10.0]}).to_excel(f, index=False)
    with pytest.raises(DataFetchError, match="缺少必需列"):
        import_file(str(f))


def test_import_corrupt_file_clear_error(tmp_path):
    f = tmp_path / "broken.xlsx"
    f.write_bytes(b"not an excel file")
    with pytest.raises(DataFetchError, match="无法读取文件"):
        import_file(str(f))


def test_legacy_import_csv_alias_and_adapter():
    """向后兼容：旧调用方式（import_csv / CsvImportAdapter().import_csv）仍可用。"""
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                     encoding="utf-8", newline="") as f:
        _sample_cn().to_csv(f, index=False)
        p = f.name
    try:
        df1 = import_csv(p)
        df2 = CsvImportAdapter().import_csv(p)
        pd.testing.assert_frame_equal(df1, df2)
    finally:
        os.unlink(p)


def test_csv_and_xlsx_produce_identical_result(tmp_path):
    """同一份内容：CSV 与 Excel 导入结果完全一致（同一规范化路径）。"""
    fc, fx = tmp_path / "same.csv", tmp_path / "same.xlsx"
    _sample_cn().to_csv(fc, index=False)
    _sample_cn().to_excel(fx, index=False)
    pd.testing.assert_frame_equal(import_file(str(fc)), import_file(str(fx)))
