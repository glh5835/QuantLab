"""pytest 共享夹具：把数据库与数据目录重定向到临时目录，测试互不污染。"""
import sys
from pathlib import Path

import pytest

# 让测试能 import quantlab
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantlab import config  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """每个测试使用独立的临时数据库与数据目录。"""
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "clean").mkdir(parents=True)
    (data_dir / "exports").mkdir(parents=True)
    monkeypatch.setattr(config, "DB_PATH", data_dir / "test.db")
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "RAW_DIR", data_dir / "raw")
    monkeypatch.setattr(config, "CLEAN_DIR", data_dir / "clean")
    monkeypatch.setattr(config, "EXPORT_DIR", data_dir / "exports")
    from quantlab import db
    db.init_db()
    yield
