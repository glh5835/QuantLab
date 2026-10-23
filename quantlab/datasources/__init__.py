"""数据源适配器包：导入各模块以触发注册装饰器。"""
from . import base, demo_adapter, csv_adapter  # noqa: F401

try:  # yfinance 为可选依赖：未安装时其余适配器仍可用
    from . import yfinance_adapter  # noqa: F401
except ImportError:
    pass
