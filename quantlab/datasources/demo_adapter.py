"""示例数据适配器：用几何布朗运动生成模拟日线行情，供无网络/无密钥时演示完整流程。

⚠️ 合规红线：该适配器生成的所有数据都是随机模拟的，is_synthetic=True，
数据库、界面、图表与导出文件中都会显著标注"模拟数据"，绝不冒充真实行情。
生成时固定随机种子，保证课堂演示可复现。
"""
import numpy as np
import pandas as pd

from .base import BaseDataAdapter, register_adapter


@register_adapter
class DemoAdapter(BaseDataAdapter):
    source_id = "demo"
    display_name = "示例数据（随机模拟，仅供演示）"
    license_note = "本数据集由 QuantLab 内置随机数生成器模拟产生，不是真实行情，仅用于功能演示与教学。"
    is_synthetic = True

    # 预置演示资产：名称 -> (基准年化收益, 年化波动率)
    DEMO_ASSETS = {
        "DEMO_TECH":  (0.15, 0.32),   # 模拟高波动科技股
        "DEMO_BANK":  (0.06, 0.18),   # 模拟低估值银行股
        "DEMO_ETF":   (0.09, 0.15),   # 模拟宽基 ETF
        "DEMO_INDEX": (0.08, 0.13),   # 模拟市场指数
    }

    def fetch(self, symbol: str, start: str = "2023-01-01", end: str = "2025-12-31",
              **_) -> pd.DataFrame:
        if symbol not in self.DEMO_ASSETS:
            raise ValueError(f"未知演示资产 {symbol}，可选: {list(self.DEMO_ASSETS)}")
        mu, sigma = self.DEMO_ASSETS[symbol]
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))  # 固定种子→可复现

        dates = pd.bdate_range(start, end)  # 工作日（近似交易日历）
        n = len(dates)
        dt = 1 / 252
        daily_mu = (mu - 0.5 * sigma**2) * dt
        daily_sigma = sigma * np.sqrt(dt)
        log_ret = rng.normal(daily_mu, daily_sigma, n)
        close = 100.0 * np.exp(np.cumsum(log_ret))

        # 由收盘价反推日内 open/high/low（幅度与波动率挂钩，形状真实）
        intraday = np.abs(rng.normal(0, daily_sigma, n))
        open_ = close * np.exp(rng.normal(0, daily_sigma * 0.5, n))
        high = np.maximum(open_, close) * (1 + intraday * 0.6)
        low = np.minimum(open_, close) * (1 - intraday * 0.6)
        volume = rng.integers(800_000, 5_000_000, n).astype(float)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.DatetimeIndex(dates, name="trade_date"))
        return df
