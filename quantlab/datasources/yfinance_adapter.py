"""yfinance 适配器：通过 Yahoo Finance 公开接口获取股票/ETF/指数日线行情。

许可说明：yfinance 是开源库，抓取 Yahoo Finance 的公开行情接口，数据仅供个人
学习研究使用，不适用于商业用途。免费接口存在限流，代码中已实现重试与友好报错。
"""
import time

from .base import BaseDataAdapter, DataFetchError, register_adapter

_RENAME = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
           "Volume": "volume", "Adj Close": "adj_close"}


@register_adapter
class YFinanceAdapter(BaseDataAdapter):
    source_id = "yfinance"
    display_name = "Yahoo Finance (yfinance)"
    license_note = (
        "Yahoo Finance 公开行情接口（通过开源库 yfinance），免费但有限流；"
        "数据仅供个人学习与研究，不含实时行情保证。"
    )

    def check_availability(self) -> dict:
        """用一笔极小请求探测接口可用性（下载 5 天数据）。"""
        try:
            df = self.fetch("AAPL", days_back=5)
            if len(df) == 0:
                return {"ok": False, "message": "接口可达但返回空数据，可能是限流或地区限制。"}
            return {"ok": True, "message": f"接口正常，测试返回 {len(df)} 条记录。"}
        except Exception as e:  # noqa: BLE001 - 可用性检查需要捕获一切异常
            return {"ok": False, "message": f"不可用：{e}"}

    def fetch(self, symbol: str, start: str = None, end: str = None,
              days_back: int = None, max_retries: int = 3) -> "pd.DataFrame":
        try:
            import yfinance as yf
        except ImportError as e:
            raise DataFetchError("未安装 yfinance。请运行: pip install yfinance") from e

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                import datetime as dt
                if days_back:
                    start = (dt.date.today() - dt.timedelta(days=days_back)).isoformat()
                    end = dt.date.today().isoformat()
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start, end=end, interval="1d", auto_adjust=False)
                if df is None or df.empty:
                    raise DataFetchError(
                        f"代码 {symbol} 未返回数据。请检查代码是否正确（如 600519.SS=上证、"
                        f"00700.HK=港股、AAPL=美股），以及网络/地区是否可访问 Yahoo。")
                df = df.rename(columns=_RENAME)
                keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
                df = df[keep].copy()
                df.index = df.index.tz_localize(None)
                df.index.name = "trade_date"
                return df.sort_index()
            except DataFetchError:
                raise  # 业务性错误（如代码不存在）不重试
            except Exception as e:  # noqa: BLE001 - 网络类错误需要重试
                last_err = e
                time.sleep(2 * attempt)  # 退避后重试
        raise DataFetchError(
            f"下载 {symbol} 失败（已重试 {max_retries} 次）：{last_err}。"
            f"建议：检查网络连接；若持续失败可能是接口限流，可稍后再试或改用 CSV 导入 / 示例数据。")
