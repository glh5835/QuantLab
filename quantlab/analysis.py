"""自动数据分析（EDA）计算层：所有指标在此计算，界面层只负责画图。

正确性约定：
- 日收益率用简单收益 close.pct_change()（对数收益仅作扩展）。
- 年化假设：252 个交易日（A 股与国际惯例通用）。
- 指标无法可靠计算时返回 None 并说明原因，绝不返回虚构数值。
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change(fill_method=None).dropna()


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """日收益率描述性统计：均值/标准差/分位数/最值/偏度/峰度 + 年化指标。"""
    r = daily_returns(df)
    if len(r) < 2:
        return pd.DataFrame()
    stats = {
        "交易日数": len(r),
        "日均收益率": r.mean(),
        "日收益率标准差": r.std(),
        "最小值": r.min(), "25%分位": r.quantile(0.25),
        "中位数": r.median(), "75%分位": r.quantile(0.75), "最大值": r.max(),
        "偏度": r.skew(), "超额峰度": r.kurt(),
        "年化收益率(简单)": ((1 + r.mean()) ** TRADING_DAYS - 1),
        "年化波动率": r.std() * np.sqrt(TRADING_DAYS),
        "正收益天数占比": (r > 0).mean(),
    }
    out = pd.DataFrame({"指标": list(stats.keys()), "数值": list(stats.values())})
    out["数值"] = out["数值"].astype(float)
    return out


def max_drawdown(df: pd.DataFrame) -> dict:
    """最大回撤：历史最高点到之后最低点的最大跌幅，附起止日期。"""
    close = df["close"]
    if len(close) < 2:
        return {"mdd": None, "reason": "数据不足（少于 2 个交易日）"}
    cummax = close.cummax()
    dd = close / cummax - 1
    trough = dd.idxmin()
    peak = close.loc[:trough].idxmax()
    return {
        "mdd": float(dd.min()),
        "peak_date": str(peak.date()), "trough_date": str(trough.date()),
        "recovery_to": str(close.index.max().date()),
        "recovered": bool(close.loc[trough:].max() >= close.loc[peak]),
        "drawdown_series": dd,  # 完整回撤序列，供画图
    }


def rolling_metrics(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """滚动指标：窗口收益率与滚动年化波动率。"""
    out = pd.DataFrame(index=df.index)
    out["滚动收益率"] = df["close"].pct_change(window, fill_method=None)
    out["滚动年化波动率"] = df["close"].pct_change(fill_method=None).rolling(window).std() * np.sqrt(TRADING_DAYS)
    return out.dropna(how="all")


def correlation_matrix(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """多资产日收益率相关性矩阵。dfs: {资产名: OHLCV DataFrame}"""
    rets = pd.DataFrame({k: daily_returns(v) for k, v in dfs.items()})
    return rets.corr()


def annualized_sharpe(df: pd.DataFrame, rf_annual: float = 0.0) -> dict:
    """年化夏普比率（模块4先给出，回测阶段会复用同一公式）。"""
    r = daily_returns(df)
    if len(r) < 30 or r.std() == 0:
        return {"sharpe": None, "reason": "样本不足 30 天或收益无波动，无法可靠计算夏普比率。"}
    excess = r.mean() - rf_annual / TRADING_DAYS
    return {"sharpe": float(excess / r.std() * np.sqrt(TRADING_DAYS))}


def missing_overview(df: pd.DataFrame) -> pd.DataFrame:
    """缺失值与数据质量可视化所需的逐字段统计。"""
    out = pd.DataFrame({
        "缺失数量": df.isna().sum(),
        "缺失占比": df.isna().mean(),
        "零值数量": (df == 0).sum(),
    })
    return out
