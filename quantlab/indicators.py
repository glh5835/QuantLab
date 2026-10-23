"""技术指标计算中心（模块 A）：统一指标接口，纯函数、因果计算、可独立测试。

设计约定（与任务书 A3 对应）：
- 所有指标只使用"当前时点及之前"的数据（因果性）。可用截断等价性测试验证：
  用完整数据算出的第 t 行，必然等于只用前 t 行数据算出的结果。
- 滚动窗口初期按指标定义保留 NaN，不补零。
- 函数不修改传入数据（先 copy）。
- 参数非法立即抛 ValueError；数据未排序/日期重复抛错，多资产必须拆成单资产分别计算。
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


class IndicatorError(ValueError):
    """指标计算错误：参数非法或数据不满足前提。"""


def prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """指标计算前的统一预处理：单资产、按日期升序、无重复日期、必需字段齐全。

    返回副本（不修改原始数据集）。多资产数据请先拆分再逐个调用——滚动指标
    跨资产计算没有意义，这里用"必须有唯一升序日期索引"强制拦住这种错误。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise IndicatorError("索引必须是 DatetimeIndex（交易日期）。")
    if df.index.has_duplicates:
        raise IndicatorError("存在重复交易日期：请先清洗（去重）后再计算指标。多资产请拆分为单资产。")
    missing = [c for c in ("close",) if c not in df.columns]
    if missing:
        raise IndicatorError(f"缺少必需字段: {missing}")
    out = df.copy()
    return out.sort_index()


def _check_window(n: int, name: str):
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise IndicatorError(f"{name} 窗口必须为正整数，收到 {n}")


# ---------------- 趋势指标 ----------------

def sma(close: pd.Series, window: int) -> pd.Series:
    """简单移动平均 SMA_t = mean(P_{t-n+1..t})；前 window-1 个位置为 NaN。"""
    _check_window(window, "SMA")
    return close.rolling(window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    """指数移动平均（递归式）：EMA_t = α·P_t + (1-α)·EMA_{t-1}，α = 2/(span+1)。

    使用 pandas ewm(span, adjust=False)，以首个有效值为种子，不引入未来数据。
    """
    _check_window(span, "EMA")
    return close.ewm(span=span, adjust=False).mean()


# ---------------- 动量指标 ----------------

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """相对强弱指标（Wilder 平滑的常见 pandas 实现）。

    delta = P_t - P_{t-1}；gain=max(delta,0)、loss=max(-delta,0)；
    平均涨/跌幅用递归平滑 avg_t = (1-1/n)·avg_{t-1} + (1/n)·x_t（ewm alpha=1/n）；
    RSI = 100·avg_gain/(avg_gain+avg_loss)。
    边界约定（明确规则，不随意）：avg_loss=0 且 avg_gain>0 → 100；
    两者同为 0（完全无涨跌）→ 50；首个交易日无 delta → NaN。
    """
    _check_window(window, "RSI")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rsi_val = 100 * avg_gain / (avg_gain + avg_loss)
    rsi_val = rsi_val.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi_val = rsi_val.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    rsi_val.iloc[:1] = np.nan  # 首日无涨跌定义
    rsi_val.name = "rsi"
    return rsi_val


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD：DIF = EMA_fast - EMA_slow；DEA = DIF 的 EMA(signal)；
    柱状值 HIST = 2×(DIF-DEA)（国内常用口径，与行情软件一致，已在文档注明）。"""
    _check_window(fast, "MACD快线")
    _check_window(slow, "MACD慢线")
    _check_window(signal, "MACD信号")
    if fast >= slow:
        raise IndicatorError(f"MACD 要求快速均线窗口 < 慢速均线窗口，收到 fast={fast}, slow={slow}。")
    dif = ema(close, fast) - ema(close, slow)
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = 2 * (dif - dea)
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": hist})


# ---------------- 波动率指标 ----------------

def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """布林带：mid = SMA(n)；std = 滚动总体标准差（ddof=0，与主流行情软件一致）；
    upper = mid + k·std；lower = mid - k·std。恒有 upper ≥ mid ≥ lower。"""
    _check_window(window, "布林带")
    if num_std <= 0:
        raise IndicatorError(f"布林带带宽系数必须为正数，收到 {num_std}")
    mid = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    return pd.DataFrame({
        "bb_mid": mid, "bb_upper": mid + num_std * std, "bb_lower": mid - num_std * std})


def rolling_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """滚动年化波动率：日收益率滚动标准差 × √252。"""
    _check_window(window, "滚动波动率")
    ret = close.pct_change(fill_method=None)
    return ret.rolling(window).std() * np.sqrt(TRADING_DAYS)


# ---------------- 基础收益特征 ----------------

def returns(close: pd.Series) -> pd.Series:
    """单期简单收益率 P_t/P_{t-1} - 1。"""
    return close.pct_change(fill_method=None)


def cum_returns(close: pd.Series, periods: int) -> pd.Series:
    """多期累计收益率 P_t/P_{t-periods} - 1。"""
    _check_window(periods, "累计收益率")
    return close.pct_change(periods, fill_method=None)


def log_returns(close: pd.Series) -> pd.Series:
    """对数收益率 ln(P_t/P_{t-1})。"""
    return np.log(close / close.shift(1))


def rolling_max_drawdown(close: pd.Series, window: int = 60) -> pd.Series:
    """历史滚动最大回撤：窗口内"峰值到谷值"的最大跌幅（0 表示窗口内无回撤）。"""
    _check_window(window, "滚动最大回撤")

    def _mdd(seg: np.ndarray) -> float:
        peak = np.maximum.accumulate(seg)
        return float((seg / peak - 1.0).min())

    return close.rolling(window).apply(_mdd, raw=True)


# ---------------- 指标注册表（供页面/特征工程统一枚举） ----------------

INDICATORS = {
    "sma":  {"func": lambda df, w: sma(df["close"], w), "default": 20, "min": 2,
             "desc": "简单移动平均：过去 n 日收盘价均值", "overlay": True},
    "ema":  {"func": lambda df, w: ema(df["close"], w), "default": 20, "min": 2,
             "desc": "指数移动平均：越近的价格权重越大（α=2/(n+1) 递归）", "overlay": True},
    "rsi":  {"func": lambda df, w: rsi(df["close"], w), "default": 14, "min": 2,
             "desc": "相对强弱指标（0~100）：>70 超买区、<30 超卖区（Wilder 平滑）", "overlay": False},
    "macd": {"func": lambda df, f, s, g: macd(df["close"], f, s, g),
             "default": (12, 26, 9), "min": (2, 3, 2),
             "desc": "DIF=EMA快-EMA慢；DEA=DIF的EMA；柱=2×(DIF-DEA)", "overlay": False},
    "bollinger": {"func": lambda df, w, k: bollinger(df["close"], w, k),
                  "default": (20, 2.0), "min": (2, 0.1),
                  "desc": "中轨=SMA(n)，上下轨=中轨±k×滚动总体标准差", "overlay": True},
    "rolling_vol": {"func": lambda df, w: rolling_volatility(df["close"], w),
                    "default": 20, "min": 2, "desc": "滚动年化波动率（日收益std×√252）",
                    "overlay": False},
    "returns": {"func": lambda df: returns(df["close"]), "default": (), "min": (),
                "desc": "单期简单收益率", "overlay": False},
    "cum_returns": {"func": lambda df, p: cum_returns(df["close"], p), "default": (5,),
                    "min": (1,), "desc": "多期累计收益率", "overlay": False},
    "log_returns": {"func": lambda df: log_returns(df["close"]), "default": (), "min": (),
                    "desc": "对数收益率 ln(P_t/P_{t-1})", "overlay": False},
    "rolling_mdd": {"func": lambda df, w: rolling_max_drawdown(df["close"], w),
                    "default": (60,), "min": (2,), "desc": "窗口内最大回撤（≤0）",
                    "overlay": False},
}


def compute_indicator(df: pd.DataFrame, name: str, params: tuple) -> pd.DataFrame | pd.Series:
    """统一入口：按名称与参数计算指标。df 须为单资产 OHLCV。"""
    if name not in INDICATORS:
        raise IndicatorError(f"未知指标 {name}，可选: {list(INDICATORS)}")
    data = prepare_ohlcv(df)
    return INDICATORS[name]["func"](data, *params)
