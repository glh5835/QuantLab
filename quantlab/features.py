"""机器学习特征工程：全部特征只用"截至 t 时点"的数据（因果），复用 indicators 计算。

与任务书 B2/B6 对应：
- 每个特征都在注册表中说明计算依据与"数据可用时间"（均为 t 收盘后即可用）。
- 特征构造与目标标签构造严格分离（标签在 ml_lab.py，只用未来价格，永不进入特征）。
- 滚动窗口初期的 NaN 保留原样，由 Pipeline 内的缺失值填补器在训练集上学习处理。
"""
import numpy as np
import pandas as pd

from . import indicators as I


class FeatureError(ValueError):
    pass


# 特征注册表：key -> (构造函数, 默认参数, 说明)
# 所有构造函数只允许使用当前行及之前的数据（截断等价性测试可证）。
FEATURES = {
    "ret_1":  (lambda df: I.returns(df["close"]), (), "过去 1 期收益率（t 收盘相对 t-1 收盘）"),
    "ret_5":  (lambda df: I.cum_returns(df["close"], 5), (), "过去 5 期累计收益率"),
    "ret_10": (lambda df: I.cum_returns(df["close"], 10), (), "过去 10 期累计收益率"),
    "sma_dev": (lambda df: df["close"] / I.sma(df["close"], 20) - 1, (), "收盘价相对 SMA20 的偏离度"),
    "ema_dev": (lambda df: df["close"] / I.ema(df["close"], 20) - 1, (), "收盘价相对 EMA20 的偏离度"),
    "rsi_14": (lambda df: I.rsi(df["close"], 14), (), "14 期 RSI"),
    "macd_dif": (lambda df: I.macd(df["close"])["dif"] / df["close"], (), "DIF/收盘价（无量纲化）"),
    "macd_dea": (lambda df: I.macd(df["close"])["dea"] / df["close"], (), "DEA/收盘价"),
    "macd_hist": (lambda df: I.macd(df["close"])["hist"] / df["close"], (), "MACD 柱/收盘价"),
    "vol_20": (lambda df: I.rolling_volatility(df["close"], 20), (), "20 期滚动年化波动率"),
    "volume_chg": (lambda df: df["volume"].pct_change(fill_method=None), (),
                   "成交量变化率（无成交量字段时该列恒为 NaN，会被安全剔除）"),
}


def build_features(df: pd.DataFrame, feature_keys: list[str]) -> pd.DataFrame:
    """按特征列表构造特征矩阵。行 = 交易日 t，所有特征只依赖 ≤ t 的数据。

    传入单资产 OHLCV（内部会做排序/去重校验）；不修改原始数据。
    """
    if not feature_keys:
        raise FeatureError("至少选择一个特征。")
    unknown = [k for k in feature_keys if k not in FEATURES]
    if unknown:
        raise FeatureError(f"未知特征 {unknown}，可选: {list(FEATURES)}")
    data = I.prepare_ohlcv(df)
    cols = {}
    for k in feature_keys:
        func, _, _ = FEATURES[k]
        col = func(data)
        cols[k] = col
    out = pd.DataFrame(cols)
    # 某特征整列 NaN（如无成交量）时剔除该列，避免无意义的全空特征
    out = out.dropna(axis=1, how="all")
    if out.shape[1] == 0:
        raise FeatureError("所有选定特征均无法计算（数据不足或缺少字段）。")
    return out


def feature_catalog() -> list[dict]:
    """特征目录（供界面展示：名称 + 计算依据）。"""
    return [{"key": k, "desc": v[2]} for k, v in FEATURES.items()]
