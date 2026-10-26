"""策略层：统一接口接入回测引擎（新增策略不改引擎）。

接口约定（与任务书"回测中的未来信息泄漏"逐条对应）：
- 每个策略实现 target_position(df_hist) → 0.0~1.0：在交易日 t 收盘后调用，
  df_hist 是"截至 t（含）"的行情——策略物理上拿不到 t 之后的数据。
- 返回的目标仓位最早在 t+1 的可交易时点由引擎执行（详见 backtest.py）。
"""
import pandas as pd

from . import db
from .indicators import sma, IndicatorError


class StrategyError(ValueError):
    pass


class BaseStrategy:
    """策略基类：子类只需实现 target_position。"""
    name = "base"
    #: 给回测报告看的策略说明
    desc = ""

    def target_position(self, df_hist: pd.DataFrame) -> float:
        raise NotImplementedError

    def warmup_days(self) -> int:
        """开始产生有效信号前需要的暖机交易日数（引擎在该天数之前不调用信号）。"""
        return 0


class BuyHoldStrategy(BaseStrategy):
    """策略 A：买入并持有基准。首个信号日即满仓，之后一直持有。"""
    name = "buy_hold"
    desc = "基准策略：回测开始后尽快建立目标仓位并持有到底（用于与其他策略比较）。"

    def target_position(self, df_hist) -> float:
        return 1.0


class MACrossoverStrategy(BaseStrategy):
    """策略 B：均线交叉。快线上穿慢线 → 目标仓位 1；下穿 → 0。窗口可配置，快<慢。"""
    name = "ma_cross"

    def __init__(self, fast: int = 5, slow: int = 20):
        if fast >= slow:
            raise StrategyError(f"均线交叉要求快速窗口 < 慢速窗口，收到 fast={fast}, slow={slow}。")
        if fast < 1 or slow < 2:
            raise StrategyError("均线窗口必须为正整数。")
        self.fast, self.slow = fast, slow
        self.desc = f"均线交叉：SMA{fast} 上穿 SMA{slow} 买入持有，下穿退出（t 收盘出信号，t+1 成交）。"

    def warmup_days(self) -> int:
        return self.slow + 1

    def target_position(self, df_hist) -> float:
        close = df_hist["close"]
        if len(close) < self.slow:
            return 0.0
        f = sma(close, self.fast).iloc[-1]
        s = sma(close, self.slow).iloc[-1]
        if pd.isna(f) or pd.isna(s):
            return 0.0
        return 1.0 if f > s else 0.0


class MLSignalStrategy(BaseStrategy):
    """策略 C：机器学习信号策略。

    两种构造方式：
    1) experiment_id：读取实验存档的预测文件（feature_time=t 的预测只用 ≤t 的特征
       + 冻结模型，在 t 收盘后可用）。默认只用 test 段（真正的样本外预测）。
    2) predictions DataFrame（滚动验证用）：列 feature_time/y_pred，由每个滚动窗口
       独立训练的模型直接生成，不经过全局存档。
    - direction 任务：预测涨（1）→ 目标仓位 1，否则 0。
    - return 任务：预测收益 > buy_threshold → 1；< exit_threshold → 0；
      介于两者之间 → 维持前一日状态（迟滞，减少频繁换手）。
    """
    name = "ml_signal"

    def __init__(self, experiment_id: int = None, buy_threshold: float = 0.0,
                 exit_threshold: float = 0.0, split: str = "test",
                 predictions: "pd.DataFrame" = None, task: str = None):
        if predictions is not None:
            exp = None
            preds = predictions.copy()
            preds["feature_time"] = pd.to_datetime(preds["feature_time"])
            preds = preds.sort_values("feature_time")
            self.task = task
            src_desc = "滚动窗口独立模型"
        else:
            exp = db.get_experiment(experiment_id)
            if not exp:
                raise StrategyError(f"实验 #{experiment_id} 不存在。")
            if not exp["predictions_path"]:
                raise StrategyError("该实验没有预测文件（训练失败或文件缺失），无法生成信号。")
            self.task = exp["task"]
            preds = pd.read_csv(exp["predictions_path"], parse_dates=["feature_time"])
            preds = preds[preds["split"] == split].sort_values("feature_time")
            src_desc = f"实验 #{experiment_id}（{split} 段样本外预测）"
        if len(preds) == 0:
            raise StrategyError("没有可用的预测记录。")
        self.experiment_id = experiment_id
        self.split = split
        self._pred_by_time = dict(zip(preds["feature_time"], preds["y_pred"]))
        self._last_target = 0.0
        self.buy_th, self.exit_th = buy_threshold, exit_threshold
        self.desc = (f"机器学习信号（{src_desc}，{self.task} 任务）："
                     f"{'预测上涨→持有' if self.task == 'direction' else f'预测收益>{buy_threshold:.4f}→持有，<{exit_threshold:.4f}→退出'}"
                     f"；模型与阈值均已冻结，信号只在 t 收盘后可用。")

    def warmup_days(self) -> int:
        return 0  # 信号可用时间由预测文件自身的时间戳决定

    def target_position(self, df_hist) -> float:
        t = df_hist.index[-1]
        pred = self._pred_by_time.get(t)
        if pred is None:
            return self._last_target  # 当日无预测 → 维持原状态（不虚构信号）
        if self.task == "direction":
            target = 1.0 if pred >= 0.5 else 0.0
        else:
            if pred > self.buy_th:
                target = 1.0
            elif pred < self.exit_th:
                target = 0.0
            else:
                target = self._last_target
        self._last_target = target
        return target
