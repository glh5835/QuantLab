"""风险分析测试：指标手算对照 / 不可计算状态 / 回撤区间 / 分段 / 压力测试 / 预警。"""
import numpy as np
import pandas as pd
import pytest

from quantlab import risk as R


def _eq(values, start="2024-01-01"):
    return pd.Series(np.asarray(values, dtype=float),
                     index=pd.bdate_range(start, periods=len(values)))


# ---------------- 指标手算 ----------------

def test_risk_metrics_hand_computed():
    # 净值 [100, 110, 99]：总收益 -1%；两天收益 [+10%, -10%]
    eq = _eq([100, 110, 99])
    m = R.risk_metrics(eq, initial=100)
    assert np.isclose(m["total_return"]["value"], -0.01)
    assert np.isclose(m["mdd"]["value"], 99 / 110 - 1)  # 手算最大回撤
    # 夏普手算：mean=0, std>0 → sharpe=0（两天涨跌对称）
    assert np.isclose(m["sharpe"]["value"], 0.0)


def test_sortino_downside_and_calmar():
    # 构造已知下行偏差的序列：收益 [+2%, -1%, +2%, -1%, ...]（补 1.0 前值保住首期收益）
    rets = np.asarray([0.02, -0.01] * 20)
    eq = _eq(100 * np.cumprod(np.concatenate([[1.0], 1 + rets])))
    m = R.risk_metrics(eq, initial=100)
    dv = m["downside_vol"]["value"] / np.sqrt(252)  # 反推日下行偏差
    expected_dv = np.sqrt((0.01 ** 2 * 20) / 40)     # 手算：负收益平方和 / 总样本数
    assert np.isclose(dv, expected_dv)
    assert m["sortino"]["value"] > 0                  # 均值收益为正 → 索提诺为正
    assert m["calmar"]["value"] > 0                   # 有回撤 → 卡玛可计算
    assert np.isclose(m["calmar"]["value"],
                      m["cagr"]["value"] / abs(m["mdd"]["value"]))


def test_beta_and_excess_return():
    # 策略收益 = 2×基准收益 → Beta=2；构造对齐净值
    bench_ret = np.array([0.01, -0.02, 0.015, 0.01, -0.005] * 10)
    strat_ret = 2 * bench_ret
    idx = pd.bdate_range("2024-01-01", periods=51)
    bench = pd.Series(100 * np.cumprod(1 + np.concatenate([[0], bench_ret])), index=idx)
    strat = pd.Series(100 * np.cumprod(1 + np.concatenate([[0], strat_ret])), index=idx)
    m = R.risk_metrics(strat, initial=100, benchmark=bench)
    assert np.isclose(m["beta"]["value"], 2.0, atol=1e-6)
    assert m["excess_return"]["value"] > 0  # 2 倍杠杆化收益（正漂移时）超过基准


def test_non_computable_states():
    eq = _eq([100] * 50)  # 恒定净值：零波动
    m = R.risk_metrics(eq, initial=100)
    assert m["sharpe"]["value"] is None and "标准差为 0" in m["sharpe"]["reason"]
    assert m["calmar"]["value"] is None and "无回撤" in m["calmar"]["reason"]
    assert m["sortino"]["value"] is None and "不可计算" in m["sortino"]["reason"]
    m2 = R.risk_metrics(_eq([100, 101]), initial=100)  # 样本不足
    assert m2["downside_vol"]["value"] is None
    m3 = R.risk_metrics(_eq([100, 110, 99]), initial=100)  # 无基准
    assert m3["beta"]["value"] is None and "基准" in m3["beta"]["reason"]


# ---------------- 回撤区间 ----------------

def test_drawdown_episodes_recovered_and_ongoing():
    # 100→120→60（回撤 -50%，未恢复）→130（恢复，创新高）→100（新回撤 -23%，未恢复）
    eq = _eq([100, 120, 60, 130, 100])
    a = R.drawdown_analysis(eq)
    eps = a["episodes"]
    assert len(eps) == 2
    e1 = eps[0]  # 120→60，130 恢复
    assert np.isclose(e1["mdd"], -0.5) and e1["recovered"] is True
    assert e1["recovery_date"] is not None
    e2 = eps[1]  # 130→100，序列结束仍未恢复
    assert e2["recovered"] is False and e2["recovery_date"] is None
    assert "未恢复" in e2["note"]  # 不伪造恢复日期
    assert np.isclose(a["max"]["mdd"], -0.5)
    assert a["max"]["peak_date"] == "2024-01-02" and a["max"]["trough_date"] == "2024-01-03"


def test_drawdown_monotonic_up_no_episodes():
    a = R.drawdown_analysis(_eq(range(100, 120)))
    assert a["episodes"] == []


# ---------------- 分段分析 ----------------

def test_period_analysis_yearly():
    idx = pd.bdate_range("2023-01-02", periods=520)  # 跨 2023-2024 两年
    eq = pd.Series(np.linspace(100, 150, 520), index=idx)
    periods = R.period_analysis(eq, freq="Y")
    years = [p["period"] for p in periods]
    assert "2023" in years and "2024" in years
    # 各年复利 ≈ 总收益 50%（跨年两段的首尾是相邻但不同的采样点，允许微小复利误差）
    total = (1 + periods[0]["return"]) * (1 + periods[1]["return"]) - 1
    assert np.isclose(total, 0.5, atol=1e-2)
    assert all(p["n_days"] >= 2 for p in periods)


def test_period_analysis_aliases():
    # 旧接口和 Pandas 3.0 新偏移别名均应得到同样的分组结果。
    idx = pd.bdate_range("2023-01-02", periods=520)
    eq = pd.Series(np.linspace(100, 150, len(idx)), index=idx)
    yearly_old = R.period_analysis(eq, freq="Y")
    yearly_new = R.period_analysis(eq, freq="YE")
    quarterly_old = R.period_analysis(eq, freq="Q")
    quarterly_new = R.period_analysis(eq, freq="QE")
    assert yearly_old == yearly_new
    assert quarterly_old == quarterly_new
    assert yearly_old and quarterly_old
    assert all("Q" in item["period"] for item in quarterly_new)


# ---------------- 压力测试 ----------------

def _fake_runner(calls):
    """runner：返回按 overrides 变化的假指标（验证框架逻辑，真实重跑在页面用回测引擎）。"""
    def runner(cfg_overrides=None, df_transform=None, date_range=None):
        calls.append((cfg_overrides, df_transform is not None, date_range))
        fee_mult = 1
        if cfg_overrides and cfg_overrides.get("commission_rate") is not None:
            fee_mult = 3
        return {"total_return": 0.10 - 0.01 * fee_mult, "sharpe": {"value": 0.5}}
    return runner


def test_stress_scenarios_marked_and_diffed():
    calls = []
    base = {"commission_rate": 0.0003, "slippage": 0.001, "target_fraction": 0.95,
            "min_commission": 5}
    scenarios = R.default_scenarios()
    results = R.run_stress(_fake_runner(calls), base, scenarios,
                           price_shock={"start": "2024-06-01", "end": "2024-06-30",
                                        "shock": -0.2})
    names = [r["name"] for r in results]
    assert "手续费×3" in names and "延迟成交 t+2" in names
    assert any("价格冲击" in n for n in names)  # 名称含区间后缀，用子串匹配
    # 真实参数情景 hypothetical=False；价格冲击必须标记假设
    by_name = {r["name"]: r for r in results}
    assert by_name["手续费×3"]["hypothetical"] is False
    shock = [r for r in results if "价格冲击" in r["name"]][0]
    assert shock["hypothetical"] is True and "并非真实历史事件" in shock["desc"]
    assert len(calls) == len(scenarios) + 1  # 每情景 + 价格冲击各一次


def test_stress_price_shock_transform():
    from quantlab.risk import StressScenario, run_stress
    seen = {}

    def runner(cfg_overrides=None, df_transform=None, date_range=None):
        df = pd.DataFrame({"open": [100.0] * 5, "close": [100.0] * 5},
                          index=pd.bdate_range("2024-06-03", periods=5))
        seen["df"] = df_transform(df) if df_transform else df
        return {"total_return": 0.0}

    run_stress(runner, {}, [], price_shock={"start": "2024-06-04",
                                            "end": "2024-06-06", "shock": -0.2})
    df = seen["df"]
    assert df["close"].iloc[0] == 100.0            # 冲击区间外不变
    assert np.isclose(df["close"].iloc[1], 80.0)   # 区间内 -20%
    assert np.isclose(df["close"].iloc[4], 100.0)


# ---------------- 风险预警 ----------------

def test_risk_warnings_dd_threshold():
    eq = _eq([100, 120, 90, 95, 130, 100])  # -25% 回撤触发；恢复后再跌 -23% 再触发
    warns = R.risk_warnings(eq, dd_threshold=0.20)
    dd_warns = [w for w in warns if w["metric"] == "最大回撤"]
    assert len(dd_warns) == 2
    assert np.isclose(dd_warns[0]["value"], -0.25)
    assert dd_warns[0]["date"] == "2024-01-03"
    # bdate 日历：第 6 个交易日是 01-08（跳过周末）
    assert dd_warns[1]["date"] == "2024-01-08"


def test_risk_warnings_vol_threshold():
    # 前段平静、后段高波动 → 滚动波动率一次越限
    rets = np.concatenate([np.full(30, 0.001), np.random.default_rng(1).normal(0, 0.03, 60)])
    eq = _eq(100 * np.cumprod(1 + rets))
    warns = R.risk_warnings(eq, dd_threshold=0.99, vol_threshold=0.30, vol_window=20)
    vol_warns = [w for w in warns if w["metric"] == "滚动年化波动率"]
    assert len(vol_warns) >= 1
    assert all(w["value"] > w["threshold"] for w in vol_warns)
