"""阶段 3C 页面：参数优化中心——网格搜索（后台任务）/ 进度 / 敏感性 / 过拟合提示。"""
import json
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from ui_common import init_page, empty_state, status_label

from quantlab import db, tasks as task_sys
from quantlab import optimizer as OP
from quantlab.backtest import BacktestConfig
from quantlab.optimizer import OptimizerError
from quantlab.storage import load_csv
from quantlab.reports import save_report, build_html_report

init_page("🎛️ 参数优化中心", "有限网格搜索 · 验证段选参 / 测试段仅终评 · 敏感性与过拟合提示")


def _opt_worker(task_id, cancel, dataset_id, strategy_type, grid, cfg_kw, objective,
                max_mdd, min_trades):
    ds = db.get_dataset(dataset_id)
    path = ds["clean_path"] if (ds["cleaned"] and ds["clean_path"]) else ds["raw_path"]
    df = load_csv(path)
    cfg = BacktestConfig(**cfg_kw)

    def cb(i, total, params):
        db.update_task(task_id, progress=i / total,
                       message=f"[{i}/{total}] 正在回测 {params}")
        db.append_task_log(task_id, f"回测组合 {i}/{total}: {params}")

    try:
        res = OP.run_grid_search(df, strategy_type, grid, cfg=cfg, objective=objective,
                                 max_mdd=max_mdd, min_trades=min_trades, progress_cb=cb)
        rid = OP.save_optimization_run(dataset_id, ds["name"], strategy_type, grid, res)
        db.update_task(task_id, message=f"完成：优化记录 #{rid}，最优 {res['best']['params']}")
        db.append_task_log(task_id, f"✅ 已存档优化记录 #{rid}")
    except Exception as e:  # noqa: BLE001
        db.update_task(task_id, status="failed", error=str(e), finished_at=db.now_str())
        raise


datasets = db.list_datasets()
if not datasets:
    empty_state("没有可优化的数据集。")
    st.stop()

st.subheader("① 配置搜索")
c1, c2 = st.columns(2)
ds_opts = {f"#{d['id']} {d['name']}（{d['rows']}行）": d["id"] for d in datasets}
dataset_id = ds_opts[c1.selectbox("数据集", list(ds_opts.keys()))]
strategy_type = c2.selectbox("策略", ["ma_cross：均线交叉（快/慢窗口）",
                                      "ml_signal：ML 信号（买卖阈值）"]).split("：")[0]

grid = {}
if strategy_type == "ma_cross":
    g1, g2 = st.columns(2)
    grid["fast"] = [int(x) for x in g1.text_input("快线候选（逗号分隔）", "3, 5, 8").split(",") if x.strip()]
    grid["slow"] = [int(x) for x in g2.text_input("慢线候选（逗号分隔）", "20, 30").split(",") if x.strip()]
else:
    exps = db.list_experiments(50)
    if not exps:
        st.info("还没有 ML 实验。请先在「🤖 机器学习实验室」训练。")
        st.stop()
    e_opts = {f"#{e['id']} {e['name']}": e["id"] for e in exps}
    grid["experiment_id"] = [e_opts[st.selectbox("选择实验", list(e_opts.keys()))]]
    g1, g2 = st.columns(2)
    grid["buy_threshold"] = [float(x) for x in g1.text_input("买入阈值候选", "0.0, 0.005").split(",") if x.strip()]
    grid["exit_threshold"] = [float(x) for x in g2.text_input("退出阈值候选", "0.0").split(",") if x.strip()]

o1, o2, o3 = st.columns(3)
objective = o1.selectbox("优化目标（在验证段上最大化）",
                         ["sharpe：验证段夏普", "return：验证段收益",
                          "calmar：验证段卡玛"]).split("：")[0]
max_mdd = float(o2.number_input("最大回撤上限（0=不设）", 0.0, 0.9, 0.0, 0.05))
min_trades = int(o3.number_input("最低交易次数", 0, 100, 0, 1))
st.caption(f"组合数上限 {OP.MAX_COMBINATIONS}；快线≥慢线的组合自动排除；"
           f"搜索只用训练+验证段，测试段仅在参数确定后评估一次。")

if st.button("🚀 开始网格搜索（后台任务）", type="primary", use_container_width=True):
    try:
        OP.build_grid(grid)  # 预检：非法/超限立即报错
        tid = db.create_task("optimize", f"{strategy_type}@{dataset_id}", {"grid": str(grid)})
        tasks = __import__("quantlab.tasks", fromlist=["run_in_background"])
        tasks.run_in_background(tid, _opt_worker, dataset_id, strategy_type, grid,
                                {"commission_rate": 0.0003, "min_commission": 5.0,
                                 "slippage": 0.001}, objective,
                                max_mdd if max_mdd > 0 else None, min_trades)
        st.success(f"任务 #{tid} 已提交。")
        st.rerun()
    except OptimizerError as e:
        st.error(f"搜索配置被拒绝：{e}")

# ---------------- 任务进度 ----------------
st.subheader("② 任务进度")
opt_tasks = [t for t in db.list_tasks(30) if t["task_type"] == "optimize"]
if opt_tasks:
    t = opt_tasks[0]
    st.markdown(f"最新任务 #{t['id']} · {status_label(t['status'])} · {t['message']}")
    if t["status"] == "running":
        st.progress(min(t["progress"], 1.0))
        time.sleep(3)
        st.rerun()
    if t["status"] == "failed":
        st.error(t["error"][:500])
else:
    st.caption("暂无优化任务。")

# ---------------- 结果 ----------------
st.subheader("③ 优化结果")
runs = db.list_optimization_runs(20)
if not runs:
    st.info("尚无优化记录。")
    st.stop()
r_opts = {f"#{r['id']} {r['strategy_type']} 于 {r['created_at']}（{r['n_combinations']}组合）": r
          for r in runs}
run = r_opts[st.selectbox("选择优化记录", list(r_opts.keys()))]
combos = json.loads(run["results_json"])
best = json.loads(run["best_params_json"])
test_m = json.loads(run["test_metrics_json"])

st.success(f"最优参数：`{best}`（验证段目标 {run['objective']} 最优）。"
           f"测试段（从未参与选参）总收益 "
           f"{test_m.get('total_return', float('nan')):.2%}，夏普 "
           f"{test_m.get('sharpe', float('nan')):.2f}。")

rows = []
for c in combos:
    if "error" in c:
        rows.append({"参数": str(c["params"]), "验证收益": "错误", "验证夏普": "—",
                     "验证回撤": "—", "交易数": "—"})
        continue
    vm = c["val_metrics"]
    rows.append({"参数": str(c["params"]),
                 "验证收益": f"{vm.get('total_return', float('nan')):.2%}",
                 "验证夏普": f"{vm.get('sharpe', float('nan')):.2f}"
                 if vm.get("sharpe") is not None else "—",
                 "验证回撤": f"{vm.get('mdd', float('nan')):.2%}",
                 "交易数": c.get("n_trades", "—")})
cdf = pd.DataFrame(rows)
st.dataframe(cdf, use_container_width=True, hide_index=True)

# 敏感性图（均线策略：快/慢二维热力）
if run["strategy_type"] == "ma_cross" and len(combos) > 1:
    hm = [{"fast": c["params"]["fast"], "slow": c["params"]["slow"],
           "sharpe": c["val_metrics"].get("sharpe") if "val_metrics" in c else None}
          for c in combos if "val_metrics" in c]
    if all(h["sharpe"] is not None for h in hm):
        pivot = pd.DataFrame(hm).pivot(index="fast", columns="slow", values="sharpe")
        fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale="RdYlGn",
                        title="验证段夏普：参数敏感性热力图")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

# 邻居稳定性 + 过拟合提示（复用核心函数）
res_dict = {"best": {**{"params": best, "val_metrics": {},
                        "n_trades": combos[0].get("n_trades", 0),
                        "train_metrics": combos[0].get("train_metrics", {})},
                     **combos[0]},
            "combinations": combos, "objective": run["objective"],
            "n_combinations": run["n_combinations"], "test_metrics": test_m}
ns = OP.neighbor_stability(res_dict, run["strategy_type"])
st.markdown("**邻近参数稳定性**：" + ns["note"] +
            (f"（最优点 {ns.get('best_value'):.2f} vs 邻近均值 "
             f"{ns.get('neighbor_mean'):.2f}）" if ns.get("neighbor_mean") is not None else ""))
for w in OP.overfit_warnings(res_dict):
    st.warning(f"⚠️ {w}")

if st.button("📤 导出优化报告（HTML）"):
    content = build_html_report(
        f"参数优化报告 · {run['strategy_type']}",
        {"优化ID": f"#{run['id']}", "数据集": run["dataset_name"],
         "目标": run["objective"], "组合数": run["n_combinations"],
         "test参与选参": "否（仅终评）"},
        [("全部组合结果", cdf), ("最优参数", pd.DataFrame([best])),
         ("测试段终评", pd.DataFrame([{"指标": k, "值": v} for k, v in test_m.items()
                                    if not isinstance(v, (dict, list))]))],
        warnings=OP.overfit_warnings(res_dict))
    p = save_report(content, f"optimization_report_{run['id']}")
    st.success(f"已导出：`{p}`")
    with open(p, "rb") as f:
        st.download_button("⬇️ 下载", f.read(), file_name=p.name, mime="text/html")
