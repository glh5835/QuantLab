"""模块 6（阶段2-B）：机器学习实验室。

选数据集 → 选任务/特征/模型 → 时间序列三分训练 → 三段指标对比 → 实验存档 → 供回测使用。
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui_common import init_page, empty_state, synthetic_badge

from quantlab import db
from quantlab import ml_lab as ML
from quantlab.features import FEATURES, feature_catalog
from quantlab.ml_lab import MLError, MODELS

init_page("🤖 机器学习实验室", "时间序列三分（防泄漏）· Pipeline 预处理 · 实验存档 · 样本外信号")

datasets = db.list_datasets()
if not datasets:
    empty_state("没有可用的数据集。", "先到「数据采集中心」采集数据（建议 ≥ 300 个交易日）。")
    st.stop()

with st.expander("📖 先读我：这个实验室怎么保证不'作弊'（防数据泄漏）", expanded=False):
    st.markdown("""
1. **特征只用过去**：第 t 行特征只由 ≤ t 的行情算出（有截断等价性自动化测试保证）。
2. **标签只用未来**：标签 = (t, t+h] 的未来收益，与特征严格分开构造。
3. **按时间三分**：训练集 < 验证集 < 测试集，绝不随机打乱；训练段末尾 h 个样本的
   标签区间会伸入验证集 → 用 **purge（隔离）** 剔除，避免标签信息重叠。
4. **预处理只学训练集**：缺失填补与标准化封装在 sklearn Pipeline 里，只在训练集 fit，
   验证/测试集只做 transform——杜绝"先全局标准化再划分"的经典泄漏。
5. **测试集只看一次**：以下展示的测试集指标来自冻结模型的样本外预测；
   若根据测试集表现反复调参再重训，结果就不再是独立测试，请注意解读。
""")

# ---------------- 实验配置 ----------------
st.subheader("① 配置实验")
c1, c2 = st.columns([2, 1])
opts = {f"#{d['id']} {d['name']}（{d['rows']} 行）": d["id"] for d in datasets}
ds = db.get_dataset(opts[c1.selectbox("选择数据集", list(opts.keys()))])
synthetic_badge(ds)

task_label = c2.selectbox("预测任务", ["direction：预测未来 h 期涨跌方向（分类）",
                                     "return：预测未来 h 期收益率（回归）"])
task = "direction" if task_label.startswith("direction") else "return"
horizon = c2.number_input("预测期限 h（交易日）", 1, 20, 1)

feats = st.multiselect("选择特征（全部只使用 ≤t 的数据）", list(FEATURES.keys()),
                       default=["ret_1", "ret_5", "sma_dev", "rsi_14", "vol_20", "volume_chg"])
with st.expander("特征说明（计算依据）"):
    for f in feature_catalog():
        st.markdown(f"- **{f['key']}**：{f['desc']}")

mc1, mc2, mc3 = st.columns(3)
model_avail = [k for k, v in MODELS.items()
               if (v["clf"] if task == "direction" else v["reg"])]
model_key = mc1.selectbox("模型", model_avail, format_func=lambda k: f"{k} — {MODELS[k]['desc']}")
seed = mc2.number_input("随机种子", 0, 9999, 42)
exp_name = mc3.text_input("实验名称（留空自动生成）", "")

ratio = st.slider("训练/验证/测试 时间比例", 0.5, 0.9, 0.7, 0.05,
                  format="%.2f", help="验证=测试=(1-训练)/2，按时间顺序切分，不随机打乱")

if st.button("🚀 训练模型", type="primary", use_container_width=True):
    if not feats:
        st.error("请至少选择一个特征。")
    else:
        with st.spinner("构造特征 → 构造标签 → 时间三分(purge) → Pipeline 训练 → 评估存档…"):
            try:
                res = ML.run_experiment(ds["id"], task, int(horizon), feats, model_key,
                                        name=exp_name or None, seed=int(seed),
                                        ratios=(float(ratio), (1 - ratio) / 2, (1 - ratio) / 2))
                st.session_state["last_exp"] = res["exp_id"]
            except MLError as e:
                st.error(f"实验被拒绝：{e}")
            except Exception as e:  # noqa: BLE001
                st.error(f"训练失败：{type(e).__name__}: {e}")
            else:
                st.success(f"✅ 实验 #{res['exp_id']} 完成。purge 隔离 {res['purged_n']} 个边界样本。")
                st.markdown(f"**标签定义**：{res['label_rule']}")

# ---------------- 结果展示 ----------------
exps = db.list_experiments()
if not exps:
    st.info("尚无实验记录。配置上方参数并点击「训练模型」。")
    st.stop()

st.subheader("② 实验结果与对比")
sel_exp = st.selectbox("选择实验",
                       [f"#{e['id']} {e['name']}（{e['task']}/{e['model_type']}, {e['created_at']}）"
                        for e in exps], index=0)
exp = exps[0] if False else db.get_experiment(int(sel_exp.split("#")[1].split(" ")[0]))
m = json.loads(exp["metrics_json"])

c1, c2, c3 = st.columns(3)
for col, key in zip((c1, c2, c3), ("train", "val", "test")):
    seg = m.get(key, {})
    label = {"train": "训练集", "val": "验证集", "test": "测试集（样本外）"}[key]
    main_metric = seg.get("accuracy", seg.get("rmse"))
    if main_metric is None:
        col.metric(label, "无数据")
        continue
    if exp["task"] == "direction":
        col.metric(label, f"Accuracy {seg['accuracy']:.3f}",
                   f"F1 {seg['f1']:.3f} · {seg['n']} 样本",
                   delta_color="off")
    else:
        col.metric(label, f"RMSE {seg['rmse']:.4f}",
                   f"R² {seg['r2']:.3f} · {seg['n']} 样本", delta_color="off")

st.markdown(f"""
- **数据**：{exp['dataset_name']} v{exp['dataset_version']}（数据集 #{exp['dataset_id']}）
- **划分**：训练 {exp['train_range']}（{exp['train_n']}）｜验证 {exp['val_range']}（{exp['val_n']}）
  ｜测试 {exp['test_range']}（{exp['test_n']}）｜purge 隔离 {exp['purged_n']} 个样本
- **特征**：{', '.join(json.loads(exp['features_json']))}
- **模型**{exp['model_type']}（seed={exp['seed']}）｜模型文件：`{exp['model_path']}`
""")

if exp["task"] == "direction":
    cm = m["test"]["confusion"]
    st.markdown("**测试集混淆矩阵**（tn / fp / fn / tp）")
    cm_df = pd.DataFrame([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                         index=["实际·跌/平", "实际·涨"], columns=["预测·跌/平", "预测·涨"])
    st.dataframe(cm_df, use_container_width=True)
    seg = m["test"]
    detail = pd.DataFrame({
        "指标": ["Accuracy", "Precision(涨)", "Recall(涨)", "F1", "ROC-AUC", "样本数"],
        "数值": [f"{seg['accuracy']:.4f}", f"{seg['precision']:.4f}", f"{seg['recall']:.4f}",
                 f"{seg['f1']:.4f}",
                 f"{seg.get('roc_auc', float('nan')):.4f}" if "roc_auc" in seg else "不适用",
                 seg["n"]],
    })
else:
    rows = []
    for k in ("train", "val", "test"):
        seg = m.get(k, {})
        rows.append({"数据段": {"train": "训练", "val": "验证", "test": "测试"}[k],
                     "MAE": f"{seg.get('mae', float('nan')):.5f}",
                     "RMSE": f"{seg.get('rmse', float('nan')):.5f}",
                     "R²": f"{seg.get('r2', float('nan')):.4f}", "样本": seg.get("n")})
    detail = pd.DataFrame(rows)
st.dataframe(detail if isinstance(detail, pd.DataFrame) else detail, use_container_width=True, hide_index=True)

st.info("⚠️ 解读提醒：训练集与验证/测试集指标差距大 → 过拟合；验证/测试指标接近随机水平"
        "（分类 Accuracy≈0.5）说明特征缺乏预测力——这是常见且诚实的结果，不代表系统故障。"
        "预测准确率 ≠ 策略盈利能力，请到「⚖️ 策略回测」用样本外信号验证。")

# ---------------- 实验历史 ----------------
st.subheader("③ 实验历史")
hist = pd.DataFrame([{
    "ID": e["id"], "名称": e["name"], "任务": e["task"], "h": e["horizon"],
    "模型": e["model_type"],
    "test主指标": (f"{json.loads(e['metrics_json']).get('test', {}).get('accuracy', float('nan')):.3f}"
                  if e["task"] == "direction" and e["metrics_json"] not in ("", "{}")
                  else (f"{json.loads(e['metrics_json']).get('test', {}).get('rmse', float('nan')):.4f}"
                        if e["metrics_json"] not in ("", "{}") else "—")),
    "测试区间": e["test_range"], "状态": e["status"], "创建": e["created_at"],
} for e in exps])
st.dataframe(hist, use_container_width=True, hide_index=True)

st.info("👉 下一步：到「⚖️ 策略回测」选择本实验的 **ML 信号策略**（默认只用 test 段样本外预测），"
        "用真实手续费与滑点检验样本外盈利能力。")
