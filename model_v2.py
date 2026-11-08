"""Nested selection of NIR preprocessing and PLS components."""

import json
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from model import ROOT, OUTER_SEED, INNER_SEED, load_data, metrics
from spectral import SpectralTransform


OUT = ROOT / "results_v2"
METHODS = ("raw", "snv", "derivative", "snv_derivative")
COMPONENTS = (5, 10, 15, 20, 25, 30)


def pipeline(method, components):
    return Pipeline([
        ("preprocess", SpectralTransform(method)),
        ("pls", PLSRegression(n_components=components, scale=True, max_iter=500)),
    ])


def select_model(x, y):
    """Choose pooled inner-fold MAE, ties resolved by candidate order."""
    splits = list(KFold(n_splits=4, shuffle=True, random_state=INNER_SEED).split(x))
    scores = []
    for method in METHODS:
        # All transformations are independent for each spectrum and learn no population statistics.
        transformed = SpectralTransform(method).fit_transform(x)
        for count in COMPONENTS:
            predictions = np.empty(len(y))
            for train, valid in splits:
                regressor = PLSRegression(n_components=count, scale=True, max_iter=500)
                regressor.fit(transformed[train], y[train])
                predictions[valid] = regressor.predict(transformed[valid]).ravel()
            scores.append({"method": method, "components": count,
                           "mae": float(np.mean(np.abs(predictions - y)))})
    best = min(scores, key=lambda score: score["mae"])
    return best, scores


def bootstrap_differences(y, old, new, iterations=2000):
    """Percentile intervals conditional on the fixed paired OOF predictions."""
    rng = np.random.default_rng(2026)
    differences = []
    for _ in range(iterations):
        ix = rng.integers(0, len(y), size=len(y))
        old_metrics = metrics(y[ix], old[ix])
        new_metrics = metrics(y[ix], new[ix])
        differences.append([old_metrics["mae"] - new_metrics["mae"],
                            old_metrics["sep"] - new_metrics["sep"]])
    return np.percentile(differences, [2.5, 97.5], axis=0).T.tolist()


def figures(y, old, new, baseline, test):
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    low, high = min(y.min(), new.min()), max(y.max(), new.max())
    axes[0].scatter(y, new, s=14, alpha=0.65)
    axes[0].plot([low, high], [low, high], "k--", linewidth=1)
    axes[0].set(xlabel="Measured", ylabel="Out-of-fold prediction", title="Nested CV predictions")
    axes[1].scatter(y, new - y, s=14, alpha=0.65)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Measured", ylabel="Prediction - measured", title="Out-of-fold residuals")
    fig.tight_layout()
    fig.savefig(OUT / "oof_diagnostics.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["Mean", "First PLSR", "Selected PLSR"]
    axes[0].bar(labels, [metrics(y, p)["mae"] for p in (baseline, old, new)])
    axes[0].set(ylabel="Out-of-fold MAE", title="Same outer splits")
    bins = np.linspace(min(y.min(), test.min()), max(y.max(), test.max()), 21)
    axes[1].hist(y, bins=bins, density=True, alpha=0.6, label="Calibration measured")
    axes[1].hist(test, bins=bins, density=True, alpha=0.6, label="Test predicted")
    axes[1].set(xlabel="Chemical content", ylabel="Density", title="Distributions")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png")
    plt.close(fig)


def write_report(summary):
    m, old, baseline = summary["selected"], summary["first"], summary["baseline"]
    rows = "\n".join(
        f"| {f['fold']} | {f['method']} | {f['components']} | {f['sep']:.4f} | {f['mae']:.4f} |"
        for f in summary["folds"]
    )
    mae_ci, sep_ci = summary["difference_ci"]
    text = f"""# 近红外定量模型优化评价报告

## 结论

在原来的 320 条校正数据、相同随机外层五折上，本版**嵌套选模的折外预测** SEP 为 **{m['sep']:.4f}**、MAE 为 **{m['mae']:.4f}**；首版分别为 {old['sep']:.4f}、{old['mae']:.4f}，训练折均值基线为 {baseline['sep']:.4f}、{baseline['mae']:.4f}。这不是独立测试精度，也无法据此证明已达到未给定的可用标准。成分名称、单位及验收阈值在数据说明中未提供。

## 数据、预处理与选模

校正集 320 条、测试集 80 条，每条光谱含 2074 个波数/吸光度点。按标注工作簿的光谱文件名匹配，检查文件集合、数值、点数与波数轴。测试集没有真值，没有用于选参或评价。

在运行本版前规定候选：全波段 PLS 回归搭配四种逐光谱处理——原始、SNV（每条光谱减自身均值再除自身标准差，`ddof=0`）、Savitzky–Golay 一阶导数（窗口 15 点、二阶多项式、默认插值边界、以点序为间隔），以及 SNV 后一阶导数。每种处理比较潜变量数 5、10、15、20、25、30，共 24 组；`PLSRegression(scale=True, max_iter=500)` 的中心化和缩放只在每个训练子集上拟合。逐光谱变换不学习其他样本的统计量。

保持首版外层 `KFold(5, shuffle=True, random_state=42)`：每次留出 64 条。每个外层训练集内采用 `KFold(4, shuffle=True, random_state=123)`，按合并内层折外 MAE 最小选出**处理方式和潜变量数**（精确并列按上述处理顺序、潜变量升序），仅用外层训练集拟合后预测外层留出样本。每条校正样本恰有一个折外预测。均值基线取对应外层训练集真值均值。最终模型用全部校正集、同一内层选模规则重新选择并拟合，选中 **{summary['final']['method']} / {summary['final']['components']} 个潜变量**。

## 折外结果

竞赛口径：`SEP = sqrt(sum((实际值 - 预测值)^2)/(n - 1))`，不扣偏差；`MAE = sum(abs(实际值 - 预测值))/n`。下表指标均从全部 320 条折外预测直接汇总；偏差为预测减实测。

| 方法 | SEP | MAE | 偏差 | R² |
|---|---:|---:|---:|---:|
| 训练折均值基线 | {baseline['sep']:.4f} | {baseline['mae']:.4f} | {baseline['bias']:.4f} | {baseline['r2']:.4f} |
| 首版 PLSR | {old['sep']:.4f} | {old['mae']:.4f} | {old['bias']:.4f} | {old['r2']:.4f} |
| 本版嵌套选模 | {m['sep']:.4f} | {m['mae']:.4f} | {m['bias']:.4f} | {m['r2']:.4f} |

| 外层折 | 内层选中处理 | 潜变量数 | 折 SEP | 折 MAE |
|---|---|---:|---:|---:|
{rows}

![折外诊断](results_v2/oof_diagnostics.png)

图 1：左侧是折外实测与预测，虚线为理想值；右侧为预测减实测。实测不低于 30 的 {summary['high_count']} 条样本折外平均残差 {summary['high_bias']:.4f}、MAE {summary['high_mae']:.4f}；该分组是事后描述，不作为选参依据。

![基线和分布](results_v2/comparison.png)

图 2：左侧为同一外层划分的 MAE 对比；右侧比较校正实测与测试预测的归一化密度，测试预测分布不等于测试真值分布。

## 对比不确定性及局限

- 本版相对首版的折外 MAE 变化（首版减本版）为 {old['mae'] - m['mae']:.4f}，SEP 变化为 {old['sep'] - m['sep']:.4f}。对 320 对固定折外预测做 2000 次样本级配对 bootstrap（种子 2026），变化的 95% 百分位区间分别为 MAE [{mae_ci[0]:.4f}, {mae_ci[1]:.4f}]、SEP [{sep_ci[0]:.4f}, {sep_ci[1]:.4f}]；正数表示本版误差更低。
- 这些区间仅描述固定折外预测的样本重抽样波动，不包含重新训练、拆分、候选设计和模型选择的不确定性；样本间可能相关，区间不能作为独立样本显著性证明。首版外层结果已被观察并用于发起本次优化，因此两版比较是探索性结果，而非全新独立确认。
- 不知道来源、批次和重复测量的分组；随机拆分可能泄露相关样本结构并高估泛化。需要获取分组元数据并进行组外验证，最好以独立批次或新仪器样本外部验证，明确可接受误差后才能判定实际可用性。
- 本版候选集有限，未评估更广泛方法；不要利用本次外层结果反复调整模型并声称其仍为无偏最终评价。所有学习型步骤必须放在训练折内；需要新的独立数据确认进一步选择。

## 测试预测与复现

[80 条逐文件预测](results_v2/test_predictions.csv) 的范围为 {summary['test_min']:.4f}–{summary['test_max']:.4f}；校正真值范围 {summary['y_min']:.4f}–{summary['y_max']:.4f}，其中 {summary['outside_count']} 条预测超出此范围。即使预测在范围内，也不能证明样本位于光谱训练分布内。测试无真值，不计算其 SEP 或 MAE。折外结果和各候选内层分数分别见 [折外预测](results_v2/oof_predictions.csv)、[选模记录](results_v2/inner_scores.json)。

`python -m pip install -r requirements.txt` 后在仓库根运行 `python model_v2.py`；首版结果需保留在 `results/` 供同折对比，缺失时先运行 `python model.py`。本次环境 Python {summary['python']}、NumPy {summary['numpy']}、SciPy {summary['scipy']}、scikit-learn {summary['sklearn']}。详细约定见 [Agent.md](Agent.md)。joblib 仅加载可信来源文件。
"""
    (ROOT / "模型优化评价报告.md").write_text(text, encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    names, y, axis, x, paths, x_test = load_data()
    first = pd.read_csv(ROOT / "results" / "oof_predictions.csv")
    if list(first["spectrum"]) != list(names) or not np.allclose(first["actual"], y, rtol=0, atol=1e-10):
        raise ValueError("First model OOF data does not match calibration labels")
    prediction = np.full(len(y), np.nan)
    baseline = np.full(len(y), np.nan)
    fold_ids = np.zeros(len(y), dtype=int)
    folds, inner_scores = [], {}
    with threadpool_limits(limits=1):
        for fold, (train, valid) in enumerate(KFold(5, shuffle=True, random_state=OUTER_SEED).split(x), 1):
            best, scores = select_model(x[train], y[train])
            fitted = pipeline(best["method"], best["components"]).fit(x[train], y[train])
            prediction[valid] = fitted.predict(x[valid]).ravel()
            baseline[valid] = y[train].mean()
            fold_ids[valid] = fold
            folds.append({"fold": fold, **best, **metrics(y[valid], prediction[valid])})
            inner_scores[str(fold)] = scores
            print(f"Fold {fold}: {best['method']}, {best['components']}, MAE={folds[-1]['mae']:.4f}", flush=True)
        best, scores = select_model(x, y)
        fitted = pipeline(best["method"], best["components"]).fit(x, y)
        test_prediction = fitted.predict(x_test).ravel()
    inner_scores["final"] = scores
    if not np.array_equal(first["fold"].to_numpy(), fold_ids) or not np.allclose(first["baseline_prediction"], baseline):
        raise ValueError("First model outer folds or baseline differ")
    if not np.isfinite(prediction).all() or not np.isfinite(test_prediction).all() or (fold_ids == 0).any():
        raise ValueError("Incomplete or non-finite predictions")
    first_prediction = first["prediction"].to_numpy()
    summary = {"selected": metrics(y, prediction), "first": metrics(y, first_prediction),
               "baseline": metrics(y, baseline), "folds": folds, "final": best,
               "difference_ci": bootstrap_differences(y, first_prediction, prediction),
               "y_min": float(y.min()), "y_max": float(y.max()),
               "test_min": float(test_prediction.min()), "test_max": float(test_prediction.max()),
               "outside_count": int(((test_prediction < y.min()) | (test_prediction > y.max())).sum()),
               "high_count": int((y >= 30).sum()), "high_bias": float(np.mean(prediction[y >= 30] - y[y >= 30])),
               "high_mae": float(np.mean(np.abs(prediction[y >= 30] - y[y >= 30]))),
               "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
               "sklearn": sklearn.__version__}
    pd.DataFrame({"spectrum": names, "fold": fold_ids, "actual": y, "prediction": prediction,
                  "first_prediction": first_prediction, "baseline_prediction": baseline}).to_csv(
                      OUT / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"spectrum": [p.name for p in paths], "prediction": test_prediction}).to_csv(
        OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
    joblib.dump({"model": fitted, "wavenumber": axis, "method": best["method"],
                 "components": best["components"]}, OUT / "final_model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "inner_scores.json").write_text(json.dumps(inner_scores, indent=2) + "\n", encoding="utf-8")
    figures(y, first_prediction, prediction, baseline, test_prediction)
    write_report(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
