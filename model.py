"""Fit and evaluate the first near-infrared PLS regression model."""

from pathlib import Path
import json
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "近红外光谱数据与说明-2025年"
OUT = ROOT / "results"
COMPONENTS = range(1, 16)
OUTER_SEED = 42
INNER_SEED = 123


def metrics(y, prediction):
    residual = np.asarray(prediction) - np.asarray(y)
    if residual.size < 2:
        raise ValueError("SEP requires at least two observations")
    return {
        "sep": float(np.sqrt(np.sum(residual ** 2) / (residual.size - 1))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "r2": float(r2_score(y, prediction)),
    }


def load_spectra(paths, reference=None):
    spectra = []
    axis = reference
    for path in paths:
        values = np.loadtxt(path, delimiter=",", ndmin=2)
        if values.shape != (2074, 2) or not np.isfinite(values).all():
            raise ValueError(f"Invalid spectrum shape or values: {path.name}")
        if axis is None:
            axis = values[:, 0].copy()
        if not np.allclose(values[:, 0], axis, rtol=0, atol=1e-8):
            raise ValueError(f"Wavenumber axis mismatch: {path.name}")
        spectra.append(values[:, 1])
    return axis, np.asarray(spectra)


def load_data():
    labels = pd.read_excel(DATA / "校正集化学成分.xlsx", engine="openpyxl")
    if list(labels.columns) != ["序号", "光谱名称", "化学成分含量"]:
        raise ValueError("Unexpected calibration workbook columns")
    names = labels["光谱名称"]
    y = pd.to_numeric(labels["化学成分含量"], errors="raise").to_numpy(dtype=float)
    if len(labels) != 320 or names.isna().any() or names.duplicated().any() or not np.isfinite(y).all():
        raise ValueError("Invalid or duplicate calibration labels")
    if any(Path(name).name != name for name in names):
        raise ValueError("Calibration names must be filenames")
    calibration = DATA / "校正集"
    actual_names = {p.name for p in calibration.glob("*.csv")}
    if set(names) != actual_names:
        raise ValueError("Calibration filenames do not match workbook")
    axis, x = load_spectra([calibration / name for name in names])
    test_paths = sorted((DATA / "测试集").glob("*.csv"), key=lambda p: int(p.stem))
    if len(test_paths) != 80:
        raise ValueError("Expected 80 test spectra")
    _, x_test = load_spectra(test_paths, axis)
    return names.to_numpy(), y, axis, x, test_paths, x_test


def choose_components(x, y):
    """Choose lowest inner pooled MAE; on ties choose fewer components."""
    split = list(KFold(n_splits=4, shuffle=True, random_state=INNER_SEED).split(x))
    scores = []
    for count in COMPONENTS:
        prediction = np.empty(len(y))
        for train, valid in split:
            model = PLSRegression(n_components=count, scale=True, max_iter=500)
            model.fit(x[train], y[train])
            prediction[valid] = model.predict(x[valid]).ravel()
        scores.append(float(np.mean(np.abs(prediction - y))))
    best = int(np.argmin(scores)) + 1
    return best, scores


def make_plots(axis, x, y, prediction, baseline, test_prediction):
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160})
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in np.linspace(0, len(x) - 1, 32, dtype=int):
        ax.plot(axis, x[i], alpha=0.55, linewidth=0.7)
    ax.set(xlabel="Wavenumber (cm$^{-1}$)", ylabel="Absorbance", title="Calibration spectra (32 representative indices)")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(OUT / "spectra.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    lo = min(y.min(), prediction.min())
    hi = max(y.max(), prediction.max())
    axes[0].scatter(y, prediction, s=14, alpha=0.65)
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0].set(xlabel="Measured", ylabel="Out-of-fold prediction", title="Nested CV predictions")
    axes[1].scatter(y, prediction - y, s=14, alpha=0.65)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Measured", ylabel="Prediction - measured", title="Out-of-fold residuals")
    fig.tight_layout()
    fig.savefig(OUT / "oof_diagnostics.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(y, bins=20, alpha=0.65, label="Calibration measured")
    axes[0].hist(test_prediction, bins=20, alpha=0.65, label="Test predicted")
    axes[0].set(xlabel="Chemical content", ylabel="Samples", title="Value distributions")
    axes[0].legend(fontsize=8)
    names = ["PLSR", "Mean baseline"]
    axes[1].bar(names, [metrics(y, prediction)["mae"], metrics(y, baseline)["mae"]])
    axes[1].set(ylabel="Out-of-fold MAE", title="Comparison")
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png")
    plt.close(fig)


def write_report(summary):
    m, b = summary["plsr"], summary["baseline"]
    folds = "\n".join(
        f"| {f['fold']} | {f['components']} | {f['sep']:.4f} | {f['mae']:.4f} |"
        for f in summary["folds"]
    )
    text = f"""# 近红外化学成分定量：首个 PLSR 模型评价报告

## 结论与适用范围

320 条校正样本的五折**嵌套交叉验证折外预测**，PLSR 的 SEP 为 **{m['sep']:.4f}**、MAE 为 **{m['mae']:.4f}**；同期训练折均值基线分别为 {b['sep']:.4f}、{b['mae']:.4f}。这仅是现有校正样本随机拆分下的内部估计，不是独立测试集精度。化学成分名称及单位在原始说明中未给出，以下数值沿用原表尺度。

## 数据与方法

- 校正集 320 条，每条 2074 个波数/吸光度点；按工作簿“光谱名称”一一匹配，不使用序号推断对应关系。测试集 80 条，只有光谱、没有化学真值。读入时检查文件匹配、有限值、点数及所有波数轴一致性。
- 原始吸光度全波段输入 PLS 回归；`PLSRegression(scale=True)` 在每次拟合时仅用对应训练子集计算中心化和缩放。没有对全体数据预拟合预处理，也未用测试光谱参与选参。
- 外层五折 `KFold(shuffle=True, random_state={OUTER_SEED})`：每次留出 64 条作为外层验证。每个外层训练集内部使用四折 `KFold(shuffle=True, random_state={INNER_SEED})`，在潜变量数 1–15 中按**合并内层折外 MAE 最小**选参；并列取较小值。外层模型仅用该外层训练集拟合，预测其留出折；每条样本恰有一次折外预测。
- 均值基线使用每个外层训练折的化学值均值预测其留出折；其指标与 PLSR 使用同一组外层拆分。
- 最终模型在完整 320 条校正数据上，以相同内层四折规则独立选参（{summary['final_components']} 个潜变量）并重新拟合；只用于 80 条测试光谱预测。

![校正光谱](results/spectra.png)

图 1：按原表顺序等间隔抽取 32 条校正光谱，仅用于展示；建模使用全部光谱。

## 指标与结果

按提供的竞赛口径，`SEP = sqrt(sum((实际值 - 预测值)^2) / (n - 1))`，不扣除偏差；`MAE = sum(abs(实际值 - 预测值)) / n`。汇总指标对全部 320 条折外预测直接计算，而非五折指标平均。偏差定义为预测值减实际值的均值；R² 是对全部折外预测计算的辅助指标。

| 方法 | 样本数 | SEP | MAE | 偏差 | R² |
|---|---:|---:|---:|---:|---:|
| 嵌套交叉验证 PLSR | 320 | {m['sep']:.4f} | {m['mae']:.4f} | {m['bias']:.4f} | {m['r2']:.4f} |
| 外层训练折均值基线 | 320 | {b['sep']:.4f} | {b['mae']:.4f} | {b['bias']:.4f} | {b['r2']:.4f} |

| 外层折 | 内层选定潜变量数 | 折 SEP | 折 MAE |
|---|---:|---:|---:|
{folds}

![折外诊断](results/oof_diagnostics.png)

图 2：左图为折外预测与实测的对应关系，虚线为理想值；右图为折外残差。偏离虚线或零线的点表示该随机拆分下的预测误差。
事后描述性诊断：实测值不低于 30 的 {summary['high_count']} 条样本中，平均残差为 {summary['high_bias']:.4f}，MAE 为 {summary['high_mae']:.4f}，显示高值区域有低估倾向；该阈值并非预先设定，结论需要独立数据确认。

![分布与基线对比](results/comparison.png)

图 3：左图比较校正实测值与无标签测试集**预测值**的分布，不能解释为测试真值分布；两组样本量分别为 320 和 80，柱高为计数，不能直接比较比例。右图比较同拆分下折外 MAE。

## 测试集预测与局限

- [逐文件预测](results/test_predictions.csv) 含 80 个光谱文件名及预测值，预测范围 {summary['test_min']:.4f}–{summary['test_max']:.4f}；[折外逐样本结果](results/oof_predictions.csv) 可供追溯。预测是估计值，不是化验结果，也没有经过截断或四舍五入后再评估。
- 校正真值范围 {summary['y_min']:.4f}–{summary['y_max']:.4f}；测试预测中 {summary['outside_count']} 条超出此范围。即使预测在该范围内，也不能证明光谱处于训练分布内。
- 缺少样本来源、批次或重复测量的分组信息；随机拆分可能将相关样本分到不同折，导致精度过于乐观，不能据此推断跨批次、跨仪器或未来样本表现。应取得分组元数据并采用分组外验证。
- 测试集无化学真值，无法计算其 SEP、MAE 或确认预测准确性；需独立化验后以相同口径评估。单次随机五折存在抽样不确定性；未提供置信区间或独立外部验证。
- 最终选出的 {summary['final_components']} 个潜变量{'位于候选上限 15；这提示更大的候选范围值得在未来独立设计的嵌套验证中探索，但不证明其更优' if summary['final_components'] == 15 else '未触及候选上限 15'}。当前报告不基于外层结果反复扩大搜索范围。
- 当前仅探索原始光谱和潜变量数量，未比较预处理、异常值策略或其他模型；任何未来方案选择均须在外层训练数据内部完成，避免使用外层验证结果反复调参。

## 复现

在仓库根目录执行 `python -m pip install -r requirements.txt` 后运行 `python model.py`。产物包含本报告、图、折外预测、测试预测、指标 JSON 和最终模型。详细说明见 [Agent.md](Agent.md)。依赖已固定；Python {summary['python']}、scikit-learn {summary['sklearn']}、NumPy {summary['numpy']} 为本次运行版本。不同平台的数值库可能造成极小浮点差异。仅加载可信来源的 joblib 模型文件。
"""
    (ROOT / "模型评价报告.md").write_text(text, encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    names, y, axis, x, test_paths, x_test = load_data()
    prediction = np.full(len(y), np.nan)
    baseline = np.full(len(y), np.nan)
    fold_ids = np.zeros(len(y), dtype=int)
    folds = []
    with threadpool_limits(limits=1):
        outer = KFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
        for fold, (train, valid) in enumerate(outer.split(x), start=1):
            count, _ = choose_components(x[train], y[train])
            model = PLSRegression(n_components=count, scale=True, max_iter=500)
            model.fit(x[train], y[train])
            prediction[valid] = model.predict(x[valid]).ravel()
            baseline[valid] = y[train].mean()
            fold_ids[valid] = fold
            folds.append({"fold": fold, "components": count, **metrics(y[valid], prediction[valid])})
            print(f"Outer fold {fold}: components={count}, MAE={folds[-1]['mae']:.4f}", flush=True)
        count, scores = choose_components(x, y)
        final = PLSRegression(n_components=count, scale=True, max_iter=500)
        final.fit(x, y)
        test_prediction = final.predict(x_test).ravel()

    if not np.isfinite(prediction).all() or not np.isfinite(test_prediction).all() or (fold_ids == 0).any():
        raise ValueError("Incomplete or non-finite predictions")
    pd.DataFrame({"spectrum": names, "fold": fold_ids, "actual": y, "prediction": prediction,
                  "baseline_prediction": baseline}).to_csv(OUT / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"spectrum": [p.name for p in test_paths], "prediction": test_prediction}).to_csv(
        OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
    joblib.dump({"model": final, "wavenumber": axis, "components": count}, OUT / "final_model.joblib")
    summary = {"plsr": metrics(y, prediction), "baseline": metrics(y, baseline), "folds": folds,
               "final_components": count, "final_inner_mae": scores, "y_min": float(y.min()),
               "y_max": float(y.max()), "test_min": float(test_prediction.min()),
               "test_max": float(test_prediction.max()),
               "outside_count": int(((test_prediction < y.min()) | (test_prediction > y.max())).sum()),
               "high_count": int((y >= 30).sum()),
               "high_bias": float(np.mean(prediction[y >= 30] - y[y >= 30])),
               "high_mae": float(np.mean(np.abs(prediction[y >= 30] - y[y >= 30]))),
               "python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__}
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    make_plots(axis, x, y, prediction, baseline, test_prediction)
    write_report(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
