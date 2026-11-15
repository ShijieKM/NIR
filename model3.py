"""Nested comparison of PLS, VIP-selected PLS, SVR, and ridge for NIR."""

import json
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from threadpoolctl import threadpool_limits

import model
from model import INNER_SEED, OUTER_SEED, ROOT, load_data, metrics
from spectral import SpectralTransform


OUT = ROOT / "results_v3"


def build_candidates():
    candidates = []
    for method in ("raw", "snv", "snv_derivative"):
        for components in (5, 10, 15, 20):
            candidates.append({"family": "pls", "preprocessing": method,
                               "components": components})
    for variables in (30, 75, 150, 300):
        for components in (3, 8, 15):
            candidates.append({"family": "vip_pls", "preprocessing": "snv",
                               "variables": variables, "vip_components": 15,
                               "components": components})
    for c_value in (1, 10, 100):
        for gamma in ("scale", 0.001):
            candidates.append({"family": "svr_rbf", "preprocessing": "snv",
                               "C": c_value, "gamma": gamma, "epsilon": 0.1})
    for alpha in (0.1, 1, 10, 100):
        candidates.append({"family": "ridge", "preprocessing": "snv", "alpha": alpha})
    return candidates


CANDIDATES = build_candidates()


class VIPSelector(BaseEstimator, TransformerMixin):
    """Rank wavelengths using training-only PLS VIP scores and keep the top k."""

    def __init__(self, n_features=75, n_components=15):
        self.n_features = n_features
        self.n_components = n_components

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[1] < 2:
            raise ValueError("VIP selection requires a non-empty 2-D spectrum matrix and matching targets")
        n_components = min(self.n_components, x.shape[0] - 1, x.shape[1])
        if n_components < 1:
            raise ValueError("Not enough training observations for VIP selection")
        fit = PLSRegression(n_components=n_components, scale=True, max_iter=500)
        fit.fit(x, y)
        weights = fit.x_weights_
        norms = np.linalg.norm(weights, axis=0)
        if np.any(norms <= np.finfo(float).eps):
            raise ValueError("PLS produced zero-norm weights during VIP selection")
        explained_y = np.sum(fit.x_scores_ ** 2, axis=0) * np.square(fit.y_loadings_.reshape(-1))
        total = explained_y.sum()
        if total <= np.finfo(float).eps:
            raise ValueError("PLS explained no response variance during VIP selection")
        vip = np.sqrt(x.shape[1] * (np.square(weights / norms) @ explained_y) / total)
        count = min(int(self.n_features), x.shape[1])
        if count < 1:
            raise ValueError("VIP selection must retain at least one wavelength")
        self.scores_ = vip
        self.selected_indices_ = np.sort(np.argsort(vip, kind="stable")[-count:])
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError("Spectrum length differs from VIP training data")
        return x[:, self.selected_indices_]


def make_estimator(candidate):
    family = candidate["family"]
    preprocessing = SpectralTransform(candidate["preprocessing"])
    if family == "pls":
        return Pipeline([
            ("preprocess", preprocessing),
            ("regressor", PLSRegression(n_components=candidate["components"],
                                        scale=True, max_iter=500)),
        ])
    if family == "vip_pls":
        return Pipeline([
            ("preprocess", preprocessing),
            ("select", VIPSelector(candidate["variables"], candidate["vip_components"])),
            ("regressor", PLSRegression(n_components=candidate["components"],
                                        scale=True, max_iter=500)),
        ])
    if family == "svr_rbf":
        target_scaled = TransformedTargetRegressor(
            regressor=SVR(kernel="rbf", C=candidate["C"], gamma=candidate["gamma"],
                          epsilon=candidate["epsilon"]),
            transformer=StandardScaler(),
        )
        return Pipeline([
            ("preprocess", preprocessing),
            ("scale", StandardScaler()),
            ("regressor", target_scaled),
        ])
    if family == "ridge":
        target_scaled = TransformedTargetRegressor(
            regressor=Ridge(alpha=candidate["alpha"]),
            transformer=StandardScaler(),
        )
        return Pipeline([
            ("preprocess", preprocessing),
            ("scale", StandardScaler()),
            ("regressor", target_scaled),
        ])
    raise ValueError(f"Unknown model family: {family}")


def select_candidate(x, y):
    """Select the lowest pooled inner OOF MAE; preserve declared candidate order on ties."""
    split = list(KFold(n_splits=4, shuffle=True, random_state=INNER_SEED).split(x))
    scores = []
    for candidate in CANDIDATES:
        prediction = np.full(len(y), np.nan)
        for train, valid in split:
            estimator = make_estimator(candidate)
            estimator.fit(x[train], y[train])
            prediction[valid] = estimator.predict(x[valid]).reshape(-1)
        if not np.isfinite(prediction).all():
            raise ValueError(f"Incomplete inner predictions for candidate {candidate}")
        scores.append({**candidate, "mae": float(np.mean(np.abs(prediction - y)))})
    best = min(scores, key=lambda row: row["mae"])
    return best, scores


def bootstrap_differences(y, prior, selected, iterations=2000):
    rng = np.random.default_rng(20261115)
    delta = {"mae": [], "sep": []}
    for _ in range(iterations):
        ix = rng.integers(0, len(y), size=len(y))
        old_metrics, new_metrics = metrics(y[ix], prior[ix]), metrics(y[ix], selected[ix])
        delta["mae"].append(old_metrics["mae"] - new_metrics["mae"])
        delta["sep"].append(old_metrics["sep"] - new_metrics["sep"])
    return {key: np.percentile(values, [2.5, 97.5]).tolist() for key, values in delta.items()}


def plot_results(y, predictions, test_prediction, fold_records):
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160})
    lower = min(float(np.min(y)), *(float(np.min(p)) for p in predictions.values()))
    upper = max(float(np.max(y)), *(float(np.max(p)) for p in predictions.values()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y, predictions["model3"], s=14, alpha=0.65)
    axes[0].plot([lower, upper], [lower, upper], "k--", linewidth=1)
    axes[0].set(xlabel="Measured", ylabel="Model 3 out-of-fold prediction",
                title="Nested CV: prediction vs measured")
    axes[1].scatter(y, predictions["model3"] - y, s=14, alpha=0.65)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Measured", ylabel="Prediction - measured", title="Nested CV residuals")
    fig.tight_layout()
    fig.savefig(OUT / "oof_diagnostics.png")
    plt.close(fig)

    names = ["Mean baseline", "Model 1", "Model 2", "Model 3"]
    values = [metrics(y, predictions[key])["mae"]
              for key in ("baseline", "model1", "model2", "model3")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, values)
    axes[0].set(ylabel="Out-of-fold MAE", title="Same outer splits")
    counts = {}
    for row in fold_records:
        counts[row["family"]] = counts.get(row["family"], 0) + 1
    axes[1].bar(list(counts), list(counts.values()))
    axes[1].set(ylabel="Outer folds selected", title="Model 3 selection frequency")
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(y.min(), test_prediction.min()), max(y.max(), test_prediction.max()), 21)
    ax.hist(y, bins=bins, density=True, alpha=0.6, label="Calibration measured")
    ax.hist(test_prediction, bins=bins, density=True, alpha=0.6, label="Test predicted")
    ax.set(xlabel="Chemical content", ylabel="Density",
           title="Calibration values and test predictions (not test truth)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "test_prediction_distribution.png")
    plt.close(fig)


def write_report(summary):
    final = summary["final_candidate"]
    if final["family"] == "pls":
        final_text = (f"{final['preprocessing']} + PLS "
                      f"({final['components']} components)")
    elif final["family"] == "vip_pls":
        final_text = (f"SNV + VIP ({final['variables']} variables) + PLS "
                      f"({final['components']} components)")
    elif final["family"] == "svr_rbf":
        final_text = (f"SNV + RBF-SVR (C={final['C']}, "
                      f"gamma={final['gamma']}, epsilon={final['epsilon']})")
    else:
        final_text = f"SNV + Ridge (alpha={final['alpha']})"
    rows = "\n".join(
        f"| {row['fold']} | {row['family']} | {row['preprocessing']} | "
        f"{row.get('components', '-')} | {row.get('variables', '-')} | "
        f"{row['sep']:.4f} | {row['mae']:.4f} |"
        for row in summary["folds"]
    )
    methods = summary["models"]
    ci = summary["model2_minus_model3_oof_bootstrap_ci"]
    report = f"""# 模型 3：NIR 嵌套算法选择与评价

## 结论

在 320 条校正样本上使用与模型 1/2 相同的五折外层划分，对预先限定的候选算法执行**严格嵌套选模**后，模型 3 的折外 SEP 为 **{methods['model3']['sep']:.4f}**、MAE 为 **{methods['model3']['mae']:.4f}**。模型 1、模型 2 和训练折均值基线在相同外层样本上的指标分别见下表。本结果是限定候选集合内模型选择流程的交叉验证估计，不是对所有可能算法的保证。

**独立测试集说明：**测试标签此前已用于模型 1/2 的评价，已经不再是模型 3 的确认性独立验证集。模型 3 的测试预测仅在其嵌套交叉验证和全量校正集选参完成、预测文件冻结后作事后核对；即使报告测试误差，也不得据此调参、选模或宣称无偏模型比较。

## 数据与模型设计

校正集 320 个样本、每个样本 2074 个波数变量；标签由工作簿光谱文件名严格关联。测试集为 80 条光谱。模型 3 不读取测试真值进行任何拟合、选参或性能估计。

候选共 34 组，预先固定如下：

1. **全谱 PLS**：原始吸光度、SNV、SNV 后 Savitzky–Golay 一阶导数；每种尝试 5、10、15、20 个潜变量。
2. **VIP 筛选 + PLS**：SNV 后用训练子集拟合 PLS 并计算 VIP 变量重要性，保留排名前 30、75、150 或 300 个波数，再尝试 3、8、15 个潜变量。筛选器在每一个内层/外层训练折重新拟合，验证折只应用其筛选结果。
3. **RBF-SVR**：SNV 光谱后作训练折内逐变量标准化，`C` 为 1、10、100，`gamma` 为 `scale`、0.001，`epsilon=0.1`。响应值标准化也只用当前训练折，预测时逆变换回原尺度。
4. **Ridge**：SNV 后训练折内逐变量标准化，响应值训练折内标准化，`alpha` 为 0.1、1、10、100。

对所有候选使用相同外层 `KFold(5, shuffle=True, random_state=42)`；外层每次留出 64 条。每个外层训练集内部使用 `KFold(4, shuffle=True, random_state=123)`，按合并内层折外 MAE 最小选候选，完全并列时按报告列出的候选生成顺序决定。每个外层验证样本只预测一次。预处理、VIP 筛选、特征缩放、响应缩放和超参数选择都仅在当折训练数据拟合。全量校正集内层选参后再拟合最终模型 3。

![模型 3 折外诊断](results_v3/oof_diagnostics.png)

图 1：左图是模型 3 外层折外预测与实测的关系；右图是外层折外残差。

![模型比较与选择](results_v3/comparison.png)

图 2：左图为相同外层验证样本上的折外 MAE；右图显示每种算法在五个外层训练集被内层选中的次数。选择次数描述稳定性，不是算法优劣检验。

![测试预测分布](results_v3/test_prediction_distribution.png)

图 3：测试图层为模型 3 对 80 条光谱的**预测**，不是测试实测值。

## 折外结果

竞赛口径：`SEP = sqrt(sum((实际值 - 预测值)^2) / (n - 1))`（不扣偏差）；`MAE = mean(abs(实际值 - 预测值))`。汇总由全部 320 条折外预测计算。

| 方法 | 样本数 | SEP | MAE | 偏差（预测 − 实测） | R² |
|---|---:|---:|---:|---:|---:|
| 训练折均值基线 | 320 | {methods['baseline']['sep']:.4f} | {methods['baseline']['mae']:.4f} | {methods['baseline']['bias']:.4f} | {methods['baseline']['r2']:.4f} |
| 模型 1 原始 PLSR | 320 | {methods['model1']['sep']:.4f} | {methods['model1']['mae']:.4f} | {methods['model1']['bias']:.4f} | {methods['model1']['r2']:.4f} |
| 模型 2 SNV-PLS | 320 | {methods['model2']['sep']:.4f} | {methods['model2']['mae']:.4f} | {methods['model2']['bias']:.4f} | {methods['model2']['r2']:.4f} |
| 模型 3 嵌套选择 | 320 | {methods['model3']['sep']:.4f} | {methods['model3']['mae']:.4f} | {methods['model3']['bias']:.4f} | {methods['model3']['r2']:.4f} |

| 外层折 | 内层选中算法 | 预处理 | PLS 成分数 | VIP 保留变量数 | 折 SEP | 折 MAE |
|---:|---|---|---:|---:|---:|---:|
{rows}

与模型 2 的折外配对 MAE 差（模型 2 减模型 3）为 {methods['model2']['mae'] - methods['model3']['mae']:.4f}；SEP 差为 {methods['model2']['sep'] - methods['model3']['sep']:.4f}。以固定配对折外预测做 2000 次样本级 bootstrap，MAE 差的 95% 区间为 [{ci['mae'][0]:.4f}, {ci['mae'][1]:.4f}]，SEP 差为 [{ci['sep'][0]:.4f}, {ci['sep'][1]:.4f}]。正值代表模型 3 误差较低；区间只反映当前配对样本重抽样波动，不涵盖重新训练、样本组相关或候选集合设计不确定性。由于模型 3 是看到先前模型 1/2 结果后继续探索，这种跨版本比较也属于探索性。

## 最终模型与局限

完整 320 条校正集的内层选中方案：**{final_text}**；相应内层合并折外 MAE 为 {summary['final_inner_mae']:.4f}。80 条测试预测写入 [逐文件预测](results_v3/test_predictions.csv)，校正集折外预测见 [OOF 结果](results_v3/oof_predictions.csv)，全部候选每折分数见 [搜索记录](results_v3/inner_scores.json)，最终 estimator 与波数轴保存在 `results_v3/final_model.joblib`。

- 交叉验证是在未知样本独立同分布假设下的随机拆分；没有样本批次/来源分组，潜在同组相关会使误差偏乐观。
- 候选清单有意保持有限；全谱高维相关、VIP 筛选以及小样本下的 SVR/Ridge 参数均有不确定性。不能把这一次嵌套交叉验证推广为所有模型/全部采集条件下的无偏结论。
- 测试集此前已参与模型 1/2 的评价，模型 3 的测试数据核对是事后分析；报告不以它做算法选择，不声称为模型 3 的独立确认性指标。
- 化学成分名称、单位及可接受误差阈值仍未提供。仅凭这里的数字无法判断能否用于质量放行或其他业务决策；需要预先定义验收标准，并用新的外部批次样本验证。

## 方法依据

- 嵌套交叉验证可避免在同一 CV 结果上选参并报告乐观性能；scikit-learn 官方说明：[Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)。
- 将预处理和特征选择纳入训练流程，以避免交叉验证信息泄漏；scikit-learn 官方说明：[Common pitfalls: data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)。
- PLS 是光谱化学计量常用的潜变量回归框架；Wold, Sjöström & Eriksson, *PLS-regression: a basic tool of chemometrics*, Chemometrics and Intelligent Laboratory Systems 58 (2001), 109–130, [DOI](https://doi.org/10.1016/S0169-7439(01)00155-1)。
- VIP 变量筛选属于训练折内的可比较选项，而不是天然可靠的波长发现结论。经典 NIR 变量筛选方法还包括 CARS：Li et al., *Key wavelengths screening using competitive adaptive reweighted sampling method for multivariate calibration*, Analytica Chimica Acta 648 (2009), 77–84, [DOI](https://doi.org/10.1016/j.aca.2009.06.046)。此版本没有实现 CARS；避免将迭代启发式搜索成本及复杂度误作自动提升保证。

## 复现

安装依赖后依次执行：

```powershell
python model.py
python model_v2.py
python model3.py
python -m unittest discover -s tests
```

使用仓库锁定依赖，固定随机种子，并在单线程下训练；脚本不读取测试标签。只有在最终预测文件生成且不再更改后，才可进行单独的事后测试核对。更多项目约定见 [Agent.md](Agent.md)。"""
    (ROOT / "模型3评价报告.md").write_text(report, encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    names, y, axis, x, test_paths, x_test = load_data()
    first = pd.read_csv(ROOT / "results" / "oof_predictions.csv")
    second = pd.read_csv(ROOT / "results_v2" / "oof_predictions.csv")
    for previous in (first, second):
        if list(previous["spectrum"]) != list(names) or not np.allclose(previous["actual"], y, rtol=0, atol=1e-10):
            raise ValueError("Previous OOF data does not match calibration labels")
    if not np.array_equal(first["fold"].to_numpy(), second["fold"].to_numpy()):
        raise ValueError("Previous model folds do not match")
    prediction = np.full(len(y), np.nan)
    baseline = np.full(len(y), np.nan)
    fold_ids = np.zeros(len(y), dtype=int)
    fold_records, inner_scores = [], {}
    with threadpool_limits(limits=1):
        outer = KFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
        for fold, (train, valid) in enumerate(outer.split(x), start=1):
            if not np.all(first.loc[valid, "fold"].to_numpy() == fold):
                raise ValueError("Outer splits differ from recorded Model 1/2 folds")
            best, scores = select_candidate(x[train], y[train])
            estimator = make_estimator(best).fit(x[train], y[train])
            prediction[valid] = estimator.predict(x[valid]).reshape(-1)
            baseline[valid] = np.mean(y[train])
            fold_ids[valid] = fold
            fold_records.append({"fold": fold, **best, **metrics(y[valid], prediction[valid])})
            inner_scores[str(fold)] = scores
            print(f"Fold {fold}: {best['family']} {best}, MAE={fold_records[-1]['mae']:.4f}", flush=True)

        final_candidate, final_scores = select_candidate(x, y)
        final_model = make_estimator(final_candidate).fit(x, y)
        test_prediction = final_model.predict(x_test).reshape(-1)
    inner_scores["final"] = final_scores
    if not np.isfinite(prediction).all() or not np.isfinite(test_prediction).all() or np.any(fold_ids == 0):
        raise ValueError("Model 3 produced missing or non-finite predictions")

    previous_metrics = {
        "baseline": metrics(y, baseline),
        "model1": metrics(y, first["prediction"].to_numpy()),
        "model2": metrics(y, second["prediction"].to_numpy()),
    }
    summary = {
        "models": {**previous_metrics, "model3": metrics(y, prediction)},
        "folds": fold_records,
        "final_candidate": final_candidate,
        "final_inner_mae": final_candidate["mae"],
        "model2_minus_model3_oof_bootstrap_ci": bootstrap_differences(
            y, second["prediction"].to_numpy(), prediction),
        "y_min": float(np.min(y)), "y_max": float(np.max(y)),
        "test_min": float(np.min(test_prediction)), "test_max": float(np.max(test_prediction)),
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "outer_seed": OUTER_SEED, "inner_seed": INNER_SEED,
        "candidate_count": len(CANDIDATES),
    }
    previous_test = pd.read_csv(ROOT / "results_v2" / "test_predictions.csv")
    if list(previous_test["spectrum"]) != [p.name for p in test_paths]:
        raise ValueError("Model 2 frozen test predictions do not match current test spectra")
    pd.DataFrame({
        "spectrum": names, "fold": fold_ids, "actual": y, "prediction": prediction,
        "model1_prediction": first["prediction"].to_numpy(),
        "model2_prediction": second["prediction"].to_numpy(),
        "baseline_prediction": baseline,
    }).to_csv(OUT / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"spectrum": [p.name for p in test_paths], "prediction": test_prediction}).to_csv(
        OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
    joblib.dump({"model": final_model, "wavenumber": axis,
                 "candidate": final_candidate}, OUT / "final_model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    (OUT / "inner_scores.json").write_text(json.dumps(inner_scores, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    plot_results(y, {"baseline": baseline, "model1": first["prediction"].to_numpy(),
                     "model2": second["prediction"].to_numpy(), "model3": prediction},
                 test_prediction, fold_records)
    write_report(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
