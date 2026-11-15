"""Evaluate frozen test predictions against a separately supplied truth workbook."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from model import ROOT


PREDICTION_FILES = {
    "首版冻结 PLSR": ROOT / "results" / "test_predictions.csv",
    "优化版冻结 PLSR": ROOT / "results_v2" / "test_predictions.csv",
}
LABEL_COLUMNS = ["序号", "化学成分含量"]


def _file_ids(names, label):
    if len(names) != 80 or len(set(names)) != len(names):
        raise ValueError(f"{label} must contain exactly 80 unique spectrum filenames")
    ids = []
    for name in names:
        path = Path(str(name))
        if path.name != str(name) or path.suffix.lower() != ".csv" or not path.stem.isdigit():
            raise ValueError(f"Invalid spectrum filename in {label}: {name!r}")
        ids.append(int(path.stem))
    if len(set(ids)) != 80:
        raise ValueError(f"{label} has duplicate numeric spectrum IDs")
    return set(ids)


def load_aligned_data(labels_path, prediction_files=None):
    """Validate a one-to-one numeric-ID/filename join; never join by row order."""
    prediction_files = prediction_files or PREDICTION_FILES
    labels = pd.read_excel(labels_path, engine="openpyxl")
    if list(labels.columns) != LABEL_COLUMNS:
        raise ValueError(f"Expected label columns {LABEL_COLUMNS!r}, got {list(labels.columns)!r}")
    if len(labels) != 80 or labels.isna().any().any():
        raise ValueError("Truth workbook must contain exactly 80 complete rows")
    raw_ids = pd.to_numeric(labels["序号"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(raw_ids).all() or not np.equal(raw_ids, np.floor(raw_ids)).all():
        raise ValueError("Truth workbook IDs must be finite integers")
    ids = raw_ids.astype(int)
    if len(np.unique(ids)) != 80:
        raise ValueError("Truth workbook contains duplicate sample IDs")
    truth_values = pd.to_numeric(labels["化学成分含量"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(truth_values).all():
        raise ValueError("Truth workbook contains non-finite target values")

    spectra_dir = ROOT / "近红外光谱数据与说明-2025年" / "测试集"
    actual_spectrum_names = {path.name for path in spectra_dir.glob("*.csv")}
    if len(actual_spectrum_names) != 80:
        raise ValueError(f"Expected 80 test spectra in {spectra_dir}")

    frames = {}
    expected_ids = set(ids)
    for model_name, path in prediction_files.items():
        frame = pd.read_csv(path)
        if list(frame.columns) != ["spectrum", "prediction"]:
            raise ValueError(f"Unexpected columns in frozen predictions: {path}")
        if frame["spectrum"].isna().any() or frame["prediction"].isna().any():
            raise ValueError(f"Frozen predictions contain missing values: {path}")
        predicted_ids = _file_ids(frame["spectrum"].astype(str).tolist(), model_name)
        if set(frame["spectrum"].astype(str)) != actual_spectrum_names:
            raise ValueError(f"Frozen prediction filenames do not match the test spectrum directory: {model_name}")
        if predicted_ids != expected_ids:
            missing = sorted(expected_ids - predicted_ids)
            extra = sorted(predicted_ids - expected_ids)
            raise ValueError(f"Sample IDs do not match for {model_name}: missing={missing}, extra={extra}")
        frame = frame.copy()
        frame["sample_id"] = frame["spectrum"].str.replace(r"\.csv$", "", regex=True).astype(int)
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate sample IDs in {model_name}")
        frames[model_name] = frame.set_index("sample_id").sort_index()

    truth = pd.Series(truth_values, index=ids, name="actual").sort_index()
    if not np.array_equal(truth.index.to_numpy(), frames[next(iter(frames))].index.to_numpy()):
        raise ValueError("Truth and prediction IDs do not align exactly")
    return truth, frames


def score(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or predicted.shape != actual.shape or len(actual) != 80:
        raise ValueError("Expected paired vectors of exactly 80 observations")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual values and predictions must be finite")
    residual = predicted - actual
    absolute = np.abs(residual)
    return {
        "n": int(len(actual)),
        "actual_min": float(np.min(actual)),
        "actual_max": float(np.max(actual)),
        "sep": float(np.sqrt(np.sum(residual ** 2) / (len(actual) - 1))),
        "mae": float(np.mean(absolute)),
        "bias_prediction_minus_actual": float(np.mean(residual)),
        "r2": float(r2_score(actual, predicted)),
        "median_absolute_error": float(np.median(absolute)),
        "p90_absolute_error": float(np.quantile(absolute, 0.9)),
        "max_absolute_error": float(np.max(absolute)),
        "underpredicted_count": int(np.sum(residual < 0)),
        "overpredicted_count": int(np.sum(residual > 0)),
    }


def bootstrap_intervals(actual, predictions, iterations=5000, seed=20261115):
    """Paired sample bootstrap intervals; reflects fixed-sample resampling only."""
    actual = np.asarray(actual, dtype=float)
    rng = np.random.default_rng(seed)
    names = list(predictions)
    samples = {name: {"sep": [], "mae": []} for name in names}
    differences = {"sep": [], "mae": []}
    for _ in range(iterations):
        ix = rng.integers(0, len(actual), size=len(actual))
        fold = {name: score(actual[ix], np.asarray(predictions[name])[ix]) for name in names}
        for name in names:
            samples[name]["sep"].append(fold[name]["sep"])
            samples[name]["mae"].append(fold[name]["mae"])
        if len(names) == 2:
            differences["sep"].append(fold[names[0]]["sep"] - fold[names[1]]["sep"])
            differences["mae"].append(fold[names[0]]["mae"] - fold[names[1]]["mae"])
    intervals = {
        name: {metric: np.percentile(values, [2.5, 97.5]).tolist()
               for metric, values in by_metric.items()}
        for name, by_metric in samples.items()
    }
    difference_intervals = {
        metric: np.percentile(values, [2.5, 97.5]).tolist()
        for metric, values in differences.items()
    } if len(names) == 2 else {}
    return intervals, difference_intervals


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_plot(actual, predictions, output_path, legend_labels=None):
    names = list(predictions)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    low = min(np.min(actual), *(np.min(predictions[name]) for name in names))
    high = max(np.max(actual), *(np.max(predictions[name]) for name in names))
    legend = legend_labels or {names[0]: "Frozen PLSR v1", names[1]: "Frozen PLSR v2"}
    for name in names:
        axes[0].scatter(actual, predictions[name], s=22, alpha=0.65, label=legend[name])
    axes[0].plot([low, high], [low, high], "k--", linewidth=1)
    axes[0].set(xlabel="Measured", ylabel="Prediction", title="Independent test: prediction vs measured")
    axes[0].legend(fontsize=8)

    for name in names:
        axes[1].scatter(actual, predictions[name] - actual, s=22, alpha=0.65, label=legend[name])
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Measured", ylabel="Prediction - measured", title="Residuals")

    for name in names:
        axes[2].hist(np.abs(predictions[name] - actual), bins=16, alpha=0.55, label=legend[name])
    axes[2].set(xlabel="Absolute error", ylabel="Samples", title="Absolute error distribution")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(output_dir, labels_path, truth, frames, summary, intervals, difference_intervals, hashes):
    names = list(frames)
    actual = truth.to_numpy(dtype=float)
    optimized_prediction = frames[names[-1]]["prediction"].to_numpy(dtype=float)
    high_mask = actual >= 30
    rows = "\n".join(
        f"| {name} | {summary[name]['n']} | {summary[name]['sep']:.4f} | {summary[name]['mae']:.4f} | "
        f"{summary[name]['bias_prediction_minus_actual']:.4f} | {summary[name]['r2']:.4f} | "
        f"{summary[name]['median_absolute_error']:.4f} | {summary[name]['p90_absolute_error']:.4f} | "
        f"{summary[name]['underpredicted_count']}/{summary[name]['overpredicted_count']} |"
        for name in names
    )
    interval_rows = "\n".join(
        f"| {name} | [{intervals[name]['sep'][0]:.4f}, {intervals[name]['sep'][1]:.4f}] | "
        f"[{intervals[name]['mae'][0]:.4f}, {intervals[name]['mae'][1]:.4f}] |"
        for name in names
    )
    comparison = ""
    if len(names) == 2:
        comparison = (
            f"\n两个模型预测均在看到本次测试真值前冻结。本表比较属于对两个冻结模型的**事后比较**，"
            "不能据此再选出模型并声称该测试集仍提供了无偏的选模性能估计。"
            f"配对 bootstrap 中“{names[0]} 减 {names[1]}”SEP 差值的 95% 区间为 "
            f"[{difference_intervals['sep'][0]:.4f}, {difference_intervals['sep'][1]:.4f}]，"
            f"MAE 差值区间为 [{difference_intervals['mae'][0]:.4f}, {difference_intervals['mae'][1]:.4f}]；"
            "正值表示后一模型误差更低。"
        )
    text = f"""# 测试集独立评价报告

## 评价范围

本报告使用用户另行提供的测试集真实值，对此前保存并冻结的模型预测进行**独立测试集评价**。真实值工作簿为 `{Path(labels_path).name}`，共 80 行，字段为“序号”和“化学成分含量”，没有光谱文件名列。本次按工作簿的整数序号与预测 CSV 文件名数字部分关联，并验证真实值 ID 与两版预测文件 ID 集合均为一一对应的 80 个样本；没有按行位置配对。源表及真实值未复制到仓库，也未写入本报告、汇总 JSON 或误差表。

评价前已存在的预测文件：首版 `results/test_predictions.csv`（SHA-256 `{hashes['first_predictions']}`），优化版 `results_v2/test_predictions.csv`（SHA-256 `{hashes['optimized_predictions']}`）。用于评估的原始工作簿按用户明确授权与本次工作一并纳入**私有**仓库（SHA-256 `{hashes['truth_workbook']}`）；逐样本真值及逐样本误差未另外导出为 CSV 或报告。另报告冻结模型文件及建模代码的哈希用于追溯。此次只读取真实值、计算指标和生成诊断；没有以测试真值重训练、校准、调参、修改预测或选择模型。

## 独立测试指标

比赛定义为 `SEP = sqrt(sum((实际值 - 预测值)^2) / (n - 1))`（**不扣除残差偏差**），`MAE = mean(abs(实际值 - 预测值))`。

| 冻结预测 | 样本数 | SEP | MAE | 偏差（预测 − 实际） | R² | 绝对误差中位数 | 绝对误差 P90 | 低估/高估数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

![独立测试诊断图](test_evaluation.png)

图中散点及残差仅用于诊断，不包含逐条真实值导出。偏差的正值表示整体高估，负值表示整体低估。P90 为绝对误差的第 90 百分位。

优化版绝对误差最大值为 {summary[names[-1]]['max_absolute_error']:.4f}；优化版实测值范围为 {summary[names[-1]]['actual_min']:.4f}–{summary[names[-1]]['actual_max']:.4f}（原始表未给单位）。按此前探索报告中使用的实测值 30 作为事后诊断切点，优化版该区间包含 {int(np.sum(high_mask))} 条样本，MAE 为 {float(np.mean(np.abs(optimized_prediction[high_mask] - actual[high_mask]))):.4f}、偏差为 {float(np.mean(optimized_prediction[high_mask] - actual[high_mask])):.4f}；此切点不代表业务阈值，也未用于选模。

## 与校正集交叉验证区分

| 评估集 | 样本数 | 优化版 PLSR SEP | 优化版 PLSR MAE |
|---|---:|---:|---:|
| 校正集嵌套交叉验证折外预测 | 320 | 3.4672 | 2.5147 |
| 独立测试集冻结预测 | 80 | {summary[names[-1]]['sep']:.4f} | {summary[names[-1]]['mae']:.4f} |

这两行对应不同样本与评估设计，交叉验证值不是测试标签的一部分。独立测试结果可直接描述这 80 个配对样本；不要在看到它之后调整模型或用其筛选模型。

## 样本 bootstrap 区间与解释

对已冻结的 80 对样本按样本对进行 5000 次有放回重抽样（固定种子 20261115），得到如下 95% 百分位区间：

| 冻结预测 | SEP 95% 区间 | MAE 95% 区间 |
|---|---:|---:|
{interval_rows}

这些区间反映当前 80 对已观测预测与实测的样本重抽样波动；不包括重新采集样本、训练、选模和潜在批次/样本相关结构带来的不确定性。{comparison}

优化版的 SEP 和 MAE 在这批样本上略低于首版，但上述配对差值区间均跨过 0，因此不能据此认定优化版具有可靠优势。

测试集上的实测性能可作为这批独立样本的直接验证，但是否“可用”还需要化学成分名称/单位、业务验收阈值及样本代表性。测试集表现不替代未来批次或新仪器的外部验证。校正集的嵌套交叉验证结果仍见仓库中的[优化版模型报告](../模型优化评价报告.md)；不要把交叉验证值与本报告的独立测试值混为一谈。

## 复现及隐私

在仓库根目录执行：

```powershell
python evaluate_test.py --labels "D:\\路径\\测试集的真实值.xlsx"
python -m unittest discover -s tests
```

评估输出保存在被 Git 忽略的 `results_external/`：本报告、诊断图和只含汇总指标的 JSON。标签工作簿及逐样本实际值/残差不会写入输出。代码不修改预测文件。正式发布前请重新检查文件权限和数据授权；不要将真实值表、包含逐样本真值的图表或本地 `results_external/` 加入 Git。"""
    (output_dir / "test_evaluation.md").write_text(text, encoding="utf-8")


def evaluate(labels_path, output_dir=None):
    output_dir = Path(output_dir) if output_dir else ROOT / "results_external"
    output_dir.mkdir(parents=True, exist_ok=True)
    truth, frames = load_aligned_data(labels_path)
    actual = truth.to_numpy()
    predictions = {name: frame["prediction"].to_numpy(dtype=float) for name, frame in frames.items()}
    summary = {name: score(actual, predicted) for name, predicted in predictions.items()}
    intervals, difference_intervals = bootstrap_intervals(actual, predictions)
    hashes = {
        "first_predictions": _sha256(PREDICTION_FILES["首版冻结 PLSR"]),
        "optimized_predictions": _sha256(PREDICTION_FILES["优化版冻结 PLSR"]),
        "first_model": _sha256(ROOT / "results" / "final_model.joblib"),
        "optimized_model": _sha256(ROOT / "results_v2" / "final_model.joblib"),
        "first_code": _sha256(ROOT / "model.py"),
        "optimized_code": _sha256(ROOT / "model_v2.py"),
        "truth_workbook": _sha256(labels_path),
    }
    payload = {
        "n": len(truth),
        "label_filename": Path(labels_path).name,
        "join": "workbook integer sample ID == prediction CSV filename numeric stem",
        "models": summary,
        "bootstrap": {
            "iterations": 5000,
            "seed": 20261115,
            "intervals": intervals,
            "paired_difference_first_minus_optimized": difference_intervals,
            "scope": "fixed paired predictions; does not include refit or model-selection uncertainty",
        },
        "sha256": hashes,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    create_plot(actual, predictions, output_dir / "test_evaluation.png")
    write_report(output_dir, labels_path, truth, frames, summary, intervals, difference_intervals, hashes)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, help="Path to the externally supplied truth workbook")
    parser.add_argument("--output-dir", default=str(ROOT / "results_external"),
                        help="Output directory (default: ignored results_external/)")
    args = parser.parse_args()
    result = evaluate(args.labels, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
