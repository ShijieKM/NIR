"""Exploratory post-hoc comparison of frozen model 2 and model 3 predictions."""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_test import bootstrap_intervals, create_plot, load_aligned_data, score
from model import ROOT


PREDICTION_FILES = {
    "冻结模型2": ROOT / "results_v2" / "test_predictions.csv",
    "冻结模型3": ROOT / "results_v3" / "test_predictions.csv",
}


def evaluate(labels_path, output_dir=None):
    output_dir = Path(output_dir) if output_dir else ROOT / "results_v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    truth, frames = load_aligned_data(labels_path, PREDICTION_FILES)
    actual = truth.to_numpy(dtype=float)
    predictions = {name: frame["prediction"].to_numpy(dtype=float)
                   for name, frame in frames.items()}
    metrics = {name: score(actual, value) for name, value in predictions.items()}
    intervals, differences = bootstrap_intervals(actual, predictions, iterations=2000)
    high = actual >= 30
    model3_prediction = predictions["冻结模型3"]
    report = f"""# 模型 3 测试集事后核对

> **重要：该测试集此前已用于模型 1/2 的评价，独立确认性用途已经消耗。**以下结果是在模型 3 和预测文件冻结后，对同一测试集做的探索性/事后核对；不能用于选模、调参或宣称模型 3 的独立无偏性能。

按“序号”与测试光谱文件名数字编号核验 80 个唯一样本的一一对应后，冻结模型 2/3 的预测评价如下。竞赛 SEP 为 `sqrt(SSE/(n-1))`，不扣偏差；MAE 为平均绝对误差。

| 已冻结模型 | 样本数 | SEP | MAE | 偏差（预测 − 实测） | R² | 最大绝对误差 |
|---|---:|---:|---:|---:|---:|---:|
| 模型 2（SNV-PLS） | 80 | {metrics['冻结模型2']['sep']:.4f} | {metrics['冻结模型2']['mae']:.4f} | {metrics['冻结模型2']['bias_prediction_minus_actual']:.4f} | {metrics['冻结模型2']['r2']:.4f} | {metrics['冻结模型2']['max_absolute_error']:.4f} |
| 模型 3（嵌套算法选择） | 80 | {metrics['冻结模型3']['sep']:.4f} | {metrics['冻结模型3']['mae']:.4f} | {metrics['冻结模型3']['bias_prediction_minus_actual']:.4f} | {metrics['冻结模型3']['r2']:.4f} | {metrics['冻结模型3']['max_absolute_error']:.4f} |

![模型3测试集事后诊断](test_evaluation.png)

按 2000 次样本级配对 bootstrap，差值“模型 2 减模型 3”的 MAE 95% 区间为 [{differences['mae'][0]:.4f}, {differences['mae'][1]:.4f}]，SEP 区间为 [{differences['sep'][0]:.4f}, {differences['sep'][1]:.4f}]。正数表示模型 3 误差更低。该区间仅描述固定预测的重抽样变化，且属于已消耗测试集上的事后比较，不能作为确认性选模依据。

实测值不低于 30 的 {int(high.sum())} 条样本中，模型 3 的 MAE 为 {np.mean(np.abs(model3_prediction[high] - actual[high])):.4f}，平均偏差为 {np.mean(model3_prediction[high] - actual[high]):.4f}。此为事后描述切片，不是业务阈值或调参依据。

模型 3 的主要选择和性能估计仍以[校正集五折嵌套交叉验证报告](../模型3评价报告.md)为准。由于缺少化学成分单位、验收容差及未来批次验证，不能据本次事后测试核对断言可供实际放行使用。
"""
    (output_dir / "test_evaluation.md").write_text(report, encoding="utf-8")
    create_plot(actual, predictions, output_dir / "test_evaluation.png",
                {"冻结模型2": "Frozen model 2", "冻结模型3": "Frozen model 3"})
    payload = {
        "n": int(len(actual)),
        "status": "exploratory_post_hoc_test_check_test_set_already_used_for_models_1_and_2",
        "models": metrics,
        "bootstrap_intervals": intervals,
        "paired_difference_model2_minus_model3": differences,
        "high_value_threshold_30_post_hoc": {
            "n": int(high.sum()),
            "model3_mae": float(np.mean(np.abs(model3_prediction[high] - actual[high]))),
            "model3_bias_prediction_minus_actual": float(np.mean(model3_prediction[high] - actual[high])),
        },
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=str(
        ROOT / "近红外光谱数据与说明-2025年" / "测试集的真实值.xlsx"))
    parser.add_argument("--output-dir", default=str(ROOT / "results_v3"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.labels, args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
