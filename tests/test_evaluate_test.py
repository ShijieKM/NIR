import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_test import load_aligned_data, score


class IndependentEvaluationTests(unittest.TestCase):
    def test_metrics_use_competition_sep_without_bias_correction(self):
        actual = np.arange(80, dtype=float)
        prediction = actual + 2
        result = score(actual, prediction)
        self.assertEqual(result["n"], 80)
        self.assertAlmostEqual(result["sep"], np.sqrt(80 * 4 / 79))
        self.assertAlmostEqual(result["mae"], 2)
        self.assertAlmostEqual(result["bias_prediction_minus_actual"], 2)

    def test_join_uses_sample_id_not_workbook_row_order(self):
        ids = np.arange(321, 401)
        labels = pd.DataFrame({"序号": ids[::-1], "化学成分含量": ids[::-1] / 10})
        prediction = pd.DataFrame({"spectrum": [f"{i}.csv" for i in ids],
                                   "prediction": ids.astype(float)})
        with tempfile.TemporaryDirectory() as temp:
            truth_path = Path(temp) / "truth.xlsx"
            prediction_path = Path(temp) / "prediction.csv"
            labels.to_excel(truth_path, index=False)
            prediction.to_csv(prediction_path, index=False)
            truth, predictions = load_aligned_data(
                truth_path, {"frozen": prediction_path}
            )
        aligned = predictions["frozen"]
        self.assertEqual(len(truth), 80)
        self.assertEqual(aligned.loc[321, "spectrum"], "321.csv")
        self.assertEqual(truth.loc[321], 32.1)
        self.assertEqual(aligned.loc[400, "spectrum"], "400.csv")

    def test_mismatched_label_ids_are_rejected(self):
        labels = pd.DataFrame({"序号": np.arange(320, 400), "化学成分含量": np.ones(80)})
        prediction = pd.DataFrame({"spectrum": [f"{i}.csv" for i in range(321, 401)],
                                   "prediction": np.ones(80)})
        with tempfile.TemporaryDirectory() as temp:
            truth_path = Path(temp) / "truth.xlsx"
            prediction_path = Path(temp) / "prediction.csv"
            labels.to_excel(truth_path, index=False)
            prediction.to_csv(prediction_path, index=False)
            with self.assertRaisesRegex(ValueError, "Sample IDs do not match"):
                load_aligned_data(truth_path, {"frozen": prediction_path})

    def test_duplicate_truth_ids_are_rejected(self):
        ids = np.arange(321, 401)
        ids[-1] = ids[-2]
        labels = pd.DataFrame({"序号": ids, "化学成分含量": np.ones(80)})
        prediction = pd.DataFrame({"spectrum": [f"{i}.csv" for i in range(321, 401)],
                                   "prediction": np.ones(80)})
        with tempfile.TemporaryDirectory() as temp:
            truth_path = Path(temp) / "truth.xlsx"
            prediction_path = Path(temp) / "prediction.csv"
            labels.to_excel(truth_path, index=False)
            prediction.to_csv(prediction_path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate sample IDs"):
                load_aligned_data(truth_path, {"frozen": prediction_path})


if __name__ == "__main__":
    unittest.main()
