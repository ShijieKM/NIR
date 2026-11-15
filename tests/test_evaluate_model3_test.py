import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from evaluate_model3_test import evaluate


class Model3PostHocEvaluationTests(unittest.TestCase):
    def test_posthoc_evaluation_scores_frozen_predictions_without_exporting_truth_rows(self):
        ids = np.arange(321, 401)
        actual = np.linspace(10, 30, 80)
        model2 = pd.DataFrame({"spectrum": [f"{i}.csv" for i in ids], "prediction": actual + 1})
        model3 = pd.DataFrame({"spectrum": [f"{i}.csv" for i in ids], "prediction": actual + 0.5})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            labels_path = root / "truth.xlsx"
            model2_path = root / "model2.csv"
            model3_path = root / "model3.csv"
            output = root / "out"
            pd.DataFrame({"序号": ids[::-1], "化学成分含量": actual[::-1]}).to_excel(
                labels_path, index=False)
            model2.to_csv(model2_path, index=False)
            model3.to_csv(model3_path, index=False)
            with patch("evaluate_model3_test.PREDICTION_FILES",
                       {"冻结模型2": model2_path, "冻结模型3": model3_path}):
                result = evaluate(labels_path, output)
            self.assertEqual(result["status"],
                             "exploratory_post_hoc_test_check_test_set_already_used_for_models_1_and_2")
            self.assertAlmostEqual(result["models"]["冻结模型3"]["mae"], 0.5)
            self.assertTrue((output / "test_evaluation.md").is_file())
            self.assertFalse((output / "per_sample.csv").exists())


if __name__ == "__main__":
    unittest.main()
