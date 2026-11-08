import json
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import model


class ModelTests(unittest.TestCase):
    def test_metrics_use_requested_sep_without_bias_removal(self):
        result = model.metrics([1, 2, 3], [3, 4, 5])
        self.assertAlmostEqual(result["sep"], np.sqrt(6))
        self.assertAlmostEqual(result["mae"], 2)
        self.assertAlmostEqual(result["bias"], 2)

    def test_outputs_match_data_and_model(self):
        names, y, axis, x, paths, x_test = model.load_data()
        oof = pd.read_csv(model.OUT / "oof_predictions.csv")
        test = pd.read_csv(model.OUT / "test_predictions.csv")
        summary = json.loads((model.OUT / "metrics.json").read_text(encoding="utf-8"))
        stored = joblib.load(model.OUT / "final_model.joblib")
        self.assertEqual(list(oof["spectrum"]), list(names))
        np.testing.assert_allclose(oof["actual"], y, rtol=0, atol=1e-12)
        self.assertEqual(list(test["spectrum"]), [p.name for p in paths])
        np.testing.assert_array_equal(stored["wavenumber"], axis)
        self.assertEqual(stored["components"], summary["final_components"])
        np.testing.assert_allclose(stored["model"].predict(x_test).ravel(), test["prediction"], rtol=1e-11, atol=1e-11)
        expected = np.zeros(len(y), dtype=int)
        for fold, (train, valid) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(x), 1):
            expected[valid] = fold
            np.testing.assert_allclose(oof.loc[valid, "baseline_prediction"], y[train].mean())
        np.testing.assert_array_equal(oof["fold"], expected)
        for name, column in [("plsr", "prediction"), ("baseline", "baseline_prediction")]:
            for key, value in model.metrics(y, oof[column]).items():
                self.assertAlmostEqual(value, summary[name][key], places=10)


if __name__ == "__main__":
    unittest.main()
