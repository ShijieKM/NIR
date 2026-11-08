import json
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

import model
import model_v2
from spectral import SpectralTransform


class OptimizedModelTests(unittest.TestCase):
    def test_transform_is_per_spectrum_and_rejects_constant_snv(self):
        x = np.arange(60, dtype=float).reshape(3, 20) ** 2 + 1
        for method in model_v2.METHODS:
            transform = SpectralTransform(method).fit(x)
            np.testing.assert_allclose(transform.transform(x)[0], transform.transform(x[:1])[0], atol=1e-14)
        with self.assertRaises(ValueError):
            SpectralTransform("snv").fit_transform(np.ones((2, 20)))

    def test_outputs_and_fold_models(self):
        names, y, axis, x, paths, x_test = model.load_data()
        oof = pd.read_csv(model_v2.OUT / "oof_predictions.csv")
        predictions = pd.read_csv(model_v2.OUT / "test_predictions.csv")
        summary = json.loads((model_v2.OUT / "metrics.json").read_text(encoding="utf-8"))
        scores = json.loads((model_v2.OUT / "inner_scores.json").read_text(encoding="utf-8"))
        stored = joblib.load(model_v2.OUT / "final_model.joblib")
        self.assertEqual(list(oof["spectrum"]), list(names))
        np.testing.assert_allclose(oof["actual"], y, rtol=0, atol=1e-12)
        self.assertEqual(list(predictions["spectrum"]), [p.name for p in paths])
        np.testing.assert_array_equal(stored["wavenumber"], axis)
        self.assertEqual(stored["method"], summary["final"]["method"])
        self.assertEqual(stored["components"], summary["final"]["components"])
        with threadpool_limits(limits=1):
            np.testing.assert_allclose(stored["model"].predict(x_test).ravel(), predictions["prediction"],
                                       rtol=1e-11, atol=1e-11)
            for fold, (train, valid) in enumerate(KFold(5, shuffle=True, random_state=42).split(x), 1):
                selected = min(scores[str(fold)], key=lambda row: row["mae"])
                record = summary["folds"][fold - 1]
                self.assertEqual((record["method"], record["components"]),
                                 (selected["method"], selected["components"]))
                self.assertTrue((oof.loc[valid, "fold"] == fold).all())
                np.testing.assert_allclose(oof.loc[valid, "baseline_prediction"], y[train].mean())
                estimator = model_v2.pipeline(selected["method"], selected["components"]).fit(x[train], y[train])
                np.testing.assert_allclose(estimator.predict(x[valid]).ravel(), oof.loc[valid, "prediction"],
                                           rtol=1e-10, atol=1e-10)
        final = min(scores["final"], key=lambda row: row["mae"])
        self.assertEqual(final, summary["final"])
        for key, column in [("selected", "prediction"), ("first", "first_prediction"),
                            ("baseline", "baseline_prediction")]:
            for metric, value in model.metrics(y, oof[column]).items():
                self.assertAlmostEqual(value, summary[key][metric], places=10)


if __name__ == "__main__":
    unittest.main()
