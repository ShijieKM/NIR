import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

import model
import model3


class Model3Tests(unittest.TestCase):
    def test_candidate_space_and_vip_selection(self):
        self.assertEqual(len(model3.CANDIDATES), 34)
        rng = np.random.default_rng(7)
        x = rng.normal(size=(80, 40))
        y = 2 * x[:, 5] - x[:, 21] + rng.normal(scale=0.1, size=80)
        selector = model3.VIPSelector(n_features=8, n_components=5).fit(x[:60], y[:60])
        self.assertEqual(len(selector.selected_indices_), 8)
        self.assertTrue(np.all(selector.selected_indices_[:-1] < selector.selected_indices_[1:]))
        self.assertEqual(selector.transform(x[60:]).shape, (20, 8))
        changed_validation = model3.VIPSelector(n_features=8, n_components=5).fit(
            np.vstack([x[:60], x[60:] * 1000])[:60], y[:60])
        np.testing.assert_array_equal(selector.selected_indices_, changed_validation.selected_indices_)

    def test_inner_selection_fits_candidates_on_inner_train_only(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(60, 25))
        y = 3 * x[:, 3] + rng.normal(scale=0.2, size=60)
        candidates = [
            {"family": "pls", "preprocessing": "raw", "components": 2},
            {"family": "ridge", "preprocessing": "snv", "alpha": 1.0},
        ]
        with patch.object(model3, "CANDIDATES", candidates):
            best, scores = model3.select_candidate(x, y)
        self.assertEqual(len(scores), 2)
        self.assertTrue(np.isfinite([row["mae"] for row in scores]).all())
        self.assertEqual(best, min(scores, key=lambda row: row["mae"]))

    def test_each_model_family_fits_and_predicts(self):
        rng = np.random.default_rng(23)
        x = rng.normal(size=(64, 40))
        y = 2 * x[:, 7] - 0.5 * x[:, 31] + rng.normal(scale=0.1, size=64)
        candidates = [
            {"family": "pls", "preprocessing": "snv", "components": 3},
            {"family": "vip_pls", "preprocessing": "snv", "variables": 12,
             "vip_components": 5, "components": 3},
            {"family": "svr_rbf", "preprocessing": "snv", "C": 10,
             "gamma": "scale", "epsilon": 0.1},
            {"family": "ridge", "preprocessing": "snv", "alpha": 1.0},
        ]
        for candidate in candidates:
            estimator = model3.make_estimator(candidate).fit(x[:48], y[:48])
            prediction = estimator.predict(x[48:])
            self.assertEqual(prediction.shape, (16,))
            self.assertTrue(np.isfinite(prediction).all())

    def test_final_oof_and_frozen_test_predictions(self):
        names, y, axis, x, paths, x_test = model.load_data()
        oof = pd.read_csv(model3.OUT / "oof_predictions.csv")
        test = pd.read_csv(model3.OUT / "test_predictions.csv")
        summary = json.loads((model3.OUT / "metrics.json").read_text(encoding="utf-8"))
        scores = json.loads((model3.OUT / "inner_scores.json").read_text(encoding="utf-8"))
        saved = joblib.load(model3.OUT / "final_model.joblib")
        self.assertEqual(list(oof["spectrum"]), list(names))
        self.assertEqual(list(test["spectrum"]), [path.name for path in paths])
        np.testing.assert_allclose(oof["actual"], y, rtol=0, atol=1e-12)
        self.assertEqual(summary["candidate_count"], len(model3.CANDIDATES))
        self.assertEqual(summary["final_candidate"],
                         min(scores["final"], key=lambda row: row["mae"]))
        np.testing.assert_array_equal(saved["wavenumber"], axis)
        with threadpool_limits(limits=1):
            np.testing.assert_allclose(saved["model"].predict(x_test).ravel(), test["prediction"],
                                       rtol=1e-10, atol=1e-10)
            for fold, (train, valid) in enumerate(
                    KFold(5, shuffle=True, random_state=42).split(x), start=1):
                selected = min(scores[str(fold)], key=lambda row: row["mae"])
                self.assertEqual(summary["folds"][fold - 1]["family"], selected["family"])
                np.testing.assert_allclose(
                    model3.make_estimator(selected).fit(x[train], y[train]).predict(x[valid]).ravel(),
                    oof.loc[valid, "prediction"], rtol=1e-9, atol=1e-9)
        for key in ("baseline", "model1", "model2", "model3"):
            col = {"baseline": "baseline_prediction", "model1": "model1_prediction",
                   "model2": "model2_prediction", "model3": "prediction"}[key]
            for metric, value in model.metrics(y, oof[col]).items():
                self.assertAlmostEqual(summary["models"][key][metric], value, places=10)


if __name__ == "__main__":
    unittest.main()
