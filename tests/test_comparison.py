import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.audit import file_hash
from src.comparison import baseline_features, fit_baselines, load_held_out, select_validation, transfer_contributions
from src.evaluation import prediction_table, save_evaluation
from src.preprocessing import save_json


def payload(targets, shift=0):
    count = len(targets)
    return {"scaled_features": np.arange(count * 6).reshape(count, 2, 1, 3) / 20 + shift,
            "imf_mask": np.ones((count, 2, 3), dtype=bool), "position_one_hot": np.eye(3)[np.arange(count) % 3],
            "targets": np.array(targets, dtype=float), "patient_ids": np.array([f"p{i:03d}" for i in range(count)]),
            "positions": np.array([("Supine", "Left", "Right")[i % 3] for i in range(count)]),
            "sample_ids": np.array([f"s{i}" for i in range(count)])}


class ComparisonTests(unittest.TestCase):
    def test_baseline_input_ignores_ids_and_targets_but_keeps_mask_and_position(self):
        first = payload([0, 10, 600])
        second = dict(first, targets=np.array([7, 8, 9]), patient_ids=np.array(["x", "y", "z"]))
        np.testing.assert_array_equal(baseline_features(first), baseline_features(second))
        self.assertEqual(baseline_features(first).shape, (3, 15))
        first["scaled_features"][0, 0, 0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            baseline_features(first)

    def test_selection_cannot_use_test_or_position_rows(self):
        table = pd.DataFrame([{"Model": "A", "Experiment": "scratch", "Subset": "validation", "position": "Overall", "MAE": 1., "RMSE": 5.},
                              {"Model": "B", "Experiment": "scratch", "Subset": "validation", "position": "Overall", "MAE": 2., "RMSE": 4.}])
        self.assertEqual(select_validation(table)["Model"], "B")
        for modified in (table.assign(Subset="test"), table.assign(position="Right")):
            with self.assertRaisesRegex(ValueError, "validation"):
                select_validation(modified)

    def test_transfer_change_preserves_negative_improvement_and_zero_denominator(self):
        table = pd.DataFrame([{"Model": "ResNet50", "Subset": "test", "position": "Overall", "Experiment": name, "MAE": value, "RMSE": value}
                              for name, value in (("scratch", 0.), ("feature_extraction", 2.), ("fine_tuning", 1.))])
        result = transfer_contributions(table)
        self.assertEqual(result.iloc[0].MAE_reduction, -2.)
        self.assertTrue(pd.isna(result.iloc[0].MAE_reduction_percent))
        self.assertEqual(result.iloc[2].RMSE_reduction_percent, 50.)

    def test_baselines_fit_scaler_and_median_on_training_only(self):
        train, validation = payload([0, 2, 4, 600]), payload([1000, 2000, 3000], shift=100)
        with tempfile.TemporaryDirectory() as directory:
            fit_baselines(train, validation, directory, alphas=(0.1, 100.))
            ridge = joblib.load(Path(directory) / "baselines/Ridge/best.joblib")
            dummy = joblib.load(Path(directory) / "baselines/DummyMedian/best.joblib")
            np.testing.assert_allclose(ridge.named_steps["standardscaler"].mean_, baseline_features(train).mean(axis=0))
            self.assertEqual(dummy.constant_.item(), np.median(train["targets"]))
            search = pd.read_csv(Path(directory) / "ridge_validation_search.csv")
            best = search.sort_values(["RMSE", "MAE", "alpha"]).iloc[0]
            self.assertEqual(ridge.named_steps["ridge"].alpha, best.alpha)
            self.assertFalse((Path(directory) / "predictions_test.csv").exists())

    def test_test_loader_requires_unchanged_frozen_protocol_and_runs_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "checkpoint.txt"
            asset.write_text("frozen checkpoint")
            protocol = root / "frozen.json"
            save_json(protocol, {"test_evaluation_authorized": True, "assets": {str(asset): file_hash(asset)}, "contract": {}})
            digest = file_hash(protocol)
            def loaded(*args):
                self.assertTrue((root / "test_access_started.json").is_file())
                return "test payload"
            with patch("src.comparison.load_payload", side_effect=loaded) as loader:
                self.assertEqual(load_held_out(protocol, digest, root), "test payload")
                with self.assertRaisesRegex(RuntimeError, "already"):
                    load_held_out(protocol, digest, root)
                self.assertEqual(loader.call_count, 1)
                asset.write_text("changed checkpoint")
                with self.assertRaisesRegex(ValueError, "asset changed"):
                    load_held_out(protocol, digest, root)
                self.assertEqual(loader.call_count, 1)
                save_json(protocol, {"changed": True})
                with self.assertRaisesRegex(ValueError, "protocol changed"):
                    load_held_out(protocol, digest, root)

    def test_test_metrics_are_named_test_and_reject_mislabeled_rows(self):
        frame = prediction_table(payload([0, 10, 600]), [-1, 12, 590], "Ridge", "baseline", "test", "full")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "subset"):
                save_evaluation(frame, directory)
            saved = save_evaluation(frame, directory, subset="test")
            self.assertTrue((Path(directory) / "predictions_test.csv").is_file())
            self.assertFalse((Path(directory) / "predictions_validation.csv").exists())
            self.assertEqual(saved.iloc[0].negative_predictions, 1)


if __name__ == "__main__":
    unittest.main()
