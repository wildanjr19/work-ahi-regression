import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import regression_metrics
from src.model_utils import EXPERIMENTS
from src.reporting import summarize_full_experiments


class ReportingTests(unittest.TestCase):
    def test_smoke_and_mixed_protocols_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [{"experiment": name, "mode": "smoke", "architecture": "ResNet50", "data_fingerprint": "same"}
                    for name in EXPERIMENTS]
            (root / "summary.json").write_text(json.dumps(runs))
            with self.assertRaisesRegex(ValueError, "Smoke"):
                summarize_full_experiments(root)
            for run in runs:
                run["mode"] = "full"
            runs[0]["data_fingerprint"] = "different"
            (root / "summary.json").write_text(json.dumps(runs))
            with self.assertRaisesRegex(ValueError, "protocol"):
                summarize_full_experiments(root)

    def test_selection_uses_rmse_and_counts_feature_extraction_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = np.array([0., 10., 20., 30.])
            errors = {"scratch": np.array([7., 7., 7., 7.]),
                      "feature_extraction": np.array([0., 0., 0., 10.]),
                      "fine_tuning": np.array([4., 4., 4., 4.])}
            runs = []
            for name in EXPERIMENTS:
                output = root / name
                output.mkdir()
                metrics = regression_metrics(actual, actual + errors[name])
                pd.DataFrame([{"position": "Overall", **metrics}]).to_csv(output / "metrics_validation.csv", index=False)
                pd.DataFrame({"val_loss": [100., 80.]}).to_csv(output / "history.csv", index=False)
                runs.append({"architecture": "ResNet50", "experiment": name, "mode": "full", "data_fingerprint": "same",
                             "training_seconds": 2., "epochs_completed": 2, "parameters": {"trainable": 1},
                             "gpu_memory_training": {"peak_allocated_bytes": 0}, "optimizer_updates": 2,
                             "amp_skipped_steps": 0, "config": {"checkpoint_monitor": "val_loss"},
                             "checkpoint_sha256": "fixture", "research_status": "fixture"})
            (root / "summary.json").write_text(json.dumps(runs))
            comparison, _, selected = summarize_full_experiments(root)
            self.assertEqual(selected["experiment"], "fine_tuning")
            self.assertFalse(selected["test_evaluated"])
            cost = comparison.loc[comparison.Experiment == "fine_tuning", "cumulative_training_seconds"].iloc[0]
            self.assertEqual(cost, 4.)


if __name__ == "__main__":
    unittest.main()
