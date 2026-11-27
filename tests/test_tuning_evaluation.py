import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.audit import file_hash
from src.comparison import run_comparison
from src.config import tuning_candidates
from src.evaluation import evaluate_predictions, prediction_table
from src.model_utils import ARCHITECTURES, EXPERIMENTS
from src.preprocessing import save_json
from src.tuning_evaluation import collect_tuned_cnn_results, run_tuning_evaluation
from test_comparison import payload


class TuningEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.validation = payload([0, 10, 600])
        self.contract = {"fingerprint": "synthetic", "hashes": {}, "metadata": {"sources": {}, "status": "synthetic"}}
        configurations = json.loads(json.dumps([asdict(config) for config in tuning_candidates()]))
        for architecture in ARCHITECTURES:
            root = self.root / architecture / "full"
            root.mkdir(parents=True)
            protocol = {"architecture": architecture, "mode": "full", "initialization": "independent",
                        "test_accessed": False, "data_fingerprint": "synthetic", "preprocessing_hashes": {},
                        "code_hashes": {"fixture": "synthetic"}, "candidates": configurations}
            save_json(root / "protocol.json", protocol)
            digest = file_hash(root / "protocol.json")
            records, selected = [], {}
            for experiment in EXPERIMENTS:
                for index, config in enumerate(configurations):
                    candidate_id = f"lr_{config['initial_learning_rate']:.0e}_dropout_{config['dropout_rates'][0]:.1f}"
                    record = {"architecture": architecture, "experiment": experiment, "mode": "full",
                              "initialization": "independent", "data_fingerprint": "synthetic",
                              "protocol_sha256": digest, "code_hashes": protocol["code_hashes"],
                              "config": config, "candidate_id": candidate_id, "directory": f"{experiment}/{candidate_id}",
                              "checkpoint": f"{experiment}/{candidate_id}/best.pt", "status": "succeeded",
                              "RMSE": float(index + 1), "MAE": float(index + 1), "MSE": float((index + 1)**2),
                              "epochs_completed": 1, "best_epoch": 1}
                    if index == 0:
                        folder = root / record["directory"]
                        folder.mkdir(parents=True)
                        saved = {name: record[name] for name in ("architecture", "experiment", "mode", "initialization",
                                 "data_fingerprint", "protocol_sha256", "code_hashes", "config")}
                        torch.save(dict(saved, epoch=1, model_state_dict={}), folder / "best.pt")
                        record["checkpoint_sha256"] = file_hash(folder / "best.pt")
                        run = dict(record, test_accessed=False, parent_feature_checkpoint_sha256=None,
                                   preprocessing_hashes={}, validation_samples=len(self.validation["targets"]),
                                   compute_device="cpu", parameters={"total": 2, "trainable": 1}, training_seconds=2.,
                                   gpu_memory_training={"peak_allocated_bytes": 0}, amp_skipped_steps=0,
                                   versions={"torch": "synthetic"}, input_normalization="synthetic")
                        save_json(folder / "run.json", run)
                        save_json(folder / "status.json", record)
                        pd.DataFrame({"loss": [2.], "mae": [1.], "val_loss": [1.], "val_mae": [1.]}).to_csv(folder / "history.csv", index=False)
                        frame = prediction_table(self.validation, self.validation["targets"] + 1,
                                                 architecture, experiment, "validation", "full")
                        frame.to_csv(folder / "predictions_validation.csv", index=False)
                        evaluate_predictions(frame).to_csv(folder / "metrics_validation.csv", index=False)
                        selected[experiment] = record
                    records.append(record)
            save_json(root / "candidates.json", records)
            save_json(root / "completion.json", {"status": "completed", "test_accessed": False, "failed": 0,
                      "succeeded": len(records), "protocol_sha256": digest, "data_fingerprint": "synthetic"})
            save_json(root / "selection_validation.json", {"mode": "full", "subset": "validation", "test_evaluated": False,
                      "protocol_sha256": digest, "data_fingerprint": "synthetic", "selected_conditions": selected})

    def test_collector_keeps_independent_cost_and_rejects_changed_ranking(self):
        metrics, _, costs, _, _ = collect_tuned_cnn_results(self.contract, self.validation, self.root)
        self.assertEqual(len(metrics), len(ARCHITECTURES) * len(EXPERIMENTS) * 4)
        np.testing.assert_array_equal(costs.training_seconds, costs.cumulative_training_seconds)
        self.assertTrue(all(value["initialization"] == "independent" for value in costs.checkpoint_expected))
        path = self.root / "ResNet50/full/selection_validation.json"
        selection = json.loads(path.read_text())
        selection["selected_conditions"]["scratch"]["candidate_id"] = "not-the-validation-winner"
        save_json(path, selection)
        with self.assertRaisesRegex(ValueError, "ranking"):
            collect_tuned_cnn_results(self.contract, self.validation, self.root)

    def test_smoke_wrong_fingerprint_and_corrupt_checkpoint_are_rejected(self):
        path = self.root / "ResNet50/full/protocol.json"
        original = path.read_bytes()
        protocol = json.loads(original)
        for field, value in (("mode", "smoke"), ("data_fingerprint", "other")):
            save_json(path, dict(protocol, **{field: value}))
            with self.assertRaisesRegex(ValueError, "protocol"):
                collect_tuned_cnn_results(self.contract, self.validation, self.root)
        path.write_bytes(original)
        checkpoint = next((self.root / "ResNet50/full").rglob("best.pt"))
        checkpoint.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            collect_tuned_cnn_results(self.contract, self.validation, self.root)

    def test_approval_existing_output_and_historical_paths_block_before_data(self):
        with patch("src.comparison.load_development_data", side_effect=AssertionError("No data access")) as loader:
            with self.assertRaises(PermissionError):
                run_tuning_evaluation()
            with self.assertRaisesRegex(ValueError, "artifacts/tuning"):
                run_tuning_evaluation(self.root / "historical", test_evaluation_approved=True)
            with self.assertRaises(FileExistsError):
                run_comparison(self.root, test_evaluation_approved=True)
            loader.assert_not_called()

    def test_freeze_precedes_single_test_access_without_cnn_training(self):
        output = self.root / "evaluation"
        processed = self.root / "processed"
        processed.mkdir()
        test = dict(payload([0, 10, 600]), patient_components=np.array(["c1", "c2", "c3"]))
        observed = []

        def load(split, contract, *args):
            observed.append(split)
            if split == "test":
                frozen = json.loads((output / "frozen_protocol.json").read_text())
                chosen = json.loads((output / "selection_validation.json").read_text())
                self.assertTrue(chosen["selected_before_test"])
                self.assertEqual(chosen["selected"], frozen["selection"])
                self.assertEqual(len(frozen["checkpoint_expectations"]), 9)
                self.assertTrue((output / "test_access_started.json").is_file())
                return test
            return self.validation

        def checkpoint(path, device, *, expected):
            self.assertEqual(expected["mode"], "full")
            self.assertEqual(expected["initialization"], "independent")
            self.assertEqual(file_hash(path), expected["checkpoint_sha256"])
            return object()

        with patch("src.comparison.TUNING_DIR", self.root), \
                patch("src.comparison.PROCESSED_DIR", processed), \
                patch("src.comparison.configure_runtime", return_value={}), \
                patch("src.comparison.load_development_data", return_value=({}, self.contract)), \
                patch("src.comparison.load_payload", side_effect=load), \
                patch("src.comparison.load_checkpoint", side_effect=checkpoint), \
                patch("src.comparison.make_dataset", return_value=None), \
                patch("src.comparison.predict_model", return_value=np.array([-1., 12., 590.])), \
                patch("src.comparison.plot_diagnostics"), \
                patch("src.comparison_plots.plot_comparison", return_value=[]), \
                patch("src.comparison_plots.research_discussion", return_value="Synthetic discussion"), \
                patch("src.model_utils.train_experiment", side_effect=AssertionError("No CNN training")) as trainer:
            metrics, _, choices = run_comparison(output, test_evaluation_approved=True, tuning_root=self.root)
            self.assertEqual(observed, ["train", "validation", "test"])
            self.assertEqual(len(metrics), 11 * 4 * 2)
            self.assertEqual(choices["cnn"]["RMSE"], 1.)
            self.assertEqual(int(metrics[(metrics.Model == "ResNet50") & (metrics.Experiment == "scratch")
                                         & (metrics.Subset == "test") & (metrics.position == "Overall")].iloc[0].negative_predictions), 1)
            trainer.assert_not_called()
            with self.assertRaises(FileExistsError):
                run_comparison(output, test_evaluation_approved=True, tuning_root=self.root)
            self.assertEqual(observed.count("test"), 1)


if __name__ == "__main__":
    unittest.main()
