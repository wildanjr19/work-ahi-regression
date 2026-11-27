import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.audit import file_hash
from src.config import POSITIONS, TRAINING, SmokeConfig, tuning_candidates
from src.model_utils import load_checkpoint, train_experiment
from src.reporting import summarize_grid_search
from src.tuning import run_grid_search, validate_development_data


def synthetic_development(processed_dir=None):
    data = {}
    generator = np.random.default_rng(42)
    for split in ("train", "validation"):
        indices = np.arange(8) % 3
        data[split] = {
            "images": generator.integers(0, 256, (8, 32, 32, 3), dtype=np.uint8),
            "position_one_hot": np.eye(3, dtype=np.float32)[indices],
            "targets": np.arange(8, dtype=np.float64) + 10,
            "patient_ids": np.array([f"{split}_patient_{index:03d}" for index in range(8)]),
            "positions": np.array(POSITIONS)[indices],
            "sample_ids": np.array([f"{split}_sample_{index:03d}" for index in range(8)]),
        }
    return data, {"fingerprint": "synthetic", "hashes": {"synthetic": "fixture"},
                  "metadata": {"status": "synthetic"}}


class TinyRegressor(nn.Module):
    """Model kecil untuk menguji loop aktual tanpa bobot eksternal atau data pasien."""
    def __init__(self, architecture, weights=None, phase="scratch", config=TRAINING):
        super().__init__()
        self.architecture, self.config, self.phase = architecture, config, phase
        self.backbone = nn.Conv2d(3, 2, 1)
        self.backbone.requires_grad_(phase != "feature_extraction")
        self.head = nn.Linear(5, 1)

    def forward(self, sensor_image, sleep_position):
        features = self.backbone(sensor_image.float() / 255).mean(dim=(2, 3))
        return self.head(torch.cat((features, sleep_position), dim=1)).float()


class TuningRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target in ("src.tuning.TUNING_DIR", "src.model_utils.TUNING_DIR"):
            self.stack.enter_context(patch(target, self.root))
        self.loader = self.stack.enter_context(patch("src.tuning.load_development_data", side_effect=synthetic_development))
        self.runtime = self.stack.enter_context(patch("src.tuning.configure_runtime", return_value={"compute_device": "synthetic"}))

    def test_disabled_and_approval_guards_have_no_data_or_output_side_effects(self):
        self.assertEqual(run_grid_search("ResNet50")["status"], "disabled")
        with self.assertRaisesRegex(PermissionError, "approval"):
            run_grid_search("ResNet50", mode="full", enabled=True)
        with patch("src.tuning.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                run_grid_search("ResNet50", mode="full", enabled=True, full_training_approved=True)
        self.loader.assert_not_called()
        self.runtime.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_output_and_paths_outside_tuning_are_rejected(self):
        output = self.root / "ResNet50/smoke"
        output.mkdir(parents=True)
        marker = output / "keep.txt"
        marker.write_text("original")
        with self.assertRaises(FileExistsError):
            run_grid_search("ResNet50", enabled=True)
        with self.assertRaisesRegex(ValueError, "inside artifacts/tuning"):
            run_grid_search("VGG16", output_root=self.root.parent, enabled=True)
        self.assertEqual(marker.read_text(), "original")
        self.loader.assert_not_called()

    def test_test_data_and_invalid_inputs_fail_before_output_is_created(self):
        data, contract = synthetic_development()
        self.loader.side_effect = None
        self.loader.return_value = (dict(data, test={}), contract)
        with self.assertRaisesRegex(ValueError, "no test set"):
            run_grid_search("ResNet50", enabled=True)
        for change in ("patient_overlap", "nonfinite", "position", "pixels", "sample_overlap", "shape"):
            data, _ = synthetic_development()
            if change == "patient_overlap":
                data["validation"]["patient_ids"][0] = data["train"]["patient_ids"][0]
            elif change == "nonfinite":
                data["train"]["targets"][0] = np.nan
            elif change == "position":
                data["train"]["position_one_hot"][0] = [1, 1, 0]
            elif change == "pixels":
                data["train"]["images"] = data["train"]["images"].astype(float)
            elif change == "sample_overlap":
                data["validation"]["sample_ids"][0] = data["train"]["sample_ids"][0]
            else:
                data["validation"]["images"] = data["validation"]["images"][:, :, :31]
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_development_data(data)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_smoke_runs_actual_training_and_reload_without_fe_parent_or_test_arrays(self):
        with patch("src.model_utils.AHIRegressor", TinyRegressor), \
                patch("src.model_utils.plot_diagnostics", return_value=[]), \
                patch("torch.cuda.is_available", return_value=False), \
                patch("numpy.load", side_effect=AssertionError("No research NPZ")):
            summary = run_grid_search("ResNet50", enabled=True)
        self.assertEqual(summary["succeeded"], 3)
        self.assertEqual(summary["attempted"], 3)
        self.assertFalse(summary["selection_available"])
        root = Path(summary["output_directory"])
        self.assertFalse((root / "selection_validation.json").exists())
        records = json.loads((root / "candidates.json").read_text())
        for record in records:
            run = json.loads((root / record["directory"] / "run.json").read_text())
            self.assertEqual(run["initialization"], "independent")
            self.assertIsNone(run["parent_feature_checkpoint_sha256"])
            self.assertFalse(run["test_accessed"])
            self.assertEqual(run["optimizer_updates"], 1)
            expected = {key: run[key] for key in ("architecture", "experiment", "mode", "config",
                                                  "initialization", "data_fingerprint", "protocol_sha256",
                                                  "code_hashes", "checkpoint_sha256")}
            with patch("src.model_utils.AHIRegressor", TinyRegressor):
                load_checkpoint(root / record["checkpoint"], expected=expected)
                for field in ("mode", "architecture", "experiment", "initialization", "data_fingerprint",
                              "protocol_sha256", "code_hashes", "config", "checkpoint_sha256"):
                    changed = dict(expected, **{field: "wrong"})
                    with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Checkpoint"):
                        load_checkpoint(root / record["checkpoint"], expected=changed)
        with self.assertRaisesRegex(ValueError, "Smoke"):
            summarize_grid_search(root)

    def fake_training(self, architecture, experiment, data, contract, output_dir, **kwargs):
        self.assertEqual(set(data), {"train", "validation"})
        self.assertEqual(kwargs["initialization"], "independent")
        config = kwargs["config"]
        output_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = output_dir / "best.pt"
        checkpoint.write_bytes(b"synthetic checkpoint")
        metrics = pd.DataFrame([{"position": "Overall", "samples": 8, "MAE": config.dropout_rates[0],
                                 "MSE": 4., "RMSE": 2.}])
        return {"checkpoint_sha256": file_hash(checkpoint), "parameters": {"trainable": 1},
                "training_seconds": 2., "epochs_completed": 1, "best_epoch": 1,
                "optimizer_updates": 1, "amp_skipped_steps": 0}, metrics, checkpoint

    def test_full_grid_orchestration_and_tie_break_use_validation_without_fe_cost(self):
        with patch("src.tuning.torch.cuda.is_available", return_value=True), \
                patch("src.tuning.train_experiment", side_effect=self.fake_training) as trainer:
            summary = run_grid_search("VGG16", mode="full", enabled=True, full_training_approved=True)
        self.assertEqual(trainer.call_count, 36)
        root = Path(summary["output_directory"])
        selection = json.loads((root / "selection_validation.json").read_text())
        self.assertEqual(len(selection["selected_conditions"]), 3)
        for record in selection["selected_conditions"].values():
            self.assertEqual(record["config"]["dropout_rates"], [0.1, 0.1])
            self.assertEqual(record["cumulative_training_seconds"], 2.)
        self.assertFalse(selection["test_evaluated"])
        records = json.loads((root / "candidates.json").read_text())
        for field, value in (("mode", "smoke"), ("data_fingerprint", "changed"), ("status", "running")):
            modified = [dict(record) for record in records]
            modified[0][field] = value
            (root / "candidates.json").write_text(json.dumps(modified))
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize_grid_search(root)
        (root / "candidates.json").write_text(json.dumps(records))
        checkpoint = root / records[0]["checkpoint"]
        checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            summarize_grid_search(root)

    def test_failed_candidates_are_recorded_and_do_not_stop_other_candidates(self):
        calls = 0
        def training(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise FloatingPointError("synthetic nonfinite loss")
            return self.fake_training(*args, **kwargs)
        with patch("src.tuning.torch.cuda.is_available", return_value=True), \
                patch("src.tuning.train_experiment", side_effect=training):
            summary = run_grid_search("VGG19", mode="full", enabled=True, full_training_approved=True)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["succeeded"], 35)
        root = Path(summary["output_directory"])
        records = json.loads((root / "candidates.json").read_text())
        self.assertEqual(records[0]["status"], "failed")
        self.assertEqual(records[0]["error_type"], "FloatingPointError")
        self.assertTrue((root / records[0]["directory"] / "status.json").is_file())
        selection = json.loads((root / "selection_validation.json").read_text())
        self.assertEqual(selection["failed"], 1)
        self.assertFalse(selection["all_candidates_succeeded"])

    def test_all_failed_condition_prevents_selection(self):
        def training(architecture, experiment, *args, **kwargs):
            if experiment == "feature_extraction":
                raise RuntimeError("synthetic failure")
            return self.fake_training(architecture, experiment, *args, **kwargs)
        with patch("src.tuning.torch.cuda.is_available", return_value=True), \
                patch("src.tuning.train_experiment", side_effect=training):
            summary = run_grid_search("ResNet50", mode="full", enabled=True, full_training_approved=True)
        self.assertEqual(summary["failed"], 12)
        self.assertFalse(summary["selection_available"])
        self.assertFalse((Path(summary["output_directory"]) / "selection_validation.json").exists())

    def test_independent_training_rejects_parent_and_historical_output(self):
        data, contract = synthetic_development()
        config = tuning_candidates()[0]
        with self.assertRaisesRegex(ValueError, "no FE parent"):
            train_experiment("ResNet50", "fine_tuning", data, contract, self.root / "candidate",
                             config=config, feature_checkpoint="old.pt", initialization="independent")
        with self.assertRaisesRegex(ValueError, "inside artifacts/tuning"):
            train_experiment("ResNet50", "scratch", data, contract, self.root.parent,
                             config=config, initialization="independent")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_scheduler_early_stopping_and_best_weight_restore_share_validation_monitor(self):
        data, contract = synthetic_development()
        contract.update(tuning_protocol_sha256="synthetic", code_hashes={"synthetic": "fixture"})
        config = tuning_candidates()[-2]
        completed = 0
        def epoch(model, loader, device, criterion, optimizer=None, scaler=None, config=None):
            nonlocal completed
            if optimizer is not None:
                completed += 1
                with torch.no_grad():
                    model.head.bias.fill_(completed)
            loss = 1. if completed == 1 else 2.
            return {"loss": loss, "mae": 1., "rmse": float(np.sqrt(loss)),
                    "updates": 1 if optimizer is not None else 0, "skipped": 0}
        with patch("src.model_utils.AHIRegressor", TinyRegressor), \
                patch("src.model_utils.epoch_pass", side_effect=epoch), \
                patch("src.model_utils.plot_diagnostics", return_value=[]), \
                patch("torch.cuda.is_available", return_value=False):
            run, _, checkpoint = train_experiment(
                "ResNet50", "scratch", data, contract, self.root / "controlled", config=config,
                smoke=SmokeConfig(epochs=15), initialization="independent")
            restored = load_checkpoint(checkpoint)
        self.assertEqual(run["epochs_completed"], 11)
        self.assertEqual(run["best_epoch"], 1)
        self.assertEqual(restored.head.bias.item(), 1.)
        history = json.loads((checkpoint.parent / "history.json").read_text())
        self.assertEqual(history["learning_rate"][7], config.initial_learning_rate * config.reduce_lr_factor)

    def test_nonfinite_training_failure_keeps_progress_and_all_statuses(self):
        with patch("src.model_utils.AHIRegressor", TinyRegressor), \
                patch("src.model_utils.epoch_pass", side_effect=FloatingPointError("synthetic loss")), \
                patch("torch.cuda.is_available", return_value=False):
            summary = run_grid_search("ResNet50", enabled=True)
        self.assertEqual(summary["status"], "failed")
        root = Path(summary["output_directory"])
        for record in json.loads((root / "candidates.json").read_text()):
            self.assertEqual(record["status"], "failed")
            progress = json.loads((root / record["directory"] / "progress.json").read_text())
            self.assertGreater(progress["parameters"]["trainable"], 0)

    def test_interrupt_is_recorded_and_not_swallowed(self):
        with patch("src.tuning.train_experiment", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_grid_search("ResNet50", enabled=True)
        root = self.root / "ResNet50/smoke"
        records = json.loads((root / "candidates.json").read_text())
        self.assertEqual([record["status"] for record in records], ["interrupted", "pending", "pending"])
        self.assertFalse((root / "completion.json").exists())


if __name__ == "__main__":
    unittest.main()
