"""Evaluasi lanjutan kandidat tuning terpilih tanpa training ulang CNN."""

import gc
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .audit import file_hash
from .config import TUNING_DIR, tuning_candidates
from .evaluation import evaluate_predictions
from .model_utils import ARCHITECTURES, EXPERIMENTS


def collect_tuned_cnn_results(contract, validation, tuning_root):
    from .comparison import validate_prediction_rows

    metrics, predictions, costs, assets, histories = [], [], [], {}, {}
    versions, normalizations = set(), set()
    configurations = {json.dumps(asdict(config), sort_keys=True) for config in tuning_candidates()}
    for architecture in ARCHITECTURES:
        root = Path(tuning_root) / architecture / "full"
        documents = {}
        for name in ("protocol", "candidates", "completion", "selection_validation"):
            path = root / f"{name}.json"
            documents[name] = json.loads(path.read_text(encoding="utf-8"))
            assets[str(path.resolve())] = file_hash(path)
        protocol, records = documents["protocol"], documents["candidates"]
        completion, selection = documents["completion"], documents["selection_validation"]
        digest = file_hash(root / "protocol.json")
        if (protocol["architecture"] != architecture or protocol["mode"] != "full"
                or protocol["initialization"] != "independent" or protocol["test_accessed"]
                or protocol["data_fingerprint"] != contract["fingerprint"]
                or protocol["preprocessing_hashes"] != contract["hashes"]
                or completion["status"] != "completed" or completion["test_accessed"]
                or completion["failed"] != 0 or completion["succeeded"] != len(records)
                or completion["protocol_sha256"] != digest
                or completion["data_fingerprint"] != contract["fingerprint"]
                or selection["protocol_sha256"] != digest or selection["mode"] != "full"
                or selection["subset"] != "validation" or selection["test_evaluated"]
                or selection["data_fingerprint"] != contract["fingerprint"]):
            raise ValueError("Tuning results differ from the completed validation-only protocol")
        if len(records) != len(EXPERIMENTS) * len(configurations):
            raise ValueError("The complete independent grid is required")
        if {json.dumps(config, sort_keys=True) for config in protocol["candidates"]} != configurations:
            raise ValueError("Tuning protocol grid differs")
        if set(selection["selected_conditions"]) != set(EXPERIMENTS):
            raise ValueError("All three selected tuning conditions are required")
        for record in records:
            if (record["status"] != "succeeded" or record["mode"] != "full"
                    or record["architecture"] != architecture or record["initialization"] != "independent"
                    or record["data_fingerprint"] != contract["fingerprint"]
                    or record["protocol_sha256"] != digest or record["code_hashes"] != protocol["code_hashes"]
                    or not np.isfinite([record["RMSE"], record["MAE"], record["MSE"]]).all()):
                raise ValueError("Candidate metadata differs from the tuning protocol")
        for experiment in EXPERIMENTS:
            candidates = [record for record in records if record["experiment"] == experiment]
            if (len(candidates) != len(configurations)
                    or {json.dumps(record["config"], sort_keys=True) for record in candidates} != configurations):
                raise ValueError("Missing or duplicate grid configurations")
            record = min(candidates, key=lambda item: (item["RMSE"], item["MAE"], item["candidate_id"]))
            if record != selection["selected_conditions"][experiment]:
                raise ValueError("Selected candidate differs from validation ranking")
            folder = (root / record["directory"]).resolve()
            checkpoint = (root / record["checkpoint"]).resolve()
            if (not folder.is_relative_to(root.resolve()) or checkpoint != folder / "best.pt"
                    or file_hash(checkpoint) != record["checkpoint_sha256"]):
                raise ValueError("Candidate checkpoint path or checksum differs")
            run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
            expected = {name: record[name] for name in (
                "architecture", "experiment", "mode", "initialization", "config", "data_fingerprint",
                "protocol_sha256", "code_hashes", "checkpoint_sha256",
            )}
            if (any(run.get(name) != value for name, value in expected.items()) or run["test_accessed"]
                    or run["parent_feature_checkpoint_sha256"] is not None
                    or run["preprocessing_hashes"] != contract["hashes"]
                    or run["validation_samples"] != len(validation["targets"])):
                raise ValueError("Selected run differs from the independent candidate")
            history = pd.read_csv(folder / "history.csv")
            if (len(history) != run["epochs_completed"] or not np.isfinite(history.to_numpy()).all()
                    or record["epochs_completed"] != run["epochs_completed"]):
                raise ValueError("Selected candidate history differs")
            expected["epoch"] = int(history.val_loss.argmin()) + 1
            if run["best_epoch"] != expected["epoch"] or record["best_epoch"] != expected["epoch"]:
                raise ValueError("Selected best epoch differs")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if any(json.dumps(saved.get(name), sort_keys=True) != json.dumps(value, sort_keys=True)
                   for name, value in expected.items() if name != "checkpoint_sha256"):
                raise ValueError("Selected checkpoint metadata differs")
            del saved
            gc.collect()
            frame = pd.read_csv(folder / "predictions_validation.csv", dtype={"Patient ID": "string"},
                                float_precision="round_trip")
            validate_prediction_rows(frame, validation, architecture, experiment, "validation")
            evaluated = evaluate_predictions(frame)
            pd.testing.assert_frame_equal(evaluated, pd.read_csv(folder / "metrics_validation.csv"),
                                          check_exact=False, rtol=1e-10, atol=1e-10)
            overall = evaluated[evaluated.position == "Overall"].iloc[0]
            np.testing.assert_allclose([overall[name] for name in ("RMSE", "MAE", "MSE")],
                                       [record[name] for name in ("RMSE", "MAE", "MSE")], rtol=1e-10, atol=1e-10)
            metrics.append(evaluated.assign(Model=architecture, Experiment=experiment, Subset="validation"))
            predictions.append(frame)
            histories[f"{architecture}/{experiment}"] = history
            costs.append({"Model": architecture, "Experiment": experiment, "kind": "CNN",
                          "candidate_id": record["candidate_id"], "initialization": "independent",
                          "compute_device": run["compute_device"], "epochs": len(history),
                          "best_epoch": expected["epoch"], "total_parameters": run["parameters"]["total"],
                          "trainable_parameters": run["parameters"]["trainable"],
                          "training_seconds": run["training_seconds"],
                          "cumulative_training_seconds": run["training_seconds"],
                          "peak_allocated_MiB": run["gpu_memory_training"]["peak_allocated_bytes"] / 1024**2,
                          "amp_skipped_steps": run["amp_skipped_steps"], "checkpoint": str(checkpoint),
                          "checkpoint_sha256": record["checkpoint_sha256"], "checkpoint_expected": expected,
                          "last_val_loss": history.val_loss.iloc[-1], "best_val_loss": history.val_loss.min(),
                          "train_loss_at_best_epoch": history.loss.iloc[expected["epoch"] - 1]})
            for filename in ("best.pt", "run.json", "status.json", "history.csv", "predictions_validation.csv", "metrics_validation.csv"):
                path = folder / filename
                assets[str(path)] = file_hash(path)
            versions.add(json.dumps(run["versions"], sort_keys=True))
            normalizations.add(run["input_normalization"])
    if len(versions) != 1 or len(normalizations) != 1:
        raise ValueError("Tuned CNN versions or normalization differ")
    return pd.concat(metrics, ignore_index=True), pd.concat(predictions, ignore_index=True), pd.DataFrame(costs), assets, histories


def run_tuning_evaluation(output_root=None, *, tuning_root=TUNING_DIR / "grid_v1", test_evaluation_approved=False):
    from .comparison import run_comparison

    tuning_root = Path(tuning_root).resolve()
    output_root = (tuning_root / "evaluation_v1") if output_root is None else Path(output_root).resolve()
    if (not tuning_root.is_relative_to(TUNING_DIR.resolve())
            or not output_root.is_relative_to(TUNING_DIR.resolve())):
        raise ValueError("Tuning evaluation must remain inside artifacts/tuning")
    return run_comparison(output_root, test_evaluation_approved=test_evaluation_approved, tuning_root=tuning_root)
