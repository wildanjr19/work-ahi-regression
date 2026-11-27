"""Grid search bersama; hanya train/validation, nonaktif tanpa enabled=True."""

import gc
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .audit import file_hash
from .config import POSITIONS, PROCESSED_DIR, PROJECT_ROOT, SMOKE, TUNING_DIR, tuning_candidates
from .model_utils import ARCHITECTURES, EXPERIMENTS, configure_runtime, load_development_data, train_experiment
from .preprocessing import save_json
from .reporting import summarize_grid_search


def validate_development_data(data):
    if set(data) != {"train", "validation"}:
        raise ValueError("Tuning accepts train and validation only; no test set")
    required = {"images", "position_one_hot", "targets", "patient_ids", "positions", "sample_ids"}
    for split, payload in data.items():
        if not required.issubset(payload):
            raise ValueError(f"Missing development arrays: {split}")
        count = len(payload["targets"])
        images = payload["images"]
        if not count or any(len(payload[name]) != count for name in required):
            raise ValueError("Development arrays must be nonempty and aligned")
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3 or min(images.shape[1:3]) < 32:
            raise ValueError("Expected raw NHWC uint8 images with three channels")
        if payload["targets"].shape != (count,) or not np.isfinite(payload["targets"]).all():
            raise ValueError("Targets must remain finite one-dimensional values")
        positions = payload["positions"].tolist()
        one_hot = payload["position_one_hot"]
        if (one_hot.shape != (count, len(POSITIONS)) or not set(positions).issubset(POSITIONS)
                or not np.array_equal(one_hot, np.eye(len(POSITIONS))[[POSITIONS.index(value) for value in positions]])):
            raise ValueError("Position one-hot differs from Supine, Left, Right")
        if any(not isinstance(value, (str, np.str_)) for value in payload["patient_ids"]):
            raise ValueError("Patient IDs must remain strings")
        if len(set(payload["sample_ids"])) != count:
            raise ValueError("Duplicate development sample IDs")
    for name in ("patient_ids", "sample_ids"):
        if set(data["train"][name]) & set(data["validation"][name]):
            raise ValueError(f"Development overlap: {name}")
    if data["train"]["images"].shape[1:] != data["validation"]["images"].shape[1:]:
        raise ValueError("Train and validation image shapes differ")


def run_grid_search(architecture, *, processed_dir=PROCESSED_DIR, output_root=None,
                    mode="smoke", enabled=False, full_training_approved=False):
    """Smoke tiga kandidat tetap; full menguji 12 kandidat pada setiap strategi."""
    if architecture not in ARCHITECTURES or mode not in ("smoke", "full"):
        raise ValueError("Invalid tuning architecture or mode")
    if not enabled:
        return {"status": "disabled", "architecture": architecture, "mode": mode}
    if mode == "full" and not full_training_approved:
        raise PermissionError("Full grid search requires approval after the pipeline smoke report")
    if mode == "full" and not torch.cuda.is_available():
        raise RuntimeError("Full grid search requires CUDA")
    base = TUNING_DIR if output_root is None else Path(output_root)
    root = (base / architecture / mode).resolve()
    if not root.is_relative_to(TUNING_DIR.resolve()):
        raise ValueError("Tuning output must be inside artifacts/tuning")
    if root.exists():
        raise FileExistsError("Tuning output exists; choose a new output_root")
    data, contract = load_development_data(processed_dir)
    validate_development_data(data)
    if not contract.get("fingerprint") or not contract.get("hashes"):
        raise ValueError("Preprocessing fingerprint and hashes are required")
    runtime = configure_runtime()
    candidates = tuning_candidates()
    if mode == "smoke":
        candidates = tuple(config for config in candidates
                           if config.initial_learning_rate == 1e-3 and config.dropout_rates == (0.3, 0.3))
    root.mkdir(parents=True, exist_ok=False)
    code_hashes = {name: file_hash(PROJECT_ROOT / name) for name in (
        "src/config.py", "src/model_utils.py", "src/tuning.py", "src/reporting.py",
        "src/evaluation.py", "src/preprocessing.py", "docs/desain_eksperimen.md", "docs/search_space.md",
    )}
    protocol = {"architecture": architecture, "mode": mode, "initialization": "independent",
                "data_fingerprint": contract["fingerprint"], "preprocessing_hashes": contract["hashes"],
                "data_status": contract["metadata"]["status"], "code_hashes": code_hashes,
                "runtime": runtime, "candidates": [asdict(config) for config in candidates],
                "smoke_config": asdict(SMOKE) if mode == "smoke" else None,
                "criterion": "validation RMSE, then MAE for ties", "test_previously_exposed": True,
                "test_accessed": False, "refit_train_validation": False,
                "full_training_approved": full_training_approved if mode == "full" else False}
    save_json(root / "protocol.json", protocol)
    protocol_hash = file_hash(root / "protocol.json")
    contract = dict(contract, tuning_protocol_sha256=protocol_hash, code_hashes=code_hashes)
    records = []
    for experiment in EXPERIMENTS:
        for config in candidates:
            candidate_id = f"lr_{config.initial_learning_rate:.0e}_dropout_{config.dropout_rates[0]:.1f}"
            records.append({"architecture": architecture, "experiment": experiment, "mode": mode,
                            "candidate_id": candidate_id, "directory": f"{experiment}/{candidate_id}",
                            "config": asdict(config), "seed": config.seed, "status": "pending",
                            "initialization": "independent", "data_fingerprint": contract["fingerprint"],
                            "protocol_sha256": protocol_hash, "code_hashes": code_hashes})
    save_json(root / "candidates.json", records)
    for record in records:
        output_dir = root / record["directory"]
        config = next(value for value in candidates if asdict(value) == record["config"])
        started = time.perf_counter()
        record["status"] = "running"
        save_json(root / "candidates.json", records)
        try:
            run, metrics, checkpoint = train_experiment(
                architecture, record["experiment"], data, contract, output_dir, mode=mode,
                full_training_approved=full_training_approved, config=config, smoke=SMOKE,
                initialization="independent",
            )
            overall = metrics.loc[metrics.position == "Overall"]
            if len(overall) != 1 or not np.isfinite(overall[["MAE", "MSE", "RMSE"]].to_numpy()).all():
                raise ValueError("Exactly one finite overall validation row is required")
            if file_hash(checkpoint) != run["checkpoint_sha256"]:
                raise ValueError("Candidate checkpoint changed after training")
            record.update(status="succeeded", RMSE=float(overall.iloc[0].RMSE),
                          MAE=float(overall.iloc[0].MAE), MSE=float(overall.iloc[0].MSE),
                          checkpoint=f"{record['directory']}/best.pt", checkpoint_sha256=run["checkpoint_sha256"],
                          parameters=run["parameters"], training_seconds=run["training_seconds"],
                          cumulative_training_seconds=run["training_seconds"],
                          epochs_completed=run["epochs_completed"], best_epoch=run["best_epoch"],
                          optimizer_updates=run["optimizer_updates"], amp_skipped_steps=run["amp_skipped_steps"])
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            raise
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc())
        finally:
            record["elapsed_seconds"] = time.perf_counter() - started
            # Status tetap tersimpan walau konstruksi model gagal sebelum direktori kandidat dibuat.
            output_dir.mkdir(parents=True, exist_ok=True)
            save_json(output_dir / "status.json", record)
            save_json(root / "candidates.json", records)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    selection = summarize_grid_search(root) if mode == "full" else None
    succeeded = sum(record["status"] == "succeeded" for record in records)
    summary = {"architecture": architecture, "mode": mode, "attempted": len(records),
               "succeeded": succeeded, "failed": len(records) - succeeded,
               "status": "completed" if succeeded == len(records) else "completed_with_failures",
               "protocol_sha256": protocol_hash, "data_fingerprint": contract["fingerprint"],
               "selection_available": selection is not None, "test_accessed": False,
               "output_directory": str(root)}
    if succeeded == 0:
        summary["status"] = "failed"
    save_json(root / "completion.json", summary)
    return summary
