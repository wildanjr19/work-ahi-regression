"""Ringkasan validation dari eksperimen penuh yang sudah tersimpan."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import file_hash
from .config import tuning_candidates
from .model_utils import EXPERIMENTS
from .preprocessing import save_json


def summarize_full_experiments(output_root):
    root = Path(output_root)
    runs = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if {run["experiment"] for run in runs} != set(EXPERIMENTS) or len(runs) != len(EXPERIMENTS):
        raise ValueError("All three experiment conditions are required")
    if any(run["mode"] != "full" for run in runs):
        raise ValueError("Smoke results cannot enter full-experiment comparison")
    if len({(run["architecture"], run["data_fingerprint"]) for run in runs}) != 1:
        raise ValueError("Architecture and preprocessing protocol must match")
    rows, position_rows = [], []
    feature_time = next(run["training_seconds"] for run in runs if run["experiment"] == "feature_extraction")
    for run in runs:
        metrics = pd.read_csv(root / run["experiment"] / "metrics_validation.csv")
        overall = metrics.loc[metrics.position == "Overall"]
        if len(overall) != 1:
            raise ValueError("Exactly one overall validation row is required")
        history = pd.read_csv(root / run["experiment"] / "history.csv")
        row = overall.iloc[0].to_dict()
        rows.append({"Model": run["architecture"], "Experiment": run["experiment"],
                     "Subset": "validation", **row, "epochs": run["epochs_completed"],
                     "best_epoch": int(np.argmin(history[run["config"]["checkpoint_monitor"]])) + 1,
                     "trainable_parameters": run["parameters"]["trainable"],
                     "training_seconds": run["training_seconds"],
                     "cumulative_training_seconds": run["training_seconds"] + (feature_time if run["experiment"] == "fine_tuning" else 0),
                     "peak_allocated_MiB": run["gpu_memory_training"]["peak_allocated_bytes"] / 1024 ** 2,
                     "optimizer_updates": run["optimizer_updates"], "amp_skipped_steps": run["amp_skipped_steps"]})
        position_rows.append(metrics.assign(Model=run["architecture"], Experiment=run["experiment"], Subset="validation"))
    comparison = pd.DataFrame(rows)
    if not np.isfinite(comparison[["MAE", "MSE", "RMSE"]].to_numpy()).all():
        raise ValueError("Validation error metrics must be finite")
    comparison.to_csv(root / "comparison_validation.csv", index=False)
    positions = pd.concat(position_rows, ignore_index=True)
    positions.to_csv(root / "position_metrics_validation.csv", index=False)
    selected = comparison.sort_values(["RMSE", "MAE", "Experiment"], kind="stable").iloc[0]
    selected_run = next(run for run in runs if run["experiment"] == selected.Experiment)
    selection = {"model": selected.Model, "experiment": selected.Experiment,
                 "criterion": "validation RMSE, then MAE for ties", "subset": "validation",
                 "validation_samples": int(selected.samples), "RMSE": float(selected.RMSE), "MAE": float(selected.MAE),
                 "checkpoint": f"{selected.Experiment}/best.pt", "checkpoint_sha256": selected_run["checkpoint_sha256"],
                 "data_fingerprint": selected_run["data_fingerprint"],
                 "research_status": selected_run["research_status"], "test_evaluated": False,
                 "scope": "selection within one architecture; cross-architecture comparison remains pending"}
    save_json(root / "selection_validation.json", selection)
    return comparison, positions, selection


def summarize_grid_search(output_root):
    """Pilih kandidat penuh per strategi; FT langsung tidak menambahkan biaya FE."""
    root = Path(output_root)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    records = json.loads((root / "candidates.json").read_text(encoding="utf-8"))
    if protocol["mode"] != "full" or any(record["mode"] != "full" for record in records):
        raise ValueError("Smoke candidates cannot enter model selection")
    digest = file_hash(root / "protocol.json")
    expected_configs = {json.dumps(config, sort_keys=True) for config in protocol["candidates"]}
    if expected_configs != {json.dumps(asdict(config), sort_keys=True) for config in tuning_candidates()}:
        raise ValueError("Full search must use the entire agreed grid")
    if len(records) != len(EXPERIMENTS) * len(expected_configs):
        raise ValueError("Every strategy must attempt the complete grid")
    for experiment in EXPERIMENTS:
        selected = [record for record in records if record["experiment"] == experiment]
        if (len(selected) != len(expected_configs)
                or {json.dumps(record["config"], sort_keys=True) for record in selected} != expected_configs):
            raise ValueError("Missing or duplicate candidate configurations")
    for record in records:
        if (record["status"] not in ("succeeded", "failed") or record["initialization"] != "independent"
                or record["architecture"] != protocol["architecture"]
                or record["code_hashes"] != protocol["code_hashes"]
                or record["data_fingerprint"] != protocol["data_fingerprint"]
                or record["protocol_sha256"] != digest):
            raise ValueError("Candidate protocol differs or search is unfinished")
    pd.DataFrame(records).drop(columns=["config", "code_hashes", "traceback"], errors="ignore").to_csv(
        root / "comparison_validation.csv", index=False)
    successful = [record for record in records if record["status"] == "succeeded"]
    for record in successful:
        if not np.isfinite([record["RMSE"], record["MAE"], record["MSE"]]).all():
            raise ValueError("Validation selection requires finite metrics")
        checkpoint = (root / record["checkpoint"]).resolve()
        if not checkpoint.is_relative_to(root.resolve()) or file_hash(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError("Selected candidate checkpoint changed")
    if {record["experiment"] for record in successful} != set(EXPERIMENTS):
        return None
    chosen = {}
    for experiment in EXPERIMENTS:
        candidates = [record for record in successful if record["experiment"] == experiment]
        chosen[experiment] = min(candidates, key=lambda record: (record["RMSE"], record["MAE"], record["candidate_id"]))
    selection = {"architecture": protocol["architecture"], "mode": "full", "subset": "validation",
                 "criterion": protocol["criterion"], "protocol_sha256": digest,
                 "data_fingerprint": protocol["data_fingerprint"], "selected_conditions": chosen,
                 "architecture_winner": min(chosen.values(), key=lambda record: (record["RMSE"], record["MAE"], record["experiment"])),
                 "attempted": len(records), "failed": len(records) - len(successful),
                 "all_candidates_succeeded": len(successful) == len(records),
                 "test_evaluated": False, "test_previously_exposed": True,
                 "scope": "within one architecture; cross-architecture selection remains pending"}
    save_json(root / "selection_validation.json", selection)
    return selection
