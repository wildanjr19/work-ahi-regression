"""Ringkasan validation dari eksperimen penuh yang sudah tersimpan."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

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
