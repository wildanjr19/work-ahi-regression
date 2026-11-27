"""Baseline, pemilihan validation, dan satu evaluasi test setelah protokol dibekukan."""

import gc
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .audit import file_hash
from .config import ARTIFACTS_DIR, POSITIONS, PROCESSED_DIR, PROJECT_ROOT, TRAINING, TUNING_DIR
from .evaluation import evaluate_predictions, plot_diagnostics, prediction_table, regression_metrics, save_evaluation
from .model_utils import ARCHITECTURES, EXPERIMENTS, configure_runtime, load_checkpoint, load_development_data, make_dataset, predict_model, torch
from .preprocessing import save_json, validate_payload

RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def baseline_features(payload):
    count = len(payload["targets"])
    matrix = np.concatenate((payload["scaled_features"].reshape(count, -1),
                             payload["imf_mask"].reshape(count, -1),
                             payload["position_one_hot"]), axis=1).astype(np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Baseline features must be finite")
    return matrix


def select_validation(table):
    if table.empty or not table["Subset"].eq("validation").all() or not table.position.eq("Overall").all():
        raise ValueError("Selection requires overall validation results only")
    if not np.isfinite(table[["RMSE", "MAE"]].to_numpy()).all():
        raise ValueError("Selection metrics must be finite")
    return table.sort_values(["RMSE", "MAE", "Model", "Experiment"], kind="stable").iloc[0].to_dict()


def transfer_contributions(metrics):
    rows = []
    cnn = metrics[metrics.Model.isin(ARCHITECTURES)]
    for (subset, model, position), group in cnn.groupby(["Subset", "Model", "position"], sort=False):
        indexed = group.set_index("Experiment")
        for before, after in (("scratch", "feature_extraction"), ("scratch", "fine_tuning"), ("feature_extraction", "fine_tuning")):
            row = {"Subset": subset, "Model": model, "position": position, "before": before, "after": after}
            for metric in ("MAE", "RMSE"):
                base, changed = float(indexed.loc[before, metric]), float(indexed.loc[after, metric])
                row[f"{metric}_reduction"] = base - changed
                row[f"{metric}_reduction_percent"] = 100 * (base - changed) / base if base > 0 else None
            rows.append(row)
    return pd.DataFrame(rows)


def load_payload(split, contract, processed_dir=PROCESSED_DIR):
    path = Path(processed_dir) / f"{split}.npz"
    if file_hash(path) != contract["metadata"]["files"][path.name]["sha256"]:
        raise ValueError(f"Processed checksum differs: {split}")
    manifest = pd.read_csv(Path(processed_dir) / "split_manifest.csv", dtype={"id": "string"}, float_precision="round_trip")
    with np.load(path, allow_pickle=False) as stored:
        payload = {name: stored[name] for name in stored.files}
    validate_payload(payload, manifest[manifest.split == split])
    return payload


def validate_prediction_rows(frame, payload, model, experiment, subset):
    for column, values in (("Patient ID", payload["patient_ids"]), ("Sleep Position", payload["positions"]), ("Sample ID", payload["sample_ids"])):
        if not np.array_equal(frame[column].astype(str), np.asarray(values).astype(str)):
            raise ValueError(f"Prediction alignment differs: {column}")
    np.testing.assert_allclose(frame["Actual AHI"], payload["targets"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(frame["Residual"], frame["Actual AHI"] - frame["Predicted AHI"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(frame["Absolute Error"], np.abs(frame["Residual"]), rtol=1e-12, atol=1e-12)
    if not all(frame[column].eq(value).all() for column, value in (
        ("Model", model), ("Experiment", experiment), ("Subset", subset), ("Run Type", "full")
    )):
        raise ValueError("Prediction identity differs from its run")


def collect_cnn_results(contract, validation, experiments_root=ARTIFACTS_DIR / "experiments"):
    metrics, predictions, costs, assets, histories = [], [], [], {}, {}
    versions, normalization = set(), set()
    expected_config = json.loads(json.dumps(asdict(TRAINING)))
    for architecture in ARCHITECTURES:
        root = Path(experiments_root) / architecture
        summary_path = root / "summary.json"
        records = json.loads(summary_path.read_text(encoding="utf-8"))
        if len(records) != 3 or {run["experiment"] for run in records} != set(EXPERIMENTS):
            raise ValueError("Exactly three full conditions are required per architecture")
        feature_run = next(run for run in records if run["experiment"] == "feature_extraction")
        assets[str(summary_path.resolve())] = file_hash(summary_path)
        for run in records:
            name = run["experiment"]
            folder = root / name
            if (run["mode"], run["architecture"], run["data_fingerprint"], run["config"]) != (
                "full", architecture, contract["fingerprint"], expected_config
            ) or run["test_accessed"]:
                raise ValueError("CNN run differs from the common development protocol")
            if run != json.loads((folder / "run.json").read_text(encoding="utf-8")):
                raise ValueError("Run summary differs from run.json")
            if run["validation_samples"] != len(validation["targets"]):
                raise ValueError("Validation sample count differs")
            checkpoint = folder / "best.pt"
            if file_hash(checkpoint) != run["checkpoint_sha256"]:
                raise ValueError("CNN checkpoint changed")
            if name == "fine_tuning" and run["parent_feature_checkpoint_sha256"] != feature_run["checkpoint_sha256"]:
                raise ValueError("Fine-tuning parent differs from full feature extraction")
            history = pd.read_csv(folder / "history.csv")
            if len(history) != run["epochs_completed"] or not np.isfinite(history.to_numpy()).all():
                raise ValueError("Training history differs from run metadata")
            best_epoch = int(history.val_loss.argmin()) + 1
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (saved["epoch"], saved["mode"], saved["architecture"], saved["experiment"], saved["data_fingerprint"], json.loads(json.dumps(saved["config"]))) != (
                best_epoch, "full", architecture, name, contract["fingerprint"], expected_config
            ):
                raise ValueError("Checkpoint protocol or best epoch differs")
            del saved
            frame = pd.read_csv(folder / "predictions_validation.csv", dtype={"Patient ID": "string"})
            validate_prediction_rows(frame, validation, architecture, name, "validation")
            recalculated = evaluate_predictions(frame)
            pd.testing.assert_frame_equal(recalculated, pd.read_csv(folder / "metrics_validation.csv"), check_exact=False, rtol=1e-10, atol=1e-10)
            metrics.append(recalculated.assign(Model=architecture, Experiment=name, Subset="validation"))
            predictions.append(frame)
            key = f"{architecture}/{name}"
            histories[key] = history
            costs.append({"Model": architecture, "Experiment": name, "kind": "CNN", "compute_device": run["compute_device"],
                          "epochs": len(history), "best_epoch": best_epoch, "total_parameters": run["parameters"]["total"],
                          "trainable_parameters": run["parameters"]["trainable"], "training_seconds": run["training_seconds"],
                          "cumulative_training_seconds": run["training_seconds"] + (feature_run["training_seconds"] if name == "fine_tuning" else 0),
                          "peak_allocated_MiB": run["gpu_memory_training"]["peak_allocated_bytes"] / 1024**2,
                          "amp_skipped_steps": run["amp_skipped_steps"], "checkpoint": str(checkpoint.resolve()),
                          "checkpoint_sha256": run["checkpoint_sha256"], "last_val_loss": history.val_loss.iloc[-1],
                          "best_val_loss": history.val_loss.min(), "train_loss_at_best_epoch": history.loss.iloc[best_epoch - 1]})
            for filename in ("best.pt", "run.json", "history.csv", "predictions_validation.csv", "metrics_validation.csv"):
                path = folder / filename
                assets[str(path.resolve())] = file_hash(path)
            versions.add(json.dumps(run["versions"], sort_keys=True))
            normalization.add(run["input_normalization"])
    if len(versions) != 1 or len(normalization) != 1:
        raise ValueError("CNN versions or normalization differ")
    return pd.concat(metrics, ignore_index=True), pd.concat(predictions, ignore_index=True), pd.DataFrame(costs), assets, histories


def fit_baselines(train, validation, output_root, alphas=RIDGE_ALPHAS):
    if not alphas or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("Ridge alphas must be positive and finite")
    x_train, x_validation = baseline_features(train), baseline_features(validation)
    y_train = train["targets"]
    started = time.perf_counter()
    dummy = DummyRegressor(strategy="median").fit(x_train, y_train)
    dummy_seconds = time.perf_counter() - started
    candidates, best = [], None
    grid_started = time.perf_counter()
    with threadpool_limits(limits=TRAINING.intra_op_threads):
        for alpha in alphas:
            candidate = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="cholesky"))
            fit_started = time.perf_counter()
            candidate.fit(x_train, y_train)
            seconds = time.perf_counter() - fit_started
            scores = regression_metrics(validation["targets"], candidate.predict(x_validation))
            candidates.append({"alpha": alpha, **scores, "fit_seconds": seconds})
            order = (scores["RMSE"], scores["MAE"], alpha)
            if best is None or order < best[0]:
                best = (order, candidate, seconds)
    ridge_seconds = time.perf_counter() - grid_started
    pd.DataFrame(candidates).to_csv(Path(output_root) / "ridge_validation_search.csv", index=False)
    metrics, predictions, costs = [], [], []
    for name, estimator, seconds in (("DummyMedian", dummy, dummy_seconds), ("Ridge", best[1], ridge_seconds)):
        folder = Path(output_root) / "baselines" / name
        frame = prediction_table(validation, estimator.predict(x_validation), name, "baseline", "validation", "full")
        evaluated = save_evaluation(frame, folder)
        checkpoint = folder / "best.joblib"
        joblib.dump(estimator, checkpoint)
        reloaded = joblib.load(checkpoint)
        np.testing.assert_allclose(reloaded.predict(x_validation), frame["Predicted AHI"], rtol=1e-12, atol=1e-12)
        info = {"Model": name, "Experiment": "baseline", "kind": "baseline", "compute_device": "CPU",
                "training_seconds": seconds, "cumulative_training_seconds": seconds,
                "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_hash(checkpoint),
                "train_samples": len(y_train), "feature_count": x_train.shape[1],
                "selected_alpha": float(best[0][2]) if name == "Ridge" else None,
                "selected_fit_seconds": best[2] if name == "Ridge" else dummy_seconds,
                "feature_representation": "C-order scaled_features, IMF mask, position one-hot; no IDs or target",
                "scaler_fit_subset": "train" if name == "Ridge" else None, "refit_train_validation": False}
        save_json(folder / "run.json", info)
        metrics.append(evaluated.assign(Model=name, Experiment="baseline", Subset="validation"))
        predictions.append(frame)
        costs.append(info)
    return pd.concat(metrics, ignore_index=True), pd.concat(predictions, ignore_index=True), pd.DataFrame(costs)


def verify_frozen_protocol(protocol_path, expected_hash):
    if file_hash(protocol_path) != expected_hash:
        raise ValueError("Frozen protocol changed")
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    for name, expected in protocol["assets"].items():
        if file_hash(Path(name)) != expected:
            raise ValueError(f"Frozen asset changed: {Path(name).name}")
    return protocol


def load_held_out(protocol_path, expected_hash, output_root, processed_dir=PROCESSED_DIR):
    protocol = verify_frozen_protocol(protocol_path, expected_hash)
    if protocol["test_evaluation_authorized"] is not True:
        raise PermissionError("Test evaluation is not authorized in this frozen protocol")
    marker = Path(output_root) / "test_access_started.json"
    if marker.exists():
        raise RuntimeError("Test exposure already started; use saved results without fitting again")
    save_json(marker, {"started_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": expected_hash})
    return load_payload("test", protocol["contract"], processed_dir)


def run_comparison(output_root=ARTIFACTS_DIR / "comparison", test_evaluation_approved=False, *, tuning_root=None):
    if not test_evaluation_approved:
        raise PermissionError("Test evaluation requires explicit approval")
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError("Comparison output exists; read saved results instead of exposing test again")
    if tuning_root is not None and not output_root.resolve().is_relative_to(TUNING_DIR.resolve()):
        raise ValueError("Tuning evaluation must remain inside artifacts/tuning")
    runtime = configure_runtime()
    development, contract = load_development_data()
    train = load_payload("train", contract)
    validation = load_payload("validation", contract)
    del development
    if tuning_root is None:
        metrics, predictions, costs, assets, histories = collect_cnn_results(contract, validation)
    else:
        from .tuning_evaluation import collect_tuned_cnn_results
        metrics, predictions, costs, assets, histories = collect_tuned_cnn_results(contract, validation, tuning_root)
    output_root.mkdir(parents=True)
    save_json(output_root / "runtime.json", runtime)
    baseline_metrics, baseline_predictions, baseline_costs = fit_baselines(train, validation, output_root)
    metrics = pd.concat((metrics, baseline_metrics), ignore_index=True)
    predictions = pd.concat((predictions, baseline_predictions), ignore_index=True)
    costs = pd.concat((costs, baseline_costs), ignore_index=True)
    overall = metrics[metrics.position == "Overall"].reset_index(drop=True)
    choices = {"cnn": select_validation(overall[overall.Model.isin(ARCHITECTURES)]),
               "baseline": select_validation(overall[~overall.Model.isin(ARCHITECTURES)]),
               "all_models": select_validation(overall)}
    metrics.to_csv(output_root / "metrics_validation.csv", index=False)
    overall.to_csv(output_root / "comparison_validation.csv", index=False)
    predictions.to_csv(output_root / "predictions_validation.csv", index=False)
    costs.to_csv(output_root / "training_costs.csv", index=False)
    for path in PROCESSED_DIR.iterdir():
        if path.is_file():
            assets[str(path.resolve())] = file_hash(path)
    for source in contract["metadata"]["sources"].values():
        path = PROJECT_ROOT / "data" / source["file"]
        if file_hash(path) != source["sha256"]:
            raise ValueError("Original workbook changed since preprocessing")
        assets[str(path.resolve())] = source["sha256"]
    for row in baseline_costs.to_dict("records"):
        path = Path(row["checkpoint"])
        assets[str(path)] = file_hash(path)
        assets[str(path.parent / "run.json")] = file_hash(path.parent / "run.json")
    for filename in ("comparison.py", "comparison_plots.py", "config.py", "model_utils.py", "evaluation.py", "preprocessing.py", "reporting.py", "tuning.py", "tuning_evaluation.py"):
        path = PROJECT_ROOT / "src" / filename
        assets[str(path)] = file_hash(path)
    for filename in ("metrics_validation.csv", "comparison_validation.csv", "predictions_validation.csv", "training_costs.csv", "ridge_validation_search.csv"):
        path = output_root / filename
        assets[str(path.resolve())] = file_hash(path)
    expectations = {f"{row['Model']}/{row['Experiment']}": row["checkpoint_expected"]
                    for row in costs.to_dict("records") if row["kind"] == "CNN" and tuning_root is not None}
    protocol = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "test_evaluation_authorized": True,
                "selection_criterion": "validation RMSE, then MAE, then stable model/experiment labels",
                "selection": choices,
                "cnn_training_config": ({key: value["config"] for key, value in expectations.items()}
                                        if tuning_root is not None else asdict(TRAINING)),
                "ridge_alphas": list(RIDGE_ALPHAS),
                "ridge_solver": "cholesky", "ridge_scaler": "StandardScaler fitted on training only",
                "no_refit_after_selection": True, "no_cnn_retraining": True,
                "test_role": "report all frozen conditions; never select or tune from test results",
                "uncertainty": "single seed, no confidence interval or significance test",
                "research_status": contract["metadata"]["status"], "contract": contract, "assets": assets,
                "initialization": "independent" if tuning_root is not None else "historical",
                "checkpoint_expectations": expectations, "test_previously_exposed": True,
                "test_context": "follow_up_after_previous_exposure"}
    protocol_path = output_root / "frozen_protocol.json"
    save_json(protocol_path, protocol)
    protocol_hash = file_hash(protocol_path)
    save_json(output_root / "selection_validation.json", {"criterion": protocol["selection_criterion"], "selected": choices,
                                                        "protocol_sha256": protocol_hash, "selected_before_test": True})
    print("Validation selection frozen before test:", choices, flush=True)
    test = load_held_out(protocol_path, protocol_hash, output_root)
    x_test = baseline_features(test)
    test_metrics, test_predictions = [], []
    for record in costs.to_dict("records"):
        name, experiment = record["Model"], record["Experiment"]
        print(f"Frozen test inference: {name}/{experiment}", flush=True)
        if record["kind"] == "CNN":
            expected = expectations.get(f"{name}/{experiment}")
            model = load_checkpoint(Path(record["checkpoint"]), "cuda", expected=expected)
            values = predict_model(model, make_dataset(test, TRAINING.batch_size), torch.device("cuda"))
            del model
            gc.collect()
            torch.cuda.empty_cache()
        else:
            values = joblib.load(record["checkpoint"]).predict(x_test)
        frame = prediction_table(test, values, name, experiment, "test", "full")
        folder = output_root / "test" / name / experiment
        evaluated = save_evaluation(frame, folder, subset="test")
        plot_diagnostics(None, frame, folder / "figures", f"{name} / {experiment} / frozen test")
        test_metrics.append(evaluated.assign(Model=name, Experiment=experiment, Subset="test"))
        test_predictions.append(frame)
    test_metrics = pd.concat(test_metrics, ignore_index=True)
    test_predictions = pd.concat(test_predictions, ignore_index=True)
    test_metrics.to_csv(output_root / "metrics_test.csv", index=False)
    test_metrics[test_metrics.position == "Overall"].to_csv(output_root / "comparison_test.csv", index=False)
    test_predictions.to_csv(output_root / "predictions_test.csv", index=False)
    all_metrics = pd.concat((metrics, test_metrics), ignore_index=True)
    contributions = transfer_contributions(all_metrics)
    contributions.to_csv(output_root / "transfer_contributions.csv", index=False)
    from .comparison_plots import plot_comparison, research_discussion
    initialization = protocol["initialization"]
    figures = plot_comparison(all_metrics, costs, histories, pd.concat((predictions, test_predictions)), output_root / "figures",
                              initialization=initialization)
    discussion = research_discussion(all_metrics, costs, choices, contributions, contract, initialization=initialization)
    (output_root / "research_discussion.md").write_text(discussion, encoding="utf-8")
    verify_frozen_protocol(protocol_path, protocol_hash)
    save_json(output_root / "completion.json", {"status": "complete", "protocol_sha256": protocol_hash,
              "frozen_conditions": len(costs), "test_samples": len(test["targets"]),
              "test_patients": len(set(test["patient_ids"])), "test_components": len(set(test["patient_components"])),
              "test_positions": {position: int((test["positions"] == position).sum()) for position in POSITIONS},
              "figures": figures, "test_diagnostic_figures": len(costs) * 5, "cnn_retrained": False,
              "test_previously_exposed": True, "test_context": protocol["test_context"]})
    return all_metrics, costs, choices
