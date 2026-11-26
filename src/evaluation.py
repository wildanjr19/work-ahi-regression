"""Evaluasi regresi; residual didefinisikan sebagai AHI aktual dikurangi prediksi."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import POSITIONS, PREPROCESSING
from .preprocessing import save_json


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape or not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("Targets and predictions must align and be finite")
    if not len(actual):
        return {"samples": 0, "MAE": None, "MSE": None, "RMSE": None, "R2": None, "negative_predictions": 0}
    residual = actual - predicted
    mse = float(np.mean(residual ** 2))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r2 = None if len(actual) < 2 or denominator == 0 else 1 - float(np.sum(residual ** 2)) / denominator
    return {"samples": len(actual), "MAE": float(np.mean(np.abs(residual))), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2,
            "negative_predictions": int((predicted < 0).sum())}


def prediction_table(payload, predicted, model, experiment, subset, run_type):
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    actual = np.asarray(payload["targets"], dtype=np.float64)
    regression_metrics(actual, predicted)
    return pd.DataFrame({
        "Patient ID": payload["patient_ids"], "Sleep Position": payload["positions"],
        "Actual AHI": actual, "Predicted AHI": predicted, "Residual": actual - predicted,
        "Absolute Error": np.abs(actual - predicted), "Model": model, "Experiment": experiment,
        "Subset": subset, "Run Type": run_type, "Sample ID": payload["sample_ids"],
    })


def evaluate_predictions(frame):
    rows = []
    for position in ("Overall", *POSITIONS):
        selected = frame if position == "Overall" else frame[frame["Sleep Position"] == position]
        rows.append({"position": position, **regression_metrics(selected["Actual AHI"], selected["Predicted AHI"])})
    return pd.DataFrame(rows)


def save_evaluation(frame, output_dir, subset="validation"):
    if subset not in ("validation", "test") or not frame["Subset"].eq(subset).all():
        raise ValueError("Evaluation subset must match prediction rows")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"predictions_{subset}.csv", index=False)
    loaded = pd.read_csv(output_dir / f"predictions_{subset}.csv", dtype={"Patient ID": "string"})
    metrics = evaluate_predictions(frame)
    recalculated = evaluate_predictions(loaded)
    pd.testing.assert_frame_equal(metrics, recalculated, check_exact=False, rtol=1e-10, atol=1e-10)
    metrics.to_csv(output_dir / f"metrics_{subset}.csv", index=False)
    save_json(output_dir / f"metrics_{subset}.json", metrics.where(pd.notna(metrics), None).to_dict("records"))
    return metrics


def plot_diagnostics(history, frame, output_dir, title, dpi=PREPROCESSING.figure_dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    actual = frame["Actual AHI"].to_numpy()
    predicted = frame["Predicted AHI"].to_numpy()
    residual = actual - predicted
    paths = []

    def save(figure, name):
        figure.suptitle(title, fontsize=11)
        figure.tight_layout()
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        paths.append(path.name)

    for metric, label in (("loss", "Loss"), ("mae", "MAE")) if history is not None else ():
        figure, axis = plt.subplots(figsize=(6, 4))
        epochs = np.arange(1, len(history[metric]) + 1)
        axis.plot(epochs, history[metric], "o-", label="Training", color="#2166ac")
        axis.plot(epochs, history[f"val_{metric}"], "s-", label="Validation", color="#b35806")
        axis.set(xlabel="Epoch", ylabel=label)
        axis.legend()
        save(figure, f"{metric}_curves")

    figure, axis = plt.subplots(figsize=(5, 4))
    axis.scatter(actual, predicted, color="#2166ac")
    low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    axis.plot([low, high], [low, high], color="black", linestyle=":", label="Prediksi = aktual")
    axis.set(xlabel="AHI aktual", ylabel="AHI prediksi")
    axis.legend()
    save(figure, "actual_vs_predicted")

    figure, axis = plt.subplots(figsize=(5, 4))
    axis.scatter(predicted, residual, color="#2166ac")
    axis.axhline(0, color="black", linestyle=":")
    axis.set(xlabel="AHI prediksi", ylabel="Residual (aktual - prediksi)")
    save(figure, "residuals")

    figure, axis = plt.subplots(figsize=(5, 4))
    axis.hist(residual, bins=min(20, max(1, len(residual))), color="#2166ac", edgecolor="white")
    axis.set(xlabel="Residual (aktual - prediksi)", ylabel="Jumlah sampel")
    save(figure, "error_distribution")

    figure, axis = plt.subplots(figsize=(6, 4))
    x = np.array([POSITIONS.index(value) for value in frame["Sleep Position"]])
    axis.scatter(x - 0.08, actual, color="#2166ac", label="Aktual")
    axis.scatter(x + 0.08, predicted, color="#b35806", label="Prediksi")
    axis.set_xticks(range(len(POSITIONS)), [f"{value} (n={int((x == index).sum())})" for index, value in enumerate(POSITIONS)])
    axis.set(ylabel="AHI", xlabel="Posisi tidur")
    axis.legend()
    save(figure, "predictions_by_position")

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter((actual + predicted) / 2, residual, color="#2166ac")
    axis.axhline(residual.mean(), color="black", label="Bias deskriptif")
    if len(residual) > 1:
        spread = 1.96 * residual.std(ddof=1)
        for value in (residual.mean() - spread, residual.mean() + spread):
            axis.axhline(value, color="#b35806", linestyle=":")
    axis.set(xlabel="Rata-rata aktual dan prediksi", ylabel="Aktual - prediksi")
    axis.text(0.01, 0.02, "Deskriptif; pengukuran berulang pasien, tanpa koreksi/CI", transform=axis.transAxes, fontsize=8)
    axis.legend()
    save(figure, "bland_altman")
    return paths
