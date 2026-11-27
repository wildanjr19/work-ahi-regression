"""Grafik akademik dan pembahasan dari metrik yang sudah dihitung."""

from pathlib import Path

import numpy as np

from .config import POSITIONS, PREPROCESSING
from .model_utils import ARCHITECTURES, EXPERIMENTS


def plot_comparison(metrics, costs, histories, predictions, output_dir, *, initialization="historical"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    identities = list(costs[["Model", "Experiment"]].itertuples(index=False, name=None))
    labels = [f"{model}\n{experiment.replace('_', ' ')}" for model, experiment in identities]
    colors = ["#2166ac" if model in ARCHITECTURES else "#b35806" for model, _ in identities]

    def save(figure, name):
        figure.tight_layout()
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=PREPROCESSING.figure_dpi, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        paths.append(path.name)

    for subset in ("validation", "test"):
        table = metrics[metrics.Subset == subset].set_index(["Model", "Experiment", "position"])
        for metric in ("MAE", "RMSE", "R2"):
            figure, axis = plt.subplots(figsize=(12, 5))
            values = [table.loc[(model, experiment, "Overall"), metric] for model, experiment in identities]
            axis.bar(range(len(labels)), values, color=colors)
            axis.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=8)
            axis.axhline(0, color="black", linewidth=0.7)
            axis.set(ylabel=metric, title=f"{subset}: {metric}; test tidak menentukan pemilihan")
            save(figure, f"{subset}_{metric.lower()}_comparison")

            figure, axes = plt.subplots(1, len(POSITIONS), figsize=(18, 6))
            for axis, position in zip(axes, POSITIONS):
                selected = [table.loc[(model, experiment, position), metric] for model, experiment in identities]
                count = int(table.loc[(*identities[0], position), "samples"])
                axis.bar(range(len(labels)), selected, color=colors)
                axis.set_xticks(range(len(labels)), labels, rotation=80, ha="right", fontsize=7)
                axis.axhline(0, color="black", linewidth=0.7)
                axis.set(title=f"{position} (n={count})", ylabel=metric)
            figure.suptitle(f"{subset}: performa per posisi", fontsize=12)
            save(figure, f"{subset}_position_{metric.lower()}")

        columns = ["MAE", "MSE", "RMSE", "R2"]
        values = np.array([[table.loc[(model, experiment, "Overall"), metric] for metric in columns] for model, experiment in identities])
        low, span = values.min(axis=0), np.ptp(values, axis=0)
        normalized = np.divide(values - low, span, out=np.zeros_like(values), where=span != 0)
        normalized[:, :3] = 1 - normalized[:, :3]
        figure, axis = plt.subplots(figsize=(8, 7))
        axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for row in range(len(values)):
            for column in range(len(columns)):
                axis.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center",
                          color="white" if normalized[row, column] > 0.65 else "black", fontsize=9)
        axis.set_xticks(range(4), columns)
        axis.set_yticks(range(len(labels)), [label.replace("\n", ": ") for label in labels], fontsize=8)
        axis.set_title(f"{subset}: angka asli; warna relatif per kolom (lebih gelap lebih baik)", fontsize=10)
        save(figure, f"{subset}_metrics_heatmap")

        frames = predictions[predictions.Subset == subset]
        low = min(frames["Actual AHI"].min(), frames["Predicted AHI"].min())
        high = max(frames["Actual AHI"].max(), frames["Predicted AHI"].max())
        figure, axes = plt.subplots(4, 3, figsize=(12, 14))
        for axis, (model, experiment) in zip(axes.flat, identities):
            frame = frames[(frames.Model == model) & (frames.Experiment == experiment)]
            axis.scatter(frame["Actual AHI"], frame["Predicted AHI"], s=9, alpha=0.65, color="#2166ac")
            axis.plot([low, high], [low, high], ":", color="black")
            axis.set(xlim=(low, high), ylim=(low, high), xlabel="AHI aktual", ylabel="AHI prediksi",
                     title=f"{model}: {experiment.replace('_', ' ')}")
            axis.title.set_fontsize(9)
        for axis in list(axes.flat)[len(identities):]:
            axis.set_visible(False)
        figure.suptitle(f"{subset}: seluruh sampel, batas sumbu sama; tanpa clipping", fontsize=12)
        save(figure, f"{subset}_actual_vs_predicted_comparison")

    cnn_costs = costs[costs.kind == "CNN"]
    cnn_labels = [f"{row.Model}\n{row.Experiment.replace('_', ' ')}" for row in cnn_costs.itertuples()]
    figure, axis = plt.subplots(figsize=(11, 5))
    x = np.arange(len(cnn_costs))
    axis.bar(x - 0.18, cnn_costs.training_seconds, width=0.36, label="Tahap ini", color="#2166ac")
    cumulative_label = "Mandiri; tanpa biaya FE terpisah" if initialization == "independent" else "Termasuk parent feature extraction"
    axis.bar(x + 0.18, cnn_costs.cumulative_training_seconds, width=0.36, label=cumulative_label, color="#b35806")
    axis.set_xticks(x, cnn_labels, rotation=55, ha="right", fontsize=8)
    axis.set(ylabel="Detik training GPU", title="Biaya training CNN; baseline CPU dicatat terpisah")
    axis.legend(fontsize=8)
    save(figure, "cnn_training_time")

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(x, cnn_costs.trainable_parameters, color="#2166ac")
    axis.set_xticks(x, cnn_labels, rotation=55, ha="right", fontsize=8)
    axis.set(yscale="log", ylabel="Parameter trainable (skala log)", title="Kapasitas yang dilatih berbeda antararsitektur")
    save(figure, "cnn_trainable_parameters")

    for metric, title in (("loss", "MSE loss"), ("mae", "MAE")):
        figure, axes = plt.subplots(3, 3, figsize=(14, 10))
        for row, model in enumerate(ARCHITECTURES):
            for column, experiment in enumerate(EXPERIMENTS):
                axis = axes[row, column]
                history = histories[f"{model}/{experiment}"]
                epochs = np.arange(1, len(history) + 1)
                axis.plot(epochs, history[metric], label="Training", color="#2166ac")
                axis.plot(epochs, history[f"val_{metric}"], label="Validation", color="#b35806")
                axis.set(title=f"{model}: {experiment}", xlabel="Epoch", ylabel=title)
                axis.title.set_fontsize(9)
                axis.legend(fontsize=8)
        figure.suptitle(f"Kurva tersimpan: {title}; CNN tidak dilatih ulang", fontsize=12)
        save(figure, f"cnn_{metric}_curves")
    return paths


def research_discussion(metrics, costs, choices, contributions, contract, *, initialization="historical"):
    lines = ["# Pembahasan hasil aktual", "", "## Pemilihan dan evaluasi terpisah", ""]
    for group, label in (("cnn", "CNN utama"), ("baseline", "Baseline terpilih")):
        selected = choices[group]
        rows = metrics[(metrics.Model == selected["Model"]) & (metrics.Experiment == selected["Experiment"]) & (metrics.position == "Overall")].set_index("Subset")
        validation, test = rows.loc["validation"], rows.loc["test"]
        lines.append(f"{label}: {selected['Model']}/{selected['Experiment']}, dipilih sebelum test. Validation MAE {validation.MAE:.4f}, RMSE {validation.RMSE:.4f}; test MAE {test.MAE:.4f}, RMSE {test.RMSE:.4f}, R2 {test.R2:.4f}.")
        lines.append("")
    lines += ["## Kontribusi transfer learning", "", "Penurunan error bernilai negatif berarti memburuk. Semua pasangan kondisi tersedia pada tabel kontribusi; test hanya melaporkan kondisi beku.", ""]
    overall = contributions[contributions.position == "Overall"]
    for model in ARCHITECTURES:
        rows = overall[(overall.Model == model) & (overall.before == "scratch") & (overall.after == "fine_tuning")].set_index("Subset")
        lines.append(f"- {model}, scratch ke fine-tuning: penurunan RMSE validation {rows.loc['validation', 'RMSE_reduction_percent']:+.2f}%; test {rows.loc['test', 'RMSE_reduction_percent']:+.2f}%.")
    timing = ("FT langsung dimulai dari ImageNet dan head baru; biaya FT tidak memasukkan FE terpisah."
              if initialization == "independent" else "Epoch fine-tuning dimulai setelah feature extraction; waktu kumulatif memasukkan biaya parent.")
    lines += ["", "## Konvergensi, biaya, dan overfitting", "", f"Tabel biaya mencatat epoch terbaik/selesai dan parameter trainable. {timing} Baseline memakai CPU dan biaya pencarian alpha; CNN memakai GPU. Batas epoch sama tidak berarti komputasi identik.", ""]
    for row in costs[(costs.kind == "CNN") & (costs.Experiment == "fine_tuning")].itertuples():
        lines.append(f"- {row.Model}: checkpoint epoch {int(row.best_epoch)}/{int(row.epochs)}; validation loss terbaik/akhir {row.best_val_loss:.2f}/{row.last_val_loss:.2f}.")
    lines += ["", "Kurva memperlihatkan risiko overfitting: training loss rendah tidak menjamin validation membaik. Evaluasi memakai checkpoint terbaik. Perbedaan parameter dan blok trainable membatasi klaim efisiensi dan keunggulan arsitektur.", "", "## Posisi tidur", ""]
    selected = choices["cnn"]
    positions = metrics[(metrics.Model == selected["Model"]) & (metrics.Experiment == selected["Experiment"]) & (metrics.position != "Overall")]
    for subset in ("validation", "test"):
        rows = positions[positions.Subset == subset]
        worst, best = rows.loc[rows.RMSE.idxmax()], rows.loc[rows.RMSE.idxmin()]
        lines.append(f"CNN terpilih pada {subset}: RMSE terbesar {worst.position} {worst.RMSE:.4f} (n={int(worst.samples)}); terkecil {best.position} {best.RMSE:.4f} (n={int(best.samples)}). Metrik per posisi seluruh model tersedia, termasuk R2 negatif.")
        lines.append("")
    lines += ["## Batas interpretasi", "", "Satu split dan seed belum mengukur variasi antar-run; tidak ada CI atau uji signifikansi. Test telah terpapar dan tidak boleh dipakai tuning sambil tetap disebut independen. Jika menghitung CI kelak, bootstrap harus menjaga dependensi pasien dan komponen salinan.", "", "Sebanyak 411 ID memiliki pengukuran beberapa posisi. Identitas biologis, provenance, oversampling, serta salinan bertarget berbeda belum terkonfirmasi. Target ekstrem termasuk AHI 600 dipertahankan; RMSE lebih peka terhadap error besar daripada MAE. Tidak ada clipping atau penghapusan error buruk. Bland-Altman hanya deskriptif.", "", "Transfer ImageNet ke pseudo-image, grid 50 x 11 x 3, padding, kuantisasi, dan resize dapat membatasi informasi. Nomor IMF sama tidak membuktikan sinkronisasi antar sensor. Baseline tabular mempertahankan fitur sebelum kuantisasi; perbedaannya menguji seluruh pipeline citra, bukan hanya arsitektur.", "", f"Status: {contract['metadata']['status']}. Koreksi sumber memerlukan preprocessing dan eksperimen baru. Hasil internal ini belum menunjukkan generalisasi eksternal atau validasi klinis.", ""]
    return "\n".join(lines)
