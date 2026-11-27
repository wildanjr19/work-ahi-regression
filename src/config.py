"""Konfigurasi eksperimen; path dihitung relatif terhadap lokasi proyek."""

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
AUDIT_DIR = ARTIFACTS_DIR / "audit"
PROCESSED_DIR = ARTIFACTS_DIR / "processed_data"
TUNING_DIR = ARTIFACTS_DIR / "tuning"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

# Urutan ini juga menentukan channel citra dan kolom one-hot.
SENSOR_FILES = {
    "Flow": "Flow Patient.xlsx",
    "PulseRate": "PulseRate.xlsx",
    "Pleth": "Pleth.xlsx",
}
POSITIONS = ("Supine", "Left", "Right")
FEATURE_COLUMNS = (
    "Energi IMF",
    "Energi Relatif (%)",
    "Frekuensi Dominan (Hz)",
    "Rentang Frek (Hz)",
    "Mean Period (s)",
    "Mean Amplitudo",
    "STD Amplitudo",
    "Instant Freq (Hz)",
    "STD Instant Freq (Hz)",
    "Entropi",
    "Korelasi dengan sinyal asli",
)
EXPECTED_COLUMNS = ("id", "Posisi", "IMF", *FEATURE_COLUMNS, "AHI")
SAMPLE_KEY = ("id", "Posisi")
ROW_KEY = (*SAMPLE_KEY, "IMF")


@dataclass(frozen=True)
class PreprocessingConfig:
    seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    max_imfs: int = 50
    image_size: tuple[int, int] = (224, 224)
    resize_method: str = "nearest"
    scaling_percentiles: tuple[float, float] = (1.0, 99.0)
    # Reservasi nol untuk padding; fitur asli akan dipetakan ke 1..255.
    padding_pixel: int = 0
    observed_pixel_range: tuple[int, int] = (1, 255)
    ahi_audit_threshold: float = 100.0
    figure_dpi: int = 300
    duplicate_grouping: str = "connected_components"
    provisional: bool = True

    def __post_init__(self):
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1) > 1e-9:
            raise ValueError("Split fractions must be positive and sum to one.")
        if self.max_imfs < 1 or min(self.image_size) < 32:
            raise ValueError("IMF count must be positive and CNN dimensions at least 32.")
        low, high = self.scaling_percentiles
        if not 0 <= low < high <= 100:
            raise ValueError("Scaling percentiles must satisfy 0 <= low < high <= 100.")
        pixel_low, pixel_high = self.observed_pixel_range
        if not (0 <= self.padding_pixel <= 255 and 0 <= pixel_low < pixel_high <= 255):
            raise ValueError("Pixel values must be within 0..255.")
        if pixel_low <= self.padding_pixel <= pixel_high:
            raise ValueError("Padding must be distinct from observed feature pixels.")
        if self.resize_method != "nearest" or self.duplicate_grouping != "connected_components":
            raise ValueError("This pipeline implements nearest resizing and connected-component grouping only.")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 8
    optimizer: str = "adam"
    weight_decay: float = 0.0
    loss: str = "mse"
    initial_learning_rate: float = 1e-3
    fine_tuning_learning_rate: float = 1e-5
    scratch_epochs: int = 50
    feature_extraction_epochs: int = 50
    fine_tuning_epochs: int = 50
    early_stopping_patience: int = 10
    restore_best_weights: bool = True
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 5
    min_learning_rate: float = 1e-7
    checkpoint_monitor: str = "val_loss"
    checkpoint_save_best_only: bool = True
    terminate_on_nan: bool = True
    head_units: tuple[int, int] = (128, 64)
    dropout_rates: tuple[float, float] = (0.3, 0.3)
    output_dtype: str = "float32"
    output_activation: str = "linear"
    pretrained_weights: str = "IMAGENET1K_V1"
    fine_tuning_prefixes: tuple[tuple[str, str], ...] = (
        ("ResNet50", "layer4"), ("VGG16", "24"), ("VGG19", "28")
    )
    freeze_batch_normalization: bool = True
    mixed_precision: bool = True
    prefetch_batches: int = 1
    intra_op_threads: int = 4
    inter_op_threads: int = 2
    reload_rtol: float = 1e-5
    reload_atol: float = 1e-4
    require_gpu_for_full_training: bool = True
    num_workers: int = 0
    amp_initial_scale: float = 128.0


@dataclass(frozen=True)
class SmokeConfig:
    batch_size: int = 8
    train_batches: int = 1
    validation_batches: int = 1
    epochs: int = 1


PREPROCESSING = PreprocessingConfig()
TRAINING = TrainingConfig()
SMOKE = SmokeConfig()

SEARCH_SPACE = {
    "learning_rate": (1e-6, 1e-5, 1e-4, 1e-3),
    "dropout": (0.1, 0.3, 0.5),
}


def tuning_candidates():
    """Konfigurasi mandiri dengan grid sama untuk seluruh arsitektur dan strategi."""
    # Kedua field LR disamakan; default TRAINING tetap kompatibel dengan protokol lama.
    return tuple(
        replace(TRAINING, initial_learning_rate=learning_rate,
                fine_tuning_learning_rate=learning_rate, dropout_rates=(dropout, dropout))
        for learning_rate, dropout in product(SEARCH_SPACE["learning_rate"], SEARCH_SPACE["dropout"])
    )
