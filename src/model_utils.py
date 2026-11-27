"""Regresi dua input dengan PyTorch; smoke test terpisah dari eksperimen penuh."""

import gc
import hashlib
import importlib.metadata
import json
import os
import random
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from .audit import command_result, file_hash
from .config import ARTIFACTS_DIR, FEATURE_COLUMNS, POSITIONS, PROCESSED_DIR, SMOKE, TRAINING, TUNING_DIR, TrainingConfig, tuning_candidates
from .evaluation import plot_diagnostics, prediction_table, save_evaluation
from .preprocessing import save_json, validate_manifest

EXPERIMENTS = ("scratch", "feature_extraction", "fine_tuning")
ARCHITECTURES = {
    "ResNet50": (models.resnet50, models.ResNet50_Weights),
    "VGG16": (models.vgg16, models.VGG16_Weights),
    "VGG19": (models.vgg19, models.VGG19_Weights),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(config=TRAINING):
    set_seed(config.seed)
    torch.set_num_threads(config.intra_op_threads)
    if torch.get_num_interop_threads() != config.inter_op_threads:
        torch.set_num_interop_threads(config.inter_op_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.hub.set_dir(str(ARTIFACTS_DIR / "torch_cache"))
    available = torch.cuda.is_available()
    if config.mixed_precision and not available:
        raise ValueError("Disable mixed_precision explicitly for a CPU-only smoke test")
    return {"framework": "PyTorch", "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "compute_device": "CUDA" if available else "CPU",
            "gpu_name": torch.cuda.get_device_name(0) if available else None,
            "gpu_capability": list(torch.cuda.get_device_capability(0)) if available else None,
            "config": asdict(config), "gpu_memory": gpu_memory(),
            "nvidia_smi": command_result(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader,nounits"])}


def gpu_memory():
    if not torch.cuda.is_available():
        return {"status": "not_measured_no_cuda"}
    free, total = torch.cuda.mem_get_info()
    return {"status": "measured", "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "free_device_bytes": free, "total_device_bytes": total}


def load_development_data(processed_dir=PROCESSED_DIR):
    processed_dir = Path(processed_dir)
    metadata = json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(processed_dir / "split_manifest.csv", dtype={"id": "string"})
    validate_manifest(manifest)
    if metadata["feature_order"] != list(FEATURE_COLUMNS) or metadata["positions"] != list(POSITIONS):
        raise ValueError("Processed feature/position order differs from project contract")
    if not metadata["validation"]["save_reload_equal"] or metadata["validation"]["encoded_input_overlap"]:
        raise ValueError("Processed data validation has not passed")
    data = {}
    # Test NPZ tidak dibaca selama pengembangan dan smoke test.
    for split in ("train", "validation"):
        path = processed_dir / f"{split}.npz"
        if file_hash(path) != metadata["files"][path.name]["sha256"]:
            raise ValueError(f"Processed checksum differs: {split}")
        with np.load(path, allow_pickle=False) as stored:
            payload = {key: stored[key] for key in ("images", "position_one_hot", "targets", "patient_ids", "positions", "sample_ids")}
        selected = manifest[manifest.split == split]
        for name, column in (("patient_ids", "id"), ("sample_ids", "sample_id"), ("positions", "Posisi"), ("targets", "AHI")):
            matches = np.allclose(payload[name], selected[column], rtol=1e-12, atol=1e-12) if name == "targets" else np.array_equal(payload[name], selected[column])
            if not matches:
                raise ValueError(f"Manifest alignment differs: {split}/{name}")
        if payload["images"].dtype != np.uint8 or not np.isfinite(payload["targets"]).all():
            raise ValueError("Expected raw uint8 pixels and finite targets")
        data[split] = payload
    hashes = {name: file_hash(processed_dir / name) for name in (
        "metadata.json", "split_manifest.csv", "scaling_parameters.json", "feature_order.json"
    )}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return data, {"metadata": metadata, "hashes": hashes, "fingerprint": fingerprint}


class SensorDataset(Dataset):
    def __init__(self, payload):
        self.payload = payload

    def __len__(self):
        return len(self.payload["targets"])

    def __getitem__(self, index):
        image = torch.from_numpy(self.payload["images"][index]).permute(2, 0, 1)
        position = torch.as_tensor(self.payload["position_one_hot"][index], dtype=torch.float32)
        target = torch.tensor([self.payload["targets"][index]], dtype=torch.float32)
        return {"sensor_image": image, "sleep_position": position}, target


def make_dataset(payload, batch_size, training=False, config=TRAINING):
    generator = torch.Generator().manual_seed(config.seed)
    options = {"prefetch_factor": config.prefetch_batches} if config.num_workers else {}
    return DataLoader(SensorDataset(payload), batch_size=batch_size, shuffle=training,
                      generator=generator, num_workers=config.num_workers,
                      pin_memory=torch.cuda.is_available(), **options)


class AHIRegressor(nn.Module):
    def __init__(self, architecture, weights=None, phase="scratch", config=TRAINING):
        super().__init__()
        if config.output_dtype != "float32" or config.output_activation != "linear":
            raise ValueError("AHI output must remain linear float32")
        factory, weight_enum = ARCHITECTURES[architecture]
        reference = weight_enum[config.pretrained_weights]
        network = factory(weights=reference if weights is not None else None, progress=False)
        self.architecture, self.config = architecture, config
        if architecture == "ResNet50":
            self.backbone = nn.Sequential(OrderedDict(
                (name, module) for name, module in network.named_children() if name not in ("avgpool", "fc")
            ))
            features = network.fc.in_features
        else:
            self.backbone = network.features
            features = 512
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        transforms = reference.transforms()
        self.register_buffer("image_mean", torch.tensor(transforms.mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(transforms.std).view(1, 3, 1, 1))
        layers, previous = [], features + len(POSITIONS)
        for units, rate in zip(config.head_units, config.dropout_rates):
            layers.extend((nn.Linear(previous, units), nn.ReLU(), nn.Dropout(rate)))
            previous = units
        layers.append(nn.Linear(previous, 1))
        self.head = nn.Sequential(*layers)
        self.configure_phase(phase)

    def normalize_images(self, images):
        # Grid sudah di-resize nearest-neighbor; crop/bilinear bawaan TorchVision tidak diterapkan.
        return (images.float() / 255.0 - self.image_mean) / self.image_std

    def configure_phase(self, phase):
        if phase not in EXPERIMENTS:
            raise ValueError("Invalid training phase")
        self.phase = phase
        self.backbone.requires_grad_(phase == "scratch")
        if phase == "fine_tuning":
            start = dict(self.config.fine_tuning_prefixes)[self.architecture]
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad = name.startswith(start + ".") if self.architecture == "ResNet50" else int(name.split(".")[0]) >= int(start)
        if phase != "scratch" and self.config.freeze_batch_normalization:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.phase != "scratch":
            self.backbone.eval()
        return self

    def forward(self, sensor_image, sleep_position):
        images = self.normalize_images(sensor_image)
        if self.phase == "feature_extraction":
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)
        features = self.pool(features).flatten(1)
        return self.head(torch.cat((features, sleep_position.float()), dim=1)).float()


def build_regression_model(architecture, image_shape, weights, train_backbone, config=TRAINING):
    if len(image_shape) != 3 or image_shape[-1] != 3 or min(image_shape[:2]) < 32:
        raise ValueError("Expected a channels-last image shape with at least 32 x 32 pixels")
    return AHIRegressor(architecture, weights, "scratch" if train_backbone else "feature_extraction", config)


def build_candidate_model(architecture, experiment, config):
    """Bentuk kandidat tuning baru; FT langsung tidak memuat checkpoint FE."""
    if architecture not in ARCHITECTURES or experiment not in EXPERIMENTS:
        raise ValueError("Invalid candidate architecture or experiment")
    if config not in tuning_candidates():
        raise ValueError("Candidate configuration must match the agreed grid and fixed protocol")
    set_seed(config.seed)
    weights = None if experiment == "scratch" else config.pretrained_weights
    return AHIRegressor(architecture, weights=weights, phase=experiment, config=config)


def configure_fine_tuning(model, architecture, config=TRAINING):
    if model.architecture != architecture:
        raise ValueError("Architecture differs from checkpoint")
    model.config = config
    model.configure_phase("fine_tuning")
    return model


def parameter_counts(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}


def load_checkpoint(path, device="cpu", *, expected=None):
    if expected is not None and file_hash(path) != expected["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum differs")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if expected is not None:
        for name, value in expected.items():
            if name != "checkpoint_sha256" and (
                name not in checkpoint or json.dumps(checkpoint[name], sort_keys=True) != json.dumps(value, sort_keys=True)
            ):
                raise ValueError(f"Checkpoint protocol differs: {name}")
    config = TrainingConfig(**checkpoint["config"])
    model = AHIRegressor(checkpoint["architecture"], weights=None, phase=checkpoint["experiment"], config=config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


def predict_model(model, loader, device):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for inputs, _ in loader:
            values = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
            predictions.append(model(**values).cpu().numpy().reshape(-1))
    return np.concatenate(predictions)


def bn_snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.backbone.named_buffers()
            if name.endswith(("running_mean", "running_var", "num_batches_tracked"))}


def epoch_pass(model, loader, device, criterion, optimizer=None, scaler=None, config=TRAINING):
    training = optimizer is not None
    model.train(training)
    loss_sum, absolute_sum, squared_sum, count, updates, skipped = 0., 0., 0., 0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=config.mixed_precision and device.type == "cuda"):
                predicted = model(**inputs)
            loss = criterion(predicted.float(), targets.float())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; training terminated")
            if training:
                scaler.scale(loss).backward()
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < previous_scale:
                    skipped += 1
                else:
                    updates += 1
            residual = targets.detach().float() - predicted.detach().float()
            count += len(targets)
            loss_sum += loss.item() * len(targets)
            absolute_sum += residual.abs().sum().item()
            squared_sum += residual.square().sum().item()
    return {"loss": loss_sum / count, "mae": absolute_sum / count,
            "rmse": float(np.sqrt(squared_sum / count)), "updates": updates, "skipped": skipped}


def train_experiment(architecture, experiment, data, contract, output_root, mode="smoke",
                     feature_checkpoint=None, full_training_approved=False, config=TRAINING, smoke=SMOKE,
                     initialization="historical"):
    """Loop bersama; jalur historical mempertahankan FT dari parent FE."""
    if mode not in ("smoke", "full") or experiment not in EXPERIMENTS:
        raise ValueError("Invalid run mode or experiment")
    if initialization not in ("historical", "independent"):
        raise ValueError("Invalid initialization protocol")
    if mode == "full":
        if not full_training_approved:
            raise PermissionError("Full training requires explicit approval after smoke testing")
        if config.require_gpu_for_full_training and not torch.cuda.is_available():
            raise RuntimeError("GPU runtime must pass verification before full training")
    if set(data) != {"train", "validation"}:
        raise ValueError("Development runner accepts train and validation only; no test set")
    if config.optimizer != "adam" or config.loss not in ("mse", "huber"):
        raise ValueError("Supported protocol: Adam with MSE or validation-selected Huber")
    independent = initialization == "independent"
    output_dir = Path(output_root).resolve() if independent else Path(output_root) / experiment
    if independent:
        if feature_checkpoint is not None or config not in tuning_candidates():
            raise ValueError("Independent candidates require the agreed grid and no FE parent")
        if not output_dir.is_relative_to(TUNING_DIR.resolve()) or output_dir == TUNING_DIR.resolve():
            raise ValueError("Candidate output must be inside artifacts/tuning")
        if not contract.get("tuning_protocol_sha256") or not contract.get("code_hashes"):
            raise ValueError("Independent candidates require a recorded tuning protocol")
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    set_seed(config.seed)
    started = time.perf_counter()
    parent_hash = None
    if independent:
        model = build_candidate_model(architecture, experiment, config).to(device)
    elif experiment == "fine_tuning":
        if feature_checkpoint is None:
            raise ValueError("Fine-tuning requires the best feature-extraction checkpoint")
        feature_checkpoint = Path(feature_checkpoint)
        parent = json.loads((feature_checkpoint.parent / "run.json").read_text(encoding="utf-8"))
        if (parent["experiment"], parent["architecture"], parent["mode"], parent["data_fingerprint"]) != (
            "feature_extraction", architecture, mode, contract["fingerprint"]
        ):
            raise ValueError("Fine-tuning parent checkpoint differs from this protocol")
        parent_hash = file_hash(feature_checkpoint)
        if parent_hash != parent["checkpoint_sha256"]:
            raise ValueError("Feature-extraction checkpoint hash differs")
        model = load_checkpoint(feature_checkpoint, device)
        configure_fine_tuning(model, architecture, config)
    else:
        weights = None if experiment == "scratch" else config.pretrained_weights
        model = build_regression_model(architecture, data["train"]["images"].shape[1:], weights, experiment == "scratch", config).to(device)
    learning_rate = config.fine_tuning_learning_rate if experiment == "fine_tuning" else config.initial_learning_rate
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad),
                                 lr=learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.reduce_lr_factor,
                                                         patience=config.reduce_lr_patience, min_lr=config.min_learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda", init_scale=config.amp_initial_scale)
    criterion = nn.MSELoss() if config.loss == "mse" else nn.HuberLoss()
    counts, bn_before = parameter_counts(model), bn_snapshot(model)
    if independent:
        save_json(output_dir / "progress.json", {"status": "running", "parameters": counts,
                  "config": asdict(config), "data_fingerprint": contract["fingerprint"],
                  "protocol_sha256": contract["tuning_protocol_sha256"], "initialization": initialization})
    batch_size = smoke.batch_size if mode == "smoke" else config.batch_size
    selected = {}
    for split, payload in data.items():
        limit = (smoke.train_batches if split == "train" else smoke.validation_batches) * batch_size if mode == "smoke" else len(payload["targets"])
        selected[split] = {key: value[:limit] for key, value in payload.items()}
    train_loader = make_dataset(selected["train"], batch_size, training=True, config=config)
    validation_loader = make_dataset(selected["validation"], batch_size, config=config)
    checkpoint = output_dir / "best.pt"
    epochs = smoke.epochs if mode == "smoke" else getattr(config, f"{experiment}_epochs")
    history = {key: [] for key in ("loss", "mae", "rmse", "val_loss", "val_mae", "val_rmse", "learning_rate")}
    best, bad_epochs, updates, skipped = float("inf"), 0, 0, 0
    build_seconds = time.perf_counter() - started
    train_started = time.perf_counter()
    for epoch in range(epochs):
        trained = epoch_pass(model, train_loader, device, criterion, optimizer, scaler, config)
        validated = epoch_pass(model, validation_loader, device, criterion, config=config)
        for key in ("loss", "mae", "rmse"):
            history[key].append(trained[key])
            history[f"val_{key}"].append(validated[key])
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        updates, skipped = updates + trained["updates"], skipped + trained["skipped"]
        monitored = history[config.checkpoint_monitor][-1]
        if not np.isfinite(monitored):
            raise FloatingPointError("Non-finite validation metric")
        improved = monitored < best
        payload = {"architecture": architecture, "experiment": experiment, "config": asdict(config),
                   "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                   "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                   "epoch": epoch + 1, "data_fingerprint": contract["fingerprint"], "mode": mode}
        if independent:
            payload.update(initialization=initialization, protocol_sha256=contract["tuning_protocol_sha256"],
                           code_hashes=contract["code_hashes"])
        if improved:
            torch.save(payload, checkpoint)
            best, bad_epochs = monitored, 0
        else:
            bad_epochs += 1
        if not config.checkpoint_save_best_only:
            torch.save(payload, output_dir / f"epoch_{epoch + 1:03d}.pt")
        scheduler.step(monitored)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        print(f"{architecture}/{experiment}/{mode}: epoch {epoch + 1}, loss={trained['loss']:.4f}, val_loss={validated['loss']:.4f}", flush=True)
        if bad_epochs >= config.early_stopping_patience:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - train_started
    peak_memory = gpu_memory()
    if updates == 0:
        raise RuntimeError("No optimizer update completed; review AMP scale or numeric stability")
    bn_after = bn_snapshot(model)
    bn_unchanged = all(torch.equal(value, bn_after[key]) for key, value in bn_before.items())
    if experiment != "scratch" and not bn_unchanged:
        raise AssertionError("Batch Normalization state changed during transfer learning")
    expected = None
    if independent:
        expected = {"checkpoint_sha256": file_hash(checkpoint), "architecture": architecture,
                    "experiment": experiment, "config": asdict(config), "mode": mode,
                    "data_fingerprint": contract["fingerprint"], "initialization": initialization,
                    "protocol_sha256": contract["tuning_protocol_sha256"], "code_hashes": contract["code_hashes"]}
    loaded = load_checkpoint(checkpoint, device, expected=expected)
    model.load_state_dict(loaded.state_dict())
    original_predictions = predict_model(model, validation_loader, device)
    loaded_predictions = predict_model(loaded, validation_loader, device)
    np.testing.assert_allclose(original_predictions, loaded_predictions, rtol=config.reload_rtol, atol=config.reload_atol)
    predictions = prediction_table(selected["validation"], loaded_predictions, architecture, experiment, "validation", mode)
    metrics = save_evaluation(predictions, output_dir)
    figures = plot_diagnostics(history, predictions, output_dir / "figures", f"{architecture} / {experiment} / {mode.upper()} / validation")
    result = {"framework": "PyTorch", "architecture": architecture, "experiment": experiment, "mode": mode,
              "research_status": contract["metadata"]["status"], "data_fingerprint": contract["fingerprint"],
              "preprocessing_hashes": contract["hashes"], "config": asdict(config), "smoke_config": asdict(smoke),
              "train_samples": len(selected["train"]["targets"]), "validation_samples": len(selected["validation"]["targets"]),
              "test_accessed": False, "epochs_completed": len(history["loss"]), "batch_size": batch_size,
              "parameters": counts, "build_seconds": build_seconds, "training_seconds": training_seconds,
              "checkpoint_sha256": file_hash(checkpoint), "parent_feature_checkpoint_sha256": parent_hash,
              "reload_predictions_consistent": True, "bn_state_unchanged": bn_unchanged,
              "optimizer_updates": updates, "amp_skipped_steps": skipped,
              "trainable_backbone_parameters": [name for name, parameter in model.backbone.named_parameters() if parameter.requires_grad],
              "gpu_memory_training": peak_memory, "compute_device": str(device),
              "negative_validation_predictions": int((loaded_predictions < 0).sum()),
              "input_normalization": "raw NCHW pixels /255 then TorchVision ImageNet mean/std; no extra resize/crop",
              "figures": figures, "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "pandas", "scikit-learn")}}
    if independent:
        result.update(initialization=initialization, protocol_sha256=contract["tuning_protocol_sha256"],
                      code_hashes=contract["code_hashes"], best_epoch=int(np.argmin(history[config.checkpoint_monitor])) + 1)
    save_json(output_dir / "history.json", history)
    save_json(output_dir / "run.json", result)
    del model, loaded, optimizer, payload, train_loader, validation_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, metrics, checkpoint
