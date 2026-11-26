"""Preprocessing AHI dengan split pasien dan komponen salinan fitur."""

import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .audit import file_hash, json_safe, read_sensor_frame
from .config import (
    DATA_DIR, FEATURE_COLUMNS, POSITIONS, PREPROCESSING, PROCESSED_DIR,
    SAMPLE_KEY, SENSOR_FILES,
)

SPLITS = ("train", "validation", "test")


def save_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def validate_sensor_frame(frame, sensor, config=PREPROCESSING):
    frame = frame.copy()
    if frame.empty or frame[list(SAMPLE_KEY) + ["IMF"]].isna().any().any():
        raise ValueError(f"{sensor}: empty data or missing identity/IMF")
    for column in (*FEATURE_COLUMNS, "AHI"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    if not np.isfinite(frame[[*FEATURE_COLUMNS, "AHI"]].to_numpy()).all():
        raise ValueError(f"{sensor}: missing or non-finite numeric values")
    if not frame.Posisi.isin(POSITIONS).all() or (frame.AHI < 0).any():
        raise ValueError(f"{sensor}: invalid position or negative target")
    index = pd.to_numeric(frame.IMF.str.extract(r"^IMF([1-9]\d*)$", expand=False), errors="coerce")
    if index.isna().any() or not index.between(1, config.max_imfs).all():
        raise ValueError(f"{sensor}: invalid IMF index")
    frame["imf_index"] = index.astype(int)
    if frame.duplicated([*SAMPLE_KEY, "imf_index"]).any():
        raise ValueError(f"{sensor}: duplicate IMF key")
    if (frame.groupby(list(SAMPLE_KEY)).AHI.nunique() != 1).any():
        raise ValueError(f"{sensor}: inconsistent AHI within sample")
    return frame


def load_sensor_tables(data_dir=DATA_DIR, config=PREPROCESSING):
    tables, sources = {}, {}
    for sensor, filename in SENSOR_FILES.items():
        path = Path(data_dir) / filename
        before = file_hash(path)
        frame, sheets, changes = read_sensor_frame(sensor, path)
        if any(sheet["position_mismatches"] for sheet in sheets):
            raise ValueError(f"{sensor}: sheet and position differ")
        tables[sensor] = validate_sensor_frame(frame, sensor, config)
        tables[sensor]["sensor"] = sensor
        if before != file_hash(path):
            raise ValueError(f"{sensor}: source changed during reading")
        sources[sensor] = {"file": filename, "sha256": before, "header_changes": changes}
    return tables, sources


def build_sample_manifest(tables):
    targets, fingerprints = [], {}
    for sensor in SENSOR_FILES:
        frame = tables[sensor]
        targets.append(frame.groupby(list(SAMPLE_KEY)).AHI.first().rename(f"AHI_{sensor}"))
        signatures = {}
        for key, group in frame.groupby(list(SAMPLE_KEY), sort=True):
            ordered = group.sort_values("imf_index")
            values = ordered[["imf_index", *FEATURE_COLUMNS]].to_numpy(dtype="<f8")
            signatures[key] = hashlib.sha256(values.tobytes()).hexdigest()
        fingerprints[sensor] = signatures
    union = pd.concat(targets, axis=1)
    if (union.nunique(axis=1, dropna=True) > 1).any():
        raise ValueError("AHI differs between sensors for the same patient-position")
    complete = union.notna().all(axis=1)
    excluded = union.loc[~complete].reset_index()
    excluded["missing_sensors"] = [
        ",".join(sensor for sensor in SENSOR_FILES if pd.isna(row[f"AHI_{sensor}"]))
        for _, row in excluded.iterrows()
    ]
    manifest = union.loc[complete, ["AHI_Flow"]].rename(columns={"AHI_Flow": "AHI"}).reset_index()
    if manifest.empty:
        raise ValueError("No complete-case samples")
    manifest["sample_id"] = [f"sample_{index:06d}" for index in range(len(manifest))]
    signatures = []
    for row in manifest.itertuples():
        key = (row.id, row.Posisi)
        payload = row.Posisi + "|" + "|".join(fingerprints[sensor][key] for sensor in SENSOR_FILES)
        signatures.append(hashlib.sha256(payload.encode()).hexdigest())
    manifest["feature_fingerprint"] = signatures
    return manifest, excluded


def assign_patient_components(manifest):
    manifest = manifest.copy()
    parents = {patient: patient for patient in sorted(manifest.id.unique())}

    def find(patient):
        while parents[patient] != patient:
            parents[patient] = parents[parents[patient]]
            patient = parents[patient]
        return patient

    for _, group in manifest.groupby("feature_fingerprint", sort=True):
        ids = sorted(group.id.unique())
        for patient in ids[1:]:
            left, right = sorted((find(ids[0]), find(patient)))
            parents[right] = left
    roots = sorted({find(patient) for patient in parents})
    labels = {root: f"component_{index:05d}" for index, root in enumerate(roots)}
    manifest["patient_component"] = manifest.id.map(lambda patient: labels[find(patient)])
    assert manifest.groupby("id").patient_component.nunique().eq(1).all()
    assert manifest.groupby("feature_fingerprint").patient_component.nunique().eq(1).all()
    return manifest


def validate_manifest(manifest):
    assert manifest.sample_id.is_unique
    assert not manifest.duplicated(list(SAMPLE_KEY)).any()
    assert set(manifest.split) == set(SPLITS)
    for column in ("id", "patient_component", "feature_fingerprint"):
        assert manifest.groupby(column).split.nunique().eq(1).all(), column
        for left, right in combinations(SPLITS, 2):
            assert set(manifest.loc[manifest.split == left, column]).isdisjoint(
                manifest.loc[manifest.split == right, column]
            ), (column, left, right)


def split_manifest(manifest, config=PREPROCESSING):
    manifest = manifest.copy()
    first = GroupShuffleSplit(n_splits=1, test_size=config.test_fraction, random_state=config.seed)
    development, testing = next(first.split(manifest, groups=manifest.patient_component))
    second = GroupShuffleSplit(
        n_splits=1, test_size=config.validation_fraction / (config.train_fraction + config.validation_fraction),
        random_state=config.seed,
    )
    training, validation = next(second.split(manifest.iloc[development],
                                            groups=manifest.iloc[development].patient_component))
    manifest["split"] = ""
    manifest.loc[manifest.index[testing], "split"] = "test"
    manifest.loc[manifest.index[development[training]], "split"] = "train"
    manifest.loc[manifest.index[development[validation]], "split"] = "validation"
    validate_manifest(manifest)
    return manifest


def fit_scalers(tables, manifest, config=PREPROCESSING):
    train_ids = set(manifest.loc[manifest.split == "train", "id"])
    train_keys = pd.MultiIndex.from_frame(manifest.loc[manifest.split == "train", list(SAMPLE_KEY)])
    scalers = {"method": "percentile", "fit_subset": "train", "percentiles": list(config.scaling_percentiles),
               "feature_order": list(FEATURE_COLUMNS), "sensors": {}}
    for sensor, frame in tables.items():
        selected = pd.MultiIndex.from_frame(frame[list(SAMPLE_KEY)]).isin(train_keys)
        training = frame.loc[selected]
        assert set(training.id).issubset(train_ids)
        values = training[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        low, high = np.percentile(values, config.scaling_percentiles, axis=0)
        constant = high == low
        # Rentang persentil nol bisa terjadi pada fitur jarang berubah; jangan membagi nol.
        scalers["sensors"][sensor] = {"low": low.tolist(), "high": high.tolist(),
                                      "zero_percentile_span": constant.tolist(),
                                      "fit_rows": len(training), "fit_patients": training.id.nunique()}
    return scalers


def transform_features(values, scaler):
    values = np.asarray(values, dtype=float)
    low, high = np.asarray(scaler["low"]), np.asarray(scaler["high"])
    span = high - low
    scaled = np.divide(values - low, span, out=np.zeros_like(values), where=span != 0)
    scaled[:, span == 0] = 0.5
    return np.clip(scaled, 0, 1).astype(np.float32)


def resize_nearest(images, size):
    row_index = np.arange(size[0]) * images.shape[1] // size[0]
    column_index = np.arange(size[1]) * images.shape[2] // size[1]
    return images[:, row_index[:, None], column_index[None, :], :]


def encode_split(tables, manifest, scalers, config=PREPROCESSING):
    count = len(manifest)
    shape = (count, config.max_imfs, len(FEATURE_COLUMNS), len(SENSOR_FILES))
    scaled = np.zeros(shape, dtype=np.float32)
    images = np.full(shape, config.padding_pixel, dtype=np.uint8)
    mask = np.zeros((count, config.max_imfs, len(SENSOR_FILES)), dtype=bool)
    sample_index = {(row.id, row.Posisi): index for index, row in enumerate(manifest.itertuples())}
    clipping = []
    for channel, sensor in enumerate(SENSOR_FILES):
        frame = tables[sensor]
        scaler = scalers["sensors"][sensor]
        selected = pd.MultiIndex.from_frame(frame[list(SAMPLE_KEY)]).isin(
            pd.MultiIndex.from_frame(manifest[list(SAMPLE_KEY)])
        )
        subset = frame.loc[selected]
        values = subset[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        transformed = transform_features(values, scaler)
        sample_rows = np.array([sample_index[key] for key in subset[list(SAMPLE_KEY)].itertuples(index=False, name=None)])
        imf_rows = subset.imf_index.to_numpy() - 1
        scaled[sample_rows, imf_rows, :, channel] = transformed
        pixel_low, pixel_high = config.observed_pixel_range
        encoded = np.rint(pixel_low + transformed * (pixel_high - pixel_low)).astype(np.uint8)
        images[sample_rows, imf_rows, :, channel] = encoded
        mask[sample_rows, imf_rows, channel] = True
        for feature, name in enumerate(FEATURE_COLUMNS):
            low, high = scaler["low"][feature], scaler["high"][feature]
            clipping.append({"sensor": sensor, "feature": name, "rows": len(values),
                             "below_low": int((values[:, feature] < low).sum()),
                             "above_high": int((values[:, feature] > high).sum()),
                             "outside_fraction": float(((values[:, feature] < low) | (values[:, feature] > high)).mean()),
                             "zero_percentile_span": bool(high == low)})
    one_hot = np.eye(len(POSITIONS), dtype=np.float32)[[POSITIONS.index(value) for value in manifest.Posisi]]
    payload = {
        "images": resize_nearest(images, config.image_size), "feature_images": images,
        "scaled_features": scaled, "imf_mask": mask, "position_one_hot": one_hot,
        "targets": manifest.AHI.to_numpy(dtype=np.float64), "patient_ids": manifest.id.to_numpy(dtype=str),
        "positions": manifest.Posisi.to_numpy(dtype=str), "sample_ids": manifest.sample_id.to_numpy(dtype=str),
        "patient_components": manifest.patient_component.to_numpy(dtype=str),
    }
    validate_payload(payload, manifest, config)
    return payload, pd.DataFrame(clipping)


def validate_payload(payload, manifest, config=PREPROCESSING):
    count = len(manifest)
    assert payload["images"].shape == (count, *config.image_size, len(SENSOR_FILES))
    assert payload["feature_images"].shape == (count, config.max_imfs, len(FEATURE_COLUMNS), len(SENSOR_FILES))
    assert payload["imf_mask"].shape == (count, config.max_imfs, len(SENSOR_FILES))
    assert all(len(value) == count for value in payload.values())
    for name in ("images", "feature_images", "scaled_features", "position_one_hot", "targets"):
        assert np.isfinite(payload[name]).all(), name
    observed = np.broadcast_to(payload["imf_mask"][:, :, None, :], payload["feature_images"].shape)
    assert np.all(payload["feature_images"][~observed] == config.padding_pixel)
    low, high = config.observed_pixel_range
    assert np.all((payload["feature_images"][observed] >= low) & (payload["feature_images"][observed] <= high))
    assert np.all(payload["position_one_hot"].sum(axis=1) == 1)
    assert np.array_equal(payload["position_one_hot"].argmax(axis=1), [POSITIONS.index(value) for value in manifest.Posisi])
    for name, column in (("targets", "AHI"), ("patient_ids", "id"), ("positions", "Posisi"),
                         ("sample_ids", "sample_id"), ("patient_components", "patient_component")):
        assert np.array_equal(payload[name], manifest[column].to_numpy()), name


def split_statistics(manifest):
    rows = []
    for split in SPLITS:
        subset = manifest.loc[manifest.split == split]
        for position in ("Overall", *POSITIONS):
            selected = subset if position == "Overall" else subset.loc[subset.Posisi == position]
            rows.append({"split": split, "position": position, "samples": len(selected),
                         "patients": selected.id.nunique(), "components": selected.patient_component.nunique(),
                         "ahi_min": selected.AHI.min(), "ahi_median": selected.AHI.median(),
                         "ahi_mean": selected.AHI.mean(), "ahi_std": selected.AHI.std(),
                         "ahi_max": selected.AHI.max(), "ahi_zero": int((selected.AHI == 0).sum())})
    return pd.DataFrame(rows)


def save_processed_dataset(tables, manifest, excluded, scalers, sources, output_dir=PROCESSED_DIR, config=PREPROCESSING):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_manifest(manifest)
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    excluded.to_csv(output_dir / "excluded_incomplete_samples.csv", index=False)
    stats = split_statistics(manifest)
    stats.to_csv(output_dir / "split_statistics.csv", index=False)
    save_json(output_dir / "scaling_parameters.json", scalers)
    save_json(output_dir / "feature_order.json", list(FEATURE_COLUMNS))
    clipping, files, encoded_sources = [], {}, {}
    for split in SPLITS:
        subset = manifest.loc[manifest.split == split].reset_index(drop=True)
        payload, report = encode_split(tables, subset, scalers, config)
        path = output_dir / f"{split}.npz"
        np.savez_compressed(path, **payload)
        with np.load(path, allow_pickle=False) as stored:
            validate_payload(stored, subset, config)
            assert all(np.array_equal(stored[key], value) for key, value in payload.items())
            for image, position in zip(stored["images"], stored["position_one_hot"]):
                signature = hashlib.sha256(image.tobytes() + position.tobytes()).hexdigest()
                if signature in encoded_sources and encoded_sources[signature] != split:
                    raise ValueError("Encoded model inputs overlap between subsets; inspect clipping/quantization.")
                encoded_sources[signature] = split
        report["split"] = split
        clipping.append(report)
        files[path.name] = {"sha256": file_hash(path), "bytes": path.stat().st_size,
                            "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in payload.items()}}
    clipping = pd.concat(clipping, ignore_index=True)
    clipping.to_csv(output_dir / "clipping_report.csv", index=False)
    metadata = {
        "status": "provisional_pending_client_confirmation" if config.provisional else "review_required_before_research_use",
        "pending": ["dataset provenance/oversampling", "identical inputs with differing targets", "extreme AHI including 600"],
        "config": asdict(config), "sensors": list(SENSOR_FILES), "positions": list(POSITIONS),
        "feature_order": list(FEATURE_COLUMNS), "sources": sources,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "openpyxl")},
        "grouping": "connected patient components linked by exact raw three-sensor features at the same position; target excluded",
        "target_transform": "none; float64 source targets retained",
        "image_encoding": "uint8 1..255 observed, 0 padding; nearest resize; no backbone preprocessing applied",
        "baseline_inputs": "scaled_features flattened plus flattened imf_mask and position_one_hot; no IDs or targets",
        "samples": len(manifest), "patients": manifest.id.nunique(),
        "components": manifest.patient_component.nunique(), "excluded_incomplete": len(excluded),
        "validation": {"patient_overlap": 0, "component_overlap": 0, "raw_fingerprint_overlap": 0,
                       "encoded_input_overlap": 0, "finite_arrays": True, "save_reload_equal": True},
        "files": files,
    }
    save_json(output_dir / "metadata.json", metadata)
    return metadata, stats, clipping
