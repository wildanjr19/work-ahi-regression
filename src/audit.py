"""Audit baca-saja untuk PHASE 0. Jalankan dengan python -m src.audit."""

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import (
    AUDIT_DIR, DATA_DIR, EXPECTED_COLUMNS, FEATURE_COLUMNS, POSITIONS,
    PREPROCESSING, ROW_KEY, SAMPLE_KEY, SENSOR_FILES,
)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    # Statistik seperti simpangan baku satu observasi tidak terdefinisi.
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def command_result(arguments):
    if shutil.which(arguments[0]) is None:
        return {"status": "not_found"}
    try:
        result = subprocess.run(arguments, capture_output=True, timeout=30)
        def decode(value):
            encoding = "utf-16-le" if b"\x00" in value else "utf-8"
            return value.decode(encoding, errors="replace").strip()
        return {"returncode": result.returncode, "stdout": decode(result.stdout),
                "stderr": decode(result.stderr)}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "error", "detail": str(error)}


def inspect_environment():
    modules = {
        "numpy": "numpy", "pandas": "pandas", "scikit-learn": "sklearn",
        "matplotlib": "matplotlib", "seaborn": "seaborn", "scipy": "scipy",
        "openpyxl": "openpyxl", "joblib": "joblib", "ipykernel": "ipykernel",
        "nbformat": "nbformat", "nbclient": "nbclient", "jupyterlab": "jupyterlab",
        "notebook": "notebook", "torch": "torch", "torchvision": "torchvision",
    }
    packages = {}
    for distribution, module in modules.items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = {"version": None, "import": "not_installed"}
            continue
        try:
            importlib.import_module(module)
            packages[distribution] = {"version": version, "import": "ok"}
        except Exception as error:
            packages[distribution] = {"version": version, "import": repr(error)}
    pytorch = {"status": "not_installed", "gpu_detection": "not_tested"}
    if importlib.util.find_spec("torch"):
        try:
            import torch
            pytorch = {"status": "imported", "version": str(torch.__version__),
                       "cuda_runtime": torch.version.cuda,
                       "cuda_available": torch.cuda.is_available(),
                       "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]}
        except Exception as error:
            pytorch = {"status": "import_error", "detail": repr(error)}
    report = {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "packages": packages, "pytorch": pytorch,
        "pip_check": command_result([sys.executable, "-m", "pip", "check"]),
        "nvidia_smi": command_result(["nvidia-smi"]),
        "gpu": command_result(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,compute_cap",
                               "--format=csv,noheader,nounits"]),
        "nvcc": command_result(["nvcc", "--version"]),
    }
    if platform.system() == "Windows":
        report["wsl"] = command_result(["wsl", "--list", "--verbose"])
        report["windows"] = command_result([
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,OSArchitecture,TotalVisibleMemorySize | ConvertTo-Json",
        ])
    return report


def read_sensor_frame(sensor, path):
    sheets, frames, header_changes = [], [], []
    with pd.ExcelFile(path, engine="openpyxl") as workbook:
        if set(workbook.sheet_names) != set(POSITIONS):
            raise ValueError(f"{path.name}: unexpected sheets {workbook.sheet_names}")
        for sheet in POSITIONS:
            frame = pd.read_excel(workbook, sheet_name=sheet, dtype="string")
            raw_columns = frame.columns.tolist()
            frame.columns = [str(column).strip() for column in raw_columns]
            if frame.columns.duplicated().any() or set(frame.columns) != set(EXPECTED_COLUMNS):
                raise ValueError(f"{path.name}/{sheet}: unexpected or colliding columns")
            header_changes.extend({"sheet": sheet, "original": old, "normalized": new}
                                  for old, new in zip(raw_columns, frame.columns) if old != new)
            for column in ("id", "Posisi", "IMF"):
                frame[column] = frame[column].str.strip().replace("", pd.NA)
            sheet_info = {"sensor": sensor, "sheet": sheet, "rows": len(frame),
                          "columns": len(frame.columns), "patients": int(frame.id.nunique()),
                          "position_mismatches": int(frame.Posisi.ne(sheet).fillna(True).sum())}
            sheets.append(sheet_info)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True), sheets, header_changes


def inspect_sensor(sensor, path):
    before_hash = file_hash(path)
    frame, sheets, header_changes = read_sensor_frame(sensor, path)
    numeric_columns = [*FEATURE_COLUMNS, "AHI"]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce").astype(float)
    missing = frame.isna().sum()
    invalid_numeric = numeric.isna() & frame[numeric_columns].notna()
    for column in numeric_columns:
        frame[column] = numeric[column]
    imf_index = pd.to_numeric(frame.IMF.str.extract(r"^IMF([1-9]\d*)$", expand=False), errors="coerce")
    invalid_imf = imf_index.isna() | ~imf_index.between(1, PREPROCESSING.max_imfs)
    groups = frame.groupby(list(SAMPLE_KEY), dropna=False, sort=True)
    counts = groups.size().rename("imf_rows").reset_index()
    counts["sensor"] = sensor
    targets = groups.AHI.agg(["min", "max", "nunique", "count", "size"])
    conflicts = targets[(targets["nunique"] != 1) | (targets["count"] != targets["size"])].reset_index()
    conflicts["sensor"] = sensor
    targets["target"] = targets["min"].where(
        (targets["nunique"] == 1) & (targets["count"] == targets["size"])
    )
    signatures = []
    gap_groups = 0
    for (patient_id, position), group in groups:
        indices = pd.to_numeric(group.IMF.str.extract(r"^IMF([1-9]\d*)$", expand=False), errors="coerce")
        valid_indices = indices.dropna().astype(int)
        if len(valid_indices):
            gap_groups += int(set(range(1, valid_indices.max() + 1)) != set(valid_indices))
        ordered = group.assign(imf_index=indices).sort_values("imf_index")
        # Sidik jari ini hanya mendeteksi kesamaan persis, bukan asal oversampling.
        signature = hashlib.sha256(pd.util.hash_pandas_object(
            ordered[["IMF", *FEATURE_COLUMNS]], index=False
        ).values.tobytes()).hexdigest()
        signatures.append({"sensor": sensor, "id": patient_id, "Posisi": position,
                           "signature": signature})
    signature_frame = pd.DataFrame(signatures)
    identical = signature_frame[signature_frame.duplicated(["Posisi", "signature"], keep=False)]
    finite = numeric.replace([np.inf, -np.inf], np.nan)
    statistics = finite.describe(percentiles=[0.01, 0.25, 0.50, 0.75, 0.99]).T
    statistics.index.name = "column"
    statistics = statistics.reset_index().assign(sensor=sensor)
    summary = {
        "sensor": sensor, "file": path.name, "sha256": before_hash,
        "source_unchanged": before_hash == file_hash(path),
        "rows": len(frame), "columns": len(EXPECTED_COLUMNS),
        "patients": int(frame.id.nunique()), "samples": len(targets),
        "samples_by_position": frame.groupby("Posisi").id.nunique().to_dict(),
        "missing_by_column": {str(k): int(v) for k, v in missing.items()},
        "invalid_numeric_by_column": {str(k): int(v) for k, v in invalid_numeric.sum().items()},
        "infinite_by_column": {str(k): int(v) for k, v in np.isinf(numeric).sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_row_keys": int(frame.duplicated(list(ROW_KEY)).sum()),
        "invalid_imf_rows": int(invalid_imf.sum()),
        "imf_min": None if imf_index.dropna().empty else int(imf_index.min()),
        "imf_max": None if imf_index.dropna().empty else int(imf_index.max()),
        "imf_count_summary": counts.imf_rows.describe().to_dict(),
        "groups_with_internal_imf_gaps": gap_groups,
        "inconsistent_target_groups": len(conflicts),
        "negative_target_rows": int((numeric.AHI < 0).sum()),
        "identical_feature_samples_same_position": len(identical),
        "header_changes": header_changes,
    }
    target_table = targets[["target"]].rename(columns={"target": f"AHI_{sensor}"})
    return summary, sheets, counts, statistics, target_table, conflicts, identical


def run_audit():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    summaries, sheets, counts, statistics, targets, conflicts, identical = [], [], [], [], [], [], []
    for sensor, filename in SENSOR_FILES.items():
        result = inspect_sensor(sensor, DATA_DIR / filename)
        for destination, item in zip(
            (summaries, sheets, counts, statistics, targets, conflicts, identical), result
        ):
            destination.append(item)
    availability = pd.concat(targets, axis=1)
    for sensor, table in zip(SENSOR_FILES, targets):
        availability[f"has_{sensor}"] = availability.index.isin(table.index)
    present_columns = [f"has_{sensor}" for sensor in SENSOR_FILES]
    ahi_columns = [f"AHI_{sensor}" for sensor in SENSOR_FILES]
    complete = availability[present_columns].all(axis=1)
    cross_conflicts = availability[ahi_columns].nunique(axis=1, dropna=True) > 1
    availability["complete_case"] = complete
    availability["cross_sensor_target_conflict"] = cross_conflicts
    joint_signatures = None
    for sensor, duplicates in zip(SENSOR_FILES, identical):
        current = duplicates.set_index(list(SAMPLE_KEY))[["signature"]].rename(
            columns={"signature": f"signature_{sensor}"}
        )
        joint_signatures = current if joint_signatures is None else joint_signatures.join(current, how="inner")
    joint_signatures = joint_signatures.reset_index()
    signature_columns = [f"signature_{sensor}" for sensor in SENSOR_FILES]
    repeated_joint = joint_signatures[
        joint_signatures.duplicated(["Posisi", *signature_columns], keep=False)
    ].merge(availability.reset_index()[[*SAMPLE_KEY, "AHI_Flow"]], on=list(SAMPLE_KEY), validate="one_to_one")
    repeated_joint.to_csv(AUDIT_DIR / "identical_fused_samples.csv", index=False)
    repeated_groups = repeated_joint.groupby(["Posisi", *signature_columns])
    availability.reset_index().to_csv(AUDIT_DIR / "sample_availability.csv", index=False)
    availability.loc[cross_conflicts].reset_index().to_csv(AUDIT_DIR / "cross_sensor_target_conflicts.csv", index=False)
    pd.concat(conflicts, ignore_index=True).to_csv(AUDIT_DIR / "within_sensor_target_conflicts.csv", index=False)
    pd.concat(counts, ignore_index=True).to_csv(AUDIT_DIR / "imf_counts.csv", index=False)
    pd.concat(statistics, ignore_index=True).to_csv(AUDIT_DIR / "numeric_statistics.csv", index=False)
    pd.concat(identical, ignore_index=True).to_csv(AUDIT_DIR / "identical_feature_samples.csv", index=False)
    pd.DataFrame([item for collection in sheets for item in collection]).to_csv(AUDIT_DIR / "sheet_summary.csv", index=False)
    ahi_rows = []
    complete_targets = availability.loc[complete, "AHI_Flow"]
    for position in ("Overall", *POSITIONS):
        values = complete_targets if position == "Overall" else complete_targets[
            complete_targets.index.get_level_values("Posisi") == position
        ]
        ahi_rows.append({"position": position, **values.describe(percentiles=[0.01, 0.5, 0.95, 0.99]).to_dict(),
                         "zero": int((values == 0).sum()),
                         "above_audit_threshold": int((values > PREPROCESSING.ahi_audit_threshold).sum()),
                         "equal_600": int((values == 600).sum())})
    pd.DataFrame(ahi_rows).to_csv(AUDIT_DIR / "ahi_summary.csv", index=False)
    extreme_mask = availability[ahi_columns].max(axis=1) > PREPROCESSING.ahi_audit_threshold
    availability.loc[extreme_mask].reset_index().to_csv(AUDIT_DIR / "high_ahi_samples.csv", index=False)
    blocking = []
    for summary, sheet_collection in zip(summaries, sheets):
        for field in ("duplicate_row_keys", "invalid_imf_rows", "inconsistent_target_groups", "negative_target_rows"):
            if summary[field]:
                blocking.append(f"{summary['sensor']}: {field}={summary[field]}")
        for field in ("missing_by_column", "invalid_numeric_by_column", "infinite_by_column"):
            if sum(summary[field].values()):
                blocking.append(f"{summary['sensor']}: {field}")
        if any(item["position_mismatches"] for item in sheet_collection):
            blocking.append(f"{summary['sensor']}: sheet/position mismatch")
        if not summary["source_unchanged"]:
            blocking.append(f"{summary['sensor']}: source hash changed")
    if cross_conflicts.any():
        blocking.append(f"Cross-sensor target conflicts: {int(cross_conflicts.sum())}")
    report = {
        "timestamp_jakarta": datetime.now(timezone(timedelta(hours=7))).isoformat(),
        "sensors": summaries,
        "fusion": {
            "union_samples": len(availability), "complete_samples": int(complete.sum()),
            "patients_complete": int(availability.index[complete].get_level_values("id").nunique()),
            "excluded_incomplete_samples": int((~complete).sum()),
            "complete_by_position": availability.loc[complete].reset_index().groupby("Posisi").size().to_dict(),
            "incomplete_by_position": availability.loc[~complete].reset_index().groupby("Posisi").size().to_dict(),
            "missing_sensor_samples": {sensor: int((~availability[f"has_{sensor}"]).sum()) for sensor in SENSOR_FILES},
            "cross_sensor_target_conflicts": int(cross_conflicts.sum()),
            "identical_fused_samples_same_position": len(repeated_joint),
            "identical_fused_groups": repeated_groups.ngroups,
            "identical_fused_patients": int(repeated_joint.id.nunique()),
            "identical_fused_groups_with_different_targets": int((repeated_groups.AHI_Flow.nunique() > 1).sum()),
        },
        "ahi_audit_threshold": PREPROCESSING.ahi_audit_threshold,
        "ahi_complete_cases": ahi_rows,
        "oversampling_status": "unconfirmed; exact duplicate checks cannot establish provenance",
        "warnings": [
            "Identical multi-sensor feature matrices across patient IDs require provenance review or conservative connected-component grouping before splitting."
        ] if len(repeated_joint) else [],
        "blocking_issues": blocking,
        "scope": "phase_0_read_only_audit; no splitting, fitted preprocessing, image encoding, or training",
    }
    for filename, payload in (("dataset_audit.json", report), ("environment.json", inspect_environment())):
        (AUDIT_DIR / filename).write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"sensors": [{k: item[k] for k in ("sensor", "rows", "patients", "samples")} for item in summaries],
                      "fusion": report["fusion"], "blocking_issues": blocking}, indent=2))
    if blocking:
        raise SystemExit("Audit found blocking issues; inspect artifacts/audit before preprocessing.")


if __name__ == "__main__":
    run_audit()
