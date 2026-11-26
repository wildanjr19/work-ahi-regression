import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.audit import file_hash, inspect_sensor
from src import audit
from src.config import EXPECTED_COLUMNS, FEATURE_COLUMNS, POSITIONS, SENSOR_FILES, PreprocessingConfig


class SensorAuditTests(unittest.TestCase):
    def write_workbook(self, path, mutate=None):
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for position in POSITIONS:
                rows = []
                for index in (1, 2):
                    row = {column: 1.0 for column in FEATURE_COLUMNS}
                    row.update(id="00001", Posisi=position, IMF=f"IMF{index}", AHI=12.0)
                    rows.append(row)
                frame = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
                if mutate:
                    frame = mutate(frame, position)
                frame.to_excel(writer, sheet_name=position, index=False)

    def test_whitespace_normalization_preserves_id_and_source(self):
        def mutate(frame, position):
            return frame.rename(columns={"AHI": "AHI "}) if position == "Right" else frame
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sensor.xlsx"
            self.write_workbook(path, mutate)
            before = file_hash(path)
            summary, _, _, _, targets, conflicts, _ = inspect_sensor("Flow", path)
            self.assertEqual(summary["rows"], 6)
            self.assertEqual(summary["samples"], 3)
            self.assertEqual(summary["header_changes"][0]["original"], "AHI ")
            self.assertEqual(set(targets.index.get_level_values("id")), {"00001"})
            self.assertTrue(conflicts.empty)
            self.assertEqual(before, file_hash(path))

    def test_target_conflict_is_reported_without_averaging(self):
        def mutate(frame, position):
            if position == "Left":
                frame.loc[1, "AHI"] = 99.0
            return frame
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sensor.xlsx"
            self.write_workbook(path, mutate)
            summary, _, _, _, targets, conflicts, _ = inspect_sensor("Flow", path)
            self.assertEqual(summary["inconsistent_target_groups"], 1)
            self.assertEqual(len(conflicts), 1)
            self.assertTrue(pd.isna(targets.loc[("00001", "Left"), "AHI_Flow"]))

    def test_invalid_values_keys_and_positions_are_detected(self):
        def mutate(frame, position):
            frame["Energi IMF"] = frame["Energi IMF"].astype(object)
            if position == "Supine":
                frame.loc[0, "Energi IMF"] = "invalid"
                frame.loc[1, "IMF"] = "IMF51"
            elif position == "Left":
                frame.loc[1, "IMF"] = "IMF1"
                frame.loc[0, "Energi IMF"] = float("inf")
            else:
                frame.loc[0, "Posisi"] = "Unknown"
                frame.loc[1, "AHI"] = None
            return frame
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sensor.xlsx"
            self.write_workbook(path, mutate)
            summary, sheets, *_ = inspect_sensor("Flow", path)
            self.assertEqual(summary["invalid_numeric_by_column"]["Energi IMF"], 1)
            self.assertEqual(summary["infinite_by_column"]["Energi IMF"], 1)
            self.assertEqual(summary["invalid_imf_rows"], 1)
            self.assertEqual(summary["duplicate_row_keys"], 1)
            self.assertEqual(summary["missing_by_column"]["AHI"], 1)
            self.assertEqual(sum(sheet["position_mismatches"] for sheet in sheets), 1)

    def test_schema_collision_is_rejected(self):
        def mutate(frame, position):
            frame["AHI "] = frame["AHI"]
            return frame
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sensor.xlsx"
            self.write_workbook(path, mutate)
            with self.assertRaisesRegex(ValueError, "colliding columns"):
                inspect_sensor("Flow", path)

    def test_fused_duplicates_keep_distinct_ids_and_targets(self):
        def mutate(frame, position):
            copy = frame.assign(id="00002", AHI=24.0 if position == "Right" else 12.0)
            return pd.concat([frame, copy], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in SENSOR_FILES.values():
                self.write_workbook(root / filename, mutate)
            output = root / "audit"
            with patch.object(audit, "DATA_DIR", root), patch.object(audit, "AUDIT_DIR", output), \
                    patch.object(audit, "inspect_environment", return_value={}), redirect_stdout(StringIO()):
                audit.run_audit()
            report = json.loads((output / "dataset_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(report["fusion"]["identical_fused_samples_same_position"], 6)
            self.assertEqual(report["fusion"]["identical_fused_groups"], 3)
            self.assertEqual(report["fusion"]["identical_fused_groups_with_different_targets"], 1)
            self.assertEqual(report["blocking_issues"], [])
            manifest = pd.read_csv(output / "sample_availability.csv", dtype={"id": "string"})
            self.assertEqual(set(manifest.id), {"00001", "00002"})

    def test_cross_sensor_target_conflict_blocks_followup(self):
        def mutate(frame, position):
            return frame.assign(AHI=99.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sensor, filename in SENSOR_FILES.items():
                self.write_workbook(root / filename, mutate if sensor == "PulseRate" else None)
            output = root / "audit"
            with patch.object(audit, "DATA_DIR", root), patch.object(audit, "AUDIT_DIR", output), \
                    patch.object(audit, "inspect_environment", return_value={}), redirect_stdout(StringIO()):
                with self.assertRaises(SystemExit):
                    audit.run_audit()
            report = json.loads((output / "dataset_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(report["fusion"]["cross_sensor_target_conflicts"], 3)
            self.assertTrue(report["blocking_issues"])


class ConfigTests(unittest.TestCase):
    def test_padding_and_split_guards(self):
        self.assertEqual(len(FEATURE_COLUMNS), 11)
        self.assertEqual(PreprocessingConfig().seed, 42)
        for kwargs in ({"train_fraction": 0.8}, {"padding_pixel": 1},
                       {"scaling_percentiles": (99, 1)}, {"image_size": (10, 224)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PreprocessingConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
