import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, POSITIONS, PREPROCESSING, SENSOR_FILES
from src.preprocessing import (
    assign_patient_components, build_sample_manifest, encode_split, fit_scalers,
    save_processed_dataset, split_manifest, transform_features, validate_manifest, validate_sensor_frame,
)


def sensor_fixture():
    tables = {}
    for channel, sensor in enumerate(SENSOR_FILES):
        rows = []
        for patient in range(20):
            source = 0 if patient == 1 else patient
            for position_index, position in enumerate(POSITIONS):
                for imf in (10, 2):
                    row = {name: float(source * 100 + position_index * 20 + imf * 500 + channel + index)
                           for index, name in enumerate(FEATURE_COLUMNS)}
                    row.update(id=f"{patient:05d}", Posisi=position, IMF=f"IMF{imf}",
                               AHI=float(patient * 3 + position_index))
                    rows.append(row)
        tables[sensor] = validate_sensor_frame(pd.DataFrame(rows), sensor)
    return tables


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.tables = sensor_fixture()
        self.config = replace(PREPROCESSING, image_size=(32, 32), scaling_percentiles=(0, 100))
        manifest, self.excluded = build_sample_manifest(self.tables)
        self.manifest = split_manifest(assign_patient_components(manifest), self.config)

    def test_patient_and_duplicate_groups_never_overlap(self):
        validate_manifest(self.manifest)
        copies = self.manifest[self.manifest.id.isin(["00000", "00001"])]
        self.assertEqual(copies.patient_component.nunique(), 1)
        self.assertEqual(copies.split.nunique(), 1)
        self.assertEqual(self.manifest.patient_component.nunique(), 19)
        again = split_manifest(assign_patient_components(build_sample_manifest(self.tables)[0]), self.config)
        pd.testing.assert_frame_equal(self.manifest, again)
        broken = self.manifest.copy()
        broken.loc[copies.index[0], "split"] = next(value for value in ("train", "test") if value != copies.split.iloc[0])
        with self.assertRaises(AssertionError):
            validate_manifest(broken)

    def test_duplicate_connections_are_transitive_across_positions(self):
        manifest = pd.DataFrame({
            "id": ["a", "b", "b", "c", "d"],
            "feature_fingerprint": ["left_copy", "left_copy", "right_copy", "right_copy", "unique"],
        })
        result = assign_patient_components(manifest)
        self.assertEqual(result.loc[result.id.isin(["a", "b", "c"]), "patient_component"].nunique(), 1)
        self.assertEqual(result.patient_component.nunique(), 2)

    def test_scaler_uses_training_only_and_no_target(self):
        original = fit_scalers(self.tables, self.manifest, self.config)
        train_ids = set(self.manifest.loc[self.manifest.split == "train", "id"])
        changed = {sensor: frame.copy() for sensor, frame in self.tables.items()}
        for frame in changed.values():
            frame.loc[~frame.id.isin(train_ids), list(FEATURE_COLUMNS)] = 1e20
            frame["AHI"] = 9999.0
        self.assertEqual(original, fit_scalers(changed, self.manifest, self.config))

    def test_imf_order_padding_channels_and_clipping(self):
        scalers = fit_scalers(self.tables, self.manifest, self.config)
        selected = self.manifest.loc[self.manifest.split == "train"].iloc[:1].reset_index(drop=True)
        payload, clipping = encode_split(dict(reversed(list(self.tables.items()))), selected, scalers, self.config)
        self.assertEqual(payload["imf_mask"].sum(), 6)
        self.assertTrue(payload["imf_mask"][0, 1, :].all())
        self.assertTrue(payload["imf_mask"][0, 9, :].all())
        self.assertFalse(payload["imf_mask"][0, 0, :].any())
        self.assertTrue((payload["feature_images"][0, 0] == 0).all())
        self.assertTrue((payload["feature_images"][0, 1] >= 1).all())
        self.assertTrue((payload["scaled_features"][0, 1] < payload["scaled_features"][0, 9]).all())
        for channel, sensor in enumerate(SENSOR_FILES):
            row = self.tables[sensor].query("id == @selected.iloc[0].id and imf_index == 2 and Posisi == @selected.iloc[0].Posisi")
            low = scalers["sensors"][sensor]["low"][0]
            high = scalers["sensors"][sensor]["high"][0]
            expected = (row.iloc[0][FEATURE_COLUMNS[0]] - low) / (high - low)
            self.assertAlmostEqual(payload["scaled_features"][0, 1, 0, channel], expected, places=6)
        self.assertEqual(len(clipping), len(FEATURE_COLUMNS) * len(SENSOR_FILES))
        self.assertTrue(np.isin(payload["images"], payload["feature_images"]).all())

    def test_complete_case_and_target_conflict(self):
        changed = {sensor: frame.copy() for sensor, frame in self.tables.items()}
        pulse = changed["PulseRate"]
        changed["PulseRate"] = pulse.loc[~((pulse.id == "00019") & (pulse.Posisi == "Right"))]
        manifest, excluded = build_sample_manifest(changed)
        self.assertEqual(len(excluded), 1)
        self.assertEqual(len(manifest), len(self.manifest) - 1)
        self.assertEqual(excluded.iloc[0].missing_sensors, "PulseRate")
        changed["Pleth"].loc[changed["Pleth"].id == "00000", "AHI"] = 99.0
        with self.assertRaisesRegex(ValueError, "between sensors"):
            build_sample_manifest(changed)

    def test_roundtrip_no_pickle_and_original_targets(self):
        scalers = fit_scalers(self.tables, self.manifest, self.config)
        with tempfile.TemporaryDirectory() as directory:
            metadata, stats, _ = save_processed_dataset(
                self.tables, self.manifest, self.excluded, scalers, {}, Path(directory), self.config
            )
            self.assertEqual(metadata["status"], "provisional_pending_client_confirmation")
            self.assertEqual(stats.query("position == 'Overall'").samples.sum(), len(self.manifest))
            with np.load(Path(directory) / "test.npz", allow_pickle=False) as stored:
                selected = self.manifest.loc[self.manifest.split == "test"]
                np.testing.assert_array_equal(stored["targets"], selected.AHI)
                self.assertEqual(stored["patient_ids"].dtype.kind, "U")
            self.assertEqual(json.loads((Path(directory) / "metadata.json").read_text())["validation"]["patient_overlap"], 0)

    def test_zero_span_and_out_of_range_features(self):
        scaler = {"low": [5.0, 0.0], "high": [5.0, 10.0]}
        transformed = transform_features([[5, -10], [100, 30]], scaler)
        np.testing.assert_array_equal(transformed, [[0.5, 0], [0.5, 1]])

    def test_encoding_collisions_cannot_silently_cross_subsets(self):
        scalers = fit_scalers(self.tables, self.manifest, self.config)
        changed = {sensor: frame.copy() for sensor, frame in self.tables.items()}
        for frame in changed.values():
            frame[list(FEATURE_COLUMNS)] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Encoded model inputs overlap"):
                save_processed_dataset(changed, self.manifest, self.excluded, scalers, {}, Path(directory), self.config)


if __name__ == "__main__":
    unittest.main()
