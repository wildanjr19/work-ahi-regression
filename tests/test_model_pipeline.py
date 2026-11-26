import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation import evaluate_predictions, prediction_table, regression_metrics, save_evaluation
from src.model_utils import (
    SensorDataset, build_regression_model, configure_fine_tuning,
    parameter_counts, train_experiment, torch,
)


class EvaluationTests(unittest.TestCase):
    def test_regression_includes_zero_targets_and_negative_predictions(self):
        result = regression_metrics([0, 10], [-1, 13])
        self.assertEqual(result["MAE"], 2)
        self.assertEqual(result["MSE"], 5)
        self.assertAlmostEqual(result["RMSE"], np.sqrt(5))
        self.assertAlmostEqual(result["R2"], 0.8)
        self.assertEqual(result["negative_predictions"], 1)
        self.assertIsNone(regression_metrics([0], [-1])["R2"])
        self.assertIsNone(regression_metrics([0, 0], [-1, 3])["R2"])
        with self.assertRaises(ValueError):
            regression_metrics([1, 2], [np.inf, 2])

    def test_csv_recomputes_metrics_and_keeps_ids(self):
        payload = {"targets": np.array([0., 10.]), "patient_ids": ["00001", "00001"],
                   "positions": ["Supine", "Left"], "sample_ids": ["s1", "s2"]}
        table = prediction_table(payload, [-1, 13], "ResNet50", "scratch", "validation", "smoke")
        with tempfile.TemporaryDirectory() as directory:
            metrics = save_evaluation(table, Path(directory))
            self.assertEqual(metrics.loc[metrics.position == "Right", "samples"].iloc[0], 0)
            self.assertEqual(table["Patient ID"].iloc[0], "00001")
            self.assertEqual(table["Residual"].tolist(), [1, -3])
        self.assertEqual(evaluate_predictions(table).iloc[0].negative_predictions, 1)


class ModelProtocolTests(unittest.TestCase):
    def test_full_training_requires_approval_before_accessing_data(self):
        with self.assertRaises(PermissionError):
            train_experiment("ResNet50", "scratch", {}, {}, Path("unused"), mode="full")

    def test_preprocessing_matches_official_function_without_double_scaling(self):
        from torchvision.transforms.functional import normalize
        model = build_regression_model("ResNet50", (32, 32, 3), weights=None, train_backbone=False)
        raw = torch.tensor([[[[10]], [[20]], [[30]]]], dtype=torch.uint8)
        actual = model.normalize_images(raw)
        expected = normalize(raw.float() / 255, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.dtype, torch.float32)

    def test_fine_tuning_opens_only_last_stage_and_keeps_bn_frozen(self):
        model = build_regression_model("ResNet50", (32, 32, 3), weights=None, train_backbone=False)
        backbone = model.backbone
        before = parameter_counts(model)
        self.assertFalse(any(parameter.requires_grad for parameter in backbone.parameters()))
        with torch.no_grad():
            output = model(torch.zeros(2, 3, 32, 32, dtype=torch.uint8), torch.eye(3)[:2])
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.shape, (2, 1))
        configure_fine_tuning(model, "ResNet50")
        model.train()
        trainable = [name for name, parameter in backbone.named_parameters() if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("layer4.") for name in trainable))
        self.assertFalse(any(isinstance(module, torch.nn.BatchNorm2d) and module.training for module in backbone.modules()))
        self.assertFalse(any(parameter.requires_grad for module in backbone.modules() if isinstance(module, torch.nn.BatchNorm2d) for parameter in module.parameters()))
        after = parameter_counts(model)
        self.assertGreater(after["trainable"], before["trainable"])
        self.assertLess(after["trainable"], after["total"])

    def test_dataset_converts_axes_without_adding_metadata_as_features(self):
        payload = {"images": np.full((1, 32, 32, 3), [10, 20, 30], dtype=np.uint8),
                   "position_one_hot": np.array([[1, 0, 0]], dtype=np.float32), "targets": np.array([600.])}
        inputs, target = SensorDataset(payload)[0]
        self.assertEqual(set(inputs), {"sensor_image", "sleep_position"})
        self.assertEqual(inputs["sensor_image"].shape, (3, 32, 32))
        self.assertEqual(inputs["sensor_image"][:, 0, 0].tolist(), [10, 20, 30])
        self.assertEqual(target.item(), 600.)

    def test_vgg16_discards_classifier_and_unfreezes_only_block5(self):
        model = build_regression_model("VGG16", (32, 32, 3), weights=None, train_backbone=False)
        self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertFalse(any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules()))
        self.assertEqual(model.head[0].in_features, 512 + 3)
        self.assertFalse(any("classifier" in name for name, _ in model.named_parameters()))
        with torch.no_grad():
            output = model(torch.zeros(2, 3, 32, 32, dtype=torch.uint8), torch.eye(3)[:2])
        self.assertEqual(output.shape, (2, 1))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())
        configure_fine_tuning(model, "VGG16")
        model.train()
        opened = {name.split(".")[0] for name, parameter in model.backbone.named_parameters() if parameter.requires_grad}
        self.assertEqual(opened, {"24", "26", "28"})
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.head.training)
        self.assertTrue(all(parameter.requires_grad for parameter in model.head.parameters()))

    def test_runner_rejects_test_data(self):
        with self.assertRaisesRegex(ValueError, "no test set"):
            train_experiment("ResNet50", "scratch", {"train": {}, "validation": {}, "test": {}}, {}, Path("unused"))

    def test_vgg19_backprop_updates_only_block5_and_head(self):
        model = build_regression_model("VGG19", (32, 32, 3), weights=None, train_backbone=False)
        self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertEqual(model.head[0].in_features, 512 + 3)
        self.assertFalse(any("classifier" in name for name, _ in model.named_parameters()))
        self.assertFalse(any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules()))
        configure_fine_tuning(model, "VGG19")
        model.train()
        output = model(torch.zeros(2, 3, 32, 32, dtype=torch.uint8), torch.eye(3)[:2])
        self.assertEqual(output.shape, (2, 1))
        self.assertEqual(output.dtype, torch.float32)
        output.square().mean().backward()
        opened = set()
        for name, parameter in model.backbone.named_parameters():
            if parameter.requires_grad:
                opened.add(name.split(".")[0])
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            else:
                self.assertIsNone(parameter.grad)
        self.assertEqual(opened, {"28", "30", "32", "34"})
        self.assertTrue(all(parameter.grad is not None for parameter in model.head.parameters()))


if __name__ == "__main__":
    unittest.main()
