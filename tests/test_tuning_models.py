import gc
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from torch import nn

from src.config import SEARCH_SPACE, TRAINING, tuning_candidates
from src.model_utils import (
    ARCHITECTURES, EXPERIMENTS, build_candidate_model, load_checkpoint,
)


class TuningConfigTests(unittest.TestCase):
    def test_grid_matches_agreement_without_changing_historical_defaults(self):
        candidates = tuning_candidates()
        self.assertEqual(SEARCH_SPACE["learning_rate"], (1e-6, 1e-5, 1e-4, 1e-3))
        self.assertEqual(SEARCH_SPACE["dropout"], (0.1, 0.3, 0.5))
        self.assertEqual(len(candidates), 12)
        self.assertEqual(len(set(candidates)), 12)
        for config in candidates:
            self.assertEqual(config.initial_learning_rate, config.fine_tuning_learning_rate)
            self.assertEqual(config.dropout_rates[0], config.dropout_rates[1])
            fixed = replace(config, initial_learning_rate=TRAINING.initial_learning_rate,
                            fine_tuning_learning_rate=TRAINING.fine_tuning_learning_rate,
                            dropout_rates=TRAINING.dropout_rates)
            self.assertEqual(fixed, TRAINING)
        self.assertEqual(TRAINING.initial_learning_rate, 1e-3)
        self.assertEqual(TRAINING.fine_tuning_learning_rate, 1e-5)
        self.assertEqual(TRAINING.weight_decay, 0)

    def test_invalid_candidates_fail_before_model_allocation(self):
        config = tuning_candidates()[0]
        with patch("src.model_utils.AHIRegressor") as constructor:
            for architecture, experiment in (("Unknown", "scratch"), ("ResNet50", "test")):
                with self.subTest(architecture=architecture, experiment=experiment):
                    with self.assertRaisesRegex(ValueError, "Invalid candidate"):
                        build_candidate_model(architecture, experiment, config)
            for changes in ({"initial_learning_rate": 1.0}, {"dropout_rates": (0.1, 0.5)},
                            {"fine_tuning_learning_rate": 1e-2}, {"seed": 99},
                            {"pretrained_weights": "DEFAULT"}, {"freeze_batch_normalization": False},
                            {"loss": "huber"}, {"weight_decay": 0.1}):
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "agreed grid"):
                    build_candidate_model("ResNet50", "fine_tuning", replace(config, **changes))
            constructor.assert_not_called()


class CandidateModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(TRAINING.intra_op_threads)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_all_strategies_start_fresh_and_update_only_allowed_parameters(self):
        config = tuning_candidates()[0]
        for architecture, (factory, weight_enum) in ARCHITECTURES.copy().items():
            # Unit test tidak mengunduh bobot; probe lokal terpisah memeriksa bobot ImageNet asli.
            offline_factory = Mock(side_effect=lambda weights, progress: factory(weights=None, progress=progress))
            initial_backbone = initial_head = None
            with patch.dict(ARCHITECTURES, {architecture: (offline_factory, weight_enum)}), \
                    patch("src.model_utils.load_checkpoint", side_effect=AssertionError("No parent checkpoint")):
                for experiment in EXPERIMENTS:
                    with self.subTest(architecture=architecture, experiment=experiment):
                        torch.manual_seed(999)
                        model = build_candidate_model(architecture, experiment, config)
                        offline_factory.assert_called_with(
                            weights=None if experiment == "scratch" else weight_enum.IMAGENET1K_V1,
                            progress=False,
                        )
                        if initial_backbone is None:
                            initial_backbone = {name: value.clone() for name, value in model.backbone.state_dict().items()}
                            initial_head = {name: value.clone() for name, value in model.head.state_dict().items()}
                        else:
                            for name, value in model.backbone.state_dict().items():
                                self.assertTrue(torch.equal(value, initial_backbone[name]), name)
                            for name, value in model.head.state_dict().items():
                                self.assertTrue(torch.equal(value, initial_head[name]), name)
                        self.assertEqual([module.p for module in model.head if isinstance(module, nn.Dropout)], [0.1, 0.1])
                        model.train()
                        batch_norm = [module for module in model.backbone.modules() if isinstance(module, nn.BatchNorm2d)]
                        for name, parameter in model.backbone.named_parameters():
                            if experiment == "scratch":
                                expected = True
                            elif experiment == "feature_extraction":
                                expected = False
                            elif architecture == "ResNet50":
                                expected = name.startswith("layer4.") and ".bn" not in name and ".downsample.1." not in name
                            else:
                                expected = int(name.split(".")[0]) >= (24 if architecture == "VGG16" else 28)
                            self.assertEqual(parameter.requires_grad, expected, name)
                        if experiment != "scratch":
                            self.assertTrue(all(not module.training for module in batch_norm))
                            self.assertTrue(all(not parameter.requires_grad for module in batch_norm for parameter in module.parameters()))
                        output = model(torch.zeros(2, 3, 32, 32, dtype=torch.uint8), torch.eye(3)[:2])
                        self.assertEqual(output.shape, (2, 1))
                        self.assertEqual(output.dtype, torch.float32)
                        self.assertTrue(torch.isfinite(output).all())
                        output.square().mean().backward()
                        for name, parameter in model.backbone.named_parameters():
                            self.assertEqual(parameter.grad is not None, parameter.requires_grad, name)
                            if parameter.grad is not None:
                                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        self.assertTrue(all(parameter.grad is not None for parameter in model.head.parameters()))
                        with torch.no_grad():
                            for parameter in model.parameters():
                                parameter.add_(1)
                        del model, output
                        gc.collect()

    def test_old_checkpoint_without_weight_decay_still_loads(self):
        config = tuning_candidates()[0]
        model = build_candidate_model("ResNet50", "scratch", config).eval()
        historical_config = asdict(TRAINING)
        historical_config.pop("weight_decay")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            torch.save({"architecture": "ResNet50", "experiment": "scratch",
                        "config": historical_config, "model_state_dict": model.state_dict()}, path)
            loaded = load_checkpoint(path)
            self.assertEqual(loaded.config.weight_decay, 0)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
