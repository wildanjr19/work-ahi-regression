import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import nbformat

from src.config import PROJECT_ROOT
from src.tuning import run_grid_search


class ModelNotebookTests(unittest.TestCase):
    def notebooks(self):
        for architecture, number in (("ResNet50", "02"), ("VGG16", "03"), ("VGG19", "04")):
            path = PROJECT_ROOT / "notebooks" / f"{number}_{architecture}.ipynb"
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            yield architecture, notebook

    def test_disabled_execution_is_read_only_and_does_not_load_data(self):
        for architecture, notebook in self.notebooks():
            with self.subTest(architecture=architecture), redirect_stdout(io.StringIO()), \
                    patch("IPython.display.display"), \
                    patch("src.tuning.run_grid_search", wraps=run_grid_search) as runner, \
                    patch("src.tuning.load_development_data", side_effect=AssertionError("No dataset access")) as loader, \
                    patch.object(Path, "mkdir", side_effect=AssertionError("No output writes")):
                namespace = {}
                for cell in notebook.cells:
                    if cell.cell_type == "code":
                        self.assertEqual(cell.outputs, [])
                        self.assertIsNone(cell.execution_count)
                        exec(compile(cell.source, f"{architecture}/{cell.id}", "exec"), namespace)
                        if cell.id == "configuration":
                            namespace.update(RUN_TUNING=False, RUN_MODE="smoke", FULL_TRAINING_APPROVED=False)
                self.assertEqual(namespace["summary"]["status"], "disabled")
                self.assertFalse(namespace["RUN_TUNING"])
                self.assertFalse(namespace["FULL_TRAINING_APPROVED"])
                runner.assert_called_once_with(
                    architecture, output_root=namespace["OUTPUT_ROOT"], mode="smoke",
                    enabled=False, full_training_approved=False)
                loader.assert_not_called()

    def test_enabling_full_mode_does_not_grant_training_approval(self):
        for architecture, notebook in self.notebooks():
            with self.subTest(architecture=architecture), redirect_stdout(io.StringIO()), \
                    patch("IPython.display.display"), \
                    patch("src.tuning.load_development_data", side_effect=AssertionError("No dataset access")) as loader:
                namespace = {}
                cells = {cell.id: cell.source for cell in notebook.cells if cell.cell_type == "code"}
                exec(cells["imports"], namespace)
                exec(cells["configuration"], namespace)
                namespace.update(RUN_TUNING=True, RUN_MODE="full", FULL_TRAINING_APPROVED=False)
                with self.assertRaisesRegex(PermissionError, "approval"):
                    exec(cells["run-tuning"], namespace)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
