"""Regression checks for the curated run configuration and comparison guard.

Run with ``python3 -m unittest discover -s tests -v``. Model setup and data
selection are exercised; optimization is intercepted before any training.
"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bending_with_distributed_loading as unified


class TrainingIntercepted(Exception):
    """Stop after the production code has prepared the optimizer inputs."""


def parse_cli(*options):
    with patch.object(sys, "argv", [str(unified.__file__), *options]):
        return unified.parse_arguments()


class CuratedConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(unified.TORCH_DTYPE)
        cls.baseline = json.loads(
            (unified.ROOT / "results" / "summary.json").read_text()
        )["results"]

    @classmethod
    def tearDownClass(cls):
        torch.set_default_dtype(cls.previous_dtype)

    def training_inputs(self, arguments, distance, source=None, data_free=False):
        if source is None and not data_free:
            source = unified.ROOT / unified.EXPERIMENTS[int(distance)]
        with patch.object(unified, "train", side_effect=TrainingIntercepted) as train:
            with self.assertRaises(TrainingIntercepted):
                unified.run_case(
                    float(distance), unified.PaperProperties(), source, arguments
                )
        return train.call_args.kwargs

    def test_all_and_individual_cases_use_saved_settings_and_initialization(self):
        all_arguments = parse_cli("--case", "all")
        for expected in self.baseline:
            distance = int(expected["distance_mm"])
            with self.subTest(distance=distance):
                together = self.training_inputs(all_arguments, distance)
                alone = self.training_inputs(parse_cli("--case", str(distance)), distance)
                for name in ("physics_weight", "closure_weight", "data_weight"):
                    self.assertEqual(together[name], expected[name])
                    self.assertEqual(alone[name], expected[name])
                self.assertEqual(len(together["xi_data"]), expected["training_points"])
                self.assertEqual(together["model"].support, expected["support"])
                self.assertIs(type(together["model"]), unified.PaperPINN)
                self.assertEqual(together["adam_epochs"], 20000)
                self.assertEqual(together["lbfgs_iterations"], 2000)
                self.assertEqual(together["collocation_points"], 160)
                for name in ("xi_data", "xy_data_m"):
                    np.testing.assert_array_equal(together[name], alone[name])
                for name, value in together["model"].state_dict().items():
                    self.assertTrue(torch.equal(value, alone["model"].state_dict()[name]))
        # Running 150 mm in the middle of the batch must not mutate batch defaults.
        self.assertIsNone(all_arguments.physics_weight)
        self.assertIsNone(all_arguments.closure_weight)

    def test_explicit_weights_including_zero_take_precedence(self):
        scenarios = (
            (75, ("--physics-weight", "3", "--closure-weight", "7"), (3, 7)),
            (150, ("--physics-weight", "0", "--closure-weight", "0"), (0, 0)),
            (150, ("--physics-weight", "3"), (3, 100)),
            (150, ("--closure-weight", "0"), (100, 0)),
        )
        for distance, options, expected in scenarios:
            with self.subTest(distance=distance, options=options):
                inputs = self.training_inputs(parse_cli(*options), distance)
                self.assertEqual(
                    (inputs["physics_weight"], inputs["closure_weight"]), expected
                )

    def test_new_150mm_distance_does_not_inherit_calibrated_weights(self):
        inputs = self.training_inputs(
            parse_cli("--distance-mm", "150"), 150, data_free=True
        )
        self.assertEqual((inputs["physics_weight"], inputs["closure_weight"]), (1, 0))
        self.assertEqual(inputs["model"].support, "pinned")
        self.assertEqual(len(inputs["xi_data"]), 0)
        self.assertEqual(inputs["data_weight"], 0)

    def test_five_experiments_keep_original_weights_and_accept_overrides(self):
        actual_run_case = unified.run_case

        def prepare_variant(*args):
            with self.assertRaises(TrainingIntercepted):
                actual_run_case(*args)
            return {"rmse_m": 0.0}

        for options, weights in (((), (1, 0)), (("--physics-weight", "6", "--closure-weight", "8"), (6, 8))):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                arguments = parse_cli("--experiment-150mm", "--output-dir", directory, *options)
                with (
                    patch.object(unified, "run_case", side_effect=prepare_variant),
                    patch.object(unified, "train", side_effect=TrainingIntercepted) as train,
                    redirect_stdout(io.StringIO()),
                ):
                    unified.run_experiment_150mm(arguments, unified.PaperProperties())
                self.assertEqual(train.call_count, 5)
                self.assertEqual([len(c.kwargs["xi_data"]) for c in train.call_args_list], [2, 2, 4, 2, 2])
                self.assertEqual(
                    [type(c.kwargs["model"]) for c in train.call_args_list],
                    [unified.PaperPINN] * 3 + [unified.FreeYPaperPINN] * 2,
                )
                for call in train.call_args_list:
                    inputs = call.kwargs
                    self.assertEqual((inputs["physics_weight"], inputs["closure_weight"]), weights)


class BaselineVerificationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.baseline = unified.ROOT / "results"
        self.reset_outputs()

    def reset_outputs(self):
        shutil.copyfile(self.baseline / "summary.json", self.output / "summary.json")
        for path in self.baseline.glob("shape_*.csv"):
            shutil.copyfile(path, self.output / path.name)

    def verify(self):
        with redirect_stdout(io.StringIO()):
            return unified.verify_saved_results(self.output)

    def test_identical_outputs_pass_all_six_cases(self):
        report = self.verify()
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["cases"]), 6)
        self.assertTrue(all(case["csv_exact"] for case in report["cases"]))
        self.assertEqual(report, json.loads((self.output / "verification.json").read_text()))

    def test_single_case_output_is_valid(self):
        path = self.output / "summary.json"
        summary = json.loads(path.read_text())
        summary["results"] = [row for row in summary["results"] if row["distance_mm"] == 150]
        path.write_text(json.dumps(summary))
        report = self.verify()
        self.assertTrue(report["passed"])
        self.assertEqual([row["distance_mm"] for row in report["cases"]], [150])

    def test_corrupted_coordinates_angles_grid_and_nan_are_rejected(self):
        path = self.output / "shape_150.csv"
        for column, increment in ((0, 1e-3), (1, 1e-3), (2, 1e-3), (3, 1e-3), (1, np.nan)):
            with self.subTest(column=column, increment=increment):
                self.reset_outputs()
                values = np.loadtxt(path, delimiter=",", skiprows=1)
                values[250, column] += increment
                np.savetxt(path, values, delimiter=",", header="xi,x_m,y_m,phi_rad", comments="")
                with self.assertRaisesRegex(RuntimeError, "differ from the curated baseline"):
                    self.verify()
                report = json.loads((self.output / "verification.json").read_text())
                self.assertFalse(report["passed"])
                failed = [row["distance_mm"] for row in report["cases"] if not row["passed"]]
                self.assertEqual(failed, [150])

    def test_corrupted_metrics_and_configuration_are_rejected(self):
        path = self.output / "summary.json"
        for field, value in (("rmse_m", 0.05), ("physics_weight", 1), ("support", "pinned")):
            with self.subTest(field=field):
                self.reset_outputs()
                summary = json.loads(path.read_text())
                next(row for row in summary["results"] if row["distance_mm"] == 150)[field] = value
                path.write_text(json.dumps(summary))
                with self.assertRaises(RuntimeError):
                    self.verify()
                report = json.loads((self.output / "verification.json").read_text())
                case = next(row for row in report["cases"] if row["distance_mm"] == 150)
                self.assertFalse(case["passed"])
                self.assertIn(field, case["metric_mismatches"])

    def test_changed_paper_properties_are_rejected(self):
        path = self.output / "summary.json"
        summary = json.loads(path.read_text())
        summary["paper_properties"]["length"] = 0.3
        path.write_text(json.dumps(summary))
        with self.assertRaises(RuntimeError):
            self.verify()
        self.assertFalse(json.loads((self.output / "verification.json").read_text())["passed"])

    def test_truncated_curve_is_rejected(self):
        path = self.output / "shape_150.csv"
        values = np.loadtxt(path, delimiter=",", skiprows=1)[:-1]
        np.savetxt(path, values, delimiter=",", header="xi,x_m,y_m,phi_rad", comments="")
        with self.assertRaises(RuntimeError):
            self.verify()
        report = json.loads((self.output / "verification.json").read_text())
        case = next(row for row in report["cases"] if row["distance_mm"] == 150)
        self.assertFalse(case["passed"])
        self.assertIsNone(case["max_abs_xy_difference_m"])


if __name__ == "__main__":
    unittest.main()
