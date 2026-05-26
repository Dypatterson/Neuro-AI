"""Unit tests for theta'(beta) calibration loader (C.1.4)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from energy_memory.phase3.theta_prime_calibration import (
    load_theta_prime_calibration,
)


def _write_synthetic_calibration(path: Path, calibration: dict) -> None:
    payload = {
        "metadata": {"date": "2026-05-26", "synthetic": True},
        "calibration": calibration,
        "approximation_baseline": {
            str(b): 1.0 / float(b) for b in calibration.keys()
        },
        "note": "synthetic test fixture",
    }
    with path.open("w") as fh:
        json.dump(payload, fh)


class ThetaPrimeCalibrationTests(unittest.TestCase):
    def test_t1_loader_returns_none_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "does_not_exist.json"
            fn = load_theta_prime_calibration(missing)
            self.assertIsNone(fn)

    def test_t2_exact_beta_lookup(self):
        calibration = {
            "0.01": {
                "theta_prime": 0.55,
                "success_curve": {},
                "boundary_above_grid": False,
                "boundary_below_grid": False,
            },
            "0.1": {
                "theta_prime": 0.30,
                "success_curve": {},
                "boundary_above_grid": False,
                "boundary_below_grid": False,
            },
            "1.0": {
                "theta_prime": 0.12,
                "success_curve": {},
                "boundary_above_grid": False,
                "boundary_below_grid": False,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calib.json"
            _write_synthetic_calibration(path, calibration)
            fn = load_theta_prime_calibration(path)
            self.assertIsNotNone(fn)
            assert fn is not None  # mypy
            self.assertAlmostEqual(fn(0.01), 0.55, places=6)
            self.assertAlmostEqual(fn(0.1), 0.30, places=6)
            self.assertAlmostEqual(fn(1.0), 0.12, places=6)

    def test_t3_log_beta_interpolation(self):
        # Two calibrated points; query at the log-midpoint.
        calibration = {
            "0.1": {
                "theta_prime": 0.30,
                "success_curve": {},
                "boundary_above_grid": False,
                "boundary_below_grid": False,
            },
            "1.0": {
                "theta_prime": 0.12,
                "success_curve": {},
                "boundary_above_grid": False,
                "boundary_below_grid": False,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calib.json"
            _write_synthetic_calibration(path, calibration)
            fn = load_theta_prime_calibration(path)
            self.assertIsNotNone(fn)
            assert fn is not None  # mypy
            # Query at a beta strictly between 0.1 and 1.0 — should land
            # strictly between 0.30 and 0.12.
            mid_value = fn(0.3162)  # ~ sqrt(0.1 * 1.0), log-midpoint
            self.assertLess(mid_value, 0.30)
            self.assertGreater(mid_value, 0.12)
            # And approximately the log-midpoint = 0.5 * (0.30 + 0.12) = 0.21.
            self.assertAlmostEqual(mid_value, 0.21, places=2)


if __name__ == "__main__":
    unittest.main()
