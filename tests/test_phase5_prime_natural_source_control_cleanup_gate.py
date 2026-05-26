"""Tests for the natural source/control cleanup gate helpers."""

from __future__ import annotations

import unittest


class TestNaturalSourceControlCleanupGate(unittest.TestCase):

    def test_fixedpoint_free_shuffle_has_no_fixed_points(self):
        from scripts.phase5_prime_natural_source_control_cleanup_gate import (
            _fixedpoint_free_shuffle,
        )

        for seed in [17, 11, 23, 1, 2]:
            perm = _fixedpoint_free_shuffle(seed, 16).tolist()
            self.assertEqual(sorted(perm), list(range(16)))
            self.assertTrue(all(role != mapped for role, mapped in enumerate(perm)))

    def test_protocol_payload_rejects_failing_protocol(self):
        from scripts.phase5_prime_natural_source_control_cleanup_gate import (
            _protocol_payload,
        )

        payload = {
            "recommended_protocol": "bad",
            "protocols": [
                {"protocol_name": "bad", "passes_all_criteria": False},
            ],
        }
        with self.assertRaisesRegex(ValueError, "does not pass"):
            _protocol_payload(payload, None)


if __name__ == "__main__":
    unittest.main()
