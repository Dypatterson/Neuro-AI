"""Tests for the strict discriminator precommit helpers."""

from __future__ import annotations

import unittest


class TestStrictDiscriminatorPrecommit(unittest.TestCase):

    def test_selector_keeps_role_target_content_miss_rows(self):
        from scripts.phase5_prime_strict_discriminator_precommit import (
            STRICT_DISCRIMINATOR_ID,
            select_strict_disagreement_rows,
        )

        rows = [
            {
                "probe_index": 0,
                "target_in_role_topk": True,
                "target_in_content_topk": True,
            },
            {
                "probe_index": 1,
                "target_in_role_topk": True,
                "target_in_content_topk": False,
            },
            {
                "probe_index": 2,
                "target_in_role_topk": False,
                "target_in_content_topk": False,
            },
            {
                "probe_index": 3,
                "target_in_role_topk": True,
                "target_in_content_topk": False,
            },
        ]

        selected = select_strict_disagreement_rows(rows, max_probes=2)

        self.assertEqual(
            STRICT_DISCRIMINATOR_ID,
            "role_topk_target_content_topk_excludes_target_v1",
        )
        self.assertEqual([row["probe_index"] for row in selected], [1, 3])


if __name__ == "__main__":
    unittest.main()
