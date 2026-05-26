"""Tests for the natural source/control cleanup preflight helpers."""

from __future__ import annotations

import unittest


class TestNaturalSourceControlCleanupPreflight(unittest.TestCase):

    def test_protocol_requires_non_special_unique_targets_and_zero_opportunity(self):
        from energy_memory.phase5.natural_source_protocol import (
            eligible_triples_for_seed,
            same_scene_opportunities,
            select_queries,
        )

        rows = [
            [0, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 9, 10, 11],
        ]
        selected_context_roles = [[0, 1], [0, 1], [0, 1]]
        eligible = eligible_triples_for_seed(
            rows=rows,
            selected_context_roles=selected_context_roles,
            k_roles=4,
            cap=None,
        )

        # scene 0 role 2/3 and scene 1 role 2/3 pass. Scene 2 role 2/3 pass,
        # while the duplicated atom 9 in roles 0/1 is never queried because
        # those roles are observed.
        self.assertEqual(len(eligible), 12)
        self.assertTrue(all(item["target_atom"] not in {0, 1} for item in eligible))

        selected = select_queries(
            seed=17,
            eligible=eligible,
            n_queries=4,
            cap=None,
            k_roles=4,
            n_rows=len(rows),
        )
        opportunities = same_scene_opportunities(
            rows=rows,
            selected=selected,
            seed=17,
            k_roles=4,
        )
        self.assertEqual(opportunities["random_exact_rate"], 0.0)
        self.assertEqual(opportunities["deranged_exact_rate"], 0.0)
        self.assertEqual(opportunities["fixedpoint_free_shuffled_exact_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
