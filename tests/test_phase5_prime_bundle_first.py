"""Focused tests for the Phase 5' bundle-first diagnostic harness."""

from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TestReplayObservedPatternContextTrace(unittest.TestCase):
    def setUp(self):
        self.mod = importlib.import_module("experiments.44_phase5_prime_bundle_first")

    def _write_snapshot(self, path: Path, *, n_rows: int, dim: int) -> None:
        patterns = torch.stack(
            [
                torch.full((dim,), complex(i + 1, 0.0), dtype=torch.complex64)
                for i in range(n_rows)
            ],
            dim=0,
        )
        rows = [
            [(0, (i * 2) % 32), (1, (i * 2 + 1) % 32)]
            for i in range(n_rows)
        ]
        torch.save({"patterns": patterns, "pattern_encoder_terms": rows}, path)

    def test_load_passive_rows_can_return_aligned_pattern_tokens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snapshot.pt"
            self._write_snapshot(path, n_rows=3, dim=8)

            rows, support, tokens = self.mod._load_passive_trace_context_rows(
                path,
                N=3,
                K_roles=4,
                C_codebook=64,
                context_roles=2,
                generator=torch.Generator(device="cpu").manual_seed(17),
                return_pattern_tokens=True,
            )

        self.assertEqual(support["eligible_rows"], 3)
        self.assertEqual(support["used_rows"], 3)
        self.assertIsNotNone(tokens)
        assert tokens is not None
        self.assertEqual(tuple(tokens.shape), (3, 8))
        for row, token in zip(rows, tokens):
            original_idx = int(row[0] // 2)
            self.assertTrue(torch.allclose(token, torch.full_like(token, original_idx + 1)))

    def test_query_context_uses_selected_snapshot_pattern_tokens(self):
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        fhrr = TorchFHRR(dim=16, seed=3, device="cpu")
        roles = fhrr.random_vectors(4)
        content = fhrr.random_vectors(32)
        filler_tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
        passive_tokens = fhrr.random_vectors(2)

        tokens = self.mod._query_context_tokens(
            fhrr,
            roles,
            content,
            filler_tensor,
            scene_idx=torch.tensor([1, 0], dtype=torch.long),
            known_role=torch.tensor([0, 0], dtype=torch.long),
            query_role=torch.tensor([2, 2], dtype=torch.long),
            source="replay_observed_pattern_context_trace",
            context_roles=2,
            observed_role_plan=[[0, 1], [0, 1]],
            passive_context_tokens=passive_tokens,
        )

        self.assertIsNotNone(tokens)
        assert tokens is not None
        self.assertTrue(torch.allclose(tokens[0], passive_tokens[1]))
        self.assertTrue(torch.allclose(tokens[1], passive_tokens[0]))

    def test_run_cell_accepts_pattern_context_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snapshot.pt"
            self._write_snapshot(path, n_rows=8, dim=64)

            result = self.mod.run_cell(
                condition="candidate",
                D=64,
                N=8,
                K_roles=4,
                cue_noise=0.0,
                scene_token=True,
                scene_token_weight=0.25,
                scene_token_source="replay_observed_pattern_context_trace",
                scene_token_pool_size=0,
                context_roles=2,
                cooccurrence="skewed",
                seed=17,
                n_queries=8,
                beta=30.0,
                C_codebook=64,
                device="cpu",
                context_trace_snapshot=str(path),
            )

        self.assertEqual(result.passive_trace_rows_available, 8)
        self.assertEqual(result.passive_trace_rows_used, 8)
        self.assertEqual(result.passive_trace_rows_invalid, 0)
        self.assertEqual(result.passive_trace_rows_too_short, 0)


if __name__ == "__main__":
    unittest.main()
