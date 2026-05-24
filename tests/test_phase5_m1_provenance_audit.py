"""Tests for Phase 5 M1 provenance audit script."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _build_snapshot(
    path: Path,
    terms=None,
    kinds=None,
    *,
    n_roles=3,
    n_atoms=3,
    metadata=None,
):
    from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
    from energy_memory.phase2.encoding import build_position_vectors
    from energy_memory.phase4.consolidation import (
        ConsolidationConfig,
        ConsolidationState,
    )
    from energy_memory.phase4.snapshot import save_substrate_snapshot
    from energy_memory.substrate.torch_fhrr import TorchFHRR

    substrate = TorchFHRR(dim=64, seed=17, device="cpu")
    mem = TorchHopfieldMemory(substrate)
    cons = ConsolidationState(ConsolidationConfig(m=4), device="cpu")
    for i in range(n_atoms):
        mem.store(substrate.random_vector(), label=f"row_{i}")
        cons.add_pattern()
    positions = build_position_vectors(substrate, n_roles)
    snapshot_metadata = {"mask_token_id": 0}
    if metadata:
        snapshot_metadata.update(metadata)
    save_substrate_snapshot(
        memory=mem,
        consolidation=cons,
        path=path,
        metadata=snapshot_metadata,
        positions=positions,
        pattern_encoder_terms=terms,
        pattern_encoder_term_kinds=kinds,
    )


def _build_encoded_snapshot(path: Path):
    from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
    from energy_memory.phase2.encoding import build_position_vectors, encode_window
    from energy_memory.phase4.consolidation import (
        ConsolidationConfig,
        ConsolidationState,
    )
    from energy_memory.phase4.snapshot import save_substrate_snapshot
    from energy_memory.substrate.torch_fhrr import TorchFHRR

    substrate = TorchFHRR(dim=256, seed=41, device="cpu")
    positions = build_position_vectors(substrate, 3)
    codebook = substrate.random_vectors(40)
    windows = [
        (0, 10, 11),
        (0, 12, 13),
        (1, 2, 14),
        (3, 2, 15),
        (4, 16, 3),
        (5, 17, 3),
    ]
    mem = TorchHopfieldMemory(substrate)
    cons = ConsolidationState(ConsolidationConfig(m=4), device="cpu")
    terms = []
    for idx, window in enumerate(windows):
        mem.store(encode_window(substrate, positions, codebook, window), label=f"row_{idx}")
        cons.add_pattern()
        terms.append([(0, window[0]), (1, window[1]), (2, window[2])])
    save_substrate_snapshot(
        memory=mem,
        consolidation=cons,
        path=path,
        metadata={"mask_token_id": 99},
        positions=positions,
        pattern_encoder_terms=terms,
        pattern_encoder_term_kinds=["source_window"] * len(windows),
    )
    return codebook


def _build_role_specialized_encoded_snapshot(path: Path, *, seed=17):
    from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
    from energy_memory.phase2.encoding import build_position_vectors, encode_window
    from energy_memory.phase4.consolidation import (
        ConsolidationConfig,
        ConsolidationState,
    )
    from energy_memory.phase4.snapshot import save_substrate_snapshot
    from energy_memory.substrate.torch_fhrr import TorchFHRR

    substrate = TorchFHRR(dim=256, seed=123, device="cpu")
    positions = build_position_vectors(substrate, 3)
    codebook = substrate.random_vectors(90)
    bases = substrate.random_vectors(3)
    cluster_ids = []
    for role_index in range(3):
        ids = list(range(role_index * 8, (role_index + 1) * 8))
        cluster_ids.append(ids)
        for token_id in ids:
            codebook[token_id] = bases[role_index]

    mem = TorchHopfieldMemory(substrate)
    cons = ConsolidationState(ConsolidationConfig(m=4), device="cpu")
    terms = []
    for row_index in range(9):
        special_role = row_index % 3
        tokens = []
        for role_index in range(3):
            if role_index == special_role:
                tokens.append(cluster_ids[role_index][row_index // 3])
            else:
                tokens.append(24 + row_index * 3 + role_index)
        window = tuple(tokens)
        mem.store(encode_window(substrate, positions, codebook, window), label=f"row_{row_index}")
        cons.add_pattern()
        terms.append([(0, window[0]), (1, window[1]), (2, window[2])])

    save_substrate_snapshot(
        memory=mem,
        consolidation=cons,
        path=path,
        metadata={"mask_token_id": 99, "seed": seed},
        positions=positions,
        pattern_encoder_terms=terms,
        pattern_encoder_term_kinds=["source_window"] * len(terms),
    )
    return codebook


@unittest.skipIf(torch is None, "torch required")
class TestM1ProvenanceAudit(unittest.TestCase):

    def test_old_snapshot_fails_missing_provenance(self):
        from scripts.phase5_m1_provenance_audit import audit_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            _build_snapshot(path)
            payload = audit_snapshot(path, weight_source="count")
            self.assertEqual(payload["status"], "fail")
            self.assertIn("missing_pattern_encoder_terms", payload["failure_reasons"])

    def test_uniform_role_rows_fail_degenerate(self):
        from scripts.phase5_m1_provenance_audit import audit_snapshot

        terms = [
            [(0, 1), (1, 2), (2, 3)],
            [(0, 4), (1, 5), (2, 6)],
            [(0, 7), (1, 8), (2, 9)],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            _build_snapshot(
                path,
                terms=terms,
                kinds=["source_window"] * 3,
                n_roles=3,
            )
            payload = audit_snapshot(path, weight_source="count")
            self.assertEqual(payload["status"], "fail")
            self.assertIn("mean_role_entropy_degenerate", payload["failure_reasons"])
            self.assertIn("uniform_role_rows_degenerate", payload["failure_reasons"])

    def test_default_weight_source_is_count_and_reports_geometric_probe(self):
        from scripts.phase5_m1_provenance_audit import audit_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            _build_encoded_snapshot(path)
            payload = audit_snapshot(path)
            self.assertEqual(payload["weight_source"], "count")
            self.assertEqual(payload["status"], "fail")
            self.assertIn("mean_role_entropy_degenerate", payload["failure_reasons"])
            self.assertIsNotNone(payload["geometric_entropy"])
            self.assertIsNotNone(payload["geometric_score_range"])
            self.assertIn("row_spread_mean", payload["geometric_score_range"])

    def test_row_count_mismatch_fails(self):
        from scripts.phase5_m1_provenance_audit import audit_snapshot

        terms = [[(0, 1)], [(1, 2)], [(2, 3)]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            _build_snapshot(
                path,
                terms=terms,
                kinds=["source_window"] * 3,
                n_roles=3,
            )
            state = torch.load(path, map_location="cpu", weights_only=False)
            state["pattern_encoder_terms"] = state["pattern_encoder_terms"][:2]
            torch.save(state, path)
            payload = audit_snapshot(path, weight_source="count")
            self.assertEqual(payload["status"], "fail")
            self.assertIn(
                "pattern_encoder_terms_row_count_mismatch",
                payload["failure_reasons"],
            )

    def test_non_degenerate_synthetic_snapshot_passes(self):
        from scripts.phase5_m1_provenance_audit import audit_snapshot

        terms = [
            [(0, 1), (0, 2), (0, 3)],
            [(1, 4), (1, 5), (1, 6)],
            [(2, 7), (2, 8), (2, 9)],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            _build_snapshot(
                path,
                terms=terms,
                kinds=["source_window"] * 3,
                n_roles=3,
            )
            payload = audit_snapshot(path, weight_source="count")
            self.assertEqual(payload["status"], "pass")
            self.assertEqual(payload["role_coverage"], 3)

    def test_geometric_audit_can_pass_uniform_count_rows_with_k1_smallest_case_only(self):
        from scripts.phase5_m1_provenance_audit import audit_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            _build_encoded_snapshot(path)
            payload = audit_snapshot(
                path,
                weight_source="geometric",
                geometric_neighbor_k=1,
            )
            self.assertEqual(payload["status"], "pass")
            self.assertIn("count_role_weights_degenerate", payload["warnings"])
            self.assertTrue(payload["count_degeneracy_reasons"])
            self.assertLess(payload["entropy"]["mean_normalized"], 0.95)

    def test_codebook_prior_audit_requires_codebook_kwarg(self):
        from scripts.phase5_m1_provenance_audit import audit_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            _build_role_specialized_encoded_snapshot(path)
            payload = audit_snapshot(
                path,
                weight_source="geometric",
                geometric_mode="codebook_prior_density",
            )

            self.assertEqual(payload["status"], "fail")
            self.assertIn("codebook_required_for_mode", payload["failure_reasons"])
            self.assertIsNone(payload["geometric_entropy"])
            self.assertEqual(
                payload["geometric_config"]["active_mode"],
                "codebook_prior_density",
            )

    def test_codebook_prior_audit_validates_codebook_dim(self):
        from energy_memory.phase2.persistence import save_codebook
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        from scripts.phase5_m1_provenance_audit import audit_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            path = tmp_path / "snap.pt"
            codebook_path = tmp_path / "bad_codebook.pt"
            _build_role_specialized_encoded_snapshot(path)
            substrate = TorchFHRR(dim=128, seed=91, device="cpu")
            save_codebook(substrate.random_vectors(8), codebook_path)

            payload = audit_snapshot(
                path,
                weight_source="geometric",
                geometric_mode="codebook_prior_density",
                codebook_path=codebook_path,
            )

            self.assertEqual(payload["status"], "fail")
            self.assertIn("codebook_dim_mismatch", payload["failure_reasons"])
            self.assertEqual(payload["geometric_config"]["codebook_shape"], [8, 128])
            self.assertIsNotNone(payload["geometric_config"]["codebook_fingerprint"])

    def test_codebook_prior_audit_uses_explicit_codebook(self):
        from energy_memory.phase2.persistence import save_codebook
        from scripts.phase5_m1_provenance_audit import audit_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            path = tmp_path / "snap.pt"
            codebook_path = tmp_path / "codebook.pt"
            codebook = _build_role_specialized_encoded_snapshot(path)
            save_codebook(codebook, codebook_path)

            payload = audit_snapshot(
                path,
                weight_source="geometric",
                geometric_mode="codebook_prior",
                codebook_path=codebook_path,
            )

            self.assertEqual(payload["status"], "pass")
            self.assertIn("count_role_weights_degenerate", payload["warnings"])
            self.assertLess(payload["entropy"]["mean_normalized"], 0.1)
            self.assertEqual(
                payload["geometric_config"]["active_mode"],
                "codebook_prior_density",
            )
            self.assertEqual(payload["geometric_config"]["codebook_shape"], [90, 256])
            self.assertIsNotNone(payload["geometric_config"]["codebook_fingerprint"])

    def test_seed17_evidence_scope_reported_in_payload_and_markdown(self):
        from scripts.phase5_m1_provenance_audit import audit_snapshot, write_report

        terms = [
            [(0, 1), (0, 2), (0, 3)],
            [(1, 4), (1, 5), (1, 6)],
            [(2, 7), (2, 8), (2, 9)],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            path = tmp_path / "snap.pt"
            report = tmp_path / "audit.md"
            _build_snapshot(
                path,
                terms=terms,
                kinds=["source_window"] * 3,
                n_roles=3,
                metadata={"seed": 17},
            )
            payload = audit_snapshot(path, weight_source="count")
            write_report(payload, report)

            self.assertFalse(
                payload["evidence_scope"]["representative_phase5_evidence"]
            )
            self.assertIn("Seed 17 is wiring/provenance/degen smoke only", report.read_text())

    def test_many_empty_rows_fail_count_source(self):
        from scripts.phase5_m1_provenance_audit import audit_snapshot

        terms = [
            [],
            [],
            [(0, 1), (0, 2)],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            _build_snapshot(
                path,
                terms=terms,
                kinds=["source_window"] * 3,
                n_roles=3,
            )
            payload = audit_snapshot(path, weight_source="count")
            self.assertEqual(payload["status"], "fail")
            self.assertIn("empty_role_rows_degenerate", payload["failure_reasons"])

    def test_mask_token_participation_is_reported(self):
        from scripts.phase5_m1_provenance_audit import audit_snapshot

        terms = [
            [(0, 0), (0, 2)],
            [(1, 4), (1, 5)],
            [(2, 7), (2, 8)],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.pt"
            _build_snapshot(
                path,
                terms=terms,
                kinds=["replay_query", "source_window", "source_window"],
                n_roles=3,
            )
            payload = audit_snapshot(path)
            self.assertEqual(payload["term_checks"]["mask_token_terms"], 1)
            self.assertIn(
                "mask_token_participates_in_row_provenance",
                payload["warnings"],
            )


if __name__ == "__main__":
    unittest.main()
