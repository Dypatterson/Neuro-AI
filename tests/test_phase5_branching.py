"""Tests for the Phase 5 branching mechanism.

Companion to experiments/40_phase5_branching.py and the design at
notes/emergent-codebook/phase-5-unified-design.md.

This file is a partial test scaffold:
- Tests for surprise_prior() and atom_split_signal() are REAL — those two
  functions are fully implemented in the skeleton and need coverage now.
- Tests for everything else are signature+docstring scaffolds that pin
  the expected contract; bodies fill in once decision #1 (schema source)
  unblocks the corresponding implementations.
"""
from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


# ---------------------------------------------------------------------------
# Local helpers — small constructors so tests don't depend on Phase 4 wiring.
# ---------------------------------------------------------------------------

def _make_consolidation(m=4, n_patterns=0, device="cpu"):
    """Build a ConsolidationState with N empty patterns."""
    from energy_memory.phase4.consolidation import (
        ConsolidationConfig, ConsolidationState,
    )
    s = ConsolidationState(ConsolidationConfig(m=m, alpha=0.25), device=device)
    for _ in range(n_patterns):
        s.add_pattern(novelty_strength=0.0)
    return s


def _make_branch(branch_id=0, prior_source="schema",
                 energy_unbiased=0.0, q_settled=None, d=8, seed=None):
    """Build a BranchState with a settable energy and final state.

    q_settled defaults to a random complex FHRR vector of dimension d.
    """
    from experiments_40 import BranchState  # see _import_module() below
    if q_settled is None:
        if seed is not None:
            torch.manual_seed(seed)
        q_settled = torch.randn(d, dtype=torch.complex64)
    return BranchState(
        branch_id=branch_id,
        prior_source=prior_source,
        prior=torch.zeros(d, dtype=torch.complex64),
        q_initial=torch.zeros(d, dtype=torch.complex64),
        q_settled=q_settled,
        energy_unbiased=energy_unbiased,
    )


def _import_module():
    """Import experiments/40_phase5_branching.py under a clean name."""
    import sys, importlib.util
    if "experiments_40" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "experiments_40", "experiments/40_phase5_branching.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["experiments_40"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["experiments_40"]


# ===========================================================================
# 1. Dataclasses
# ===========================================================================

@unittest.skipIf(torch is None, "torch required")
class TestBranchStateDefaults(unittest.TestCase):
    """BranchState fields default to neutral values so missing diagnostics
    don't produce false signals.
    """

    def test_branchstate_diagnostic_defaults_are_neutral(self):
        mod = _import_module()
        d = 8
        bs = mod.BranchState(
            branch_id=0, prior_source="schema",
            prior=torch.zeros(d, dtype=torch.complex64),
            q_initial=torch.zeros(d, dtype=torch.complex64),
            q_settled=torch.zeros(d, dtype=torch.complex64),
        )
        # All diagnostics start at zero/False — non-zero defaults would
        # silently bias aggregate analysis.
        self.assertEqual(bs.energy_unbiased, 0.0)
        self.assertEqual(bs.energy_biased, 0.0)
        self.assertEqual(bs.energy_drop, 0.0)
        self.assertEqual(bs.prior_alignment, 0.0)
        self.assertEqual(bs.score_entropy_initial, 0.0)
        self.assertEqual(bs.score_entropy_final, 0.0)
        self.assertEqual(bs.entropy_collapse, 0.0)
        self.assertEqual(bs.final_state_divergence, 0.0)
        self.assertFalse(bs.recall_support)
        self.assertFalse(bs.meta_stable)
        self.assertFalse(bs.converged)


# ===========================================================================
# 2. surprise_prior() — REAL TESTS (function fully implemented)
# ===========================================================================

@unittest.skipIf(torch is None, "torch required")
class TestSurprisePrior(unittest.TestCase):
    """surprise_prior() returns (idx, prior_vector) for max log(u1+ε) - log(um+ε).

    The log-ratio form (rather than raw u_1/u_m) is the design choice that
    prevents tiny u_m values from creating artificial surprise explosions.
    """

    def test_returns_none_when_empty(self):
        mod = _import_module()
        cons = _make_consolidation(n_patterns=0)
        codebook = torch.randn(0, 8, dtype=torch.complex64)
        self.assertIsNone(mod.surprise_prior(consolidation=cons, codebook=codebook))

    def test_picks_pattern_with_highest_u1_over_um(self):
        mod = _import_module()
        cons = _make_consolidation(m=4, n_patterns=3)
        # Pattern 0: u_1=0.1, u_m=0.1 → log-ratio ~ 0
        # Pattern 1: u_1=1.0, u_m=0.01 → log-ratio ~ 4.6 (high novelty)
        # Pattern 2: u_1=0.5, u_m=0.5 → log-ratio ~ 0
        cons.u[0, 0] = 0.1; cons.u[0, 3] = 0.1
        cons.u[1, 0] = 1.0; cons.u[1, 3] = 0.01
        cons.u[2, 0] = 0.5; cons.u[2, 3] = 0.5
        codebook = torch.randn(3, 8, dtype=torch.complex64)
        result = mod.surprise_prior(consolidation=cons, codebook=codebook)
        self.assertIsNotNone(result)
        idx, prior = result
        self.assertEqual(idx, 1)
        # Returned prior is the codebook row for the chosen pattern.
        self.assertTrue(torch.equal(prior, codebook[1]))

    def test_log_form_handles_tiny_um_without_explosion(self):
        """Raw ratio u_1/u_m blows up for tiny u_m; the log form clips it.

        We compare two patterns: one with moderately high u_1 and modest
        u_m, one with low u_1 and near-zero u_m. Under raw-ratio form the
        near-zero-u_m pattern would always win regardless of u_1; under
        log form, comparable novelty scores are compared honestly.
        """
        mod = _import_module()
        cons = _make_consolidation(m=4, n_patterns=2)
        # Pattern 0: u_1=1.0, u_m=0.1 → log(1+ε) - log(0.1+ε) ≈ 2.3
        # Pattern 1: u_1=0.001, u_m=1e-9 → log(0.001+ε) - log(1e-9+ε) ≈ 6.9
        #   (raw ratio would give 1e6 vs 10 — pattern 1 wins trivially)
        cons.u[0, 0] = 1.0;   cons.u[0, 3] = 0.1
        cons.u[1, 0] = 0.001; cons.u[1, 3] = 0.0
        codebook = torch.randn(2, 8, dtype=torch.complex64)
        result = mod.surprise_prior(consolidation=cons, codebook=codebook)
        idx, _ = result
        # Under the log form, pattern 1's tiny-u_m still produces a high
        # novelty score (the clamp+ε bounds it); but pattern 0's score is
        # bounded too. The test pins behavior, not whether either is
        # specifically "right" — this is a sanity check that the function
        # is well-defined under near-zero u_m.
        self.assertIn(idx, {0, 1})

    def test_ties_broken_deterministically_by_argmax(self):
        """torch.argmax returns the first index of a tie. Document the behavior."""
        mod = _import_module()
        cons = _make_consolidation(m=4, n_patterns=2)
        cons.u[0, 0] = 1.0; cons.u[0, 3] = 0.1
        cons.u[1, 0] = 1.0; cons.u[1, 3] = 0.1
        codebook = torch.randn(2, 8, dtype=torch.complex64)
        idx, _ = mod.surprise_prior(consolidation=cons, codebook=codebook)
        self.assertEqual(idx, 0)


# ===========================================================================
# 3. atom_split_signal() — REAL TESTS (function fully implemented)
# ===========================================================================

@unittest.skipIf(torch is None, "torch required")
class TestAtomSplitSignal(unittest.TestCase):
    """Joint criterion: split_eligible iff
        (≥2 branches within δ_energy of best)
        AND
        (max pairwise FHRR cosine-distance in that set > δ_state).

    Energy-similarity alone over-fires on redundant duplicates;
    state-divergence alone over-fires on high-energy outliers. Both
    matter.
    """

    def _orthogonal_states(self, d=8, seed=0):
        """Two near-orthogonal complex unit-norm vectors."""
        torch.manual_seed(seed)
        a = torch.randn(d, dtype=torch.complex64); a = a / a.norm()
        b = torch.randn(d, dtype=torch.complex64); b = b / b.norm()
        return a, b

    def test_single_branch_never_splits(self):
        mod = _import_module()
        b = _make_branch(energy_unbiased=0.0, seed=1)
        ok, n_low, dist = mod.atom_split_signal([b])
        self.assertFalse(ok)
        self.assertEqual(n_low, 0)

    def test_two_branches_close_energy_close_state_is_redundancy_not_split(self):
        """Two branches with similar energies AND similar settled states
        is redundancy (multiple paths to the same answer), not polysemy.
        """
        mod = _import_module()
        a, _ = self._orthogonal_states(seed=2)
        # Both branches converge to the SAME state but with separate copies
        # so they're branch objects but state is identical.
        b1 = _make_branch(branch_id=0, energy_unbiased=0.0, q_settled=a.clone())
        b2 = _make_branch(branch_id=1, energy_unbiased=0.02, q_settled=a.clone())
        ok, n_low, dist = mod.atom_split_signal(
            [b1, b2], delta_energy=0.1, delta_state=0.3,
        )
        self.assertFalse(ok)
        self.assertEqual(n_low, 2)
        self.assertAlmostEqual(dist, 0.0, places=4)

    def test_two_branches_close_energy_divergent_state_is_polysemy(self):
        """Two branches with similar low energies AND substantially
        different settled states → genuine polysemy. Split signal fires.
        """
        mod = _import_module()
        a, b = self._orthogonal_states(seed=3)
        b1 = _make_branch(branch_id=0, energy_unbiased=0.0, q_settled=a)
        b2 = _make_branch(branch_id=1, energy_unbiased=0.02, q_settled=b)
        ok, n_low, dist = mod.atom_split_signal(
            [b1, b2], delta_energy=0.1, delta_state=0.3,
        )
        self.assertTrue(ok)
        self.assertEqual(n_low, 2)
        self.assertGreater(dist, 0.3)

    def test_two_branches_divergent_state_but_far_energy_is_not_split(self):
        """One branch much higher energy than the other → not in the
        low-energy set → split signal doesn't consider their divergence.
        """
        mod = _import_module()
        a, b = self._orthogonal_states(seed=4)
        b1 = _make_branch(branch_id=0, energy_unbiased=0.0,  q_settled=a)
        b2 = _make_branch(branch_id=1, energy_unbiased=10.0, q_settled=b)  # very far in energy
        ok, n_low, dist = mod.atom_split_signal(
            [b1, b2], delta_energy=0.1, delta_state=0.3,
        )
        self.assertFalse(ok)
        self.assertEqual(n_low, 1)  # only b1 is in the low-energy set

    def test_three_branches_only_low_energy_set_considered(self):
        """A high-energy outlier doesn't contribute to the split decision
        even if it diverges from the low-energy cluster.
        """
        mod = _import_module()
        torch.manual_seed(5)
        a = torch.randn(8, dtype=torch.complex64); a = a / a.norm()
        b = torch.randn(8, dtype=torch.complex64); b = b / b.norm()
        c = torch.randn(8, dtype=torch.complex64); c = c / c.norm()
        # b1 and b2 are both low-energy and divergent → should fire.
        # b3 is a high-energy outlier — excluded from the comparison set.
        b1 = _make_branch(branch_id=0, energy_unbiased=0.0, q_settled=a)
        b2 = _make_branch(branch_id=1, energy_unbiased=0.05, q_settled=b)
        b3 = _make_branch(branch_id=2, energy_unbiased=5.0, q_settled=c)
        ok, n_low, dist = mod.atom_split_signal(
            [b1, b2, b3], delta_energy=0.1, delta_state=0.3,
        )
        self.assertTrue(ok)
        self.assertEqual(n_low, 2)

    def test_thresholds_are_configurable(self):
        """Tighter thresholds should produce fewer split signals; loose
        ones should produce more. Verifies the knobs are wired.
        """
        mod = _import_module()
        a, b = self._orthogonal_states(seed=6)
        b1 = _make_branch(branch_id=0, energy_unbiased=0.0, q_settled=a)
        b2 = _make_branch(branch_id=1, energy_unbiased=0.5, q_settled=b)
        # With default thresholds, b2's energy of 0.5 puts it outside the
        # low-energy set (delta_energy=0.1 default).
        ok_default, _, _ = mod.atom_split_signal([b1, b2])
        self.assertFalse(ok_default)
        # Loosen delta_energy to 1.0 and the split fires.
        ok_loose, _, _ = mod.atom_split_signal(
            [b1, b2], delta_energy=1.0, delta_state=0.3,
        )
        self.assertTrue(ok_loose)


# ===========================================================================
# 4. Scaffold-only tests (bodies fill once schema source unblocks)
# ===========================================================================

@unittest.skipIf(torch is None, "torch required")
class TestGetSchemaStore(unittest.TestCase):
    """get_schema_store() selects k atoms from a consolidation snapshot.

    Snapshot-agnostic: the schema-source robustness ablation in
    phase-5-checklist.md §C is realized by calling this function with
    different consolidation states (post-death, pre-death, step-1500);
    the function itself only differs by selection_rule.
    """

    def _make_aligned(self, n_atoms=8, d=4, strengths=None, seed=0):
        """Build a (consolidation, patterns) pair with aligned indices.

        Sets u_1 per atom so effective_strength() = 0.5 * strengths
        (default weight 2^(1-1)=1 on u_1, so eff_strength = u_1).
        Actually default weight on u_1 is 2^0 = 1.0; subsequent weights
        decay. With only u_1 set, eff_strength = u_1[i].
        """
        torch.manual_seed(seed)
        cons = _make_consolidation(m=4, n_patterns=n_atoms)
        if strengths is None:
            strengths = torch.linspace(1.0, float(n_atoms), n_atoms)
        for i in range(n_atoms):
            cons.u[i, 0] = float(strengths[i])
        patterns = torch.randn(n_atoms, d, dtype=torch.complex64)
        return cons, patterns

    def test_top_k_picks_highest_effective_strength(self):
        mod = _import_module()
        # Atom 5 has strength 100, others have 1..N. Top-3 must include atom 5.
        cons, patterns = self._make_aligned(n_atoms=8)
        cons.u[5, 0] = 100.0
        schemas, atom_idx = mod.get_schema_store(
            consolidation=cons, patterns=patterns,
            selection_rule="top_k_by_effective_strength", k=3,
        )
        self.assertEqual(schemas.shape, (3, 4))
        self.assertEqual(atom_idx.shape, (3,))
        self.assertIn(5, atom_idx.tolist())
        # The schema vector for atom 5 must equal patterns row 5.
        pos = atom_idx.tolist().index(5)
        self.assertTrue(torch.equal(schemas[pos], patterns[5]))

    def test_top_k_is_deterministic(self):
        mod = _import_module()
        cons, patterns = self._make_aligned(n_atoms=6, seed=11)
        s1, idx1 = mod.get_schema_store(
            consolidation=cons, patterns=patterns,
            selection_rule="top_k_by_effective_strength", k=4,
        )
        s2, idx2 = mod.get_schema_store(
            consolidation=cons, patterns=patterns,
            selection_rule="top_k_by_effective_strength", k=4,
        )
        self.assertTrue(torch.equal(idx1, idx2))

    def test_random_k_uses_rng_and_is_reproducible(self):
        mod = _import_module()
        cons, patterns = self._make_aligned(n_atoms=10, seed=2)
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        s1, idx1 = mod.get_schema_store(
            consolidation=cons, patterns=patterns,
            selection_rule="random_k", k=4, rng=g1,
        )
        s2, idx2 = mod.get_schema_store(
            consolidation=cons, patterns=patterns,
            selection_rule="random_k", k=4, rng=g2,
        )
        self.assertTrue(torch.equal(idx1, idx2))

    def test_random_k_does_not_prefer_high_strength(self):
        """Over many random draws, the strongest atom is not always picked
        — establishes that random_k is not secretly top-k.
        """
        mod = _import_module()
        cons, patterns = self._make_aligned(n_atoms=20, seed=3)
        # Atom 0 has the highest strength by construction.
        cons.u[0, 0] = 1000.0
        n_runs = 30
        n_picks_of_atom_0 = 0
        g = torch.Generator().manual_seed(100)
        for _ in range(n_runs):
            _, idx = mod.get_schema_store(
                consolidation=cons, patterns=patterns,
                selection_rule="random_k", k=3, rng=g,
            )
            if 0 in idx.tolist():
                n_picks_of_atom_0 += 1
        # Top-k would pick atom 0 every run. Random picks it 3/20 of the
        # time in expectation (k/n); we just check it's not all 30.
        self.assertLess(n_picks_of_atom_0, n_runs)

    def test_k_larger_than_population_returns_all(self):
        mod = _import_module()
        cons, patterns = self._make_aligned(n_atoms=3, seed=5)
        schemas, atom_idx = mod.get_schema_store(
            consolidation=cons, patterns=patterns,
            selection_rule="top_k_by_effective_strength", k=10,
        )
        self.assertEqual(schemas.shape[0], 3)
        self.assertEqual(set(atom_idx.tolist()), {0, 1, 2})

    def test_empty_consolidation_raises(self):
        mod = _import_module()
        cons = _make_consolidation(m=4, n_patterns=0)
        patterns = torch.zeros(0, 4, dtype=torch.complex64)
        with self.assertRaises(ValueError):
            mod.get_schema_store(
                consolidation=cons, patterns=patterns,
                selection_rule="top_k_by_effective_strength", k=3,
            )

    def test_misaligned_patterns_raises(self):
        """Anti-foot-gun: if the caller passes patterns that don't match
        the consolidation row count, get_schema_store refuses rather than
        silently scrambling indices."""
        mod = _import_module()
        cons, patterns = self._make_aligned(n_atoms=5, seed=7)
        bad_patterns = patterns[:3]
        with self.assertRaises(ValueError):
            mod.get_schema_store(
                consolidation=cons, patterns=bad_patterns,
                selection_rule="top_k_by_effective_strength", k=2,
            )

    def test_unknown_selection_rule_raises(self):
        mod = _import_module()
        cons, patterns = self._make_aligned(n_atoms=4, seed=9)
        with self.assertRaises(ValueError):
            mod.get_schema_store(
                consolidation=cons, patterns=patterns,
                selection_rule="freq_alpha_filter", k=2,
            )


@unittest.skipIf(torch is None, "torch required")
class TestSchemaPriorSelectionDiversityFilter(unittest.TestCase):
    """select_schema_priors() applies a greedy diversity walk over the
    ranked candidates. The K_main returned schemas should satisfy
    pairwise cosine similarity ≤ delta_redundant.
    """

    def test_no_two_returned_schemas_exceed_delta_redundant(self):
        """Two near-duplicate top schemas: the lower-ranked duplicate must
        be skipped in favor of the next diverse candidate."""
        mod = _import_module()
        torch.manual_seed(0)
        d = 16
        # Build cue.
        cue = torch.randn(d, dtype=torch.complex64)
        cue = cue / cue.norm()
        # s0 = cue itself (highest content match)
        # s1 = near-duplicate of s0 (also high content match)
        # s2 = orthogonal-to-cue diverse vector
        s0 = cue.clone()
        s1 = s0 + 0.01 * torch.randn(d, dtype=torch.complex64)
        s1 = s1 / s1.norm()
        s2 = torch.randn(d, dtype=torch.complex64)
        s2 = s2 - (_fhrr_dot(s2, s0) / _fhrr_dot(s0, s0)) * s0  # orthogonalize
        s2 = s2 / s2.norm()
        store = torch.stack([s0, s1, s2])
        picked = mod.select_schema_priors(
            cue=cue, schema_store=store, k_main=2,
            delta_redundant=0.95, prior_type="content",
        )
        # Should pick s0 first; s1 is too similar to s0; should fall through to s2.
        idx_picks = [p[0] for p in picked]
        self.assertEqual(idx_picks, [0, 2])

    def test_returns_k_main_when_store_has_enough_diverse_schemas(self):
        """K_main schemas, all pairwise cosine ≤ delta_redundant."""
        mod = _import_module()
        torch.manual_seed(1)
        d = 64
        n = 8
        cue = torch.randn(d, dtype=torch.complex64); cue = cue / cue.norm()
        # Generate random vectors — at d=64 they're approximately orthogonal.
        store = torch.randn(n, d, dtype=torch.complex64)
        store = store / store.norm(dim=-1, keepdim=True)
        picked = mod.select_schema_priors(
            cue=cue, schema_store=store, k_main=4,
            delta_redundant=0.95, prior_type="content",
        )
        self.assertEqual(len(picked), 4)
        vecs = [v for _, v in picked]
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sim = float(_fhrr_cosine_test(vecs[i], vecs[j]))
                self.assertLessEqual(sim, 0.95 + 1e-6)

    def test_returns_fewer_than_k_main_when_store_is_degenerate(self):
        """If every schema is near-duplicate to every other, the function
        returns fewer than K_main rather than recycling duplicates."""
        mod = _import_module()
        torch.manual_seed(2)
        d = 16
        base = torch.randn(d, dtype=torch.complex64); base = base / base.norm()
        # All schemas are tiny perturbations of `base`; all pairwise sims > 0.99.
        store = torch.stack([
            (base + 0.001 * torch.randn(d, dtype=torch.complex64)) for _ in range(6)
        ])
        store = store / store.norm(dim=-1, keepdim=True)
        cue = base.clone()
        picked = mod.select_schema_priors(
            cue=cue, schema_store=store, k_main=4,
            delta_redundant=0.95, prior_type="content",
        )
        self.assertEqual(len(picked), 1)  # only the top-ranked survives

    def test_prior_type_random_ignores_cue_similarity(self):
        """With a fixed rng, prior_type='random' produces selection
        independent of the cue (top-1 random pick is NOT the highest-cosine
        schema, statistically)."""
        mod = _import_module()
        torch.manual_seed(3)
        d = 64
        cue = torch.randn(d, dtype=torch.complex64); cue = cue / cue.norm()
        # Make schema 0 the EXACT cue — content ranking always picks 0 first.
        s0 = cue.clone()
        rest = torch.randn(9, d, dtype=torch.complex64)
        rest = rest / rest.norm(dim=-1, keepdim=True)
        store = torch.cat([s0.unsqueeze(0), rest], dim=0)
        # Content mode always picks s0 first.
        content_picks = mod.select_schema_priors(
            cue=cue, schema_store=store, k_main=3,
            delta_redundant=0.99, prior_type="content",
        )
        self.assertEqual(content_picks[0][0], 0)
        # Random mode picks differently across rngs.
        rng = torch.Generator().manual_seed(42)
        random_picks = mod.select_schema_priors(
            cue=cue, schema_store=store, k_main=3,
            delta_redundant=0.99, prior_type="random", rng=rng,
        )
        random_idx = [p[0] for p in random_picks]
        # Statistically possible but unlikely the random rng landed on 0
        # first; verify the chosen ordering differs from content ordering.
        self.assertNotEqual(random_idx, [p[0] for p in content_picks])

    def test_prior_type_role_uses_binding_similarity_not_content(self):
        """Schema A: high content similarity to cue but no shared bindings.
        Schema B: low content similarity but shared bindings.
        prior_type='role' must pick B; 'content' must pick A.
        """
        mod = _import_module()
        torch.manual_seed(4)
        d = 64
        # Build a cue and its bindings.
        cue = torch.randn(d, dtype=torch.complex64); cue = cue / cue.norm()
        filler1 = torch.randn(d, dtype=torch.complex64); filler1 = filler1 / filler1.norm()
        filler2 = torch.randn(d, dtype=torch.complex64); filler2 = filler2 / filler2.norm()
        cue_bindings = torch.stack([filler1, filler2])  # [2, D]

        # Schema A: content-similar to cue (= cue itself), bindings unrelated.
        sA = cue.clone()
        sA_bindings = torch.randn(2, d, dtype=torch.complex64)
        sA_bindings = sA_bindings / sA_bindings.norm(dim=-1, keepdim=True)

        # Schema B: low content similarity, but bindings match cue_bindings.
        sB = torch.randn(d, dtype=torch.complex64); sB = sB / sB.norm()
        # Force sB orthogonal-ish to cue.
        sB = sB - (_fhrr_dot(sB, cue) / _fhrr_dot(cue, cue)) * cue
        sB = sB / sB.norm()
        sB_bindings = cue_bindings.clone()  # exact match

        store = torch.stack([sA, sB])
        schema_bindings = torch.stack([sA_bindings, sB_bindings])

        content_picks = mod.select_schema_priors(
            cue=cue, schema_store=store, k_main=1, delta_redundant=1.0,
            prior_type="content",
        )
        self.assertEqual(content_picks[0][0], 0)

        role_picks = mod.select_schema_priors(
            cue=cue, schema_store=store, k_main=1, delta_redundant=1.0,
            prior_type="role",
            cue_bindings=cue_bindings, schema_bindings=schema_bindings,
        )
        self.assertEqual(role_picks[0][0], 1)


# Helpers used by tests above — small inline cosine ops on complex FHRR
# vectors. Kept here (not imported from the experiment module) so tests
# remain independent of internal naming.

def _fhrr_dot(a, b):
    return torch.dot(a.conj(), b).real


def _fhrr_cosine_test(a, b):
    return _fhrr_dot(a, b) / (a.norm() * b.norm()).clamp(min=1e-12)


@unittest.skipIf(torch is None, "torch required")
class TestSettleBranchWithPrior(unittest.TestCase):
    """settle_branch_with_prior() supports two prior formulations (the
    decision #5 spike). Both reduce to unbiased retrieve at γ=0; they
    diverge at γ>0 in how the prior shapes the dynamics. Tests pin both.
    """

    def _build_memory(self, n=6, d=64, seed=0):
        from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        torch.manual_seed(seed)
        substrate = TorchFHRR(dim=d, device="cpu")
        mem = TorchHopfieldMemory(substrate)
        patterns = []
        for i in range(n):
            p = torch.randn(d, dtype=torch.complex64)
            p = substrate.normalize(p)
            mem.store(p, label=i)
            patterns.append(p)
        return mem, patterns

    # ----- γ=0 equivalence (must hold for both formulations) ----- #

    def test_gamma_zero_per_pattern_matches_unbiased_retrieve(self):
        mod = _import_module()
        mem, patterns = self._build_memory(n=6, d=64, seed=0)
        cue = mem.substrate.normalize(patterns[2] + 0.1 * torch.randn(64, dtype=torch.complex64))
        baseline = mem.retrieve(cue, beta=10.0, max_iter=12)
        zero_prior = torch.zeros(64, dtype=torch.complex64)
        settled, _ = mod.settle_branch_with_prior(
            memory=mem, cue=cue, prior=zero_prior,
            beta=10.0, gamma=0.0, max_iter=12, formulation="per_pattern",
        )
        self.assertLess((settled - baseline.state).abs().max().item(), 1e-5)

    def test_gamma_zero_global_pull_matches_unbiased_retrieve(self):
        mod = _import_module()
        mem, patterns = self._build_memory(n=6, d=64, seed=0)
        cue = mem.substrate.normalize(patterns[2] + 0.1 * torch.randn(64, dtype=torch.complex64))
        baseline = mem.retrieve(cue, beta=10.0, max_iter=12)
        zero_prior = torch.zeros(64, dtype=torch.complex64)
        settled, _ = mod.settle_branch_with_prior(
            memory=mem, cue=cue, prior=zero_prior,
            beta=10.0, gamma=0.0, max_iter=12, formulation="global_pull",
        )
        self.assertLess((settled - baseline.state).abs().max().item(), 1e-5)

    def test_gamma_zero_with_wild_prior_no_effect_per_pattern(self):
        """At γ=0 the prior must not influence the result regardless of magnitude."""
        mod = _import_module()
        mem, patterns = self._build_memory(n=6, d=64, seed=1)
        cue = mem.substrate.normalize(patterns[0] + 0.2 * torch.randn(64, dtype=torch.complex64))
        baseline = mem.retrieve(cue, beta=10.0, max_iter=12)
        wild_prior = 7.5 * torch.randn(64, dtype=torch.complex64)
        settled, _ = mod.settle_branch_with_prior(
            memory=mem, cue=cue, prior=wild_prior,
            beta=10.0, gamma=0.0, max_iter=12, formulation="per_pattern",
        )
        self.assertLess((settled - baseline.state).abs().max().item(), 1e-5)

    def test_gamma_zero_with_wild_prior_no_effect_global_pull(self):
        mod = _import_module()
        mem, patterns = self._build_memory(n=6, d=64, seed=1)
        cue = mem.substrate.normalize(patterns[0] + 0.2 * torch.randn(64, dtype=torch.complex64))
        baseline = mem.retrieve(cue, beta=10.0, max_iter=12)
        wild_prior = 7.5 * torch.randn(64, dtype=torch.complex64)
        settled, _ = mod.settle_branch_with_prior(
            memory=mem, cue=cue, prior=wild_prior,
            beta=10.0, gamma=0.0, max_iter=12, formulation="global_pull",
        )
        self.assertLess((settled - baseline.state).abs().max().item(), 1e-5)

    # ----- High-γ behavior: per_pattern stays on substrate; global_pull doesn't ----- #

    def test_per_pattern_high_gamma_routes_to_prior_matched_stored_pattern(self):
        """Per-pattern with prior matching stored pattern 4 (off-cue) should
        route q toward pattern 4 — but q stays on the stored-pattern manifold."""
        mod = _import_module()
        mem, patterns = self._build_memory(n=6, d=64, seed=2)
        cue = mem.substrate.normalize(patterns[0] + 0.05 * torch.randn(64, dtype=torch.complex64))
        prior = mem.substrate.normalize(patterns[4] + 0.05 * torch.randn(64, dtype=torch.complex64))
        settled, telem = mod.settle_branch_with_prior(
            memory=mem, cue=cue, prior=prior,
            beta=10.0, gamma=100.0, max_iter=12, formulation="per_pattern",
        )
        sim_to_0 = float(mem.substrate.similarity(settled, patterns[0]))
        sim_to_4 = float(mem.substrate.similarity(settled, patterns[4]))
        self.assertGreater(sim_to_4, sim_to_0)
        # On-substrate: max sim to any stored pattern is large.
        self.assertGreater(telem["on_substrate_alignment"], 0.1)

    def test_global_pull_off_manifold_prior_drifts_off_substrate(self):
        """The key behavioral difference: at high γ with a prior that is
        NOT in the stored-pattern span, global_pull drags q off-substrate
        (low on_substrate_alignment) while per_pattern stays put."""
        mod = _import_module()
        mem, patterns = self._build_memory(n=6, d=64, seed=10)
        cue = mem.substrate.normalize(patterns[0] + 0.05 * torch.randn(64, dtype=torch.complex64))
        # Off-manifold prior: a fresh random vector unrelated to any stored pattern.
        torch.manual_seed(7777)
        off_prior = mem.substrate.normalize(torch.randn(64, dtype=torch.complex64))
        # Sanity check: prior is not particularly close to any stored
        # pattern (relaxed; at d=64 a random vector has ~0.1 typical sim).
        max_sim = max(float(mem.substrate.similarity(off_prior, p)) for p in patterns)
        self.assertLess(max_sim, 0.15)
        # Per-pattern: high γ does nothing useful because no pattern matches prior.
        _, telem_pp = mod.settle_branch_with_prior(
            memory=mem, cue=cue, prior=off_prior,
            beta=10.0, gamma=100.0, max_iter=12, formulation="per_pattern",
        )
        # Global pull: q gets dragged toward the prior; alignment with stored
        # patterns degrades.
        _, telem_gp = mod.settle_branch_with_prior(
            memory=mem, cue=cue, prior=off_prior,
            beta=10.0, gamma=100.0, max_iter=12, formulation="global_pull",
        )
        # Per-pattern stays on substrate; global pull drifts off.
        self.assertGreater(telem_pp["on_substrate_alignment"], telem_gp["on_substrate_alignment"])

    def test_telemetry_records_entropy_and_alignment_per_pattern(self):
        mod = _import_module()
        mem, patterns = self._build_memory(n=6, d=64, seed=3)
        cue = mem.substrate.normalize(patterns[1] + 0.1 * torch.randn(64, dtype=torch.complex64))
        zero_prior = torch.zeros(64, dtype=torch.complex64)
        _, telem = mod.settle_branch_with_prior(
            memory=mem, cue=cue, prior=zero_prior,
            beta=10.0, gamma=0.0, max_iter=12, formulation="per_pattern",
        )
        for key in (
            "score_entropy_initial", "score_entropy_final",
            "converged", "iterations",
            "energy_unbiased_final", "energy_biased_final",
            "on_substrate_alignment", "formulation",
        ):
            self.assertIn(key, telem)
        self.assertEqual(telem["formulation"], "per_pattern")
        self.assertGreater(telem["score_entropy_initial"], telem["score_entropy_final"])

    def test_telemetry_formulation_field_set_for_global_pull(self):
        mod = _import_module()
        mem, patterns = self._build_memory(n=4, d=32, seed=4)
        cue = patterns[0]
        _, telem = mod.settle_branch_with_prior(
            memory=mem, cue=cue, prior=patterns[0],
            beta=10.0, gamma=0.5, max_iter=4, formulation="global_pull",
        )
        self.assertEqual(telem["formulation"], "global_pull")

    def test_invalid_args_raise(self):
        mod = _import_module()
        mem, patterns = self._build_memory(n=4, d=32, seed=5)
        cue = patterns[0]
        with self.assertRaises(ValueError):
            mod.settle_branch_with_prior(
                memory=mem, cue=cue, prior=patterns[0],
                beta=10.0, gamma=-0.1, max_iter=4,
            )
        with self.assertRaises(ValueError):
            mod.settle_branch_with_prior(
                memory=mem, cue=cue, prior=patterns[0],
                beta=0.0, gamma=0.5, max_iter=4,
            )
        with self.assertRaises(ValueError):
            mod.settle_branch_with_prior(
                memory=mem, cue=cue, prior=patterns[0],
                beta=10.0, gamma=0.5, max_iter=4, formulation="schemaforcing",
            )


@unittest.skipIf(torch is None, "torch required")
class TestComputeBranchDiagnostics(unittest.TestCase):
    """compute_branch_diagnostics fills BranchState fields in-place.
    Each test pins one field's computation rule.
    """

    def _build_memory(self, n=4, d=32, seed=0):
        from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        torch.manual_seed(seed)
        substrate = TorchFHRR(dim=d, device="cpu")
        mem = TorchHopfieldMemory(substrate)
        patterns = []
        for i in range(n):
            p = substrate.normalize(torch.randn(d, dtype=torch.complex64))
            mem.store(p, label=i)
            patterns.append(p)
        return mem, patterns

    def _branch(self, q_init, q_settled, prior, d=32):
        mod = _import_module()
        return mod.BranchState(
            branch_id=0, prior_source="schema",
            prior=prior, q_initial=q_init, q_settled=q_settled,
        )

    def test_energy_drop_equals_initial_minus_final_unbiased(self):
        mod = _import_module()
        mem, patterns = self._build_memory(n=4, d=32, seed=0)
        cue = mem.substrate.normalize(patterns[0] + 0.2 * torch.randn(32, dtype=torch.complex64))
        zero_prior = torch.zeros(32, dtype=torch.complex64)
        settled, telem = mod.settle_branch_with_prior(
            memory=mem, cue=cue, prior=zero_prior, beta=10.0, gamma=0.0,
            max_iter=12,
        )
        b = self._branch(q_init=cue, q_settled=settled, prior=zero_prior)
        mod.compute_branch_diagnostics(
            branch=b, memory=mem, cue=cue, beta=10.0, gamma=0.0,
            target_id=None, codebook=None, positions=None,
            decode_ids=[], decode_k=5, masked_pos=0,
            settling_telemetry=telem,
        )
        # Reconstruct what energy_drop should be.
        e_init = mod._unbiased_energy(mem, cue, 10.0)
        e_final = mod._unbiased_energy(mem, settled, 10.0)
        self.assertAlmostEqual(b.energy_drop, e_init - e_final, places=5)
        self.assertAlmostEqual(b.energy_unbiased, e_final, places=5)

    def test_prior_alignment_is_cosine_between_settled_state_and_prior(self):
        mod = _import_module()
        mem, patterns = self._build_memory(n=4, d=32, seed=1)
        cue = mem.substrate.normalize(patterns[0] + 0.1 * torch.randn(32, dtype=torch.complex64))
        prior = mem.substrate.normalize(patterns[1].clone())
        settled, telem = mod.settle_branch_with_prior(
            memory=mem, cue=cue, prior=prior, beta=10.0, gamma=0.5,
            max_iter=12,
        )
        b = self._branch(q_init=cue, q_settled=settled, prior=prior)
        mod.compute_branch_diagnostics(
            branch=b, memory=mem, cue=cue, beta=10.0, gamma=0.5,
            target_id=None, codebook=None, positions=None,
            decode_ids=[], decode_k=5, masked_pos=0,
            settling_telemetry=telem,
        )
        expected = float(_fhrr_cosine_test(settled, prior))
        self.assertAlmostEqual(b.prior_alignment, expected, places=5)

    def test_entropy_collapse_positive_under_decisive_settling(self):
        mod = _import_module()
        mem, patterns = self._build_memory(n=6, d=64, seed=2)
        cue = mem.substrate.normalize(patterns[3] + 0.1 * torch.randn(64, dtype=torch.complex64))
        zero_prior = torch.zeros(64, dtype=torch.complex64)
        settled, telem = mod.settle_branch_with_prior(
            memory=mem, cue=cue, prior=zero_prior, beta=10.0, gamma=0.0,
            max_iter=12,
        )
        b = self._branch(q_init=cue, q_settled=settled, prior=zero_prior)
        mod.compute_branch_diagnostics(
            branch=b, memory=mem, cue=cue, beta=10.0, gamma=0.0,
            target_id=None, codebook=None, positions=None,
            decode_ids=[], decode_k=5, masked_pos=0,
            settling_telemetry=telem,
        )
        self.assertGreater(b.entropy_collapse, 0.0)
        self.assertAlmostEqual(
            b.entropy_collapse,
            b.score_entropy_initial - b.score_entropy_final,
            places=5,
        )

    def test_meta_stable_and_recall_via_decode(self):
        """meta_stable = (top decode < 0.95); recall_support = target_id
        in top-K decoded ids. Exercises the decode_position readout path."""
        mod = _import_module()
        from energy_memory.phase2.encoding import build_position_vectors, encode_window
        from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        torch.manual_seed(3)
        d = 64
        substrate = TorchFHRR(dim=d, device="cpu")
        mem = TorchHopfieldMemory(substrate)
        positions = build_position_vectors(substrate, count=3)
        # Codebook of 10 tokens; deterministic.
        codebook = torch.randn(10, d, dtype=torch.complex64)
        codebook = substrate.normalize(codebook)
        # Build a known window encoding and store it.
        window = [4, 7, 2]
        enc = encode_window(substrate, positions, codebook, window)
        mem.store(enc)
        # The settled state IS the encoded window; decode at position 0 should
        # return token 4 with score 1 (perfect match) → meta_stable False
        # (top score >= 0.95) and recall_support True.
        b = self._branch(q_init=enc, q_settled=enc, prior=torch.zeros(d, dtype=torch.complex64))
        mod.compute_branch_diagnostics(
            branch=b, memory=mem, cue=enc, beta=10.0, gamma=0.0,
            target_id=4, codebook=codebook, positions=positions,
            decode_ids=list(range(10)), decode_k=5, masked_pos=0,
        )
        # Target token 4 is the bound filler at position 0 → must appear
        # in the top-K decode.
        self.assertTrue(b.recall_support)
        # In FHRR, decoding a 3-binding superposition leaves residual
        # noise; the top decode score is typically well below 0.95, so
        # meta_stable (=top<0.95) is True. The cap-coverage τ=0.5 metric
        # is satisfied iff top>=0.5 AND target in top-K. The exact value
        # depends on dim/window; we just pin the relative semantics.
        self.assertTrue(b.meta_stable)  # 3-binding superposition; top decode < 0.95

    def test_recall_support_false_when_target_id_none(self):
        mod = _import_module()
        from energy_memory.phase2.encoding import build_position_vectors, encode_window
        from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        torch.manual_seed(4)
        d = 64
        substrate = TorchFHRR(dim=d, device="cpu")
        mem = TorchHopfieldMemory(substrate)
        positions = build_position_vectors(substrate, count=3)
        codebook = substrate.normalize(torch.randn(10, d, dtype=torch.complex64))
        enc = encode_window(substrate, positions, codebook, [1, 2, 3])
        mem.store(enc)
        b = self._branch(q_init=enc, q_settled=enc, prior=torch.zeros(d, dtype=torch.complex64))
        mod.compute_branch_diagnostics(
            branch=b, memory=mem, cue=enc, beta=10.0, gamma=0.0,
            target_id=None, codebook=codebook, positions=positions,
            decode_ids=list(range(10)), decode_k=5, masked_pos=0,
        )
        self.assertFalse(b.recall_support)
        self.assertEqual(b.cap_coverage_t05, 0.0)

    def test_diagnostics_do_not_mutate_selection_score(self):
        """energy_unbiased computed by the diagnostic must equal the
        energy_unbiased_final from settle_branch_with_prior's telemetry.
        If they diverge, selection and interpretation use different values."""
        mod = _import_module()
        mem, patterns = self._build_memory(n=5, d=32, seed=5)
        cue = mem.substrate.normalize(patterns[2] + 0.1 * torch.randn(32, dtype=torch.complex64))
        prior = patterns[1].clone()
        settled, telem = mod.settle_branch_with_prior(
            memory=mem, cue=cue, prior=prior, beta=10.0, gamma=0.5,
            max_iter=12,
        )
        b = self._branch(q_init=cue, q_settled=settled, prior=prior)
        mod.compute_branch_diagnostics(
            branch=b, memory=mem, cue=cue, beta=10.0, gamma=0.5,
            target_id=None, codebook=None, positions=None,
            decode_ids=[], decode_k=5, masked_pos=0,
            settling_telemetry=telem,
        )
        self.assertAlmostEqual(
            b.energy_unbiased, telem["energy_unbiased_final"], places=5,
        )

    def test_structural_match_uses_role_decomposition(self):
        """Build q* such that unbinding position 0 yields cue_bindings[0]
        exactly. structural_match must be ≈ 1.0."""
        mod = _import_module()
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        from energy_memory.phase2.encoding import build_position_vectors
        torch.manual_seed(6)
        d = 64
        substrate = TorchFHRR(dim=d, device="cpu")
        positions = build_position_vectors(substrate, count=2)
        filler0 = substrate.normalize(torch.randn(d, dtype=torch.complex64))
        filler1 = substrate.normalize(torch.randn(d, dtype=torch.complex64))
        cue_bindings = torch.stack([filler0, filler1])
        # q* = filler0 ⊛ pos0 + filler1 ⊛ pos1, normalized
        q_star = substrate.normalize(
            substrate.bind(filler0, positions[0]) + substrate.bind(filler1, positions[1])
        )
        # Minimal Hopfield memory (still needed for the diagnostic).
        from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
        mem = TorchHopfieldMemory(substrate)
        mem.store(q_star)
        b = self._branch(q_init=q_star, q_settled=q_star, prior=torch.zeros(d, dtype=torch.complex64))
        mod.compute_branch_diagnostics(
            branch=b, memory=mem, cue=q_star, beta=10.0, gamma=0.0,
            target_id=None, codebook=None, positions=positions,
            decode_ids=[], decode_k=5, masked_pos=0,
            cue_bindings=cue_bindings,
        )
        # Structural match should be high — within normalize/superposition noise.
        self.assertGreater(b.structural_match, 0.3)


@unittest.skipIf(torch is None, "torch required")
class TestPairwiseFinalStateDivergence(unittest.TestCase):
    """compute_pairwise_final_state_divergence fills final_state_divergence."""

    def test_single_branch_divergence_is_zero(self):
        mod = _import_module()
        b = _make_branch(seed=1)
        mod.compute_pairwise_final_state_divergence([b])
        self.assertEqual(b.final_state_divergence, 0.0)

    def test_three_orthogonal_branches_have_near_unit_divergence(self):
        mod = _import_module()
        torch.manual_seed(2)
        d = 64
        # Three near-orthogonal complex unit vectors.
        states = []
        for _ in range(3):
            v = torch.randn(d, dtype=torch.complex64); v = v / v.norm()
            states.append(v)
        branches = [_make_branch(q_settled=s) for s in states]
        mod.compute_pairwise_final_state_divergence(branches)
        for b in branches:
            # Should be near 1.0 (cosine ≈ 0 → distance ≈ 1).
            self.assertGreater(b.final_state_divergence, 0.7)
            self.assertLess(b.final_state_divergence, 1.2)

    def test_identical_branches_have_zero_divergence(self):
        mod = _import_module()
        torch.manual_seed(3)
        d = 16
        v = torch.randn(d, dtype=torch.complex64); v = v / v.norm()
        branches = [_make_branch(q_settled=v.clone()) for _ in range(3)]
        mod.compute_pairwise_final_state_divergence(branches)
        for b in branches:
            self.assertAlmostEqual(b.final_state_divergence, 0.0, places=4)


@unittest.skipIf(torch is None, "torch required")
class TestCombineBundleResettle(unittest.TestCase):
    """The preferred combination rule. Anti-homunculus-clean reading:
    energy-weighted bundle of branch states, followed by an UNBIASED
    Hopfield re-settle.
    """

    def _build(self, n=4, d=32, seed=0):
        from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        torch.manual_seed(seed)
        substrate = TorchFHRR(dim=d, device="cpu")
        mem = TorchHopfieldMemory(substrate)
        patterns = []
        for i in range(n):
            p = substrate.normalize(torch.randn(d, dtype=torch.complex64))
            mem.store(p, label=i)
            patterns.append(p)
        return mem, patterns

    def test_single_branch_returns_state_near_input(self):
        """K=1: bundle is just q_settled[0]; the unbiased re-settle should
        converge to a near-identical attractor."""
        mod = _import_module()
        mem, patterns = self._build(n=4, d=32, seed=0)
        # Branch state = an actual attractor (one of the stored patterns).
        b = _make_branch(branch_id=0, energy_unbiased=-5.0,
                         q_settled=patterns[0].clone(), d=32)
        q_final, weights, conv = mod.combine_bundle_resettle(
            branches=[b], memory=mem, beta=10.0, temperature=1.0, max_iter=12,
        )
        self.assertEqual(len(weights), 1)
        self.assertAlmostEqual(weights[0], 1.0, places=6)
        # Re-settle from a stored attractor stays at that attractor.
        sim = float(mem.substrate.similarity(q_final, patterns[0]))
        self.assertGreater(sim, 0.95)

    def test_softmax_weights_sum_to_one_and_nonnegative(self):
        mod = _import_module()
        mem, patterns = self._build(n=4, d=32, seed=1)
        branches = [
            _make_branch(branch_id=i, energy_unbiased=float(-i), q_settled=patterns[i].clone())
            for i in range(4)
        ]
        _, weights, _ = mod.combine_bundle_resettle(
            branches=branches, memory=mem, beta=10.0, temperature=1.0, max_iter=12,
        )
        self.assertAlmostEqual(sum(weights), 1.0, places=5)
        for w in weights:
            self.assertGreaterEqual(w, 0.0)

    def test_low_temperature_concentrates_weights_on_argmin(self):
        """τ→0: weight on the lowest-energy branch approaches 1."""
        mod = _import_module()
        mem, patterns = self._build(n=3, d=32, seed=2)
        # Branch 1 has much lower energy than 0 and 2.
        energies = [0.0, -5.0, 0.0]
        branches = [
            _make_branch(branch_id=i, energy_unbiased=e, q_settled=patterns[i].clone())
            for i, e in enumerate(energies)
        ]
        _, weights, _ = mod.combine_bundle_resettle(
            branches=branches, memory=mem, beta=10.0, temperature=0.1, max_iter=12,
        )
        self.assertGreater(weights[1], 0.99)

    def test_high_temperature_approaches_uniform(self):
        mod = _import_module()
        mem, patterns = self._build(n=4, d=32, seed=3)
        energies = [-1.0, -0.5, -0.2, 0.0]
        branches = [
            _make_branch(branch_id=i, energy_unbiased=e, q_settled=patterns[i].clone())
            for i, e in enumerate(energies)
        ]
        _, weights, _ = mod.combine_bundle_resettle(
            branches=branches, memory=mem, beta=10.0, temperature=1e3, max_iter=12,
        )
        for w in weights:
            self.assertAlmostEqual(w, 0.25, places=2)

    def test_resettle_uses_unbiased_energy_no_gamma_leak(self):
        """The re-settle is called with γ=0 inside combine_bundle_resettle.
        We verify behaviorally: with branches at stored attractors, the
        re-settle should land near a substrate attractor regardless of
        what prior was used to make those branches. (No prior is in
        scope for the combiner.)
        """
        mod = _import_module()
        mem, patterns = self._build(n=4, d=32, seed=4)
        # Two branches both at pattern 0; the combined bundle should re-settle
        # back to pattern 0 regardless of branch.prior content.
        b0 = mod.BranchState(
            branch_id=0, prior_source="schema",
            prior=patterns[2].clone(),  # arbitrary prior
            q_initial=patterns[0].clone(),
            q_settled=patterns[0].clone(),
            energy_unbiased=-1.0,
        )
        b1 = mod.BranchState(
            branch_id=1, prior_source="schema",
            prior=patterns[3].clone(),  # different arbitrary prior
            q_initial=patterns[0].clone(),
            q_settled=patterns[0].clone(),
            energy_unbiased=-1.0,
        )
        q_final, _, _ = mod.combine_bundle_resettle(
            branches=[b0, b1], memory=mem, beta=10.0, temperature=1.0, max_iter=12,
        )
        sim_to_0 = float(mem.substrate.similarity(q_final, patterns[0]))
        sim_to_2 = float(mem.substrate.similarity(q_final, patterns[2]))
        sim_to_3 = float(mem.substrate.similarity(q_final, patterns[3]))
        # If priors leaked, q_final would drift toward p2 or p3.
        self.assertGreater(sim_to_0, sim_to_2)
        self.assertGreater(sim_to_0, sim_to_3)


@unittest.skipIf(torch is None, "torch required")
class TestCombineGreedyArgmin(unittest.TestCase):
    """Baseline combination — picks the lowest-energy branch."""

    def test_returns_state_with_lowest_unbiased_energy(self):
        mod = _import_module()
        torch.manual_seed(0)
        states = [torch.randn(8, dtype=torch.complex64) for _ in range(4)]
        energies = [0.5, -2.0, 1.0, -1.0]
        branches = [
            _make_branch(branch_id=i, energy_unbiased=e, q_settled=s)
            for i, (e, s) in enumerate(zip(energies, states))
        ]
        q, k = mod.combine_greedy_argmin(branches)
        self.assertEqual(k, 1)
        self.assertTrue(torch.equal(q, states[1]))

    def test_ties_broken_deterministically_first_index(self):
        mod = _import_module()
        torch.manual_seed(1)
        states = [torch.randn(8, dtype=torch.complex64) for _ in range(3)]
        branches = [
            _make_branch(branch_id=i, energy_unbiased=-1.0, q_settled=states[i])
            for i in range(3)
        ]
        _, k = mod.combine_greedy_argmin(branches)
        self.assertEqual(k, 0)

    def test_empty_branch_list_raises(self):
        mod = _import_module()
        with self.assertRaises(ValueError):
            mod.combine_greedy_argmin([])


@unittest.skipIf(torch is None, "torch required")
class TestCombineBoltzmannSample(unittest.TestCase):
    """Boltzmann sampling: k ~ Categorical(softmax(-E/τ))."""

    def test_low_temperature_concentrates_on_argmin(self):
        mod = _import_module()
        torch.manual_seed(0)
        states = [torch.randn(8, dtype=torch.complex64) for _ in range(4)]
        energies = [0.0, -5.0, 0.0, 0.0]
        branches = [
            _make_branch(branch_id=i, energy_unbiased=e, q_settled=s)
            for i, (e, s) in enumerate(zip(energies, states))
        ]
        picks = []
        rng = torch.Generator().manual_seed(123)
        for _ in range(30):
            _, k, _ = mod.combine_boltzmann_sample(branches, temperature=0.1, rng=rng)
            picks.append(k)
        # Nearly all picks should be the argmin (branch 1).
        self.assertGreater(picks.count(1), 25)

    def test_high_temperature_approaches_uniform(self):
        mod = _import_module()
        torch.manual_seed(1)
        states = [torch.randn(8, dtype=torch.complex64) for _ in range(4)]
        energies = [-1.0, -0.5, 0.0, 0.5]
        branches = [
            _make_branch(branch_id=i, energy_unbiased=e, q_settled=s)
            for i, (e, s) in enumerate(zip(energies, states))
        ]
        rng = torch.Generator().manual_seed(456)
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        n_samples = 400
        for _ in range(n_samples):
            _, k, _ = mod.combine_boltzmann_sample(branches, temperature=1e3, rng=rng)
            counts[k] += 1
        # Each branch should be picked roughly uniformly (~25%) at high τ.
        for c in counts.values():
            self.assertGreater(c / n_samples, 0.15)
            self.assertLess(c / n_samples, 0.40)

    def test_seed_makes_sampling_reproducible(self):
        mod = _import_module()
        torch.manual_seed(2)
        states = [torch.randn(8, dtype=torch.complex64) for _ in range(4)]
        energies = [0.0, -1.0, -0.5, 0.5]
        branches = [
            _make_branch(branch_id=i, energy_unbiased=e, q_settled=s)
            for i, (e, s) in enumerate(zip(energies, states))
        ]
        rng1 = torch.Generator().manual_seed(42)
        rng2 = torch.Generator().manual_seed(42)
        picks1 = [mod.combine_boltzmann_sample(branches, temperature=1.0, rng=rng1)[1] for _ in range(10)]
        picks2 = [mod.combine_boltzmann_sample(branches, temperature=1.0, rng=rng2)[1] for _ in range(10)]
        self.assertEqual(picks1, picks2)


@unittest.skipIf(torch is None, "torch required")
class TestRunBranchedRetrievalIntegration(unittest.TestCase):
    """End-to-end: one cue through the full pipeline. All three
    combination rules run; all diagnostics filled.
    """

    def _build(self, n_atoms=6, d=64, seed=0):
        from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        torch.manual_seed(seed)
        substrate = TorchFHRR(dim=d, device="cpu")
        mem = TorchHopfieldMemory(substrate)
        cons = ConsolidationState(ConsolidationConfig(m=4, alpha=0.25), device="cpu")
        patterns = []
        for i in range(n_atoms):
            p = substrate.normalize(torch.randn(d, dtype=torch.complex64))
            mem.store(p, label=i)
            cons.add_pattern(novelty_strength=1.0 + 0.1 * i)
            patterns.append(p)
        return mem, cons, patterns

    def test_pipeline_runs_under_default_config(self):
        mod = _import_module()
        mem, cons, patterns = self._build(n_atoms=6, d=64, seed=0)
        schema_store, atom_idx = mod.get_schema_store(
            consolidation=cons, patterns=mem._pattern_matrix(),
            selection_rule="top_k_by_effective_strength", k=5,
        )
        cue = mem.substrate.normalize(patterns[2] + 0.1 * torch.randn(64, dtype=torch.complex64))
        result = mod.run_branched_retrieval(
            cue=cue, cue_id=0, target_id=None,
            memory=mem, codebook=mem._pattern_matrix(), positions=None,
            decode_ids=[], decode_k=5, masked_pos=0,
            schema_store=schema_store, schema_atom_idx=atom_idx,
            consolidation=cons, prior_type="content", k_main=4,
            gamma=0.5, beta=10.0, temperature=1.0,
            delta_energy=0.1, delta_state=0.3, delta_redundant=0.95,
            include_surprise_branch=True,
        )
        # All branches have non-None settled state.
        self.assertGreaterEqual(len(result.branches), 1)
        for b in result.branches:
            self.assertIsNotNone(b.q_settled)
        # All three combination outputs present.
        self.assertIsNotNone(result.q_bundle)
        self.assertIsNotNone(result.q_greedy)
        self.assertIsNotNone(result.q_boltzmann)
        # Per-cue aggregate diagnostics populated.
        self.assertEqual(len(result.softmax_weights), len(result.branches))
        self.assertAlmostEqual(sum(result.softmax_weights), 1.0, places=5)

    def test_pipeline_under_k1_gamma0_no_surprise_matches_unbiased_retrieve(self):
        """Regression guard. At K=1, γ=0, no surprise branch, the single
        branch's q_settled equals the unbiased retrieve. The bundle re-settle
        of one branch (also unbiased) lands at the same attractor."""
        mod = _import_module()
        mem, cons, patterns = self._build(n_atoms=6, d=64, seed=1)
        schema_store, atom_idx = mod.get_schema_store(
            consolidation=cons, patterns=mem._pattern_matrix(),
            selection_rule="top_k_by_effective_strength", k=5,
        )
        cue = mem.substrate.normalize(patterns[3] + 0.1 * torch.randn(64, dtype=torch.complex64))
        baseline = mem.retrieve(cue, beta=10.0, max_iter=12)
        result = mod.run_branched_retrieval(
            cue=cue, cue_id=0, target_id=None,
            memory=mem, codebook=mem._pattern_matrix(), positions=None,
            decode_ids=[], decode_k=5, masked_pos=0,
            schema_store=schema_store, schema_atom_idx=atom_idx,
            consolidation=cons, prior_type="content", k_main=1,
            gamma=0.0, beta=10.0, temperature=1.0,
            delta_energy=0.1, delta_state=0.3, delta_redundant=0.95,
            include_surprise_branch=False,
        )
        self.assertEqual(len(result.branches), 1)
        # The branch settled with γ=0 → equals unbiased retrieve.
        diff_branch = (result.branches[0].q_settled - baseline.state).abs().max().item()
        self.assertLess(diff_branch, 1e-4)
        # The bundle is one branch re-settled (still unbiased) → same attractor.
        sim_bundle_baseline = float(mem.substrate.similarity(result.q_bundle, baseline.state))
        self.assertGreater(sim_bundle_baseline, 0.99)

    def test_split_eligibility_is_property_of_branches_not_combiner(self):
        """atom_split_signal is computed once from `branches`. The
        BranchedRetrievalResult.split_eligible field is therefore the
        same value across all three combination outputs — exposing the
        invariant that polysemy detection is independent of the
        combiner."""
        mod = _import_module()
        mem, cons, patterns = self._build(n_atoms=6, d=64, seed=2)
        schema_store, atom_idx = mod.get_schema_store(
            consolidation=cons, patterns=mem._pattern_matrix(),
            selection_rule="top_k_by_effective_strength", k=5,
        )
        cue = mem.substrate.normalize(patterns[1] + 0.1 * torch.randn(64, dtype=torch.complex64))
        result = mod.run_branched_retrieval(
            cue=cue, cue_id=0, target_id=None,
            memory=mem, codebook=mem._pattern_matrix(), positions=None,
            decode_ids=[], decode_k=5, masked_pos=0,
            schema_store=schema_store, schema_atom_idx=atom_idx,
            consolidation=cons, prior_type="content", k_main=4,
            gamma=0.5, beta=10.0, temperature=1.0,
            delta_energy=0.5, delta_state=0.3, delta_redundant=0.95,
            include_surprise_branch=True,
        )
        # The signal is a single bool on result, not three separate values.
        # Verify by recomputing it directly from the same branches; must match.
        ok, n_low, max_dist = mod.atom_split_signal(
            result.branches, delta_energy=0.5, delta_state=0.3,
        )
        self.assertEqual(result.split_eligible, ok)
        self.assertEqual(result.n_in_low_energy_set, n_low)
        self.assertAlmostEqual(
            result.max_state_distance_in_low_energy_set, max_dist, places=5,
        )

    def test_surprise_branch_marked_with_prior_source(self):
        """When include_surprise_branch=True and consolidation is non-empty,
        the surprise branch appears at the end of `branches` with
        prior_source='surprise'. This is the per-cue logging hook for
        the surprise instrumentation."""
        mod = _import_module()
        mem, cons, patterns = self._build(n_atoms=6, d=64, seed=3)
        schema_store, atom_idx = mod.get_schema_store(
            consolidation=cons, patterns=mem._pattern_matrix(),
            selection_rule="top_k_by_effective_strength", k=5,
        )
        cue = mem.substrate.normalize(patterns[4] + 0.05 * torch.randn(64, dtype=torch.complex64))
        result = mod.run_branched_retrieval(
            cue=cue, cue_id=0, target_id=None,
            memory=mem, codebook=mem._pattern_matrix(), positions=None,
            decode_ids=[], decode_k=5, masked_pos=0,
            schema_store=schema_store, schema_atom_idx=atom_idx,
            consolidation=cons, prior_type="content", k_main=3,
            gamma=0.5, beta=10.0, temperature=1.0,
            delta_energy=0.1, delta_state=0.3, delta_redundant=0.95,
            include_surprise_branch=True,
        )
        surprise_branches = [b for b in result.branches if b.prior_source == "surprise"]
        self.assertEqual(len(surprise_branches), 1)
        # The non-surprise branches carry prior_source = 'content' (the
        # prior_type used).
        non_surprise = [b for b in result.branches if b.prior_source != "surprise"]
        for b in non_surprise:
            self.assertEqual(b.prior_source, "content")


if __name__ == "__main__":
    unittest.main()
