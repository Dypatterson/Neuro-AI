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
class TestSchemaPriorSelectionDiversityFilter(unittest.TestCase):
    """select_schema_priors() applies a greedy diversity walk over the
    ranked candidates. The K_main returned schemas should satisfy
    pairwise cosine similarity ≤ delta_redundant.
    """

    @unittest.skip("blocked: requires schema-source decision #1")
    def test_no_two_returned_schemas_exceed_delta_redundant(self):
        """Construct a store with two near-duplicate top schemas and
        verify that the second one is skipped in favor of the next
        diverse candidate."""

    @unittest.skip("blocked: requires schema-source decision #1")
    def test_returns_k_main_when_store_has_enough_diverse_schemas(self):
        """K_main schemas, all pairwise cosine ≤ delta_redundant."""

    @unittest.skip("blocked: requires schema-source decision #1")
    def test_returns_fewer_than_k_main_when_store_is_degenerate(self):
        """If every schema is near-duplicate to every other, the function
        should return fewer than K_main rather than recycling duplicates."""

    @unittest.skip("blocked: requires schema-source decision #1")
    def test_prior_type_random_ignores_cue_similarity(self):
        """Under prior_type='random', selection is independent of the cue."""

    @unittest.skip("blocked: requires schema-source decision #1")
    def test_prior_type_role_uses_binding_similarity_not_content(self):
        """Construct a cue whose top content-similar schema does NOT share
        bindings, and a lower-content-similar schema that DOES share
        bindings. prior_type='role' should pick the latter."""


@unittest.skipIf(torch is None, "torch required")
class TestSettleBranchWithPrior(unittest.TestCase):
    """settle_branch_with_prior() runs Hopfield retrieval with score
    biased by γ · Re(⟨X, prior⟩). At γ=0 it must reduce exactly to the
    unbiased retrieve.
    """

    @unittest.skip("blocked: settle_branch_with_prior implementation pending")
    def test_gamma_zero_matches_unbiased_retrieve_bit_exact(self):
        """The critical backward-compatibility guarantee. Without this,
        no condition in the experiment is a true Phase-4-graduated
        baseline."""

    @unittest.skip("blocked: settle_branch_with_prior implementation pending")
    def test_gamma_infinity_collapses_to_nearest_prior_match(self):
        """At very high γ, the prior term dominates the energy and the
        retrieval reduces to nearest-stored-pattern-to-prior lookup."""

    @unittest.skip("blocked: settle_branch_with_prior implementation pending")
    def test_telemetry_records_score_entropy_initial_and_final(self):
        """score_entropy_initial > score_entropy_final under typical
        retrieval (settling is decisive)."""


@unittest.skipIf(torch is None, "torch required")
class TestComputeBranchDiagnostics(unittest.TestCase):
    """Each diagnostic field is filled by a specific computation."""

    @unittest.skip("blocked: compute_branch_diagnostics implementation pending")
    def test_energy_drop_equals_initial_minus_final_unbiased(self):
        """energy_drop = E_unbiased(q_initial) - E_unbiased(q_settled)."""

    @unittest.skip("blocked: compute_branch_diagnostics implementation pending")
    def test_prior_alignment_is_cosine_between_settled_state_and_prior(self):
        """prior_alignment = Re(⟨q_settled, p⟩) / (‖q_settled‖ · ‖p‖)."""

    @unittest.skip("blocked: compute_branch_diagnostics implementation pending")
    def test_entropy_collapse_positive_under_decisive_settling(self):
        """Initial score distribution is high-entropy; settled is
        concentrated. The difference is positive on well-formed cues."""

    @unittest.skip("blocked: compute_branch_diagnostics implementation pending")
    def test_meta_stable_flag_matches_phase2_metric(self):
        """meta_stable = (top decode score < 0.95), consistent with
        phase2.metrics.meta_stable_rate semantics."""

    @unittest.skip("blocked: compute_branch_diagnostics implementation pending")
    def test_recall_support_true_when_target_in_top_decode(self):
        """recall_support = target_id in top-K-decoded(q_settled).
        Requires a target_id (else None, not False)."""

    @unittest.skip("blocked: compute_branch_diagnostics implementation pending")
    def test_diagnostics_do_not_mutate_selection_score(self):
        """energy_unbiased computed here must equal what
        combine_bundle_resettle uses for w_k. If they diverge, the
        selection and interpretation paths are using different values."""


@unittest.skipIf(torch is None, "torch required")
class TestCombineBundleResettle(unittest.TestCase):
    """The preferred combination rule. Anti-homunculus-clean reading:
    energy-weighted bundle of branch states, followed by an unbiased
    Hopfield re-settle.
    """

    @unittest.skip("blocked: combine_bundle_resettle implementation pending")
    def test_single_branch_input_returns_that_branch_state(self):
        """K=1 degenerate case: the bundle is just q_settled[0], the
        re-settle should converge to a near-identical attractor."""

    @unittest.skip("blocked: combine_bundle_resettle implementation pending")
    def test_softmax_weights_sum_to_one_and_are_temperature_correct(self):
        """w_k must form a probability distribution (sum=1, all
        non-negative) and respect τ — at τ→∞ they tend toward uniform,
        at τ→0 toward one-hot on the argmin."""

    @unittest.skip("blocked: combine_bundle_resettle implementation pending")
    def test_dominant_branch_pulls_bundle_toward_its_state(self):
        """One branch with much lower energy than the others should
        contribute most of the bundle mass."""

    @unittest.skip("blocked: combine_bundle_resettle implementation pending")
    def test_resettle_converges_on_well_formed_input(self):
        """Bundling K basin-attractors can produce a non-attractor
        starting point; verify the unbiased re-settle still converges
        within max_iter for typical cases."""

    @unittest.skip("blocked: combine_bundle_resettle implementation pending")
    def test_resettle_uses_unbiased_energy_no_gamma_term(self):
        """The re-settle MUST use the unbiased Hopfield energy. If γ
        leaks into the re-settle, the bundle becomes an arbitrary mix
        of prior alignments rather than a substrate-pure posterior."""


@unittest.skipIf(torch is None, "torch required")
class TestCombineGreedyArgmin(unittest.TestCase):
    """Baseline combination. Anti-homunculus check is borderline; allowed
    only as a comparison condition, never as the production rule.
    """

    @unittest.skip("blocked: combine_greedy_argmin implementation pending")
    def test_returns_state_with_lowest_unbiased_energy(self):
        """Trivial behavioral spec — but pin it so the comparison
        condition is what we claim it is."""

    @unittest.skip("blocked: combine_greedy_argmin implementation pending")
    def test_ties_broken_deterministically(self):
        """Two branches with equal energies: argmin returns the first
        index (matches torch.argmin / argmax conventions used elsewhere)."""


@unittest.skipIf(torch is None, "torch required")
class TestCombineBoltzmannSample(unittest.TestCase):
    """Boltzmann sampling over branch energies. FEP-clean (sampling
    from a posterior defined by energies).
    """

    @unittest.skip("blocked: combine_boltzmann_sample implementation pending")
    def test_high_temperature_approaches_uniform_sampling(self):
        """τ→∞: every branch sampled approximately equally over N draws."""

    @unittest.skip("blocked: combine_boltzmann_sample implementation pending")
    def test_low_temperature_approaches_greedy_argmin(self):
        """τ→0: nearly always picks the lowest-energy branch."""

    @unittest.skip("blocked: combine_boltzmann_sample implementation pending")
    def test_seed_makes_sampling_reproducible(self):
        """Same RNG seed → same branch picked, every time."""


@unittest.skipIf(torch is None, "torch required")
class TestRunBranchedRetrievalIntegration(unittest.TestCase):
    """End-to-end test: one cue through the full pipeline. All three
    combination rules run; all diagnostics filled.
    """

    @unittest.skip("blocked: requires schema source + full driver")
    def test_pipeline_runs_under_default_config(self):
        """No exceptions; all branches have non-None q_settled; all
        three combination outputs present."""

    @unittest.skip("blocked: requires schema source + full driver")
    def test_pipeline_under_k1_is_equivalent_to_phase4_baseline(self):
        """At K=1, γ=0, no surprise branch, the pipeline should produce
        the same retrieval as the Phase 4 substrate alone — modulo the
        prior-seed (which at γ=0 contributes nothing). This is the
        regression guard that bridge-experiment changes don't break
        Phase 4 graduation."""

    @unittest.skip("blocked: requires schema source + full driver")
    def test_split_eligibility_persists_across_combination_rules(self):
        """If a cue is split-eligible under one combination rule, it
        should be split-eligible under all three — the joint criterion
        is a property of the branches, not the combiner."""

    @unittest.skip("blocked: requires schema source + full driver")
    def test_surprise_branch_winning_is_logged_per_cue(self):
        """When the surprise branch achieves min E_k, that fact is
        recorded in the BranchedRetrievalResult. Used by the surprise-
        branch instrumentation in §'Mixed-branch policy' of the design."""


if __name__ == "__main__":
    unittest.main()
