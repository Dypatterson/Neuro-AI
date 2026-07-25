"""Tests for the Bet-B continual harness (`energy_memory.betb`).

Covers the invariants that were load-bearing but unmechanized before 2026-07-25:
the anchor parity that kept Reports 122-139 comparable, the anti-homunculus
contract that three `legacy/` modules assert in prose while violating in code,
and the provenance guard that turns a declared control into an executed one.
"""

from __future__ import annotations

import inspect
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TestTaskFamilies(unittest.TestCase):

    def test_modular_family_matches_the_published_stream(self):
        """The 134-139 stream must be reproducible or the anchors stop meaning anything."""
        from energy_memory.betb import build_family

        fam = build_family("modular", p=5)
        g = torch.Generator().manual_seed(0)
        s = fam.build(K=4, frac=0.7, gen=g)

        self.assertEqual(s.n_inputs, 2)
        self.assertEqual(s.n_classes, 5)
        self.assertEqual([t.kind for t in s.tasks], ["add", "sub", "add", "sub"])
        # even tasks are fresh alphabets; odd tasks reuse the block just introduced
        self.assertEqual(s.xblock, [2])
        self.assertEqual(s.wblock, [1, 3])
        # block 1 uses a disjoint token range from block 0
        b0 = {tok for t in s.tasks[:2] for (inp, _) in t.train + t.test for tok in inp}
        b1 = {tok for t in s.tasks[2:] for (inp, _) in t.train + t.test for tok in inp}
        self.assertEqual(b0 & b1, set())

    def test_permutation_family_shape_matches_the_validated_spec(self):
        """Report 140's validated configuration. Sizes are load-bearing: the
        K=8 / 40-rows-per-cell variant measurably fails property 2."""
        from energy_memory.betb import build_family

        s = build_family("permutation", m=5, k=3, n_ops=6, heldout_frac=0.27).build(
            K=2, frac=0.7, gen=torch.Generator().manual_seed(0))
        prim, comp = s.tasks

        self.assertEqual((s.n_inputs, s.n_classes, s.vocab), (5, 125, 12))
        self.assertEqual((len(prim.train), len(comp.train)), (1044, 1914))
        self.assertEqual(len(comp.heldout), 1000)
        self.assertEqual(len(comp.heldout_groups), 8)
        self.assertEqual({len(g) for g in comp.heldout_groups}, {125})
        self.assertEqual(len({(r[0][0], r[0][1]) for r in comp.train}), 22)

    def test_composition_label_is_the_group_composition(self):
        """Every composition label must be g_j(g_i(x)) — applied in that order.

        Order matters here in a way it did not for the abelian design: in S_5,
        g_j∘g_i != g_i∘g_j in general, and that non-commutativity is the entire
        reason held-out pairs cannot be answered by pooling the two operators.
        """
        from energy_memory.betb import build_family

        fam = build_family("permutation", m=5, k=3, n_ops=6, heldout_frac=0.27)
        gen = torch.Generator().manual_seed(0)
        s = fam.build(K=2, frac=0.7, gen=gen)
        prim, comp = s.tasks

        # recover each operator's action from the primitive task
        act = {}
        for (toks, y) in prim.train + prim.test:
            op = max(toks[0], toks[1])
            state = tuple(t - (fam.n_ops + 1) for t in toks[2:])
            out, v = [], y
            for _ in range(fam.k):
                out.append(v % fam.m)
                v //= fam.m
            act.setdefault(op, {})[state] = tuple(reversed(out))

        checked = order_matters = 0
        for (toks, y) in comp.train + comp.heldout:
            oi, oj = toks[0], toks[1]
            state = tuple(t - (fam.n_ops + 1) for t in toks[2:])
            mid = act[oi][state]
            self.assertEqual(fam._label(act[oj][mid]), y,
                             "composition label is not g_j(g_i(x))")
            checked += 1
            if fam._label(act[oi][act[oj][state]]) != y:
                order_matters += 1
        self.assertGreater(checked, 0)
        self.assertGreater(order_matters, 0.3 * checked,
                           "composition is nearly commutative — the shortcut would be legal")

    def test_heldout_pairs_are_never_trainable(self):
        """The whole regime rests on this. If it leaks, the headline is meaningless."""
        from energy_memory.betb import build_family

        s = build_family("permutation").build(K=6, frac=0.7,
                                              gen=torch.Generator().manual_seed(2))
        for t in s.tasks:
            if t.kind != "composition":
                continue
            trainable = {r[0] for r in t.train} | {r[0] for r in t.test}
            held = {r[0] for r in t.heldout}
            self.assertEqual(trainable & held, set(), "held-out row leaked into training")
            tr_pairs = {(r[0][0], r[0][1]) for r in t.train} | {(r[0][0], r[0][1]) for r in t.test}
            hd_pairs = {(r[0][0], r[0][1]) for r in t.heldout}
            self.assertEqual(tr_pairs & hd_pairs, set(), "held-out PAIR appeared in training")

    def test_fix1_mirrored_slot_orders_share_one_split(self):
        """Integrity fix 1, found by adversarial re-run.

        `(op, IDENT, x)` and `(IDENT, op, x)` compute the SAME function. Splitting
        them independently put 69% of primitive-test rows into training under the
        mirrored slot order, inflating primitive accuracy 0.86-0.92 -> 0.98.
        """
        from energy_memory.betb import build_family

        s = build_family("permutation").build(K=2, frac=0.7,
                                              gen=torch.Generator().manual_seed(0))
        prim = s.tasks[0]
        key = lambda toks: (max(toks[0], toks[1]), toks[2:])
        tr = {key(r[0]) for r in prim.train}
        te = {key(r[0]) for r in prim.test}
        self.assertEqual(tr & te, set(),
                         "a (operator, state) fact is in train under one slot order "
                         "and in test under the other")
        from collections import Counter
        self.assertEqual(set(Counter(key(r[0]) for r in prim.train).values()), {2},
                         "each trained (operator, state) must appear in BOTH slot orders")

    def test_fix2_no_composite_collides_with_a_primitive(self):
        """Integrity fix 2: rejection sampling.

        Without it ~5% of held-out cells have g_j∘g_i equal to some primitive g_l
        or the identity, so the answer is recallable without composing at all
        (measured 0.428 accuracy on those cells vs 0.224 on genuine ones).
        """
        from energy_memory.betb import build_family

        fam = build_family("permutation", m=5, k=3, n_ops=6)
        ident = fam._identity()
        for seed in range(12):
            ops = fam._sample_operators(torch.Generator().manual_seed(seed))
            self.assertEqual(len(set(ops)), 6)
            self.assertNotIn(ident, ops)
            for i in range(6):
                for j in range(6):
                    if i == j:
                        continue
                    c = fam._compose(ops[j], ops[i])
                    self.assertNotEqual(c, ident, f"seed {seed}: composite is the identity")
                    self.assertNotIn(c, ops, f"seed {seed}: composite equals a primitive")

    def test_abelian_control_makes_the_shortcut_legal(self):
        """The matched control must be structurally different in exactly one way.

        In `(Z_5)^3` composition IS pooling (order-independent), so a model that
        pools the two operator embeddings gets held-out pairs right. That is why
        the control's gap collapses to +0.060 while the S_5 gap is +0.642.
        """
        from energy_memory.betb import build_family

        ab = build_family("permutation", group="cyclic", m=5, k=3, n_ops=6)
        ops = ab._sample_operators(torch.Generator().manual_seed(0))
        for i in range(6):
            for j in range(6):
                self.assertEqual(ab._compose(ops[j], ops[i]), ab._compose(ops[i], ops[j]),
                                 "cyclic control must be commutative")

        sym = build_family("permutation", group="symmetric", m=5, k=3, n_ops=6)
        sops = sym._sample_operators(torch.Generator().manual_seed(0))
        noncomm = sum(1 for i in range(6) for j in range(6)
                      if sym._compose(sops[j], sops[i]) != sym._compose(sops[i], sops[j]))
        self.assertGreater(noncomm, 0, "symmetric group sample was accidentally abelian")

        s = ab.build(K=2, frac=0.7, gen=torch.Generator().manual_seed(0))
        self.assertEqual(s.n_classes, 125)
        self.assertEqual(len(s.tasks[1].heldout_groups), 8)

    def test_blocks_use_disjoint_token_ranges(self):
        from energy_memory.betb import build_family

        s = build_family("permutation").build(K=4, frac=0.7,
                                              gen=torch.Generator().manual_seed(0))
        b0 = {t for task in s.tasks[:2] for (inp, _) in task.train for t in inp}
        b1 = {t for task in s.tasks[2:] for (inp, _) in task.train for t in inp}
        self.assertEqual(b0 & b1, set())


@unittest.skipIf(torch is None, "torch required")
class TestContinualNet(unittest.TestCase):

    def test_head_modes(self):
        from energy_memory.betb import ContinualNet

        shared = ContinualNet(20, 5, 4, n_inputs=2, head_mode="shared", seed=0)
        per = ContinualNet(20, 5, 4, n_inputs=2, head_mode="per_task", seed=0)
        self.assertTrue(hasattr(shared, "head"))
        self.assertTrue(hasattr(per, "heads"))
        self.assertEqual(len(per.heads), 4)

        x = torch.tensor([[1, 2], [3, 4]])
        # shared head ignores the task id; per-task head does not
        self.assertTrue(torch.equal(shared(x, 0), shared(x, 3)))
        self.assertFalse(torch.equal(per(x, 0), per(x, 3)))

    def test_reproduces_exp83_initialisation(self):
        """Anchor parity: same seed => same weights as the published harness.

        Reports 137-139 compare against exp83/84 numbers. If initialisation drifts,
        every cross-report anchor silently stops being an anchor.
        """
        from energy_memory.betb import ContinualNet

        torch.manual_seed(0 * 7919 + 1)
        emb = torch.nn.Embedding(20, 8)
        mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * 8, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU())
        heads = torch.nn.ModuleList([torch.nn.Linear(16, 5) for _ in range(3)])

        net = ContinualNet(20, 5, 3, n_inputs=2, embed=8, hidden=16, seed=0, head_mode="per_task")
        self.assertTrue(torch.allclose(net.emb.weight, emb.weight))
        self.assertTrue(torch.allclose(net.mlp[0].weight, mlp[0].weight))
        self.assertTrue(torch.allclose(net.heads[2].weight, heads[2].weight))

    def test_forward_concat_matches_exp83_semantics(self):
        """exp83 did cat([emb(a), emb(b)]); the general path flattens [N, n_inputs]."""
        from energy_memory.betb import ContinualNet

        net = ContinualNet(20, 5, 2, n_inputs=2, embed=8, hidden=16, seed=1, head_mode="per_task")
        a = torch.tensor([1, 5]); b = torch.tensor([2, 6])
        expected = net.heads[0](net.mlp(torch.cat([net.emb(a), net.emb(b)], -1)))
        got = net(torch.stack([a, b], dim=1), 0)
        self.assertTrue(torch.allclose(expected, got))


@unittest.skipIf(torch is None, "torch required")
class TestAntiHomunculus(unittest.TestCase):
    """CLAUDE.md measurement rule 6, enforced instead of asserted in a docstring.

    `legacy/phase4/replay_loop.py:800-826` reads a metric against a threshold and
    branches to delete, under a docstring claiming the continuous dynamic
    "replaces binary deletion" — and its guard only fires when `coverage_lambda >
    0`, which defaults to 0.0. That is why prose compliance is not evidence.
    """

    def test_step_cannot_read_a_metric(self):
        from energy_memory.betb.consolidators import RECIPES

        for name, cls in RECIPES.items():
            sig = inspect.signature(cls.step)
            params = [p for p in sig.parameters if p != "self"]
            self.assertEqual(
                params, [],
                f"{name}.step() takes {params}: a consolidator that accepts anything "
                f"beyond self can be handed a metric and become a supervisor",
            )

    def test_consolidators_are_pure_functions_of_weights(self):
        """Same weights in => same weights out, regardless of any external state."""
        from energy_memory.betb.consolidators import RECIPES

        for name, cls in RECIPES.items():
            with self.subTest(recipe=name):
                torch.manual_seed(0)
                p1 = torch.nn.Parameter(torch.randn(6, 4))
                p2 = torch.nn.Parameter(p1.detach().clone())
                torch.manual_seed(7)
                c1 = cls([p1])
                torch.manual_seed(7)
                c2 = cls([p2])
                for _ in range(3):
                    c1.step()
                    c2.step()
                self.assertTrue(torch.allclose(p1, p2), f"{name} is not deterministic in its weights")

    def test_each_recipe_actually_changes_weights(self):
        from energy_memory.betb.consolidators import RECIPES

        for name, cls in RECIPES.items():
            with self.subTest(recipe=name):
                torch.manual_seed(0)
                p = torch.nn.Parameter(torch.randn(6, 4))
                before = p.detach().clone()
                c = cls([p])
                # a consolidator's reference is its engage-time state, so it must be
                # perturbed before it has anything to pull back toward
                with torch.no_grad():
                    p.add_(torch.randn_like(p) * 0.5)
                for _ in range(5):
                    c.step()
                self.assertFalse(torch.allclose(p, before, atol=1e-6),
                                 f"{name}.step() was inert")

    def test_ewc_anchor_pulls_toward_its_frozen_reference(self):
        from energy_memory.betb import EWCAnchor

        p = torch.nn.Parameter(torch.zeros(4, 3))
        anchor = EWCAnchor([p], lam=0.5)
        with torch.no_grad():
            p.add_(1.0)
        anchor.step()
        # one step of lam=0.5 halves the distance to the frozen reference (zeros)
        self.assertTrue(torch.allclose(p, torch.full((4, 3), 0.5)))


@unittest.skipIf(torch is None, "torch required")
class TestProvenance(unittest.TestCase):

    def test_declared_but_unexecuted_control_raises(self):
        """The exact failure that put an unrun control into reports/134:15."""
        from energy_memory.betb import MissingControlError, Provenance

        prov = Provenance(
            experiment="t", git_sha="x", argv=[], seeds=[0], config={},
            declared_controls=["scratch_denominator", "frozen_model_in_context"],
            executed_controls=["scratch_denominator"],
        )
        with self.assertRaises(MissingControlError) as ctx:
            prov.verify()
        self.assertIn("frozen_model_in_context", str(ctx.exception))

    def test_fully_executed_declaration_passes(self):
        from energy_memory.betb import Provenance

        Provenance(
            experiment="t", git_sha="x", argv=[], seeds=[0], config={},
            declared_controls=["a"], executed_controls=["a", "b"],
        ).verify()

    def test_merge_refuses_mismatched_configs(self):
        """The old per-experiment mergers never checked; a headline could mix runs."""
        import json
        import tempfile
        from pathlib import Path

        from energy_memory.betb import merge_shards

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, p_val in enumerate([17, 19]):
                doc = {
                    "provenance": {"config": {"p": p_val, "K": 10}, "seeds": [i]},
                    "per_seed_raw": {str(i): {"x": i}},
                }
                fp = Path(tmp) / f"shard{i}.json"
                fp.write_text(json.dumps(doc))
                paths.append(str(fp))
            with self.assertRaises(ValueError):
                merge_shards(paths)

    def test_merge_combines_compatible_shards(self):
        import json
        import tempfile
        from pathlib import Path

        from energy_memory.betb import merge_shards

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(3):
                doc = {
                    "provenance": {"config": {"p": 17, "K": 10, "shard": i}, "seeds": [i]},
                    "per_seed_raw": {str(i): {"x": i}},
                }
                fp = Path(tmp) / f"shard{i}.json"
                fp.write_text(json.dumps(doc))
                paths.append(str(fp))
            merged = merge_shards(paths)
            self.assertEqual(sorted(merged["per_seed_raw"]), ["0", "1", "2"])
            self.assertEqual(merged["provenance"]["seeds"], [0, 1, 2])
            self.assertEqual(merged["provenance"]["n_shards_merged"], 3)


@unittest.skipIf(torch is None, "torch required")
class TestHarnessRuns(unittest.TestCase):

    def _stream(self, K=4):
        from energy_memory.betb import build_family

        fam = build_family("permutation", m=4, k=2, n_ops=4, heldout_frac=0.3)
        return fam.build(K, 0.7, torch.Generator().manual_seed(0))

    def test_all_arms_run_and_report_steps(self):
        from energy_memory.betb import ARMS_2X2, make_consolidator, run_arm

        s = self._stream()
        ewc = make_consolidator("ewc", lam=0.01)
        for arm in ARMS_2X2:
            with self.subTest(arm=arm):
                r = run_arm(arm, s, embed=8, hidden=16, max_steps=40, crit=0.95,
                            eval_every=10, replay_frac=0.5, lr=1e-3, seed=0,
                            device="cpu", head_mode="shared", consolidator_factory=ewc)
                self.assertEqual(len(r.steps), len(s))
                self.assertTrue(all(x is not None for x in r.steps))

    def test_consolidation_arm_requires_a_recipe(self):
        from energy_memory.betb import run_arm

        s = self._stream()
        with self.assertRaises(ValueError):
            run_arm("consol_only", s, embed=8, hidden=16, max_steps=10, crit=0.95,
                    eval_every=5, replay_frac=0.5, lr=1e-3, seed=0, device="cpu",
                    head_mode="shared", consolidator_factory=None)

    def test_geometric_eval_schedule_beats_grid_resolution(self):
        """The FTSR quantization bug flagged in 138/139 and never fixed.

        A fixed grid can only report multiples of `eval_every`, so a ratio of two
        such counts inherits coarse, scale-dependent rounding. Tested on the
        schedule directly rather than through training, so the assertion does not
        depend on whether a particular toy happens to converge.
        """
        from energy_memory.betb.continual import eval_checks

        fixed = eval_checks(4000, 100, "fixed")
        geo = eval_checks(4000, 100, "geometric")

        self.assertTrue(all(v % 100 == 0 for v in fixed))
        self.assertTrue(any(v % 100 != 0 for v in geo),
                        "geometric schedule landed only on grid multiples")
        # far finer resolution early, where the fast arms actually stop
        self.assertGreater(len({v for v in geo if v <= 400}),
                           len({v for v in fixed if v <= 400}))
        # relative spacing stays bounded instead of blowing up at small step counts
        s = sorted(geo)
        ratios = [b / a for a, b in zip(s, s[1:]) if a > 0]
        self.assertLess(max(ratios), 2.1)

    def test_fixed_schedule_still_reproduces_the_published_grid(self):
        """Reports 134-139 ran the fixed grid; it must stay available for anchors."""
        from energy_memory.betb.continual import eval_checks

        self.assertEqual(sorted(eval_checks(500, 100, "fixed")), [100, 200, 300, 400, 500])

    def test_retention_matrix_metrics(self):
        from energy_memory.betb import retention_matrix_metrics

        # perfect retention: diagonal preserved at the end
        m = retention_matrix_metrics([[1.0], [1.0, 1.0], [1.0, 1.0, 1.0]])
        self.assertAlmostEqual(m["ACC"], 1.0)
        self.assertAlmostEqual(m["BWT"], 0.0)
        # catastrophic forgetting of task 0 shows up as negative BWT
        m2 = retention_matrix_metrics([[1.0], [0.0, 1.0]])
        self.assertLess(m2["BWT"], 0.0)


if __name__ == "__main__":
    unittest.main()
