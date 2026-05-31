"""Tests for WS-InfoNCE within-scene predictive codebook shaping (Phase-3, Stage-1).

Covers the anti-homunculus + fence + manifold conditions the design spec
(notes/emergent-codebook/phase-3-within-scene-predictive-jepa-design.md) makes binding:
- shaping requires a FROZEN buffer (batch-offline only);
- ``C'`` stays on the FHRR unit-phasor manifold (a valid codebook H can read);
- the contrastive InfoNCE loss decreases (it actually trains);
- non-value rows are left UNCHANGED (so cues built from context atoms are isolable);
- the module imports nothing from the fenced Phase-5' energy surfaces and never calls
  ``_energy_from_scores`` (Phase-5' bright line).
"""

import ast
import inspect
import unittest

import torch

from energy_memory.phase4 import ws_infonce
from energy_memory.phase4.ws_infonce import (
    FrozenSceneBuffer,
    WSInfoNCEConfig,
    shape_codebook,
    within_scene_slot_queries,
)
from energy_memory.phase2.encoding import build_position_vectors
from energy_memory.substrate.torch_fhrr import TorchFHRR


def _toy(D=256, V=20, L=8, W=5, N=40, seed=0):
    """A topic-corpus-shaped toy: value atoms = first L rows; context tokens drawn
    from rows [L, V); each scene's target = a value atom predictable from context."""
    sub = TorchFHRR(dim=D, seed=seed, device="cpu")
    codebook = sub.random_vectors(V)
    positions = torch.stack(build_position_vectors(sub, W))
    mpos = W // 2
    ctx = [p for p in range(W) if p != mpos]
    g = torch.Generator().manual_seed(seed * 9973 + 1)
    buf = FrozenSceneBuffer()
    windows = []
    for _ in range(N):
        z = int(torch.randint(0, L, (1,), generator=g))      # latent topic == target
        row = [0] * W
        for p in ctx:
            # context token depends on topic z (so the target is recoverable)
            row[p] = L + int((z * 3 + p) % (V - L))
        row[mpos] = z
        windows.append(row)
        buf.add(row, z)                                       # target_local_idx == z (in [0,L))
    buf.freeze("cpu")
    value_row_ids = torch.arange(L)
    return sub, codebook, value_row_ids, positions, buf, mpos, windows


class FrozenBufferTest(unittest.TestCase):
    def test_add_after_freeze_raises(self):
        buf = FrozenSceneBuffer()
        buf.add([1, 2, 3], 0)
        buf.freeze("cpu")
        with self.assertRaises(RuntimeError):
            buf.add([4, 5, 6], 1)

    def test_shape_before_freeze_raises(self):
        sub, codebook, vids, positions, _buf, mpos, _ = _toy()
        unfrozen = FrozenSceneBuffer()
        unfrozen.add([0, 1, 2, 3, 4], 0)  # not frozen
        with self.assertRaises(RuntimeError):
            shape_codebook(sub, codebook, vids, positions, unfrozen, mpos,
                           WSInfoNCEConfig(epochs=1))

    def test_empty_freeze_raises(self):
        with self.assertRaises(ValueError):
            FrozenSceneBuffer().freeze("cpu")


class ManifoldTest(unittest.TestCase):
    def test_c_prime_is_unit_phasor(self):
        sub, codebook, vids, positions, buf, mpos, _ = _toy()
        c_full, c_val, _info = shape_codebook(
            sub, codebook, vids, positions, buf, mpos, WSInfoNCEConfig(epochs=30))
        # every atom magnitude == 1 -> a valid FHRR codebook the graduated H can read
        self.assertTrue(torch.allclose(c_full.abs(), torch.ones_like(c_full.abs()), atol=1e-5))
        self.assertTrue(torch.allclose(c_val.abs(), torch.ones_like(c_val.abs()), atol=1e-5))
        self.assertEqual(tuple(c_val.shape), (vids.numel(), codebook.shape[1]))

    def test_non_value_rows_unchanged(self):
        # In the topic-corpus the value atoms (first L) do not appear in scene context;
        # shaping must leave the context rows byte-identical so cues are isolable.
        sub, codebook, vids, positions, buf, mpos, _ = _toy()
        c_full, _c_val, _info = shape_codebook(
            sub, codebook, vids, positions, buf, mpos, WSInfoNCEConfig(epochs=30))
        non_value = torch.ones(codebook.shape[0], dtype=torch.bool)
        non_value[vids] = False
        self.assertTrue(torch.equal(c_full[non_value], codebook[non_value]))

    def test_value_rows_actually_move(self):
        sub, codebook, vids, positions, buf, mpos, _ = _toy()
        c_full, _c_val, info = shape_codebook(
            sub, codebook, vids, positions, buf, mpos, WSInfoNCEConfig(epochs=60))
        self.assertFalse(torch.equal(c_full[vids], codebook[vids]))
        self.assertGreater(info["codebook_drift"], 0.0)


class TrainingTest(unittest.TestCase):
    def test_contrastive_loss_decreases(self):
        sub, codebook, vids, positions, buf, mpos, _ = _toy()
        _c, _v, info = shape_codebook(
            sub, codebook, vids, positions, buf, mpos,
            WSInfoNCEConfig(epochs=200, lr=0.05, tau=0.05, contrastive=True))
        self.assertLess(info["loss_end"], info["loss_start"])

    def test_seed_fixed_determinism(self):
        # Same inputs + same config -> identical C' (a precommitted offline pass).
        a = _toy(seed=1)
        b = _toy(seed=1)
        ca, _, _ = shape_codebook(*a[:5], a[5], WSInfoNCEConfig(epochs=40))
        cb, _, _ = shape_codebook(*b[:5], b[5], WSInfoNCEConfig(epochs=40))
        self.assertTrue(torch.equal(ca, cb))


class SlotQueryTest(unittest.TestCase):
    def test_slot_query_unit_magnitude_and_shape(self):
        sub, codebook, _vids, positions, buf, mpos, _ = _toy()
        slot = within_scene_slot_queries(sub, positions, codebook, buf.windows, mpos)
        self.assertEqual(tuple(slot.shape), (len(buf), codebook.shape[1]))
        self.assertTrue(torch.allclose(slot.abs(), torch.ones_like(slot.abs()), atol=1e-5))


class FenceTest(unittest.TestCase):
    """Phase-5' bright line: the shaping module must not touch the fenced energy
    surfaces (it owns its own log-softmax)."""

    def test_no_fenced_energy_imports_or_calls(self):
        # AST identifiers only — docstrings/comments naming the fence (documentation)
        # are fine; what is forbidden is importing or CALLING the fenced surfaces.
        tree = ast.parse(inspect.getsource(ws_infonce))
        identifiers: set[str] = set()
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            elif isinstance(node, ast.alias):
                modules.add(node.name)
        banned = ("_energy_from_scores", "mhn_energy", "cue_conditioned_scene_energy",
                  "delta_e_content_minus_role", "raw_scene_energy_v0")
        for name in banned:
            self.assertNotIn(name, identifiers,
                             f"WS-InfoNCE must not call fenced surface {name!r}")
        for mod in modules:
            self.assertNotIn("bridge_readouts", mod,
                             "WS-InfoNCE must not import phase5/bridge_readouts")
            self.assertNotIn("m1_role_energy", mod,
                             "WS-InfoNCE must not import phase5/m1_role_energy")


if __name__ == "__main__":
    unittest.main()
