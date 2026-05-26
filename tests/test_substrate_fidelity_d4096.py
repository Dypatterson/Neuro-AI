"""Substrate-fidelity tests at the production dimensionality D=4096.

The audit at ``audit-phase5-2026-05-26.md`` §4.1 / §9 found that every prior
bind/unbind test in this repo stops at D ≤ 512, while production runs at
D=4096 (``experiments/44_phase5_prime_bundle_first.py:933``,
``src/energy_memory/substrate/torch_fhrr.py:27``). Phase 5 bundle-first
chains 16 binds at D=4096; that chain has never been tested. These tests
close that gap for both the pure-Python and Torch FHRR backends.

This is infrastructure, not an experiment: no headline metric, no report.
"""

from __future__ import annotations

import math
import random
import unittest

from energy_memory.substrate import FHRR


D = 4096
SEED = 20260526


def _torch_available():
    try:
        from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


class SingleBindUnbindD4096Tests(unittest.TestCase):
    """Family 1: single bind/unbind roundtrip at D=4096."""

    def test_pure_python_single_bind_unbind(self):
        random.seed(SEED)
        substrate = FHRR(dim=D, seed=SEED)
        role = substrate.random_vector()
        filler = substrate.random_vector()

        recovered = substrate.unbind(substrate.bind(role, filler), role)
        cos = substrate.similarity(filler, recovered)

        print(f"[single-bind pure-python D={D}] cosine={cos:.6f}")
        self.assertGreater(cos, 0.99)

    def test_torch_single_bind_unbind(self):
        if not _torch_available():
            self.skipTest("torch is not installed")
        import torch
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        torch.manual_seed(SEED)
        random.seed(SEED)
        substrate = TorchFHRR(dim=D, seed=SEED)
        role = substrate.random_vector()
        filler = substrate.random_vector()

        recovered = substrate.unbind(substrate.bind(role, filler), role)
        cos = substrate.similarity(filler, recovered)

        print(f"[single-bind torch D={D}] cosine={cos:.6f}")
        self.assertGreater(cos, 0.99)


class Chained16BindD4096Tests(unittest.TestCase):
    """Family 2: 16-role bundle (the actual production binding depth)."""

    N_ROLES = 16
    THRESHOLD = 0.10  # well above 1/sqrt(D) ~= 0.0156 random-pair baseline

    def test_pure_python_chained_16_bind(self):
        random.seed(SEED)
        substrate = FHRR(dim=D, seed=SEED)
        roles = substrate.random_vectors(self.N_ROLES)
        fillers = substrate.random_vectors(self.N_ROLES)

        bound_pairs = [substrate.bind(r, f) for r, f in zip(roles, fillers)]
        bundle = substrate.bundle(bound_pairs)

        recoveries = []
        for k in range(self.N_ROLES):
            recovered = substrate.unbind(bundle, roles[k])
            cos = substrate.similarity(fillers[k], recovered)
            recoveries.append(cos)

        mean_cos = sum(recoveries) / len(recoveries)
        min_cos = min(recoveries)
        print(
            f"[chained-16 pure-python D={D}] mean={mean_cos:.6f} "
            f"min={min_cos:.6f} max={max(recoveries):.6f}"
        )

        for k, cos in enumerate(recoveries):
            self.assertGreater(
                cos,
                self.THRESHOLD,
                f"role {k}: cosine={cos:.6f} below threshold {self.THRESHOLD}",
            )

    def test_torch_chained_16_bind(self):
        if not _torch_available():
            self.skipTest("torch is not installed")
        import torch
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        torch.manual_seed(SEED)
        random.seed(SEED)
        substrate = TorchFHRR(dim=D, seed=SEED)
        roles = substrate.random_vectors(self.N_ROLES)
        fillers = substrate.random_vectors(self.N_ROLES)

        bound_pairs = [substrate.bind(roles[k], fillers[k]) for k in range(self.N_ROLES)]
        bundle = substrate.bundle(bound_pairs)

        recoveries = []
        for k in range(self.N_ROLES):
            recovered = substrate.unbind(bundle, roles[k])
            cos = substrate.similarity(fillers[k], recovered)
            recoveries.append(cos)

        mean_cos = sum(recoveries) / len(recoveries)
        min_cos = min(recoveries)
        print(
            f"[chained-16 torch D={D}] mean={mean_cos:.6f} "
            f"min={min_cos:.6f} max={max(recoveries):.6f}"
        )

        for k, cos in enumerate(recoveries):
            self.assertGreater(
                cos,
                self.THRESHOLD,
                f"role {k}: cosine={cos:.6f} below threshold {self.THRESHOLD}",
            )


class BundleCapacityCurveD4096Tests(unittest.TestCase):
    """Family 3: bundle capacity curve at D=4096 for N in {4,8,16,32,64}."""

    NS = (4, 8, 16, 32, 64)

    def _measure(self, substrate, n):
        roles = substrate.random_vectors(n)
        fillers = substrate.random_vectors(n)
        bound_pairs = [substrate.bind(roles[k], fillers[k]) for k in range(n)]
        bundle = substrate.bundle(bound_pairs)
        sims = []
        for k in range(n):
            recovered = substrate.unbind(bundle, roles[k])
            sims.append(substrate.similarity(fillers[k], recovered))
        return sum(sims) / len(sims)

    def test_bundle_capacity_curve(self):
        if _torch_available():
            import torch
            from energy_memory.substrate.torch_fhrr import TorchFHRR

            torch.manual_seed(SEED)
            random.seed(SEED)
            substrate = TorchFHRR(dim=D, seed=SEED)
            backend = "torch"
        else:
            random.seed(SEED)
            substrate = FHRR(dim=D, seed=SEED)
            backend = "pure-python"

        curve = []
        for n in self.NS:
            mean_cos = self._measure(substrate, n)
            curve.append((n, mean_cos))

        print(f"[bundle-capacity curve D={D} backend={backend}]")
        for n, mean_cos in curve:
            print(f"  N={n:>3d}  mean_cosine={mean_cos:.6f}")

        # Sanity floor at N=4.
        self.assertGreaterEqual(
            curve[0][1],
            0.40,
            f"N=4 recovery {curve[0][1]:.6f} below sanity floor 0.40",
        )

        # Monotonically non-increasing within 5% tolerance.
        tol = 0.05
        for i in range(1, len(curve)):
            prev_n, prev_cos = curve[i - 1]
            cur_n, cur_cos = curve[i]
            # Allow small backward jumps due to noise: cur_cos must not
            # exceed prev_cos by more than tol*max(prev_cos, eps).
            slack = tol * max(abs(prev_cos), 1e-6)
            self.assertLessEqual(
                cur_cos,
                prev_cos + slack,
                f"recovery not monotonically non-increasing: "
                f"N={prev_n}->{cur_n}, {prev_cos:.6f}->{cur_cos:.6f}",
            )


class PurePythonTorchParityD4096Tests(unittest.TestCase):
    """Family 4: pure-Python <-> Torch FHRR parity at D=4096."""

    def test_bind_unbind_parity(self):
        if not _torch_available():
            self.skipTest("torch is not installed")
        import torch
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        torch.manual_seed(SEED)
        random.seed(SEED)

        # Construct matched random phases independently of the two
        # substrates' RNG paths, then materialize role+filler in both
        # backends from the same phase arrays. Avoids relying on
        # cross-backend RNG equivalence.
        phases_role = [random.random() * 2.0 * math.pi for _ in range(D)]
        phases_filler = [random.random() * 2.0 * math.pi for _ in range(D)]

        # Pure-Python side.
        import cmath
        py_role = tuple(cmath.exp(1j * p) for p in phases_role)
        py_filler = tuple(cmath.exp(1j * p) for p in phases_filler)
        py_bound = FHRR(dim=D).bind(py_role, py_filler)
        py_unbound = FHRR(dim=D).unbind(py_bound, py_role)

        # Torch side: cast phases to a Torch tensor on a TorchFHRR's device.
        substrate_t = TorchFHRR(dim=D, seed=SEED)
        device = substrate_t.device
        ones = torch.ones(D, dtype=torch.float32, device=device)
        t_phases_role = torch.tensor(phases_role, dtype=torch.float32, device=device)
        t_phases_filler = torch.tensor(phases_filler, dtype=torch.float32, device=device)
        t_role = torch.polar(ones, t_phases_role)
        t_filler = torch.polar(ones, t_phases_filler)
        t_bound = substrate_t.bind(t_role, t_filler)
        t_unbound = substrate_t.unbind(t_bound, t_role)

        # Compare per-component complex values. Cast pure-Python to
        # tensors on the same device for an apples-to-apples comparison.
        py_bound_t = torch.tensor(
            [(v.real, v.imag) for v in py_bound], dtype=torch.float32, device=device
        )
        py_unbound_t = torch.tensor(
            [(v.real, v.imag) for v in py_unbound], dtype=torch.float32, device=device
        )
        t_bound_pairs = torch.stack([t_bound.real, t_bound.imag], dim=-1).to(torch.float32)
        t_unbound_pairs = torch.stack([t_unbound.real, t_unbound.imag], dim=-1).to(torch.float32)

        max_abs_bind = (py_bound_t - t_bound_pairs).abs().max().item()
        max_abs_unbind = (py_unbound_t - t_unbound_pairs).abs().max().item()
        print(
            f"[parity D={D}] max_abs_bind_err={max_abs_bind:.3e} "
            f"max_abs_unbind_err={max_abs_unbind:.3e}"
        )

        # float32 rtol ~= 1e-5; allow a small absolute slack for values
        # whose magnitude approaches zero in either component.
        torch.testing.assert_close(
            py_bound_t, t_bound_pairs, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            py_unbound_t, t_unbound_pairs, rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
