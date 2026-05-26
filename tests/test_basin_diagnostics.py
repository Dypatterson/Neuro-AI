import unittest

import torch

from energy_memory.phase3.basin_diagnostics import (
    BasinTrace,
    BasinTraceBuffer,
    compute_basin_diagnostics,
    compute_basin_nc1,
    compute_basin_separability_nc2,
    compute_basin_separability_pairwise_mean,
)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp(min=1e-12)


def _random_unit(d: int, generator: torch.Generator) -> torch.Tensor:
    v = torch.randn(d, generator=generator)
    return _unit(v)


class BasinDiagnosticsTests(unittest.TestCase):
    def test_t1_collapsed_basins_nc1_near_zero(self):
        gen = torch.Generator().manual_seed(11)
        D = 128
        buf = BasinTraceBuffer(maxlen=20)
        for atom in range(4):
            v = _random_unit(D, gen)
            for _ in range(5):
                buf.append(BasinTrace(settled_state=v.clone(), top1_atom=atom))
        nc1, singletons = compute_basin_nc1(buf)
        self.assertEqual(set(nc1.keys()), {0, 1, 2, 3})
        self.assertEqual(singletons, set())
        for atom, val in nc1.items():
            self.assertLess(val, 0.01, f"atom {atom} NC1={val} not near 0")

    def test_t2_diffuse_basins_nc1_large(self):
        gen = torch.Generator().manual_seed(22)
        D = 128
        buf = BasinTraceBuffer(maxlen=20)
        for atom in range(4):
            for _ in range(5):
                buf.append(BasinTrace(settled_state=_random_unit(D, gen), top1_atom=atom))
        nc1, _ = compute_basin_nc1(buf)
        for atom, val in nc1.items():
            self.assertGreater(val, 2.0, f"atom {atom} NC1={val} not > 2")

    def test_t3_healthy_basins_nc1_intermediate(self):
        gen = torch.Generator().manual_seed(33)
        D = 128
        buf = BasinTraceBuffer(maxlen=20)
        for atom in range(4):
            centroid = _random_unit(D, gen)
            for _ in range(5):
                noise = torch.randn(D, generator=gen)
                state = _unit(centroid + 0.1 * noise)
                buf.append(BasinTrace(settled_state=state, top1_atom=atom))
        nc1, _ = compute_basin_nc1(buf)
        for atom, val in nc1.items():
            self.assertGreater(val, 0.01, f"atom {atom} NC1={val} not > 0.01")
            self.assertLess(val, 5.0, f"atom {atom} NC1={val} not < 5")

    def test_t4_etf_centroids_nc2_near_zero(self):
        D = 128
        K = 4
        gen = torch.Generator().manual_seed(44)
        Q = torch.linalg.qr(torch.randn(D, K, generator=gen))[0]
        ortho = Q.T
        mean = ortho.mean(dim=0, keepdim=True)
        etf = ortho - mean
        etf = etf / etf.norm(dim=1, keepdim=True).clamp(min=1e-12)
        buf = BasinTraceBuffer(maxlen=20)
        for atom in range(K):
            buf.append(BasinTrace(settled_state=etf[atom].clone(), top1_atom=atom))
        nc2 = compute_basin_separability_nc2(buf)
        self.assertIsNotNone(nc2)
        self.assertLess(nc2, 0.05, f"NC2={nc2} not near 0")

    def test_t5_collapsed_centroids_nc2_large(self):
        D = 128
        gen = torch.Generator().manual_seed(55)
        v = _random_unit(D, gen)
        buf = BasinTraceBuffer(maxlen=20)
        for atom in range(4):
            buf.append(BasinTrace(settled_state=v.clone(), top1_atom=atom))
        nc2 = compute_basin_separability_nc2(buf)
        self.assertIsNotNone(nc2)
        self.assertGreater(nc2, 1.0, f"NC2={nc2} not > 1.0")

    def test_t6_edge_cases(self):
        empty = BasinTraceBuffer(maxlen=5)
        diag = compute_basin_diagnostics(empty)
        self.assertEqual(diag.n_basins_observed, 0)
        self.assertEqual(diag.nc1_per_atom, {})
        self.assertIsNone(diag.separability_nc2)

        gen = torch.Generator().manual_seed(66)
        D = 32
        single_basin = BasinTraceBuffer(maxlen=5)
        for _ in range(3):
            single_basin.append(BasinTrace(settled_state=_random_unit(D, gen), top1_atom=7))
        nc2 = compute_basin_separability_nc2(single_basin)
        self.assertIsNone(nc2)

        singleton = BasinTraceBuffer(maxlen=5)
        singleton.append(BasinTrace(settled_state=_random_unit(D, gen), top1_atom=2))
        for _ in range(3):
            singleton.append(BasinTrace(settled_state=_random_unit(D, gen), top1_atom=3))
        nc1, singletons = compute_basin_nc1(singleton)
        self.assertIn(2, singletons)
        self.assertEqual(nc1[2], 1.0)

    def test_t7_complex_fhrr_tensors(self):
        gen = torch.Generator().manual_seed(77)
        D = 64
        buf = BasinTraceBuffer(maxlen=20)
        for atom in range(4):
            for _ in range(5):
                re = torch.randn(D, generator=gen)
                im = torch.randn(D, generator=gen)
                z = torch.complex(re, im)
                z = z / z.abs().pow(2).sum().sqrt().clamp(min=1e-12)
                buf.append(BasinTrace(settled_state=z, top1_atom=atom))
        diag = compute_basin_diagnostics(buf)
        self.assertEqual(len(diag.nc1_per_atom), 4)
        for val in diag.nc1_per_atom.values():
            self.assertTrue(torch.isfinite(torch.tensor(val)).item())
        self.assertIsNotNone(diag.separability_nc2)
        self.assertTrue(torch.isfinite(torch.tensor(diag.separability_nc2)).item())

    def test_t9_pairwise_mean_fallback(self):
        gen = torch.Generator().manual_seed(99)
        D = 128

        collapsed = BasinTraceBuffer(maxlen=20)
        v = _random_unit(D, gen)
        for atom in range(4):
            for _ in range(3):
                collapsed.append(BasinTrace(settled_state=v.clone(), top1_atom=atom))
        d_collapsed = compute_basin_separability_pairwise_mean(collapsed)
        self.assertLess(d_collapsed, 0.05)

        orthogonal = BasinTraceBuffer(maxlen=20)
        eye = torch.eye(D)
        for atom in range(4):
            for _ in range(3):
                orthogonal.append(BasinTrace(settled_state=eye[atom].clone(), top1_atom=atom))
        d_orth = compute_basin_separability_pairwise_mean(orthogonal)
        self.assertGreater(d_orth, 0.95)
        self.assertLess(d_orth, 1.05)

        single = BasinTraceBuffer(maxlen=5)
        for _ in range(3):
            single.append(BasinTrace(settled_state=_random_unit(D, gen), top1_atom=0))
        self.assertIsNone(compute_basin_separability_pairwise_mean(single))

    def test_t8_rolling_buffer(self):
        gen = torch.Generator().manual_seed(88)
        D = 16
        buf = BasinTraceBuffer(maxlen=5)
        traces = []
        for i in range(7):
            t = BasinTrace(settled_state=_random_unit(D, gen), top1_atom=i)
            traces.append(t)
            buf.append(t)
        self.assertEqual(len(buf), 5)
        grouped = buf.traces_by_atom()
        self.assertNotIn(0, grouped)
        self.assertNotIn(1, grouped)
        self.assertIn(6, grouped)


if __name__ == "__main__":
    unittest.main()
