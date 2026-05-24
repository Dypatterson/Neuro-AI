"""Smoke test for recommendation (c): MQAR-style associative recall.

The brainstorm's claim: MQAR-style capacity curves are the substrate's
missing unit test — they can distinguish 'role-binding works' from
'content-binding works on role-like cues' in a way the current ΔE
headline cannot. This smoke implements a minimal MQAR-like task against
the existing FHRR + Modern Hopfield substrate and measures the capacity
curve N ∈ {16, 32, 64, 128, 256, 512}.

The task. We sample N key vectors and N value vectors from the FHRR
codebook. We bind each (key, value) pair and store the bound vectors —
either by bundling them all into one super-vector (the classic HRR
'distributed memory' route) or as N separate patterns in the Hopfield
landscape (the MHN route). For each of n_queries random keys from the
stored set, we unbind the storage to recover the value and check
top-1 match against the value codebook.

We report top-1 recall as a function of N for both storage routes —
that's the capacity curve. We also vary D ∈ {1024, 4096} to confirm
the substrate's scaling.
"""
from __future__ import annotations
import math
import torch
from energy_memory.substrate.torch_fhrr import TorchFHRR
from energy_memory.memory.torch_hopfield import TorchHopfieldMemory


def hrr_bundle_recall(fhrr: TorchFHRR, N: int, n_queries: int, *, seed: int):
    """Store via single-vector bundle (Plate HRR). Cheapest variant.
    Capacity ~ D / log(N) ish."""
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    # Key + value codebooks (random unit-magnitude phasor).
    keys = fhrr.random_vectors(N)
    values = fhrr.random_vectors(N)
    # Bundle bind(k_i, v_i) — single super-memory.
    bound = keys * values  # element-wise FHRR bind
    M = fhrr.normalize(bound.sum(dim=0))
    # Query each key, unbind, top-1 against value codebook.
    correct = 0
    q_indices = torch.randint(0, N, (n_queries,), generator=g).tolist()
    for qi in q_indices:
        rec = fhrr.unbind(M, keys[qi])
        sims = fhrr.similarity_matrix(rec, values)
        pred = int(sims.argmax().item())
        if pred == qi:
            correct += 1
    return correct / n_queries


def hopfield_recall(fhrr: TorchFHRR, N: int, n_queries: int, *, seed: int, beta: float = 30.0):
    """Store each bind(k_i, v_i) as a separate MHN pattern. Use the
    existing TorchHopfieldMemory to settle the cue (unbind(bundle, k))
    and classify the settled state by similarity to values."""
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    keys = fhrr.random_vectors(N)
    values = fhrr.random_vectors(N)
    bound = keys * values  # [N, D]
    # Build MHN memory from bound patterns.
    hop = TorchHopfieldMemory(substrate=fhrr)
    for i in range(N):
        hop.store(bound[i], label=f"pair_{i}")
    correct = 0
    q_indices = torch.randint(0, N, (n_queries,), generator=g).tolist()
    for qi in q_indices:
        # Cue is the bound pair itself (perfect cue ⇒ should reproduce).
        # This is the easy version (the 'memorization' check); a harder
        # version would cue with bind(k, *something else*) and ask whether
        # the substrate finds the right pair by key alignment.
        cue = bound[qi]
        result = hop.retrieve(cue, beta=beta)
        # Unbind retrieved pattern with key to recover value.
        rec_val = fhrr.unbind(result.state, keys[qi])
        sims = fhrr.similarity_matrix(rec_val, values)
        pred = int(sims.argmax().item())
        if pred == qi:
            correct += 1
    return correct / n_queries


def hopfield_recall_keyonly(fhrr: TorchFHRR, N: int, n_queries: int, *, seed: int, beta: float = 30.0):
    """The harder version: cue with the KEY alone (not the bound pair),
    settle in the bound-vector landscape, recover the value. This is
    the 'true' associative-recall test: does the substrate find a pair
    when you give it only one factor?"""
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    keys = fhrr.random_vectors(N)
    values = fhrr.random_vectors(N)
    bound = keys * values
    hop = TorchHopfieldMemory(substrate=fhrr)
    for i in range(N):
        hop.store(bound[i], label=f"pair_{i}")
    correct = 0
    q_indices = torch.randint(0, N, (n_queries,), generator=g).tolist()
    for qi in q_indices:
        cue = keys[qi]  # KEY ONLY
        result = hop.retrieve(cue, beta=beta)
        rec_val = fhrr.unbind(result.state, keys[qi])
        sims = fhrr.similarity_matrix(rec_val, values)
        pred = int(sims.argmax().item())
        if pred == qi:
            correct += 1
    return correct / n_queries


def main():
    n_queries = 32
    seeds = [17, 11, 23]
    Ns = [16, 32, 64, 128, 256, 512]
    Ds = [1024, 4096]

    print(f"\n=== HRR-bundle recall (single super-vector storage) ===")
    print(f"{'D':>5}  {'N':>5}  {'mean top1':>10}  {'spread':>8}  per-seed")
    for D in Ds:
        for N in Ns:
            accs = []
            for seed in seeds:
                fhrr = TorchFHRR(dim=D, seed=seed, device="cpu")
                accs.append(hrr_bundle_recall(fhrr, N, n_queries, seed=seed))
            m_ = sum(accs) / len(accs); sp = max(accs) - min(accs)
            print(f"{D:>5}  {N:>5}  {m_:>10.3f}  {sp:>8.3f}  {[f'{a:.2f}' for a in accs]}")

    print(f"\n=== Hopfield, perfect-cue (the bound pair as cue — memorization check) ===")
    print(f"{'D':>5}  {'N':>5}  {'mean top1':>10}  {'spread':>8}  per-seed")
    for D in Ds:
        for N in Ns:
            accs = []
            for seed in seeds:
                fhrr = TorchFHRR(dim=D, seed=seed, device="cpu")
                accs.append(hopfield_recall(fhrr, N, n_queries, seed=seed))
            m_ = sum(accs) / len(accs); sp = max(accs) - min(accs)
            print(f"{D:>5}  {N:>5}  {m_:>10.3f}  {sp:>8.3f}  {[f'{a:.2f}' for a in accs]}")

    print(f"\n=== Hopfield, KEY-ONLY cue (true MQAR: associative recall by key) ===")
    print(f"{'D':>5}  {'N':>5}  {'mean top1':>10}  {'spread':>8}  per-seed")
    for D in Ds:
        for N in Ns:
            accs = []
            for seed in seeds:
                fhrr = TorchFHRR(dim=D, seed=seed, device="cpu")
                accs.append(hopfield_recall_keyonly(fhrr, N, n_queries, seed=seed))
            m_ = sum(accs) / len(accs); sp = max(accs) - min(accs)
            print(f"{D:>5}  {N:>5}  {m_:>10.3f}  {sp:>8.3f}  {[f'{a:.2f}' for a in accs]}")


if __name__ == "__main__":
    main()
