"""Stage-1 write-then-read gate (Phase-3 consolidation-write sub-program).

Spec: notes/emergent-codebook/phase-3-consolidation-write-design.md
Plan: brainstorm-workspace/2026-05-30-research-grounded-plan/02-forward-plan-FINAL.md

This is the cheapest-decisive immediate gate. It builds a partial-context role-
recovery toy (scene-MHN -> unbind query role -> content-MHN cleanup), validates
the toy is in the HARD regime (store-as-is recovery near chance == the 062-066
null) before trusting any "write helps" signal, then runs:

  G-A  frozen-substrate refit-readout : after the baseline contrastive write,
        freeze the content codebook; refit a fresh nearest-prototype readout on
        a HELD-OUT split. refit >> native  -> the defect is the READOUT, not the
        objective (un-park GSBC); refit also fails -> structure truly never
        written.  Decision rule grounded on Reports 066+111 + the random-codebook
        control (NOT on arxiv:2310.05644, which transfers as protocol only).

  G-D  value-codebook top_index_hits Selectivity-Δ headline, two-floor Wilson
        rule, with the mandatory same-test-set controls:
          - role-shuffle  : unbind at a fixed-point-free deranged role (the
            selectivity control; gauge-safe because roles are distinct)
          - no-negatives  : pull-only write must collapse Δ
          - random-codebook : write into a random codebook must collapse Δ
          - perfect-cue   : UPPER-BOUND arm (cue == the target content atom)

Anti-homunculus: every write is an OFFLINE/BATCH pass (runtime error-driven
BANNED); the swap-negative is a PRECOMMITTED seed-fixed role-class draw, never
sims.argmax (the existing error_driven_learner.py:106 thermostat being removed);
the readouts are offline measurements on a held-out split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional

# Make the repo root importable so `experiments.44_*` resolves when this file is
# run directly (sys.path[0] is the experiments/ dir otherwise).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from energy_memory.phase2.encoding import build_position_vectors
from energy_memory.phase3.basin_readout import (
    selectivity_delta,
    top_index_hits,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR

# Reuse the canonical batched Hopfield retrieve (returns state, top_index,
# entropy, margin) from the bundle-first harness rather than reinventing it.
import importlib

EXP44 = importlib.import_module("experiments.44_phase5_prime_bundle_first")
_retrieve = EXP44._batched_hopfield_retrieve


# --------------------------------------------------------------------------- #
# Toy construction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Toy:
    fhrr: TorchFHRR
    roles: torch.Tensor      # [K, D]
    content: torch.Tensor    # [C, D]
    fill: torch.Tensor       # [N, K] long : which content atom fills each role
    scenes: torch.Tensor     # [N, D]      : bundle of role*content bindings
    K: int
    C: int


def build_toy(*, D: int, K: int, C: int, N: int, seed: int, device: str) -> Toy:
    fhrr = TorchFHRR(dim=D, seed=seed, device=device)
    roles = torch.stack(build_position_vectors(fhrr, K), dim=0)  # [K, D]
    content = fhrr.random_vectors(C)                              # [C, D]
    # Each role in each scene is filled by a uniformly-random content atom,
    # under the seed-fixed generator (so the toy is fully reproducible).
    fill = torch.randint(
        0, C, (N, K), generator=fhrr.generator, device="cpu"
    ).to(device)
    scenes = torch.stack([
        fhrr.bundle([roles[r] * content[int(fill[s, r])] for r in range(K)])
        for s in range(N)
    ], dim=0)
    return Toy(fhrr=fhrr, roles=roles, content=content, fill=fill,
               scenes=scenes, K=K, C=C)


def _deranged_roles(K: int, seed: int, device: str) -> torch.Tensor:
    """A fixed-point-free permutation of role indices (the gauge-safe role
    shuffle: every role maps to a DIFFERENT role, so it is not a vacuous
    relabel)."""
    g = torch.Generator().manual_seed(seed * 7919 + 13)
    for _ in range(1000):
        perm = torch.randperm(K, generator=g)
        if bool((perm != torch.arange(K)).all()):
            return perm.to(device)
    # K=1 has no derangement; fall back to identity (caller guards K>=2).
    return torch.arange(K, device=device)


# --------------------------------------------------------------------------- #
# Read paths (all native; top_index_hits basin-membership)
# --------------------------------------------------------------------------- #
def _observed_roles(K: int, query_role: int, observed: int) -> list[int]:
    """The context roles included in the cue (the first ``observed`` roles other
    than the query role). Fewer observed roles == harder (key-only at observed=1)."""
    others = [r for r in range(K) if r != query_role]
    n = max(1, min(observed, len(others)))
    return others[:n]


def _partial_cue(
    toy: Toy, scene_ids: torch.Tensor, query_role: int, observed: int
) -> torch.Tensor:
    """Cue = bundle of the observed (role, filler) bindings. Recovering the query
    role's filler needs the between-role association consolidation should write."""
    fhrr = toy.fhrr
    obs = _observed_roles(toy.K, query_role, observed)
    cues = []
    for s in scene_ids.tolist():
        terms = [toy.roles[r] * toy.content[int(toy.fill[s, r])] for r in obs]
        cues.append(fhrr.bundle(terms))
    return torch.stack(cues, dim=0)


def native_recoverability(
    toy: Toy, content: torch.Tensor, scene_ids: torch.Tensor,
    *, query_role: int, unbind_role: int, observed: int, beta: float, max_iter: int,
) -> tuple[int, int, float, float]:
    """scene-MHN(cue) -> unbind at unbind_role -> content-MHN cleanup.
    Returns (hits, n, mean_entropy, mean_margin). unbind_role != query_role is
    the role-shuffle control."""
    fhrr = toy.fhrr
    cue = _partial_cue(toy, scene_ids, query_role, observed)
    scene_state, _, _, _ = _retrieve(
        fhrr, toy.scenes, cue, beta=beta, max_iter=max_iter)
    unbound = fhrr.normalize(fhrr.unbind(scene_state, toy.roles[unbind_role]))
    _, top_index, entropy, margin = _retrieve(
        fhrr, content, unbound, beta=beta, max_iter=max_iter)
    target = toy.fill[scene_ids, query_role]
    hits = top_index_hits(top_index, target)
    return hits, len(scene_ids), float(entropy.mean()), float(margin.mean())


def perfect_cue_recoverability(
    toy: Toy, content: torch.Tensor, scene_ids: torch.Tensor,
    *, query_role: int, beta: float, max_iter: int,
) -> tuple[int, int]:
    """UPPER BOUND: cue the content-MHN with the exact target content atom."""
    target = toy.fill[scene_ids, query_role]
    cue = content[target]
    _, top_index, _, _ = _retrieve(
        toy.fhrr, content, cue, beta=beta, max_iter=max_iter)
    return top_index_hits(top_index, target), len(scene_ids)


# --------------------------------------------------------------------------- #
# Write rules (all OFFLINE/BATCH; modify the content codebook)
# --------------------------------------------------------------------------- #
def _slot_queries(
    toy: Toy, scene_ids: torch.Tensor, query_role: int, observed: int,
    *, beta: float = 10.0, max_iter: int = 12,
) -> torch.Tensor:
    """The unbind output of the retrieved scene at the query role (the direction
    the write pulls the target content atom toward)."""
    fhrr = toy.fhrr
    cue = _partial_cue(toy, scene_ids, query_role, observed)
    scene_state, _, _, _ = _retrieve(
        fhrr, toy.scenes, cue, beta=beta, max_iter=max_iter)
    return fhrr.normalize(fhrr.unbind(scene_state, toy.roles[query_role]))


def consolidate(
    toy: Toy, *, rule: str, lr_pull: float, lr_push: float,
    epochs: int, beta: float, max_iter: int, seed: int, observed: int,
) -> torch.Tensor:
    """Offline batch contrastive write over all TRAIN (scene, role) observations.

    rule ∈ {none, pull_only, baseline_contrastive, swap_negative}.
    Returns the (possibly modified) content codebook. Never mutates toy.content.
    """
    fhrr = toy.fhrr
    content = toy.content.clone()
    if rule == "none":
        return content
    N = toy.scenes.shape[0]
    scene_ids = torch.arange(N, device=toy.content.device)
    g = torch.Generator().manual_seed(seed * 104729 + 7)
    for _ in range(epochs):
        pull_acc = {c: [] for c in range(toy.C)}
        push_acc = {c: [] for c in range(toy.C)}
        for qr in range(toy.K):
            sq = _slot_queries(toy, scene_ids, qr, observed,
                               beta=beta, max_iter=max_iter)  # [N, D]
            target = toy.fill[scene_ids, qr]
            for i in range(N):
                tid = int(target[i])
                pull_acc[tid].append(sq[i])
                if rule == "pull_only":
                    continue
                if rule == "baseline_contrastive":
                    # self-mined argmax negative (the phase3b thermostat).
                    sims = fhrr.similarity_matrix(sq[i], content)
                    pid = int(sims.argmax())
                    if pid != tid:
                        push_acc[pid].append(sq[i])
                elif rule == "swap_negative":
                    # PRECOMMITTED seed-fixed role-class draw: a different
                    # scene's filler for the SAME role. Never sims.argmax.
                    j = int(torch.randint(0, N, (1,), generator=g))
                    pid = int(toy.fill[j, qr])
                    if pid != tid:
                        push_acc[pid].append(sq[i])
        for tid, qs in pull_acc.items():
            if not qs:
                continue
            avg = fhrr.normalize(torch.stack(qs).sum(dim=0))
            content[tid] = fhrr.normalize(
                (1.0 - lr_pull) * content[tid] + lr_pull * avg)
        for pid, qs in push_acc.items():
            if not qs:
                continue
            avg = fhrr.normalize(torch.stack(qs).sum(dim=0))
            content[pid] = fhrr.normalize(
                (1.0 + lr_push) * content[pid] - lr_push * avg)
    return content


# --------------------------------------------------------------------------- #
# G-A: frozen-substrate refit-readout
# --------------------------------------------------------------------------- #
def refit_recoverability(
    toy: Toy, content: torch.Tensor,
    train_ids: torch.Tensor, test_ids: torch.Tensor,
    *, observed: int, beta: float, max_iter: int,
) -> tuple[int, int]:
    """Fit nearest-prototype filler classifiers from TRAIN slot-queries, then
    classify HELD-OUT slot-queries by nearest train-prototype — bypassing the
    content codebook readout entirely. If this recovers where the native read
    fails, the unbind outputs DO cluster by filler (readout defect)."""
    fhrr = toy.fhrr
    # Build per-filler prototypes from train slot-queries (all roles pooled).
    proto_sum = torch.zeros_like(content)
    proto_cnt = torch.zeros(toy.C, device=content.device)
    for qr in range(toy.K):
        sq = _slot_queries(toy, train_ids, qr, observed, beta=beta, max_iter=max_iter)
        tgt = toy.fill[train_ids, qr]
        for i in range(len(train_ids)):
            proto_sum[int(tgt[i])] += sq[i]
            proto_cnt[int(tgt[i])] += 1
    seen = proto_cnt > 0
    prototypes = torch.where(
        seen[:, None], fhrr.normalize(proto_sum), content)  # fall back to atom
    # Classify held-out slot-queries by nearest prototype (cosine argmax).
    hits, n = 0, 0
    for qr in range(toy.K):
        sq = _slot_queries(toy, test_ids, qr, observed, beta=beta, max_iter=max_iter)
        tgt = toy.fill[test_ids, qr]
        sims = (sq @ prototypes.conj().T).real / content.shape[1]  # [n, C]
        # only classify against fillers actually seen in train
        sims = sims.masked_fill(~seen[None, :], float("-inf"))
        pred = sims.argmax(dim=1)
        valid = seen[tgt]
        hits += int(((pred == tgt) & valid).sum())
        n += int(valid.sum())
    return hits, n


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_seed(args, seed: int) -> dict:
    dev = args.device
    toy_all = build_toy(D=args.D, K=args.K, C=args.C, N=args.N_train + args.N_test,
                        seed=seed, device=dev)
    n_all = args.N_train + args.N_test
    train_ids = torch.arange(args.N_train, device=dev)
    test_ids = torch.arange(args.N_train, n_all, device=dev)
    deranged = _deranged_roles(args.K, seed, dev)
    chance = 1.0 / args.C

    def native_over_roles(content, ids, *, shuffle: bool):
        hits = n = 0
        ent = marg = 0.0
        for qr in range(args.K):
            ur = int(deranged[qr]) if shuffle else qr
            h, nn, e, m = native_recoverability(
                toy_all, content, ids, query_role=qr, unbind_role=ur,
                observed=args.observed, beta=args.beta, max_iter=args.max_iter)
            hits += h; n += nn; ent += e; marg += m
        return hits, n, ent / args.K, marg / args.K

    out: dict = {"seed": seed, "chance": chance}

    # store-as-is (regime validation + the null baseline)
    sa_true = native_over_roles(toy_all.content, test_ids, shuffle=False)
    sa_shuf = native_over_roles(toy_all.content, test_ids, shuffle=True)
    pc_h = pc_n = 0
    for qr in range(args.K):
        h, nn = perfect_cue_recoverability(
            toy_all, toy_all.content, test_ids, query_role=qr,
            beta=args.beta, max_iter=args.max_iter)
        pc_h += h; pc_n += nn
    out["store_as_is"] = {
        "true_rate": sa_true[0] / sa_true[1], "true_hits": sa_true[0], "n": sa_true[1],
        "shuffled_rate": sa_shuf[0] / sa_shuf[1],
        "perfect_cue_rate": pc_h / pc_n, "mean_entropy": sa_true[2],
        "mean_margin": sa_true[3],
    }

    # Write arms + G-D Selectivity-Δ each
    out["arms"] = {}
    for rule in ["baseline_contrastive", "swap_negative", "pull_only"]:
        cb = consolidate(toy_all, rule=rule, lr_pull=args.lr_pull,
                         lr_push=args.lr_push, epochs=args.epochs,
                         beta=args.beta, max_iter=args.max_iter, seed=seed,
                         observed=args.observed)
        t = native_over_roles(cb, test_ids, shuffle=False)
        s = native_over_roles(cb, test_ids, shuffle=True)
        sd = selectivity_delta(true_hits=t[0], true_n=t[1],
                               shuffled_hits=s[0], shuffled_n=s[1], chance=chance)
        out["arms"][rule] = sd.as_dict()

    # random-codebook control (write the baseline rule into a random codebook)
    rand_cb = toy_all.fhrr.random_vectors(args.C)
    # measure native recoverability against the random codebook (should be ~chance)
    rt = native_over_roles(rand_cb, test_ids, shuffle=False)
    rs = native_over_roles(rand_cb, test_ids, shuffle=True)
    out["arms"]["random_codebook"] = selectivity_delta(
        true_hits=rt[0], true_n=rt[1], shuffled_hits=rs[0], shuffled_n=rs[1],
        chance=chance).as_dict()

    # G-A frozen-refit (on the baseline-contrastive write)
    cb = consolidate(toy_all, rule="baseline_contrastive", lr_pull=args.lr_pull,
                     lr_push=args.lr_push, epochs=args.epochs, beta=args.beta,
                     max_iter=args.max_iter, seed=seed, observed=args.observed)
    native_t = native_over_roles(cb, test_ids, shuffle=False)
    refit_h, refit_n = refit_recoverability(
        toy_all, cb, train_ids, test_ids, observed=args.observed,
        beta=args.beta, max_iter=args.max_iter)
    out["G_A"] = {
        "native_rate": native_t[0] / native_t[1],
        "refit_rate": refit_h / refit_n if refit_n else 0.0,
        "refit_minus_native": (refit_h / refit_n if refit_n else 0.0) - native_t[0] / native_t[1],
        "refit_n": refit_n,
    }
    # random-codebook G-A control: refit gap should be ~0
    refit_rh, refit_rn = refit_recoverability(
        toy_all, rand_cb, train_ids, test_ids, observed=args.observed,
        beta=args.beta, max_iter=args.max_iter)
    out["G_A_random_control"] = {
        "native_rate": rt[0] / rt[1],
        "refit_rate": refit_rh / refit_rn if refit_rn else 0.0,
        "refit_minus_native": (refit_rh / refit_rn if refit_rn else 0.0) - rt[0] / rt[1],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--observed", type=int, default=99,
                    help="context roles in the cue (default all-but-query; 1 == key-only hard regime)")
    ap.add_argument("--C", type=int, default=24)
    ap.add_argument("--N-train", type=int, default=128, dest="N_train")
    ap.add_argument("--N-test", type=int, default=128, dest="N_test")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--max-iter", type=int, default=12, dest="max_iter")
    ap.add_argument("--lr-pull", type=float, default=0.1, dest="lr_pull")
    ap.add_argument("--lr-push", type=float, default=0.05, dest="lr_push")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    results = [run_seed(args, s) for s in range(args.seeds)]

    def agg(path):
        vals = []
        for r in results:
            node = r
            for p in path:
                node = node[p]
            vals.append(node)
        return sum(vals) / len(vals)

    summary = {
        "config": vars(args),
        "chance": 1.0 / args.C,
        "regime_check": {
            "store_as_is_true_rate": agg(["store_as_is", "true_rate"]),
            "store_as_is_perfect_cue_rate": agg(["store_as_is", "perfect_cue_rate"]),
            "hard_regime": agg(["store_as_is", "true_rate"]) < 0.5 * agg(["store_as_is", "perfect_cue_rate"]),
        },
        "G_A_mean": {
            "native_rate": agg(["G_A", "native_rate"]),
            "refit_rate": agg(["G_A", "refit_rate"]),
            "refit_minus_native": agg(["G_A", "refit_minus_native"]),
            "random_control_gap": agg(["G_A_random_control", "refit_minus_native"]),
        },
        "G_D_mean": {
            rule: {
                "true_rate": agg(["arms", rule, "true_rate"]),
                "shuffled_rate": agg(["arms", rule, "shuffled_rate"]),
                "delta": agg(["arms", rule, "delta"]),
            }
            for rule in ["baseline_contrastive", "swap_negative", "pull_only", "random_codebook"]
        },
        "per_seed": results,
    }
    print(json.dumps(summary["regime_check"], indent=2))
    print(json.dumps(summary["G_A_mean"], indent=2))
    print(json.dumps(summary["G_D_mean"], indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
