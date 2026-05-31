"""WS-InfoNCE codebook shaping — composition-margin over the graduated H (Phase-3, Stage-1).

CLASSIFICATION: **DRILL-DOWN / exploratory** (secondary GENERALIZATION track),
NOT a graduation experiment. Spec:
notes/emergent-codebook/phase-3-within-scene-predictive-jepa-design.md (§§2,3,4,5).

The memorization mechanism (Reports 055/056/058) is the FLOOR. This experiment asks
the un-de-risked generalization question (F-COMPOSE / F-KILL): does a batch-offline
within-scene InfoNCE pass that shapes the value codebook into C' let the UNCHANGED
graduated H + decorrelator read a MORE role-separable codebook — and does any lift
GENERALIZE to held-out (role,filler) combinations, or is it a third memory?

HEADLINE (jepa-design.md:25,33): the held-out-roles **composition-margin**
  Δ(C') − Δ(raw C),  Newcombe CI,  in the sparse-cue regime where store-as-is fails.
In-sample Selectivity-Δ is a FLOOR GUARD ONLY (H already tracks the Bayes ceiling, no
headroom). HARD FLOOR GUARD (jepa-design.md:22): write+L2 on C' must be >= the raw-C
ceiling at EVERY cue richness in-sample; an in-sample regression = ABORT.

HONEST PRIOR (jepa-design.md:6): on real text the analytic prediction is a held-out
NULL — a third memory (in-sample pass, held-out ~ chance). This harness is built to
DETECT that cleanly (held-out Δ-vs-frac_seen), not to assume it away.

ANTI-HOMUNCULUS. The WS-InfoNCE write (phase4/ws_infonce.py) is a single batch-offline
pass over a FROZEN scene buffer; the self-target is the TRUE masked filler index of
each scene (a fixed function of data, NOT sims.argmax-mined); negatives are the
precommitted full value codebook; C' is then read by the UNCHANGED graduated path
(value_codebook=C') — a data-shaping handoff, not arbitration. Every read terminates
in a top_index basin count (recall_top_index), never an energy / min-over-branches / ΔE
(Phase-5' fence). The InfoNCE loss owns its own log-softmax; tau != read-time beta.
No runtime metric is read to gate, branch, or select.

Reuses experiments/56_gd_selectivity_panel.py building blocks verbatim (byte-identical
data + reference write/read path); shaping is the only addition.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# exp-56's module name starts with a digit -> load by path (the exp-57 pattern).
_spec = importlib.util.spec_from_file_location(
    "exp56_panel", REPO / "experiments" / "56_gd_selectivity_panel.py")
exp56 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp56)

from energy_memory.phase3.basin_readout import (
    newcombe_diff_ci, selectivity_delta, top_index_hits,
)
from energy_memory.phase4.ws_infonce import (
    FrozenSceneBuffer, WSInfoNCEConfig, shape_codebook,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR

# Byte-identical building blocks from the G-D panel.
TopicCorpus = exp56.TopicCorpus
CorpusWindows = exp56.CorpusWindows
encode_cue = exp56.encode_cue
build_position_vectors = exp56.build_position_vectors
write_H = exp56.write_H
write_read = exp56.write_read
store_read = exp56.store_read
batched_hopfield_topindex = exp56.batched_hopfield_topindex


PRESETS = {
    "synthetic": dict(corpus_source="synthetic", D=2048, W=6, L=8, Vc=16, N=500,
                      eps=0.25, max_vocab=512, seeds=[0, 1, 2], observed=[1, 2, 3]),
    "repo_sample": dict(corpus_source="repo_sample", D=1024, W=6, L=0, Vc=0, N=500,
                        eps=0.0, max_vocab=512, seeds=[0, 1, 2], observed=[1, 2, 3]),
    "wikitext": dict(corpus_source="wikitext", D=4096, W=6, L=0, Vc=0, N=1000,
                     eps=0.0, max_vocab=2000, seeds=[0, 1, 2], observed=[1, 2, 3]),
}


def _make_source(args, seed):
    if args.corpus_source == "synthetic":
        return TopicCorpus(seed=seed, W=args.W, L=args.L, Vc=args.Vc, N=args.N,
                           eps=args.eps, posdep=True)
    return CorpusWindows(seed=seed, W=args.W, N=args.N, corpus_source=args.corpus_source,
                         wikitext_name=args.wikitext_name, max_vocab=args.max_vocab,
                         repo_root=REPO)


def _value_row_ids(src, args, device):
    """Rows of the substrate codebook that constitute the value codebook (the
    learnable + read-out atoms; their LOCAL order matches `tgt`)."""
    if args.corpus_source == "synthetic":
        return torch.arange(src.L, device=device)
    return torch.tensor(src._decode_ids, dtype=torch.long, device=device)


def _deranged_cue_set(sub, positions, codebook, src, op, mpos, W, mask_id, seed, dev):
    """Per-scene fixed-point-free position derangement — byte-identical to exp-56's
    run_cell closure (same seed formula seed*100003 + si*131 + 17), so the role-shuffle
    arm of the raw-C arm reproduces the Report-056 headline exactly."""
    keys = []
    for si, w in enumerate(src.windows):
        g = torch.Generator().manual_seed(seed * 100003 + si * 131 + 17)
        rp = list(range(W))
        if len(op) >= 2:
            while True:
                pp = torch.randperm(len(op), generator=g).tolist()
                if all(i != p for i, p in enumerate(pp)):
                    break
            for i, p in enumerate(op):
                rp[p] = op[pp[i]]
        else:
            cands = [p for p in range(W) if p != op[0]]
            rp[op[0]] = cands[int(torch.randint(0, len(cands), (1,), generator=g))]
        keys.append(encode_cue(sub, positions, codebook, w, op, mpos, W,
                               mask_id, mode="true", perm=rp))
    return torch.stack(keys)


def _cue_set(sub, positions, codebook, src, op, mpos, W, mask_id, mode):
    return torch.stack([encode_cue(sub, positions, codebook, w, op, mpos, W,
                                   mask_id, mode=mode) for w in src.windows])


def _arm(sub, codebook, value_cb, K_true, K_der, tr_t, te_t, tgt, tgt_te, n_te,
         chance, args):
    """Fit write+L2 on TRAIN cues -> read TEST true/shuffled cues. Returns the
    selectivity-Δ dict + the true/shuffled hit COUNTS (for pooling + Newcombe)."""
    H, dec = write_H(K_true[tr_t], tgt[tr_t], value_cb, decorr="l2", dim=args.D,
                     lr=args.lr, epochs=args.epochs_write)
    ti_t, ent, mg = write_read(sub, H, dec, K_true[te_t], value_cb, args.beta, args.mi)
    ti_s, _, _ = write_read(sub, H, dec, K_der[te_t], value_cb, args.beta, args.mi)
    h_t = top_index_hits(ti_t, tgt_te)
    h_s = top_index_hits(ti_s, tgt_te)
    sd = selectivity_delta(true_hits=h_t, true_n=n_te, shuffled_hits=h_s,
                           shuffled_n=n_te, chance=chance)
    d = sd.as_dict()
    d.update({"true_hits": h_t, "shuffled_hits": h_s, "n": n_te,
              "mean_entropy": float(ent.mean()), "mean_margin": float(mg.mean())})
    return d, H, dec


def run_seed_split(args, seed, split):
    """One (seed, split): shape C' ONCE on the split's TRAIN windows (shaping is
    independent of `observed`), then run the read panel at every observed level for
    raw-C, C'(contrastive), C'(no-negatives ablation) + controls."""
    dev = args.device
    src = _make_source(args, seed)
    sub = TorchFHRR(dim=args.D, seed=seed, device=dev)
    codebook, value_cb_raw, tgt, chance = src.codebook_targets(sub)
    tgt = tgt.to(dev)
    positions = torch.stack(build_position_vectors(sub, src.W)).to(dev)
    mpos, mask_id, W = src.mpos, src.mask_id, src.W
    value_row_ids = _value_row_ids(src, args, dev)

    half = src.N // 2
    if split == "insample":
        tr, te = list(range(src.N)), list(range(src.N))
    else:
        tr, te = list(range(half)), list(range(half, src.N))
    tr_t = torch.tensor(tr, device=dev)
    te_t = torch.tensor(te, device=dev)

    # ---- WS-InfoNCE shaping: batch-offline over the FROZEN train scene buffer. ----
    def shape(contrastive):
        buf = FrozenSceneBuffer()
        for i in tr:
            buf.add(src.windows[i], int(tgt[i]))
        buf.freeze(dev)
        cfg = WSInfoNCEConfig(lr=args.lr_infonce, epochs=args.epochs_infonce,
                              tau=args.tau, contrastive=contrastive, seed=seed)
        return shape_codebook(sub, codebook, value_row_ids, positions, buf, mpos, cfg)

    cprime_full, cprime_val, shape_info = shape(contrastive=True)
    if args.run_noneg:
        cnn_full, cnn_val, shape_info_nn = shape(contrastive=False)
    else:
        cnn_full = cnn_val = shape_info_nn = None

    # For the topic-corpus the value atoms never appear in scene context, so cues built
    # from context atoms are identical across arms -> build once on the raw codebook.
    # For real text the value rows ARE context rows, so cues must be rebuilt per arm.
    synthetic = args.corpus_source == "synthetic"

    cells = []
    for observed in args.observed:
        op = src.observed_positions(observed)
        n_te = len(te)
        tgt_te = tgt[te_t]

        def cues(cb):
            K_true = _cue_set(sub, positions, cb, src, op, mpos, W, mask_id, "true")
            K_der = _deranged_cue_set(sub, positions, cb, src, op, mpos, W, mask_id, seed, dev)
            return K_true, K_der

        K_true_raw, K_der_raw = cues(codebook)
        if synthetic:
            K_true_cp, K_der_cp = K_true_raw, K_der_raw
            K_true_nn, K_der_nn = K_true_raw, K_der_raw
        else:
            K_true_cp, K_der_cp = cues(cprime_full)
            K_true_nn, K_der_nn = cues(cnn_full) if cnn_full is not None else (None, None)

        # frac_seen (the F-KILL x-axis): fraction of TEST cue-classes seen in TRAIN.
        seen = {tuple(src.windows[i][p] for p in op) for i in tr}
        frac_seen = sum(tuple(src.windows[i][p] for p in op) in seen for i in te) / n_te

        # ---- arms ----
        raw, H_raw, dec_raw = _arm(sub, codebook, value_cb_raw, K_true_raw, K_der_raw,
                                   tr_t, te_t, tgt, tgt_te, n_te, chance, args)
        cp, _, _ = _arm(sub, cprime_full, cprime_val, K_true_cp, K_der_cp,
                        tr_t, te_t, tgt, tgt_te, n_te, chance, args)
        arms = {"raw_write_l2": raw, "cprime_write_l2": cp}
        if cnn_full is not None:
            nn, _, _ = _arm(sub, cnn_full, cnn_val, K_true_nn, K_der_nn,
                            tr_t, te_t, tgt, tgt_te, n_te, chance, args)
            arms["cprime_no_negatives"] = nn

        # ---- composition margin: Δ(C') − Δ(raw C) on the TRUE arm (Newcombe). ----
        m_lo, m_hi = newcombe_diff_ci(cp["true_hits"], n_te, raw["true_hits"], n_te)
        composition_margin = {
            "cprime_true_rate": cp["true_rate"], "raw_true_rate": raw["true_rate"],
            "cprime_minus_raw_true": cp["true_rate"] - raw["true_rate"],
            "cprime_minus_raw_ci": [m_lo, m_hi],
            "cprime_beats_raw": bool(m_lo > 0.0),
            "cprime_selectivity_delta": cp["delta"], "raw_selectivity_delta": raw["delta"],
            "delta_of_deltas": cp["delta"] - raw["delta"],
            "F_COMPOSE_pass": bool(cp["two_floor_pass"] and m_lo > 0.0),
        }

        # ---- store-as-is (coverage-matched baseline, Dorrell FATAL-3) ----
        full_enc = torch.stack([exp56.encode_window(sub, positions, codebook, w)
                                for w in src.windows])
        ti_sa, _, _ = store_read(sub, full_enc[tr_t], K_true_raw[te_t], positions,
                                 value_cb_raw, mpos, args.beta, args.mi)
        arms["store_as_is_true_rate"] = top_index_hits(ti_sa, tgt_te) / n_te

        # ---- Control: random-codebook leak detector (read C'-H output vs random cb) ----
        # Re-fit C''s H so we can read its own recall output against a random codebook;
        # a leaky readout would recover above chance even with the value atoms scrambled.
        rand_cb = sub.random_vectors(cprime_val.shape[0])
        H_cp, dec_cp = write_H(K_true_cp[tr_t], tgt[tr_t], cprime_val, decorr="l2",
                               dim=args.D, lr=args.lr, epochs=args.epochs_write)
        rec = (dec_cp.apply(K_true_cp[te_t]) @ H_cp.transpose(0, 1)) / args.D
        ti_r, _, _ = batched_hopfield_topindex(sub, rand_cb, rec, beta=args.beta, max_iter=args.mi)
        arms["random_codebook_true_rate"] = top_index_hits(ti_r, tgt_te) / n_te

        # ---- Control: content-matched non-positional bag (read C' H with a bag cue) ----
        K_bag_cp = _cue_set(sub, positions, cprime_full, src, op, mpos, W, mask_id, "bag")
        ti_b, _, _ = write_read(sub, H_cp, dec_cp, K_bag_cp[te_t], cprime_val, args.beta, args.mi)
        arms["content_bag_cprime_true_rate"] = top_index_hits(ti_b, tgt_te) / n_te

        # ---- Control: perfect-cue upper bound (over C') ----
        ti_pc, _, _ = batched_hopfield_topindex(sub, cprime_val, cprime_val[tgt_te],
                                                beta=args.beta, max_iter=args.mi)
        arms["perfect_cue_cprime_rate"] = top_index_hits(ti_pc, tgt_te) / n_te

        cells.append({
            "seed": seed, "split": split, "observed": observed, "chance": chance,
            "frac_seen": frac_seen, "n_test": n_te,
            "arms": arms, "composition_margin": composition_margin,
        })

    return cells, shape_info, (shape_info_nn if args.run_noneg else None)


def aggregate(cells):
    """Pool TRUE/SHUFFLED hit counts across seeds at each (split, observed) for a
    pooled two-floor read + a pooled composition-margin Newcombe CI."""
    from collections import defaultdict
    by_key = defaultdict(list)
    for c in cells:
        by_key[(c["split"], c["observed"])].append(c)
    out = {}
    for (split, obs), cs in sorted(by_key.items()):
        chance = cs[0]["chance"]

        def pool(arm, field):
            return sum(c["arms"][arm][field] for c in cs)

        n = pool("raw_write_l2", "n")
        raw = selectivity_delta(true_hits=pool("raw_write_l2", "true_hits"), true_n=n,
                                shuffled_hits=pool("raw_write_l2", "shuffled_hits"),
                                shuffled_n=n, chance=chance)
        cp = selectivity_delta(true_hits=pool("cprime_write_l2", "true_hits"), true_n=n,
                               shuffled_hits=pool("cprime_write_l2", "shuffled_hits"),
                               shuffled_n=n, chance=chance)
        m_lo, m_hi = newcombe_diff_ci(pool("cprime_write_l2", "true_hits"), n,
                                      pool("raw_write_l2", "true_hits"), n)
        n_seeds = len(cs)
        entry = {
            "split": split, "observed": obs, "n_pooled": n, "n_seeds": n_seeds,
            "chance": chance,
            "frac_seen_mean": sum(c["frac_seen"] for c in cs) / n_seeds,
            "raw_write_l2": raw.as_dict(), "cprime_write_l2": cp.as_dict(),
            "cprime_minus_raw_true": cp.true_rate - raw.true_rate,
            "cprime_minus_raw_ci": [m_lo, m_hi],
            "delta_of_deltas": cp.delta - raw.delta,
            "store_as_is_true_mean": sum(c["arms"]["store_as_is_true_rate"] for c in cs) / n_seeds,
            "random_codebook_true_mean": sum(c["arms"]["random_codebook_true_rate"] for c in cs) / n_seeds,
            "content_bag_cprime_true_mean": sum(c["arms"]["content_bag_cprime_true_rate"] for c in cs) / n_seeds,
            "perfect_cue_cprime_mean": sum(c["arms"]["perfect_cue_cprime_rate"] for c in cs) / n_seeds,
            "F_COMPOSE_pass": bool(cp.two_floor_pass and m_lo > 0.0),
            "seeds_cprime_beats_raw": f"{sum(c['composition_margin']['cprime_beats_raw'] for c in cs)}/{n_seeds}",
        }
        if "cprime_no_negatives" in cs[0]["arms"]:
            nn = selectivity_delta(true_hits=pool("cprime_no_negatives", "true_hits"), true_n=n,
                                   shuffled_hits=pool("cprime_no_negatives", "shuffled_hits"),
                                   shuffled_n=n, chance=chance)
            entry["cprime_no_negatives"] = nn.as_dict()
        out[f"{split}:obs{obs}"] = entry
    return out


def floor_guard(pooled):
    """HARD FLOOR GUARD (jepa-design.md:22): in-sample, C' write+L2 must NOT regress
    below raw-C at any cue richness. Returns (pass, violations)."""
    violations = []
    for key, e in pooled.items():
        if e["split"] != "insample":
            continue
        # regression = C' true-rate significantly below raw (upper CI of (C'-raw) < 0).
        if e["cprime_minus_raw_ci"][1] < 0.0:
            violations.append({"cell": key,
                               "cprime_true": e["cprime_write_l2"]["true_rate"],
                               "raw_true": e["raw_write_l2"]["true_rate"],
                               "ci": e["cprime_minus_raw_ci"]})
    return (len(violations) == 0), violations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="synthetic")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--splits", nargs="+", default=["insample", "heldout"],
                    choices=["insample", "heldout"])
    ap.add_argument("--override-D", type=int, default=0, dest="override_D")
    ap.add_argument("--override-N", type=int, default=0, dest="override_N")
    ap.add_argument("--seeds", type=int, default=0, help="0 -> preset seed count")
    ap.add_argument("--observed", type=int, nargs="+", default=[])
    # WS-InfoNCE shaping hyperparameters
    ap.add_argument("--lr-infonce", type=float, default=0.05, dest="lr_infonce")
    ap.add_argument("--epochs-infonce", type=int, default=400, dest="epochs_infonce")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--no-noneg", action="store_false", dest="run_noneg",
                    help="skip the no-negatives ablation arm")
    # graduated write/read (matches exp-56 defaults)
    ap.add_argument("--lr", type=float, default=0.5, help="hetero write lr")
    ap.add_argument("--epochs-write", type=int, default=20, dest="epochs_write")
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--mi", type=int, default=12)
    ap.add_argument("--wikitext-name", default="wikitext-2-raw-v1", dest="wikitext_name")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    p = dict(PRESETS[args.preset])
    preset_seeds = p.pop("seeds")
    preset_observed = p.pop("observed")
    for k, v in p.items():
        setattr(args, k, v)
    if args.override_D:
        args.D = args.override_D
    if args.override_N:
        args.N = args.override_N
    seeds = list(range(args.seeds)) if args.seeds else list(preset_seeds)
    args.observed = args.observed if args.observed else list(preset_observed)

    all_cells = []
    shape_infos = []
    for seed in seeds:
        for split in args.splits:
            cells, sinfo, sinfo_nn = run_seed_split(args, seed, split)
            all_cells.extend(cells)
            shape_infos.append({"seed": seed, "split": split, "contrastive": sinfo,
                                "no_negatives": sinfo_nn})

    pooled = aggregate(all_cells)
    fg_pass, fg_violations = floor_guard(pooled)
    summary = {
        "config": {k: getattr(args, k) for k in
                   ("preset", "corpus_source", "D", "N", "W", "L", "Vc",
                    "lr_infonce", "epochs_infonce", "tau", "beta", "mi")},
        "seeds": seeds, "splits": args.splits,
        "floor_guard_pass": fg_pass, "floor_guard_violations": fg_violations,
        "pooled": pooled, "shape_info": shape_infos, "cells": all_cells,
    }

    print(json.dumps({"config": summary["config"],
                      "floor_guard_pass": fg_pass,
                      "floor_guard_violations": fg_violations,
                      "pooled": pooled}, indent=2))
    if args.out:
        outp = pathlib.Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
