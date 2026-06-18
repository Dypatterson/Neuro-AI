"""experiments/97 — Bet B / SCAN-MCD: PROGRESS-PRIORITIZED ALLOCATION (the untested central hypothesis).

CHARTER: CONTEXT-B.md §3 — "One allocation currency: learning PROGRESS, not raw prediction error. Progress (the
change in competence) is zero on noise, zero on the mastered, peaks on the learnable frontier. It is the
allocation signal — what to write, what to replay, what to seek — not the weight-update (that stays
error-driven). It is direction-blind ... so it allocates; it does NOT manufacture." This is the project's
CENTRAL architecture claim and has NEVER been instantiated on a discriminating arena (RETROSPECTIVE-addendum §4
"genuinely untested"). Every replay/sampling variant tried (134-138) used surprise/error or plain interleaving.

THE INSTANTIATION (single-task MCD reduction): a PROGRESS-WEIGHTED example sampler. Per example track a fast and
a slow loss EMA; progress = max(0, slow - fast) (fast below slow = competence rising = on the frontier). Sample
training examples weighted by progress (+ eps for exploration). The weight-update stays error-driven (standard
CE). vs UNIFORM sampling (the baseline). ANTI-HOMUNCULUS: progress is a LOCAL per-example signal; sampling
proportional to it is a fixed dynamic (the sanctioned CONTEXT-B §3 mechanism), not a metric-reading supervisor.

CONTROL: 'hardness' = sample by CURRENT loss (hardest-first) — if hardness also helps, progress isn't special
(it's just hard-example-mining). 'antiprogress' = sample by -progress (mastered/unlearnable first) — must hurt-or-null.

HEADLINE (CLAUDE.md preamble): held-out MCD exact-match, progress - uniform CI-disjoint > 0, n=8, mcd1.
Controls: uniform (floor) · hardness (is it progress or just hard-mining?) · antiprogress (must not help).
HONEST PRIOR: progress-allocation is a sample-efficiency/curriculum signal; it may speed convergence but is NOT
obviously a structure-MANUFACTURER (CONTEXT-B §3: "it allocates; it does NOT manufacture") -> likely null on
held-out compositional generalization. But it is the central untested claim, so it gets a clean test.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _load(name, fname):
    s = importlib.util.spec_from_file_location(name, REPO / "experiments" / fname)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


exp87 = _load("exp87", "87_betb_scan_gap.py")
exp94 = _load("exp94", "94_betb_scan_mcd_ccc.py")
load_pairs, build_vocab, to_ids, pad = exp87.load_pairs, exp87.build_vocab, exp87.to_ids, exp87.pad
PAD = exp87.PAD
MCD = REPO / "data" / "scan" / "mcd_split"


def train_alloc(model, pairs, in_vocab, out_vocab, *, mode, epochs, batch_size, lr, tf_ratio, seed, device,
                eps=0.05, fast=0.5, slow=0.9):
    """mode in {uniform, progress, hardness, antiprogress}. Weight-update is always error-driven CE; only the
    SAMPLING allocation differs."""
    n = len(pairs); V = model.out_vocab
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed * 104729 + 7)
    loss_fast = torch.full((n,), float("nan")); loss_slow = torch.full((n,), float("nan"))
    model.train()
    for ep in range(epochs):
        seeded = (~torch.isnan(loss_fast)).all()
        if mode == "uniform" or ep == 0 or not seeded:
            order = torch.randperm(n, generator=gen).tolist()                      # uniform (ep0 seeds the EMAs)
        else:
            if mode == "progress":
                signal = (loss_slow - loss_fast).clamp(min=0)
            elif mode == "antiprogress":
                signal = (loss_fast - loss_slow).clamp(min=0)
            else:                                                                  # hardness = current (fast) loss
                signal = loss_fast.clamp(min=0)
            w = torch.nan_to_num(signal, nan=float(signal[~torch.isnan(signal)].mean())) + eps
            order = torch.multinomial(w, n, replacement=True, generator=gen).tolist()
        for b in range(0, n, batch_size):
            idxs = order[b:b + batch_size]
            chunk = [pairs[i] for i in idxs]
            src, _ = pad([to_ids(c[0], in_vocab) for c in chunk], device)
            tgt, _ = pad([to_ids(c[1], out_vocab) for c in chunk], device)
            B, T = tgt.shape
            logits = model(src, src != PAD, tgt, tf_ratio, gen)
            ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=PAD, reduction="none").reshape(B, T)
            valid = (tgt != PAD).float()
            per_ex = (ce * valid).sum(1) / valid.sum(1).clamp(min=1)
            loss = per_ex.mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            pe = per_ex.detach().to("cpu")
            for k, i in enumerate(idxs):                                           # update per-example EMAs
                if torch.isnan(loss_fast[i]):
                    loss_fast[i] = pe[k]; loss_slow[i] = pe[k]
                else:
                    loss_fast[i] = fast * loss_fast[i] + (1 - fast) * pe[k]
                    loss_slow[i] = slow * loss_slow[i] + (1 - slow) * pe[k]
        print(f"    [{mode}] epoch {ep + 1}/{epochs}", file=sys.stderr, flush=True)


def load_mcd(split):
    return (load_pairs(str(MCD / f"tasks_train_{split}.txt")), load_pairs(str(MCD / f"tasks_test_{split}.txt")))


def run_seed(args, seed, split, arms):
    tr, te = load_mcd(split); allp = tr + te
    in_vocab, _ = build_vocab(allp, "in"); out_vocab, _ = build_vocab(allp, "out")
    max_len = max(len(p[1]) for p in allp) + 2
    res = {}
    for arm in arms:
        m = exp87.Seq2Seq(len(in_vocab), len(out_vocab), args.embed, args.hidden, seed).to(args.device)
        train_alloc(m, tr, in_vocab, out_vocab, mode=arm, epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, tf_ratio=args.tf_ratio, seed=seed, device=args.device)
        res[arm] = exp87.exact_match(m, te, in_vocab, out_vocab, max_len=max_len, batch_size=args.batch_size, device=args.device)
        print(f"  [{split} s{seed}] {arm} = {res[arm]:.4f}", file=sys.stderr, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", type=int, default=64); ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--tf-ratio", type=float, default=0.5, dest="tf_ratio")
    ap.add_argument("--seeds", type=int, default=8); ap.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    ap.add_argument("--device", default="mps"); ap.add_argument("--splits", default="mcd1")
    ap.add_argument("--arms", default="uniform,progress,hardness,antiprogress")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs, args.arms = 1, 6, "uniform,progress"

    arms = args.arms.split(",")
    out = {"experiment": "97_betb_scan_mcd_progress (progress-prioritized allocation, CONTEXT-B §3)",
           "charter": "CONTEXT-B.md §3", "config": vars(args), "by_split": {}}
    for split in args.splits.split(","):
        per = {a: [] for a in arms}
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            r = run_seed(args, seed, split, arms)
            for a in arms:
                per[a].append(r[a])
        sdict = {a: exp94.summary(per[a]) for a in arms}
        deltas = {}
        for a in ("progress", "hardness", "antiprogress"):
            if a in per and "uniform" in per and len(per[a]) == len(per["uniform"]) and len(per[a]) > 1:
                deltas[f"{a}_minus_uniform"] = exp94.paired_delta(per[a], per["uniform"])
        out["by_split"][split] = {"arms": sdict, "paired_deltas": deltas}
        print(f"\n=== {split}: " + " ".join(f"{a}={sdict[a]['mean']:.3f}" for a in arms), file=sys.stderr, flush=True)
    print(json.dumps(out, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
