"""experiments/91 — Bet B / SCAN Stage 1: the GECA REDUNDANCY GUARD (the critical guard on the 143 graduation).

CHARTER: notes/betb-scan-stage1-consolidation-precommit.md §4 (Controls & the redundancy guard) +
Report 143 (the headline graduated: factored+consolidation lifts jump-split 0.111 -> 0.868, 8/8 seeds).

THE QUESTION (precommit §3 disposition): GECA (good-enough compositional augmentation, Andreas 2020) is the
KNOWN SCAN add-jump fix. Does GECA, on this setup, ALSO reach ~0.87? If yes, the consolidation result is REAL
but REDUNDANT (the 133/139 "matches-not-beats-the-known-method" pattern): "consolidation does it" YES, but
"...that a known method can't" NO. If consolidation >> GECA, it beats the known fix.

GECA on add-jump reduces to PRIMITIVE SUBSTITUTION (precommit §4.2: "swap jump into template slots that other
verbs occupy"): jump appears in train ONLY standalone; walk/run/look appear in all templates. GECA generates
synthetic jump-in-template examples by substituting jump (and I_JUMP) into the templates the other verbs occupy.
This is label-preserving and SOUND for SCAN (verb->action is 1:1 and position-preserving). It is the engineered
data-space route to the same capability the consolidation reaches in representation space.

GENEROUS-TO-GECA BY DESIGN (makes the guard stringent): GECA is given the peer set + the held-out primitive,
both COMPUTED-FROM-BUFFER (BUILD CONDITION 1) — never a hardcoded literal:
  - peer set = tokens that appear as a complete one-token command in train  -> {jump,walk,run,look}
  - held-out = the peer verb(s) with ZERO multi-token (template) occurrences -> {jump}  (computed, not assumed)
  - verb->action = the standalone command->output map                        -> {jump:I_JUMP, walk:I_WALK, ...}

THE MECHANISTIC CONTRAST THE GUARD PINS DOWN (measured, not asserted): GECA solves add-jump by being SHOWN
(synthetic) composed-jump data; the consolidation (Report 143) reaches ~0.87 having NEVER seen a composed-jump
example — it only realigns jump's role-representation from the standalone jump->I_JUMP signal. Same capability,
different mechanism (data augmentation vs representation restructuring).

ARMS (all n=8, seeds 0-7, to match Report 143's stabilization run):
  vanilla_plain   exp87 Seq2Seq, plain add-jump train         (Stage-0 floor; ~0.003)
  vanilla_geca    exp87 Seq2Seq, GECA-augmented train         (the known fix on the SIMPLE method)
  factored_geca   exp90 Factored2, GECA-augmented train, NO consolidation
                                                              (augmentation on the consolidation's OWN substrate)
  [cited from Report 143, same seeds] factored_baseline 0.111 ; factored_consolidation 0.868
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


def _load(name, fname):
    s = importlib.util.spec_from_file_location(name, REPO / "experiments" / fname)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


exp87 = _load("exp87", "87_betb_scan_gap.py")
exp90 = _load("exp90", "90_betb_scan_factored2.py")
load_pairs, build_vocab = exp87.load_pairs, exp87.build_vocab
DATA = REPO / "data" / "scan"


# ---- GECA augmentation ------------------------------------------------------------------------------------

def compute_structure(trp):
    """Compute (peer set, template-verbs, held-out verbs, verb->action), all from the train buffer.
    BUILD CONDITION 1: nothing hardcoded; the held-out primitive is DISCOVERED as the peer verb absent from
    every template (multi-token command), not assumed to be 'jump'."""
    peer = {cmd[0] for cmd, _ in trp if len(cmd) == 1}
    assert peer == {"jump", "walk", "run", "look"}, f"peer set -> {peer}"
    verb_action = {}
    for cmd, act in trp:
        if len(cmd) == 1 and len(act) == 1:
            verb_action.setdefault(cmd[0], act[0])
    assert set(verb_action) == peer and set(verb_action.values()) == {"I_JUMP", "I_WALK", "I_RUN", "I_LOOK"}, \
        f"verb->action -> {verb_action}"
    template_verbs = {v for v in peer if any(v in cmd for cmd, _ in trp if len(cmd) > 1)}
    held_out = peer - template_verbs
    assert held_out == {"jump"}, f"held-out (computed) -> {held_out}"   # sanity: matches the documented split
    return peer, template_verbs, held_out, verb_action


def geca_augment(trp, template_verbs, held_out, verb_action):
    """Primitive-substitution GECA: for each template example using a donor verb d (one that appears in
    templates), and each held-out recipient r, emit a copy with d->r in the command and action(d)->action(r)
    in the output. Deduped against the original train set. Returns (augmented_pairs, stats)."""
    seen = {(tuple(c), tuple(a)) for c, a in trp}
    aug = []
    for cmd, act in trp:
        if len(cmd) == 1:
            continue                                                   # standalone: nothing to templatize
        donors = [d for d in template_verbs if d in cmd]
        for d in donors:
            ad = verb_action[d]
            for r in held_out:
                ar = verb_action[r]
                new_cmd = [r if t == d else t for t in cmd]
                new_act = [ar if t == ad else t for t in act]
                key = (tuple(new_cmd), tuple(new_act))
                if key not in seen:
                    seen.add(key); aug.append((new_cmd, new_act))
    stats = {"n_original": len(trp), "n_generated": len(aug), "n_total": len(trp) + len(aug),
             "example": [" ".join(aug[0][0]), " ".join(aug[0][1])] if aug else None}
    return trp + aug, stats


# ---- arms -------------------------------------------------------------------------------------------------

def run_vanilla(train_pairs, test_pairs, in_vocab, out_vocab, max_len, args, seed, tag):
    model = exp87.Seq2Seq(len(in_vocab), len(out_vocab), args.embed, args.hidden, seed).to(args.device)
    exp87.train(model, train_pairs, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, tf_ratio=args.tf_ratio, seed=seed, device=args.device)
    acc = exp87.exact_match(model, test_pairs, in_vocab, out_vocab, max_len=max_len,
                            batch_size=args.batch_size, device=args.device)
    print(f"  [{tag} seed {seed}] jump-split = {acc:.4f}", file=sys.stderr, flush=True)
    return acc


def run_factored(train_pairs, test_pairs, in_vocab, out_vocab, max_len, verb_ids, args, seed, tag):
    model = exp90.Factored2Seq2Seq(len(in_vocab), len(out_vocab), verb_ids, args.role_dim, args.fill_dim,
                                   args.hidden, seed).to(args.device)
    exp90.train(model, train_pairs, in_vocab, out_vocab, epochs=args.factored_epochs, batch_size=args.batch_size,
                lr=args.lr, tf_ratio=args.tf_ratio, seed=seed, device=args.device)
    acc = exp90.exact_match(model, test_pairs, in_vocab, out_vocab, max_len=max_len,
                            batch_size=args.batch_size, device=args.device)
    print(f"  [{tag} seed {seed}] jump-split = {acc:.4f}", file=sys.stderr, flush=True)
    return acc


# ---- stats ------------------------------------------------------------------------------------------------

def summary(v, n_boot=10000, seed=0):
    t = torch.tensor(v, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, t.numel(), (n_boot, t.numel()), generator=g)
    means = t[idx].mean(1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return {"mean": float(t.mean()), "min": float(t.min()), "max": float(t.max()),
            "ci95": [lo, hi], "per_seed": v}


def paired_delta(a, b, n_boot=10000, seed=0):
    """Paired bootstrap of mean(a) - mean(b) over aligned seeds (a,b same seeds)."""
    ta, tb = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    d = ta - tb
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, d.numel(), (n_boot, d.numel()), generator=g)
    dm = d[idx].mean(1)
    lo, hi = torch.quantile(dm, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return {"delta_mean": float(d.mean()), "ci95": [lo, hi], "per_seed": d.tolist()}


# ---- driver -----------------------------------------------------------------------------------------------

# Report 143 stabilization (8 seeds, seeds 0-7) — cited, not re-run.
R143_FACTORED_BASELINE = [0.3694523747729042, 0.0, 0.0, 0.2639501686997145, 0.0002595380223202699,
                          0.006488450558006748, 0.005060991435245263, 0.2420192058136517]
R143_FACTORED_CONSOL = [1.0, 0.9026732416298988, 0.5683882688813912, 0.9976641577991175, 0.5742278743835972,
                        0.9767713470023358, 0.9614586036854399, 0.9591227614845574]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--role-dim", type=int, default=48, dest="role_dim")
    ap.add_argument("--fill-dim", type=int, default=16, dest="fill_dim")
    ap.add_argument("--epochs", type=int, default=30)                       # vanilla arms
    ap.add_argument("--factored-epochs", type=int, default=50, dest="factored_epochs")
    ap.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tf-ratio", type=float, default=0.5, dest="tf_ratio")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--arms", default="vanilla_plain,vanilla_geca,factored_geca")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs, args.factored_epochs = 1, 8, 10

    trp = load_pairs(str(DATA / "tasks_train_addprim_jump.txt"))
    tep = load_pairs(str(DATA / "tasks_test_addprim_jump.txt"))
    allp = trp + tep
    in_vocab, _ = build_vocab(allp, "in"); out_vocab, _ = build_vocab(allp, "out")
    max_len = max(len(p[1]) for p in allp) + 2

    peer, template_verbs, held_out, verb_action = compute_structure(trp)
    verb_ids = [out_vocab[verb_action[v]] for v in ["jump", "walk", "run", "look"]]
    assert set(verb_ids) == {out_vocab[a] for a in exp90.VERB_ACTIONS}      # identical to exp90's (no arch drift)
    trp_geca, geca_stats = geca_augment(trp, template_verbs, held_out, verb_action)
    print(f"  GECA augmentation: {geca_stats}", file=sys.stderr, flush=True)

    arms = args.arms.split(",")
    res = {a: [] for a in arms}
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    for seed in seeds:
        if "vanilla_plain" in arms:
            res["vanilla_plain"].append(run_vanilla(trp, tep, in_vocab, out_vocab, max_len, args, seed, "vanilla_plain"))
        if "vanilla_geca" in arms:
            res["vanilla_geca"].append(run_vanilla(trp_geca, tep, in_vocab, out_vocab, max_len, args, seed, "vanilla_geca"))
        if "factored_geca" in arms:
            res["factored_geca"].append(run_factored(trp_geca, tep, in_vocab, out_vocab, max_len, verb_ids, args, seed, "factored_geca"))

    out = {"experiment": "91_betb_scan_geca (GECA redundancy guard)",
           "charter": "notes/betb-scan-stage1-consolidation-precommit.md §4",
           "config": vars(args), "geca_stats": geca_stats,
           "arms": {a: summary(res[a]) for a in arms if res[a]},
           "cited_report143": {"factored_baseline": summary(R143_FACTORED_BASELINE),
                               "factored_consolidation": summary(R143_FACTORED_CONSOL)}}
    # Redundancy verdict: consolidation vs the known fix (paired, seeds aligned 0-7).
    if res.get("vanilla_geca") and len(res["vanilla_geca"]) == 8:
        out["consol_minus_vanilla_geca"] = paired_delta(R143_FACTORED_CONSOL, res["vanilla_geca"])
    if res.get("factored_geca") and len(res["factored_geca"]) == 8:
        out["consol_minus_factored_geca"] = paired_delta(R143_FACTORED_CONSOL, res["factored_geca"])
    print(json.dumps(out, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
