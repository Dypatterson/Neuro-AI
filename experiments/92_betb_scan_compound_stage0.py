"""experiments/92 — Bet B / SCAN tier-A STAGE-0 GATE: a custom compound-hold-out split, vanilla + faithful GECA.

CHARTER: notes/betb-geca-resistant-regime-precommit.md §1, §3 (tier A), §4 (Stage 0 gate).
After Report 144 falsified the add-jump consolidation's distinctiveness (GECA beats it), the fix is a regime
where the KNOWN method ALSO fails. The 144 lesson, front-loaded: GECA moves from last-guard to STAGE-0 GATE.

THE GATE (precommit §4, build FIRST, ~cheap): on a custom compound-hold-out SCAN split,
  (0a) vanilla seq2seq exact-match << ceiling   (the gap is real, as for add-jump), AND
  (0b) a FAITHFUL, best-shot GECA exact-match << ceiling  (the known augmentation fix ALSO fails).
PASS (both fail) => the regime discriminates against known methods => a Stage-1 mechanism is licensed.
FAIL (GECA solves it) => the regime is GECA-saturated => harden the split or escalate to tier B (canonical MCD).

WHY a NEW GECA (not exp91's): exp91's GECA is the PRIMITIVE-substitution special case (hardcoded held_out={jump}).
A compound hold-out has no single off-manifold primitive; the faithful GECA here is general TOKEN-REWRITE GECA:
discover exchangeable command-token pairs from shared environments, infer each one's SOUND 1:1 action
substitution from train-only minimal pairs, and manufacture augmented examples. All computed-from-buffer
(BUILD CONDITION 1 — nothing hardcoded; the held-out compound is the only input).

HONEST PRIOR (documented before the run): SCAN-simple is the ENTIRE command set (train+test = random 80/20 of
all 20910 commands) and is left<->right / verb symmetric. So for a token-compound hold-out (e.g. `around right`)
every held-out test example's mirror (`around left` form) lives in train, reachable by the single left->right
rewrite => GECA is expected to manufacture ~the whole test set and SOLVE it. We run it to CONFIRM empirically
(>=3 seeds) rather than assert, per project discipline; the likely outcome is the tier-B pivot.

MODEL/eval reuse exp87 (Seq2Seq GRU enc-dec + attention, greedy exact-match). No mechanism here — Stage 0 only.
"""
from __future__ import annotations

import argparse
import collections
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
load_pairs, build_vocab = exp87.load_pairs, exp87.build_vocab
DATA = REPO / "data" / "scan"


# ---- data: the full SCAN command set + a compound-hold-out split -----------------------------------------

def load_simple_all():
    """train+test of the `simple` split = the ENTIRE SCAN command set (random 80/20), each command once."""
    return load_pairs(str(DATA / "tasks_train_simple.txt")) + load_pairs(str(DATA / "tasks_test_simple.txt"))


def contains_subseq(seq, sub):
    n = len(sub)
    return any(list(seq[i:i + n]) == list(sub) for i in range(len(seq) - n + 1))


def compound_split(allp, held_compounds):
    """Partition by the held-out contiguous command compound(s). Assert every command atom stays in train."""
    test, train = [], []
    for c, a in allp:
        (test if any(contains_subseq(c, hc) for hc in held_compounds) else train).append((c, a))
    train_toks = {t for c, _ in train for t in c}
    all_toks = {t for c, _ in allp for t in c}
    missing = all_toks - train_toks
    assert not missing, f"BUILD CONDITION violated: command tokens missing from train: {missing}"
    return train, test


# ---- faithful TOKEN-REWRITE GECA (computed-from-buffer) ---------------------------------------------------

def _environments(pairs):
    env = collections.defaultdict(collections.Counter)
    for c, _ in pairs:
        for i, t in enumerate(c):
            prev = c[i - 1] if i > 0 else "<S>"
            nxt = c[i + 1] if i < len(c) - 1 else "<E>"
            env[t][(prev, nxt)] += 1
    return env


def discover_rewrite_rules(train, min_shared_env=3):
    """Discover SOUND directed token-rewrite rules (a->b, action_sub) from the TRAIN buffer only.

    A rule is sound iff: (a,b) share >= min_shared_env command environments (exchangeable), AND every train
    minimal pair (command c with a) -> (command c with a replaced by b) that ALSO exists in train differs in
    the action by a single CONSISTENT 1:1 token substitution (same length). Returns both directions that pass.
    This excludes around<->opposite and twice<->thrice (action length changes => no 1:1 sub), and keeps
    left<->right and the verb swaps. Nothing hardcoded — the held-out compound never enters this computation."""
    env = _environments(train)
    cmd_index = {tuple(c): a for c, a in train}            # command -> action (commands unique in SCAN)
    toks = [t for t in env]
    rules = []
    for i in range(len(toks)):
        for j in range(len(toks)):
            if i == j:
                continue
            a, b = toks[i], toks[j]
            if len(set(env[a]) & set(env[b])) < min_shared_env:
                continue
            sub, sound, n = {}, True, 0
            for c, act in train:
                if a not in c:
                    continue
                c2 = tuple(b if t == a else t for t in c)
                act2 = cmd_index.get(c2)
                if act2 is None:
                    continue
                if len(act) != len(act2):
                    sound = False; break
                for x, y in zip(act, act2):
                    if x == y:
                        continue
                    if x in sub and sub[x] != y:
                        sound = False; break
                    sub[x] = y
                if not sound:
                    break
                n += 1
            if sound and n >= min_shared_env and sub:
                rules.append((a, b, sub, n))
    return rules


def geca_augment(train, held_compounds, min_shared_env=3):
    """Apply each sound rewrite rule to every train example, keep NEW (deduped) examples. Generous-to-GECA:
    keep ALL generated examples (the guard is stringent only if GECA is strong). Report how many of the
    generated examples land inside the held-out compound region (the direct measure of GECA's reach)."""
    rules = discover_rewrite_rules(train, min_shared_env)
    seen = {(tuple(c), tuple(a)) for c, a in train}
    aug, reached = [], 0
    for a, b, sub, _ in rules:
        for c, act in train:
            if a not in c:
                continue
            new_c = [b if t == a else t for t in c]
            new_a = [sub.get(t, t) for t in act]
            key = (tuple(new_c), tuple(new_a))
            if key in seen:
                continue
            seen.add(key); aug.append((new_c, new_a))
            if any(contains_subseq(new_c, hc) for hc in held_compounds):
                reached += 1
    stats = {
        "n_rules": len(rules),
        "rules": [{"a": a, "b": b, "action_sub": {k: v for k, v in sub.items()}, "n_support": n}
                  for a, b, sub, n in rules],
        "n_original": len(train), "n_generated": len(aug), "n_total": len(train) + len(aug),
        "n_generated_in_heldout_region": reached,
    }
    return train + aug, stats


# ---- arms (reuse exp87 Seq2Seq) ---------------------------------------------------------------------------

def run_vanilla(train_pairs, test_pairs, in_vocab, out_vocab, max_len, args, seed, tag):
    model = exp87.Seq2Seq(len(in_vocab), len(out_vocab), args.embed, args.hidden, seed).to(args.device)
    exp87.train(model, train_pairs, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, tf_ratio=args.tf_ratio, seed=seed, device=args.device)
    acc = exp87.exact_match(model, test_pairs, in_vocab, out_vocab, max_len=max_len,
                            batch_size=args.batch_size, device=args.device)
    print(f"  [{tag} seed {seed}] heldout-compound exact-match = {acc:.4f}", file=sys.stderr, flush=True)
    return acc


def summary(v, n_boot=10000, seed=0):
    t = torch.tensor(v, dtype=torch.float64)
    if t.numel() == 1:
        return {"mean": float(t), "min": float(t), "max": float(t), "ci95": [float(t), float(t)], "per_seed": v}
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, t.numel(), (n_boot, t.numel()), generator=g)
    means = t[idx].mean(1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return {"mean": float(t.mean()), "min": float(t.min()), "max": float(t.max()), "ci95": [lo, hi], "per_seed": v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tf-ratio", type=float, default=0.5, dest="tf_ratio")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--held", default="around right",
                    help="held-out contiguous command compound(s); ';'-separated, tokens space-separated")
    ap.add_argument("--min-shared-env", type=int, default=3, dest="min_shared_env")
    ap.add_argument("--arms", default="vanilla_plain,vanilla_geca")
    ap.add_argument("--ceiling-frac", type=float, default=0.9, dest="ceiling_frac",
                    help="gate: an arm 'fails' if mean <= ceiling_frac * simple-ceiling (~1.0)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs = 1, 3

    held_compounds = [tuple(h.strip().split()) for h in args.held.split(";") if h.strip()]
    allp = load_simple_all()
    train, test = compound_split(allp, held_compounds)
    in_vocab, _ = build_vocab(allp, "in"); out_vocab, _ = build_vocab(allp, "out")
    max_len = max(len(p[1]) for p in allp) + 2
    train_geca, geca_stats = geca_augment(train, held_compounds, args.min_shared_env)
    print(f"  held-out compounds: {held_compounds}", file=sys.stderr, flush=True)
    print(f"  split: train={len(train)} test(held-out)={len(test)} total={len(allp)}", file=sys.stderr, flush=True)
    print(f"  GECA: {geca_stats['n_rules']} sound rules, +{geca_stats['n_generated']} generated "
          f"({geca_stats['n_generated_in_heldout_region']} in held-out region)", file=sys.stderr, flush=True)

    arms = args.arms.split(",")
    res = {a: [] for a in arms}
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    for seed in seeds:
        if "vanilla_plain" in arms:
            res["vanilla_plain"].append(run_vanilla(train, test, in_vocab, out_vocab, max_len, args, seed, "vanilla_plain"))
        if "vanilla_geca" in arms:
            res["vanilla_geca"].append(run_vanilla(train_geca, test, in_vocab, out_vocab, max_len, args, seed, "vanilla_geca"))

    arm_summ = {a: summary(res[a]) for a in arms if res[a]}
    # Gate: both vanilla AND geca must FAIL (mean <= ceiling_frac, ceiling≈1.0 for SCAN simple).
    fail = {a: (arm_summ[a]["mean"] <= args.ceiling_frac) for a in arm_summ}
    gate_pass = bool(fail.get("vanilla_plain", False) and fail.get("vanilla_geca", False))
    out = {
        "experiment": "92_betb_scan_compound_stage0 (tier-A Stage-0 gate)",
        "charter": "notes/betb-geca-resistant-regime-precommit.md §1,§3,§4",
        "config": vars(args), "held_compounds": [list(h) for h in held_compounds],
        "split_sizes": {"train": len(train), "test_heldout": len(test), "total": len(allp)},
        "geca_stats": geca_stats,
        "arms": arm_summ,
        "arm_fails(mean<=ceiling_frac)": fail,
        "STAGE0_GATE_vanilla_fails_AND_geca_fails": gate_pass,
        "verdict": (
            f"tier-A Stage 0 on held={held_compounds}: vanilla={arm_summ.get('vanilla_plain',{}).get('mean',float('nan')):.4f} "
            f"geca={arm_summ.get('vanilla_geca',{}).get('mean',float('nan')):.4f} (ceiling~1.0). "
            f"GATE(both fail)={gate_pass}. PASS => regime discriminates vs known methods => Stage-1 mechanism licensed. "
            f"FAIL => GECA-saturated => harden split or escalate to tier B (canonical MCD)."),
    }
    print(json.dumps(out, indent=2))
    print(f"\n=== {out['verdict']}", file=sys.stderr, flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
