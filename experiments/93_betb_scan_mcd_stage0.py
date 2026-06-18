"""experiments/93 — Bet B / SCAN tier-B STAGE-0 GATE on canonical MCD (mcd1/mcd2/mcd3).

CHARTER: notes/betb-geca-resistant-regime-precommit.md §3 (tier B), §4 (Stage 0 gate).
After tier-A confirmed SCAN-simple compound hold-outs are GECA-saturated (Report 145: `around right`
vanilla 0.005 / lower-bound token-GECA 0.848), escalate to the regime KNOWN methods also fail.

Canonical MCD (Keysers et al. 2020, maximum compound divergence with matched atom distribution) is
empirically GECA-RESISTANT: published faithful GECA = 51.5 / 30.4 / 12.0 on mcd1/2/3 (Conklin et al. 2021,
arXiv:2106.04252, Table 2) — i.e. the known augmentation fix FAILS (<< ceiling ~1.0), the property tier-A
lacked. Data: data/scan/mcd_split/ (SegwangKim/SCAN fork, pinned SHA in PROVENANCE_*.sha; verified zero
train/test leakage, zero OOV, all atoms in train).

THE GATE (precommit §4, build FIRST): on each MCD split,
  (0a) vanilla seq2seq exact-match << ceiling   (the gap is real), AND
  (0b) a faithful GECA exact-match << ceiling    (the known fix ALSO fails).
PASS (both fail) => the regime discriminates against KNOWN methods => a Stage-1 mechanism is licensed
(and the Stage-1 bar = beat the general-purpose neural floor: vanilla ~5%, GECA ~31%, MAML ~32%, T5 <17% —
all < 50% mean MCD; the ~99-100% ceiling is held only by structure-injecting AuxSeq/symbolic-LeAR).

GECA ARM: the exp92 faithful TOKEN-REWRITE GECA (computed-from-buffer; discovers sound rewrite rules +
their 1:1 action subs). This is a LOWER BOUND on GECA power; on MCD even the published STRONG (fragment)
GECA fails (51/30/12), so a weaker in-house GECA also failing is consistent + trustworthy for 0b.

MODEL/eval reuse exp87. No mechanism here — Stage 0 only.
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


exp92 = _load("exp92", "92_betb_scan_compound_stage0.py")
exp87 = exp92.exp87
load_pairs, build_vocab = exp87.load_pairs, exp87.build_vocab
MCD = REPO / "data" / "scan" / "mcd_split"

# Published reference bar (Conklin et al. 2021, arXiv:2106.04252, Table 2) — cited, NOT re-run.
PUBLISHED = {
    "GECA(faithful,strong)": {"mcd1": 0.515, "mcd2": 0.304, "mcd3": 0.120},
    "vanilla_LSTM": {"mcd1": 0.047, "mcd2": 0.073, "mcd3": 0.018},
    "Lev-MAML": {"mcd1": 0.476, "mcd2": 0.352, "mcd3": 0.114},
    "T5-base": {"mcd1": 0.262, "mcd2": 0.079, "mcd3": 0.121},
    "AuxSeq(structure-injecting)": {"mcd1": 0.999, "mcd2": 0.901, "mcd3": 0.982},
    "LeAR(symbolic)": {"mcd1": 1.0, "mcd2": 1.0, "mcd3": 1.0},
}


def run_split(split, args):
    tr = load_pairs(str(MCD / f"tasks_train_{split}.txt"))
    te = load_pairs(str(MCD / f"tasks_test_{split}.txt"))
    allp = tr + te
    in_vocab, _ = build_vocab(allp, "in"); out_vocab, _ = build_vocab(allp, "out")
    max_len = max(len(p[1]) for p in allp) + 2

    # faithful token-rewrite GECA on this split's train buffer (reach = how many test examples it manufactures)
    tr_geca, gstats = exp92.geca_augment(tr, [], args.min_shared_env)
    test_set = {(tuple(c), tuple(a)) for c, a in te}
    reach = sum(((tuple(c), tuple(a)) in test_set) for c, a in tr_geca[len(tr):])
    gstats["n_generated_in_test"] = reach
    gstats["test_coverage"] = reach / len(te)

    res = {a: [] for a in args.arms.split(",")}
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        if "vanilla_plain" in res:
            res["vanilla_plain"].append(exp92.run_vanilla(tr, te, in_vocab, out_vocab, max_len, args, seed, f"{split}/vanilla_plain"))
        if "vanilla_geca" in res:
            res["vanilla_geca"].append(exp92.run_vanilla(tr_geca, te, in_vocab, out_vocab, max_len, args, seed, f"{split}/vanilla_geca"))
    return {"arms": {a: exp92.summary(v) for a, v in res.items() if v}, "geca_stats": gstats,
            "sizes": {"train": len(tr), "test": len(te)}}


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
    ap.add_argument("--splits", default="mcd1,mcd2,mcd3")
    ap.add_argument("--min-shared-env", type=int, default=3, dest="min_shared_env")
    ap.add_argument("--arms", default="vanilla_plain,vanilla_geca")
    ap.add_argument("--ceiling-frac", type=float, default=0.5, dest="ceiling_frac",
                    help="gate: an arm 'fails' if mean <= ceiling_frac (MCD known-method floor is <50%)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs, args.splits = 1, 3, "mcd1"

    splits = args.splits.split(",")
    by_split = {s: run_split(s, args) for s in splits}

    gate = {}
    for s in splits:
        arms = by_split[s]["arms"]
        v = arms.get("vanilla_plain", {}).get("mean", float("nan"))
        g = arms.get("vanilla_geca", {}).get("mean", float("nan"))
        gate[s] = {"vanilla_fails": bool(v <= args.ceiling_frac), "geca_fails": bool(g <= args.ceiling_frac),
                   "PASS": bool(v <= args.ceiling_frac and g <= args.ceiling_frac)}
    gate_all = all(gate[s]["PASS"] for s in splits)

    out = {
        "experiment": "93_betb_scan_mcd_stage0 (tier-B canonical MCD Stage-0 gate)",
        "charter": "notes/betb-geca-resistant-regime-precommit.md §3,§4",
        "data_provenance": "data/scan/mcd_split/ (SegwangKim/SCAN fork @ SHA in PROVENANCE_*.sha; Keysers 2020 MCD)",
        "config": vars(args), "by_split": by_split,
        "published_reference_bar": PUBLISHED,
        "gate_by_split": gate,
        "STAGE0_GATE_ALL_vanilla_AND_geca_fail": gate_all,
        "verdict": (
            "tier-B MCD Stage 0: per-split vanilla/geca exact-match below ceiling => regime discriminates vs "
            "KNOWN methods. PASS => Stage-1 mechanism licensed (bar = beat the general-purpose neural floor "
            "vanilla/GECA/MAML/T5 all <50% mean MCD; AuxSeq/LeAR ceiling is structure-injecting)."),
    }
    print(json.dumps(out, indent=2))
    print(f"\n=== GATE(all splits) = {gate_all} ; per-split = "
          f"{ {s: gate[s]['PASS'] for s in splits} }", file=sys.stderr, flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
