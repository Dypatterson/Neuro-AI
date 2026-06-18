"""experiments/96 — Bet B / SCAN-MCD: the STRUCTURE-INJECTION CEILING control (is the arena crackable in-harness?).

CHARTER: RETROSPECTIVE-addendum-2026-06-17 §4 lever 2 + the literature agent's decisive recommendation
("run the field's WINNING ingredient in-harness as a ceiling-control"). After CCC (147) and CCA (148) both
fail the matched comparison, the open question is whether the MCD arena is crackable IN OUR HARNESS at all, or
whether 147/148's nulls are partly harness/scale artifacts.

THE ORACLE (maximally structure-injecting — the absolute ceiling of "if composition were handed to the model"):
train a STANDARD seq2seq normally; at TEST, hand-inject the full compositional structure:
  1. parse each test command at its single top-level conjunction (and/after) into (clause_L, conj, clause_R);
  2. decode each clause STANDALONE with the trained model (feed the clause tokens as a fresh source);
  3. concatenate per the KNOWN SCAN conjunction semantics:  and -> out(L)++out(R) ;  after -> out(R)++out(L)
     (verified: out("A and B")=out(A)++out(B), out("A after B")=out(B)++out(A)); single-clause -> decode whole.
This injects the parse + the concatenation rule (the symbolic compositional skeleton the field's winners use).

DECISION VALUE:
  - oracle >> vanilla (high)  => the arena IS crackable in-harness; structure-injection is the answer (the BIND
    is confirmed: emergent composition nulls, hand-injected composition works) -> 147/148 nulls are REAL, not
    harness artifacts; the win requires the charter-forbidden ingredient.
  - oracle ~ vanilla (low)    => even oracle structure-injection fails in-harness -> the model can't decode
    clauses / the harness is too weak -> 147/148 nulls are partly harness/scale artifacts (do NOT bank a 'wall').
DRILL-DOWN: single-clause-test exact-match (can the model decode a standalone clause at all?) localizes the
bottleneck (clause-decoding vs composition).

PREAMBLE (CLAUDE.md): Active capability = Bet-B Stage-1 arena-crackability ceiling-control (146/147/148).
Headline = oracle exact-match vs vanilla full-command exact-match on MCD, n>=5. Controls = the trained model is
identical; only the EVAL differs (normal vs oracle decompose-decode-concat). Why now: user mandated verify-before-
deciding; this is the literature-recommended decisive ceiling.
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
exp94 = _load("exp94", "94_betb_scan_mcd_ccc.py")
load_pairs, build_vocab, to_ids, pad = exp87.load_pairs, exp87.build_vocab, exp87.to_ids, exp87.pad
PAD, SOS, EOS = exp87.PAD, exp87.SOS, exp87.EOS
MCD = REPO / "data" / "scan" / "mcd_split"


def split_clauses(cmd):
    """Parse a command (token list) at its single top-level conjunction. Returns (L, conj, R); conj in
    {'and','after',None}. Content-blind surface split (same as exp94's parser)."""
    for i, t in enumerate(cmd):
        if t in ("and", "after"):
            return cmd[:i], t, cmd[i + 1:]
    return cmd, None, []


@torch.no_grad()
def _batch_greedy(model, srcs, in_vocab, max_len, batch_size, device):
    """Greedy-decode a list of token-list sources -> list of predicted out-id lists (EOS-truncated, no EOS kept)."""
    preds = []
    for b in range(0, len(srcs), batch_size):
        chunk = srcs[b:b + batch_size]
        src, _ = pad([to_ids(s, in_vocab) for s in chunk], device)
        out = model.greedy(src, src != PAD, max_len).tolist()
        for p in out:
            if EOS in p:
                p = p[:p.index(EOS)]                      # strip EOS (and everything after)
            else:
                p = [t for t in p if t != PAD]
            preds.append(p)
    return preds


@torch.no_grad()
def oracle_exact_match(model, pairs, in_vocab, out_vocab, *, max_len, batch_size, device):
    """Oracle decompose-decode-concatenate exact-match. Also returns single-clause-test exact-match (drill-down)."""
    model.eval()
    # collect all sub-sequences to decode (clause_L and clause_R per command; whole for single-clause)
    jobs = []                                              # (kind, idx) -> which sub-decodes feed command idx
    subs = []
    for c, _ in pairs:
        L, conj, R = split_clauses(c)
        if conj is None:
            jobs.append(("whole", len(subs))); subs.append(c)
        else:
            jL = len(subs); subs.append(L)
            jR = len(subs); subs.append(R)
            jobs.append((conj, jL, jR))
    dec = _batch_greedy(model, subs, in_vocab, max_len, batch_size, device)
    correct = sc_correct = sc_total = 0
    for job, gold_pair in zip(jobs, pairs):
        gold = to_ids(gold_pair[1], out_vocab)             # includes EOS
        if job[0] == "whole":
            pred = dec[job[1]] + [EOS]
            sc_total += 1; sc_correct += int(pred == gold)
        else:
            conj, jL, jR = job
            out_L, out_R = dec[jL], dec[jR]
            pred = (out_L + out_R if conj == "and" else out_R + out_L) + [EOS]   # known SCAN conj semantics
        correct += int(pred == gold)
    return correct / len(pairs), (sc_correct / sc_total if sc_total else float("nan")), sc_total


def load_mcd(split):
    return (load_pairs(str(MCD / f"tasks_train_{split}.txt")), load_pairs(str(MCD / f"tasks_test_{split}.txt")))


def load_single_clause_aug():
    """Single-clause (conjunction-free) SCAN commands + outputs from the `simple` split — gives the model
    CLAUSE-LEVEL competence (the structure-injection the oracle needs; analogous to LeAR/AuxSeq being handed the
    atomic/clause semantics). This is the ceiling control: clause-competence + hand-injected concatenation."""
    SCAN = REPO / "data" / "scan"
    simple = load_pairs(str(SCAN / "tasks_train_simple.txt")) + load_pairs(str(SCAN / "tasks_test_simple.txt"))
    return [(c, a) for c, a in simple if "and" not in c and "after" not in c]


def run_seed(args, seed, split):
    tr, te = load_mcd(split)
    allp = tr + te
    in_vocab, _ = build_vocab(allp, "in"); out_vocab, _ = build_vocab(allp, "out")
    max_len = max(len(p[1]) for p in allp) + 2
    if getattr(args, "clause_aug", False):
        tr = tr + load_single_clause_aug() * 10            # inject clause-level competence (upsampled 10x so the
        #                                                    model reliably learns the 102 clauses; the real ceiling)
    m = exp87.Seq2Seq(len(in_vocab), len(out_vocab), args.embed, args.hidden, seed).to(args.device)
    exp87.train(m, tr, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                tf_ratio=args.tf_ratio, seed=seed, device=args.device)
    van = exp87.exact_match(m, te, in_vocab, out_vocab, max_len=max_len, batch_size=args.batch_size, device=args.device)
    orc, sc, sc_n = oracle_exact_match(m, te, in_vocab, out_vocab, max_len=max_len, batch_size=args.batch_size, device=args.device)
    print(f"  [{split} s{seed}] vanilla={van:.4f}  oracle={orc:.4f}  single-clause-EM={sc:.4f} (n={sc_n})", file=sys.stderr, flush=True)
    return {"vanilla": van, "oracle": orc, "single_clause_em": sc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tf-ratio", type=float, default=0.5, dest="tf_ratio")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--splits", default="mcd1")
    ap.add_argument("--clause-aug", action="store_true", dest="clause_aug",
                    help="inject clause-level competence via single-clause simple data (the real ceiling control)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs = 1, 8

    out = {"experiment": "96_betb_scan_mcd_oracle_ceiling (structure-injection ceiling-control)",
           "charter": "RETROSPECTIVE-addendum-2026-06-17 §4 lever 2", "config": vars(args), "by_split": {}}
    for split in args.splits.split(","):
        rows = [run_seed(args, seed, split) for seed in range(args.seed_start, args.seed_start + args.seeds)]
        agg = {k: exp94.summary([r[k] for r in rows]) for k in ("vanilla", "oracle", "single_clause_em")}
        delta = exp94.paired_delta([r["oracle"] for r in rows], [r["vanilla"] for r in rows]) if len(rows) > 1 else None
        out["by_split"][split] = {"arms": agg, "oracle_minus_vanilla": delta}
        print(f"\n=== {split}: vanilla={agg['vanilla']['mean']:.3f} ORACLE={agg['oracle']['mean']:.3f} "
              f"(Δ={delta['delta_mean']:+.3f} CI{[round(x,3) for x in delta['ci95']]})" if delta else
              f"\n=== {split}: vanilla={agg['vanilla']['mean']:.3f} ORACLE={agg['oracle']['mean']:.3f}",
              file=sys.stderr, flush=True)
    print(json.dumps(out, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
