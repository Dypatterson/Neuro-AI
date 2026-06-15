"""experiments/86 — Bet B: the COMPOSITIONAL discriminating regime (Step 1 = VALIDITY gap-diagnostic).

CHARTER: notes/betb-compositional-discriminating-regime-precommit.md (extends CONTEXT-B.md §8 with the
retrospective-§6 discriminating regime). This is a VALIDITY DIAGNOSTIC, NOT a graduation experiment
(CLAUDE.md preamble rule): it asks whether the regime is even discriminating before any mechanism is built,
exactly as Report 136's gap-diagnostic preceded the 137 mechanism attempt.

THE REGIME (nested-binary, zero model change vs exp83/85). Per alphabet block (fresh disjoint tokens):
  A  primitive 1   A(i,j) = (i + j) mod p           (the additive circuit; groks ~5.7k)
  B  primitive 2   B(i,j) = (i * j) mod p           (a genuinely different, NONLINEAR circuit; groks ~8k alone)
  C  composition   C(i,j) = A(i,j) + B(i,j) = (i + j + i*j) mod p   (combines BOTH primitives' outputs; groks ~4k)

  [2026-06-15 design note: the original C=((i+j)*j) was a PATHOLOGICAL target — it never groks (flat MLP can't
   fit it even at 30k steps), so there was no FTSR headroom. A grokkability sweep found C=A+B=(i+j+ij) groks
   reliably (4k) while genuinely using both primitives. CAVEAT surfaced by the sweep: mod-p compositions in this
   flat MLP appear bimodal — grokked-directly-from-scratch OR never — so whether this C leaves a real COMPOSITION
   gap (learnable only by composing A,B) vs is just-another-easy-polynomial is exactly what this pilot measures.]
Tokens are base+i, base+j (base = block*p); the head predicts the residue in 0..p-1 (exp83 convention).
At least one primitive (B) is nonlinear, so C is NOT a single collapsible linear map. The composition is
NOT in the target definition — it is in whether the model REUSES its learned A,B to learn C faster.

THE GAP we are testing (the simple method = replay + EWC-lite soft anchor):
  Parts transfer:        FTSR(A_j), FTSR(B_j) > 1   (protection works on the primitives)
  Composition does NOT:  FTSR(C_j) ~ 1              (recombination is the hard part protection can't assemble)
  Headroom is real:      frozen_oracle / joint show C IS reachable (FTSR_C >> 1 / joint acc ~ 1)
If instead FTSR(C_j) > 1 under replay+EWC, the regime is NOT discriminating -> climb the §6 primitive ladder
(iterate-fuel, not a dead end). A clean null here is a few hours.

ARMS (FTSR via per-task steps-to-crit; scratch = denominator):
  scratch            fresh model per task                              (FTSR denominator)
  floor              sequential, no replay, no consolidation           (anchor)
  replay_only        interleaved replay, no anchor
  replay_plus_ewc    interleaved replay + EWC-lite anchor on shared MLP (THE simple method; engages after block 0)
  frozen_oracle      replay; for x-block C tasks FREEZE the MLP, learn C on head+emb only  (composability, 136-style)
  joint (ceiling)    train all tasks jointly; report final per-type held-out acc           (C is learnable?)

SCRAMBLE (redesigned; Report 134's was invalid): run with --scramble to replace C by C' = B_rand(A_rand(i,j), j)
with fresh random Z_p tables (matched output entropy). Compare FTSR_C and scratch-difficulty across the two
invocations: the transfer lift must die on C', and scratch(C) ~ scratch(C') (difficulty-matched).

ANTI-HOMUNCULUS: the stream is a fixed schedule; replay is content-blind; EWC-lite is a fixed local L2 pull.
NO novel mechanism in Step 1 (that is Step 2). Fence-clean (no SVD as a mechanism; joint = iterative SGD).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Reuse the established harness leaves (no model change).
_s83 = importlib.util.spec_from_file_location("exp83", REPO / "experiments" / "83_betb_two_timescale.py")
exp83 = importlib.util.module_from_spec(_s83); _s83.loader.exec_module(exp83)
_s85 = importlib.util.spec_from_file_location("exp85", REPO / "experiments" / "85_betb_replay_x_consolidation.py")
exp85 = importlib.util.module_from_spec(_s85); _s85.loader.exec_module(exp85)

split_task = exp83.split_task
to_tensors = exp83.to_tensors
ContinualNet = exp83.ContinualNet
evaluate = exp83.evaluate
train_task = exp83.train_task
boot_ci = exp83.boot_ci
EWCAnchor = exp85.EWCAnchor
train_task_bf = exp85.train_task_bf

TYPES = ("A", "B", "C")                 # per-block task order
ARMS = ["scratch", "floor", "replay_only", "replay_plus_ewc", "frozen_oracle"]
FTSR_ARMS = ["floor", "replay_only", "replay_plus_ewc", "frozen_oracle"]


# ---- task tables (residue-valued targets, exp83 convention) -------------------------------------------------

def _rand_table(p, gen):
    """A fixed random binary op Z_p x Z_p -> Z_p (matched-entropy scramble primitive)."""
    return torch.randint(0, p, (p, p), generator=gen).tolist()


def make_task(typ, p, base, gen, scram=None):
    """Rows (base+i, base+j, target). typ in {A,B,C}. scram=(A_rand,B_rand) replaces C by C'=B_rand(A_rand,.)."""
    rows = []
    for i in range(p):
        for j in range(p):
            if typ == "A":
                y = (i + j) % p
            elif typ == "B":
                y = (i * j) % p
            else:  # "C" composition  C(i,j) = A(i,j) + B(i,j) = (i + j + i*j) mod p  (uses BOTH primitives)
                if scram is None:
                    y = (i + j + i * j) % p
                else:
                    Ar, Br = scram
                    y = (Ar[i][j] + Br[i][j]) % p  # A_rand + B_rand — same "sum of two ops" form, unlearned pieces
            rows.append((base + i, base + j, y))
    return rows


def build_stream(p, n_alph, frac, gen, scramble=False):
    """Per block: [A, B, C] sharing one fresh alphabet (base=block*p). task t -> type t%3, block t//3."""
    scram = (_rand_table(p, gen), _rand_table(p, gen)) if scramble else None
    stream, meta = [], []
    for blk in range(n_alph):
        base = blk * p
        for typ in TYPES:
            rows = make_task(typ, p, base, gen, scram if typ == "C" else None)
            stream.append(split_task(rows, frac, gen))
            meta.append((typ, blk))
    return stream, meta, n_alph * p


# ---- one arm ------------------------------------------------------------------------------------------------

def run_arm(arm, p, n_alph, frac, *, embed, hidden, max_steps, crit, eval_every, replay_frac, lr, weight_decay,
            ewc_lambda, ewc_start_blk, scramble, seed, device):
    g = torch.Generator().manual_seed(seed * 104729 + 7)
    stream, meta, vocab = build_stream(p, n_alph, frac, g, scramble)
    K = len(stream)
    steps = [None] * K
    ret_after = [None] * K

    if arm == "scratch":
        for k in range(K):
            m = ContinualNet(vocab, p, K, embed, hidden, seed).to(device)
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
            steps[k] = train_task(m, opt, k, stream[k][0], stream[k][1], freeze_mlp=False, max_steps=max_steps,
                                  crit=crit, eval_every=eval_every, buf=[], replay_frac=0.0, gen=g, device=device)
        return {"steps": steps, "ret_after": None, "meta": meta}

    do_replay = arm in ("replay_only", "replay_plus_ewc", "frozen_oracle")
    m = ContinualNet(vocab, p, K, embed, hidden, seed).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
    ewc = None
    buf = []
    for k in range(K):
        typ, blk = meta[k]
        if arm == "replay_plus_ewc" and blk == ewc_start_blk and ewc is None:
            ewc = EWCAnchor(m.mlp.parameters(), ewc_lambda, device)     # engage on a formed circuit
        rf = replay_frac if do_replay else 0.0
        # frozen_oracle: freeze the MLP for x-block C tasks (composability read of the existing circuit)
        freeze = (arm == "frozen_oracle" and typ == "C" and blk >= 1)
        if ewc is not None:
            steps[k] = train_task_bf(m, opt, ewc, k, stream[k][0], stream[k][1], max_steps=max_steps, crit=crit,
                                     eval_every=eval_every, buf=buf, replay_frac=rf, gen=g, device=device)
        else:
            steps[k] = train_task(m, opt, k, stream[k][0], stream[k][1], freeze_mlp=freeze, max_steps=max_steps,
                                  crit=crit, eval_every=eval_every, buf=buf, replay_frac=rf, gen=g, device=device)
        buf.append((stream[k][0], k))
        ret_after[k] = [evaluate(m, j, stream[j][1], device) for j in range(k + 1)]
    return {"steps": steps, "ret_after": ret_after, "meta": meta}


def run_joint(p, n_alph, frac, *, embed, hidden, max_steps, crit, eval_every, lr, weight_decay, scramble,
              seed, device):
    """Ceiling: train all tasks jointly; report final per-type held-out accuracy (is C learnable / reachable?)."""
    g = torch.Generator().manual_seed(seed * 104729 + 7)
    stream, meta, vocab = build_stream(p, n_alph, frac, g, scramble)
    K = len(stream)
    m = ContinualNet(vocab, p, K, embed, hidden, seed).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
    data = [(to_tensors(stream[k][0], device), k) for k in range(K)]
    for _ in range(max_steps):
        loss = 0.0
        for ((a, b, y), k) in data:
            loss = loss + F.cross_entropy(m(a, b, k), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    acc = [evaluate(m, k, stream[k][1], device) for k in range(K)]
    return {"acc": acc, "meta": meta}


# ---- aggregation --------------------------------------------------------------------------------------------

def _xblock_idx(meta, typ):
    """x-block tasks of a given type: block >= 1 (after >=1 full block of primitive experience)."""
    return [k for k, (t, blk) in enumerate(meta) if t == typ and blk >= 1]


def collect(args, seed_start, seeds):
    res = {a: [] for a in ARMS}
    joint = []
    for seed in range(seed_start, seed_start + seeds):
        for a in ARMS:
            res[a].append(run_arm(a, args.p, args.n_alph, args.frac, embed=args.embed, hidden=args.hidden,
                                  max_steps=args.max_steps, crit=args.crit, eval_every=args.eval_every,
                                  replay_frac=args.replay_frac, lr=args.lr, weight_decay=args.weight_decay,
                                  ewc_lambda=args.ewc_lambda, ewc_start_blk=args.ewc_start_blk,
                                  scramble=args.scramble, seed=seed, device=args.device))
        joint.append(run_joint(args.p, args.n_alph, args.frac, embed=args.embed, hidden=args.hidden,
                               max_steps=args.joint_steps, crit=args.crit, eval_every=args.eval_every, lr=args.lr,
                               weight_decay=args.weight_decay, scramble=args.scramble, seed=seed, device=args.device))
        meta = res["scratch"][-1]["meta"]
        def xf(a, typ):
            ks = _xblock_idx(meta, typ)
            return sum(res["scratch"][-1]["steps"][k] / res[a][-1]["steps"][k] for k in ks) / len(ks)
        print(f"[seed {seed}] replay+EWC x-FTSR  A={xf('replay_plus_ewc','A'):.2f} B={xf('replay_plus_ewc','B'):.2f} "
              f"C={xf('replay_plus_ewc','C'):.2f} | frozen C={xf('frozen_oracle','C'):.2f}",
              file=sys.stderr, flush=True)
    return res, joint


def aggregate(res, joint, p, n_alph, cfg):
    seeds = len(res["floor"])
    meta = res["scratch"][0]["meta"]
    K = len(meta)
    equiv = cfg["equiv_log"]

    def lf(arm, typ):                      # per-seed mean log-FTSR over x-block tasks of a type
        ks = _xblock_idx(meta, typ)
        return [sum(math.log(res["scratch"][s]["steps"][k] / res[arm][s]["steps"][k]) for k in ks) / len(ks)
                for s in range(seeds)]

    def raw(arm, typ):
        ks = _xblock_idx(meta, typ)
        return [sum(res["scratch"][s]["steps"][k] / res[arm][s]["steps"][k] for k in ks) / len(ks)
                for s in range(seeds)]

    ftsr_raw = {a: {t: boot_ci(raw(a, t), seed=hash((a, t)) % 9991) for t in TYPES} for a in FTSR_ARMS}
    ftsr_log = {a: {t: boot_ci(lf(a, t), seed=hash((a, t, "L")) % 9991) for t in TYPES} for a in FTSR_ARMS}

    # joint ceiling: mean held-out acc per type over x-block tasks
    def jacc(typ):
        ks = _xblock_idx(meta, typ)
        return boot_ci([sum(joint[s]["acc"][k] for k in ks) / len(ks) for s in range(seeds)], seed=7)
    joint_acc = {t: jacc(t) for t in TYPES}

    # per-block FTSR_C vs block (compounding drill-down) under the simple method
    cblocks = sorted({blk for (t, blk) in meta if t == "C" and blk >= 1})
    def cidx(blk):
        return [k for k, (t, b) in enumerate(meta) if t == "C" and b == blk][0]
    perblock_C = {f"blk{blk}": boot_ci([res["scratch"][s]["steps"][cidx(blk)] / res["replay_plus_ewc"][s]["steps"][cidx(blk)]
                                        for s in range(seeds)], seed=blk) for blk in cblocks}

    # ---- the 4 validity conditions (the Step-1 headline) ----
    sm = "replay_plus_ewc"
    A_lo, B_lo = ftsr_raw[sm]["A"][1], ftsr_raw[sm]["B"][1]
    C_ci = ftsr_raw[sm]["C"]
    parts_transfer = bool(A_lo > 1.0 and B_lo > 1.0)
    # composition does NOT transfer: CI must include 1 (NOT CI-lo>1). Gap is cleaner the lower C_ci is.
    comp_no_transfer = bool(not (C_ci[1] > 1.0))
    headroom_real = bool(ftsr_raw["frozen_oracle"]["C"][1] > 1.0 or joint_acc["C"][1] > 0.90)
    DISCRIMINATING = bool(parts_transfer and comp_no_transfer and headroom_real)

    verdict = (
        "exp86 COMPOSITIONAL regime VALIDITY (n={n}, raw FTSR over x-block tasks; scramble={scr}). "
        "Simple method (replay+EWC): A={A:.2f}[{Al:.2f},{Ah:.2f}] B={B:.2f}[{Bl:.2f},{Bh:.2f}] "
        "C={C:.2f}[{Cl:.2f},{Ch:.2f}]. Composability: frozen_oracle C={Fc:.2f}[{Fcl:.2f},], joint C-acc={Jc:.2f}. "
        "VALIDITY: parts_transfer(A,B>1)={pt}; composition_no_transfer(C~1)={cnt}; headroom_real={hr} "
        "=> DISCRIMINATING={D}. {note}"
    ).format(n=seeds, scr=cfg["scramble"],
             A=ftsr_raw[sm]["A"][0], Al=ftsr_raw[sm]["A"][1], Ah=ftsr_raw[sm]["A"][2],
             B=ftsr_raw[sm]["B"][0], Bl=ftsr_raw[sm]["B"][1], Bh=ftsr_raw[sm]["B"][2],
             C=C_ci[0], Cl=C_ci[1], Ch=C_ci[2],
             Fc=ftsr_raw["frozen_oracle"]["C"][0], Fcl=ftsr_raw["frozen_oracle"]["C"][1], Jc=joint_acc["C"][0],
             pt=parts_transfer, cnt=comp_no_transfer, hr=headroom_real, D=DISCRIMINATING,
             note=("If NOT discriminating because C transfers under the simple method: climb the §6 primitive "
                   "ladder (iterate-fuel). This is a VALIDITY diagnostic, not a graduation."))

    return {
        "experiment": "86_betb_compositional_regime (Bet B — Step 1 VALIDITY gap-diagnostic)",
        "charter": "notes/betb-compositional-discriminating-regime-precommit.md (extends CONTEXT-B.md §8)",
        "config": {**cfg, "seeds": seeds, "n_tasks": K, "x_block_C_blocks": cblocks},
        "VALIDITY_conditions": {"parts_transfer_A_B_gt_1": parts_transfer,
                                "composition_no_transfer_C_approx_1": comp_no_transfer,
                                "headroom_real": headroom_real,
                                "DISCRIMINATING": DISCRIMINATING},
        "FTSR_raw_by_type": ftsr_raw, "FTSR_logFTSR_by_type": ftsr_log,
        "joint_ceiling_heldout_acc_by_type": joint_acc,
        "perblock_FTSR_C_simple_method": perblock_C,
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=17)
    ap.add_argument("--n-alph", type=int, default=4, dest="n_alph")          # blocks; K = 3*n_alph tasks
    ap.add_argument("--frac", type=float, default=0.7)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=8000, dest="max_steps")
    ap.add_argument("--joint-steps", type=int, default=4000, dest="joint_steps")
    ap.add_argument("--crit", type=float, default=0.90)
    ap.add_argument("--eval-every", type=int, default=100, dest="eval_every")
    ap.add_argument("--replay-frac", type=float, default=0.5, dest="replay_frac")
    ap.add_argument("--ewc-lambda", type=float, default=0.01, dest="ewc_lambda")
    ap.add_argument("--ewc-start-blk", type=int, default=1, dest="ewc_start_blk")  # engage after block 0
    ap.add_argument("--equiv-log", type=float, default=0.22, dest="equiv_log")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0, dest="weight_decay")
    ap.add_argument("--scramble", action="store_true")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.n_alph, args.max_steps, args.joint_steps = 2, 2, 1500, 800

    cfg = {"p": args.p, "n_alph": args.n_alph, "frac": args.frac, "embed": args.embed, "hidden": args.hidden,
           "max_steps": args.max_steps, "joint_steps": args.joint_steps, "crit": args.crit,
           "replay_frac": args.replay_frac, "ewc_lambda": args.ewc_lambda, "ewc_start_blk": args.ewc_start_blk,
           "equiv_log": args.equiv_log, "lr": args.lr, "weight_decay": args.weight_decay, "scramble": args.scramble}

    res, joint = collect(args, args.seed_start, args.seeds)
    out = aggregate(res, joint, args.p, args.n_alph, cfg)
    out["per_seed_raw"] = res
    out["per_seed_joint"] = joint
    out["config"]["seed_start"] = args.seed_start

    print(json.dumps(out, indent=2))
    print(f"\n=== {out['verdict']}", file=sys.stderr, flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
