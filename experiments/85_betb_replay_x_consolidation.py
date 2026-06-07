"""experiments/85 — Bet B: the replay × CONSOLIDATION interaction (the thesis-critical test).

CHARTER: CONTEXT-B.md §5/§8 + the "Terminology & the framing fix (2026-06-06)" block. This is the FIRST
experiment that tests Bet B's distinctive claim, and it is posed as an INTERACTION, not a head-to-head:

  REPLAY        = re-presentation of stored experience (the substrate; interleaved rehearsal). Report 138 showed
                  interleaved replay carries ALL of the baseline's new-alphabet compounding; an offline REHEARSAL
                  pass adds nothing to transfer. So replay alone is the bar to beat (x-block FTSR ~6.3).
  CONSOLIDATION = an offline RESTRUCTURING operation via a NON-reconstruction objective (NOT re-presenting
                  examples — that is replay). It must be anti-homunculus (a fixed local objective) and respect
                  the fence (no one-shot closed-form SVD/eig; iterative/learned global is allowed).

THE QUESTION (CONTEXT-B §8 Terminology): does **replay + consolidation** manufacture new-alphabet forward
transfer that **neither replay-alone nor consolidation-alone** produces? = the superadditive INTERACTION.

2×2 FACTORIAL (replay {off,on} × consolidation {off,on}) + scratch denominator:
  scratch              fresh model per task                                  (FTSR denominator)
  floor                NO replay, NO consolidation (pure sequential)         (= exp84 no_replay_no_consol; ANCHOR)
  replay_only      R   interleaved replay, NO consolidation                  (= exp84 replay_no_consol; ANCHOR ~6.3)
  consol_only      C   NO replay, consolidation ON                           (degenerate if recipe is data-space)
  replay_plus_consol RC interleaved replay + consolidation                   (the mechanism arm)

HEADLINE (interaction, in LOG-FTSR over new-alphabet x-block tasks, paired bootstrap, equivalence margin):
  Δ_RC_vs_R = logFTSR(RC) − logFTSR(R)   does consolidation ADD on top of replay?  (the load-bearing delta;
                                          for a data-space recipe where C-only is degenerate, THIS is the test)
  Δ_RC_vs_C = logFTSR(RC) − logFTSR(C)   does replay add on top of consolidation?
  Δ_super   = [logFTSR(RC) − logFTSR(floor)] − [(logFTSR(R) − logFTSR(floor)) + (logFTSR(C) − logFTSR(floor))]
              = logFTSR(RC) + logFTSR(floor) − logFTSR(R) − logFTSR(C)   the formal superadditivity term
PASS (interaction) = Δ_RC_vs_R CI-lo > 0 AND Δ_RC_vs_C CI-lo > 0 (RC beats BOTH) AND RC compounds
  (last-block logFTSR ≫ first, CI-disjoint) AND retention(RC) held (≥ R − 0.05). Drop the "beats C" clause to
  INFORMATIVE-only when the recipe is data-space (C-only degenerate). No PASS/GRADUATES token unless the
  interaction clears; this is a graduation ATTEMPT for the §8 headline.

ANTI-HOMUNCULUS: replay is uniform/content-blind; the consolidation operation must be a FIXED local dynamic (no
metric-reading supervisor, no "if alphabet-X then ...").

RECIPE 1 = BENNA-FUSI multi-timescale synaptic consolidation (Benna & Fusi 2016) on the SHARED MLP, chosen by the
user as the first swappable recipe (design workflow wvxj66tdp ranked GERM #1 among 3 proposers, but the
domain-expert grounding flagged Benna-Fusi as the cleanest non-replay weight-space mechanism in the corpus AND
the principled, *graded* fix for why Report 137's BINARY freeze nulled). Each shared-MLP weight w becomes the
visible variable u_1 of a chain u_1..u_m with geometric capacitances C_k=2^(k-1) and a fixed coupling g; after
every optimizer step (which injects plasticity into u_1=w via Adam), the chain DIFFUSES — slower variables
protect old structure and feed it back into u_1, resisting overwrite WITHOUT hard-freezing (137's failure mode),
and the circuit still improves. It is a CONTINUOUS during-learning weight dynamic, NOT an offline pass; so
"consolidation ON" = the shared MLP is a Benna-Fusi synapse. Embeddings + heads stay fast (the two-timescale
split). Anti-homunculus: a fixed local linear relaxation, no metric read / no task identity. Fence: iterative
local updates, no SVD/eig. Weight-space → consol_only (BF, no replay) is NON-degenerate → full clean 2×2. The
offline-pass slot consolidate_restructure() is retained for FUTURE recipes (GERM/CANON fallbacks); BF does not
use it.
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

# Bit-identical leaf functions: `floor` and `replay_only` MUST reproduce exp84 (anchor).
_spec = importlib.util.spec_from_file_location("exp83", REPO / "experiments" / "83_betb_two_timescale.py")
exp83 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(exp83)
build_stream = exp83.build_stream
ContinualNet = exp83.ContinualNet
train_task = exp83.train_task
evaluate = exp83.evaluate
to_tensors = exp83.to_tensors
boot_ci = exp83.boot_ci

ARMS = ["scratch", "floor", "replay_only", "consol_only", "replay_plus_consol"]
BASE_ARMS = ["floor", "replay_only", "consol_only", "replay_plus_consol"]
RET_SEED = {a: 10 + i for i, a in enumerate(BASE_ARMS)}
XBLOCK = None   # new-alphabet tasks: even >= 2
WBLOCK = None   # within-block sanity


class BennaFusi:
    """Benna & Fusi 2016 multi-timescale synaptic consolidation on a set of parameters (here: the shared MLP).
    Each param w is the visible variable u_1 of a chain u_1..u_m; slower variables (geometric capacitances
    C_k = 2^(k-1)) protect old structure and feed it back. After each optimizer step injects plasticity into
    u_1 = w, diffuse() relaxes the chain one step with fixed coupling g and reflective boundaries:
        du_k = (g / C_k) * [ (u_{k-1} - u_k) + (u_{k+1} - u_k) ]
    The slow chain is initialized equilibrated (all = the init weight). Fixed local linear dynamic
    (anti-homunculus: no metric read, no task identity); iterative, no SVD/eig (fence-clean)."""

    def __init__(self, params, m, g, device):
        self.params = list(params)                      # live u_1 tensors (shared-MLP weights/biases)
        self.m = m
        self.g = g
        self.C = [float(2 ** k) for k in range(m)]       # C_k, k = 0..m-1 (geometric timescales)
        self.slow = [p.detach().clone().unsqueeze(0).repeat(m - 1, *([1] * p.dim())) for p in self.params]

    @torch.no_grad()
    def diffuse(self):
        for p, s in zip(self.params, self.slow):
            u = torch.cat([p.unsqueeze(0), s], dim=0)    # full chain (m, *shape): u[0]=live weight, u[1:]=slow
            up = torch.cat([u[:1], u[:-1]], dim=0)        # u_{k-1}, reflective at the live (k=0) boundary
            dn = torch.cat([u[1:], u[-1:]], dim=0)        # u_{k+1}, reflective at the deep (k=m-1) boundary
            du = torch.stack([(self.g / self.C[k]) * ((up[k] - u[k]) + (dn[k] - u[k])) for k in range(self.m)], 0)
            u = u + du
            p.copy_(u[0]); s.copy_(u[1:])


class EWCAnchor:
    """Frozen-reference L2 weight anchor (EWC-lite, uniform / no Fisher) — the CONTROL recipe for the Report 139
    attribution. At engage time, snapshot theta* = current shared-MLP weights (FROZEN). After each optimizer step,
    pull live weights toward theta* by lam: theta -= lam*(theta - theta*). Single knob lam (anchoring strength),
    tunable INDEPENDENTLY of any equilibration so protection can be matched to Benna-Fusi's C-FTSR. KEY CONTRAST vs
    BennaFusi: the reference is FROZEN and does NOT track the live weight, so the circuit is pinned toward its
    engage-time state and cannot keep improving (the predicted 137-style late-degradation), whereas BF's
    bidirectional multi-timescale chain lets the protected circuit keep improving. Exposes .diffuse() so it plugs
    into train_task_bf unchanged. Anti-homunculus (fixed local pull), fence-clean (no SVD)."""

    def __init__(self, params, lam, device):
        self.params = list(params)
        self.lam = lam
        self.ref = [p.detach().clone() for p in self.params]   # frozen snapshot at engage time

    @torch.no_grad()
    def diffuse(self):
        for p, r in zip(self.params, self.ref):
            p.add_(r - p, alpha=self.lam)                       # p -= lam*(p - r)


def train_task_bf(model, opt, bf, task_idx, tr, te, *, max_steps, crit, eval_every, buf, replay_frac, gen, device):
    """Per-task fast learning with a Benna-Fusi consolidation step after every optimizer step. Mirrors
    exp83.train_task (interleaved replay identical) + bf.diffuse(). MLP is NEVER frozen (BF protects gradedly)."""
    a, b, y = to_tensors(tr, device)
    bt = [(*to_tensors(rows, device), t) for (rows, t) in buf] if (buf and replay_frac > 0) else []
    rep = max(1, int(replay_frac * len(tr) / max(1, len(bt)))) if bt else 0
    steps = max_steps
    for step in range(1, max_steps + 1):
        loss = F.cross_entropy(model(a, b, task_idx), y)
        for (ba, bb, by, t) in bt:
            idx = torch.randint(ba.shape[0], (min(ba.shape[0], rep),), generator=gen).to(device)
            loss = loss + F.cross_entropy(model(ba[idx], bb[idx], t), by[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        bf.diffuse()                                      # <-- Benna-Fusi consolidation (graded protection)
        if step % eval_every == 0 and evaluate(model, task_idx, te, device) >= crit:
            steps = step; break
    return steps


def consolidate_restructure(model, opt, buf, cfg, gen, device):
    """OFFLINE-PASS restructuring slot, retained for FUTURE recipes (GERM / CANON fallbacks). Benna-Fusi does
    NOT use this (it is a during-learning weight dynamic). No-op when consol_mode != an offline recipe."""
    return


DURING_LEARNING_MODES = ("bennafusi", "ewc")   # consolidation recipes that run as a per-step weight dynamic


def run_arm(arm, p, K, frac, *, embed, hidden, max_steps, crit, eval_every, replay_frac, lr, weight_decay,
            consol_mode, bf_m, bf_g, ewc_lambda, bf_start_task, consol_cfg, seed, device):
    g = torch.Generator().manual_seed(seed * 104729 + 7)
    stream, bases, vocab = build_stream(p, K, frac, g)
    steps = [None] * K
    ret_after = [None] * K

    if arm == "scratch":
        for k in range(K):
            m = ContinualNet(vocab, p, K, embed, hidden, seed).to(device)
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
            steps[k] = train_task(m, opt, k, stream[k][0], stream[k][1], freeze_mlp=False, max_steps=max_steps,
                                  crit=crit, eval_every=eval_every, buf=[], replay_frac=0.0, gen=g, device=device)
        return {"steps": steps, "ret_after": None}

    do_replay = arm in ("replay_only", "replay_plus_consol")
    do_consol = arm in ("consol_only", "replay_plus_consol")
    use_consol_recipe = do_consol and consol_mode in DURING_LEARNING_MODES
    m = ContinualNet(vocab, p, K, embed, hidden, seed).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
    # The consolidation recipe attaches to the SHARED MLP only (embeddings + heads stay fast) and ENGAGES at task
    # bf_start_task (default 1): the bootstrap forms the circuit with plain training, THEN the consolidator's
    # reference is initialized to the FORMED circuit (protect-a-circuit-that-exists; avoids over-protecting random
    # init). C-off arms use plain exp83.train_task throughout → reproduce exp84 byte-identically (the anchor).
    bf = None
    buf = []
    for k in range(K):
        if use_consol_recipe and k == bf_start_task:
            if consol_mode == "bennafusi":
                bf = BennaFusi(m.mlp.parameters(), bf_m, bf_g, device)        # multi-timescale chain := formed circuit
            else:  # "ewc" — frozen-reference L2 anchor (the attribution CONTROL)
                bf = EWCAnchor(m.mlp.parameters(), ewc_lambda, device)
        if do_consol and consol_mode not in DURING_LEARNING_MODES and buf:    # offline-pass recipes (future: GERM/CANON)
            consolidate_restructure(m, opt, buf, consol_cfg, g, device)
        rf = replay_frac if do_replay else 0.0
        if bf is not None:
            steps[k] = train_task_bf(m, opt, bf, k, stream[k][0], stream[k][1], max_steps=max_steps, crit=crit,
                                     eval_every=eval_every, buf=buf, replay_frac=rf, gen=g, device=device)
        else:
            steps[k] = train_task(m, opt, k, stream[k][0], stream[k][1], freeze_mlp=False, max_steps=max_steps,
                                  crit=crit, eval_every=eval_every, buf=buf, replay_frac=rf, gen=g, device=device)
        buf.append((stream[k][0], k))
        ret_after[k] = [evaluate(m, j, stream[j][1], device) for j in range(k + 1)]
    return {"steps": steps, "ret_after": ret_after}


def _state(ci, equiv):
    mean, lo, hi = ci
    if lo > 0:
        return "POSITIVE"
    if hi < 0:
        return "NEGATIVE"
    if lo >= -equiv and hi <= equiv:
        return "EQUIVALENT"
    return "INCONCLUSIVE"


def collect(args, seed_start, seeds):
    res = {a: [] for a in ARMS}
    cc = {"consol_steps": args.consol_steps}
    for seed in range(seed_start, seed_start + seeds):
        for a in ARMS:
            res[a].append(run_arm(a, args.p, args.K, args.frac, embed=args.embed, hidden=args.hidden,
                                  max_steps=args.max_steps, crit=args.crit, eval_every=args.eval_every,
                                  replay_frac=args.replay_frac, lr=args.lr, weight_decay=args.weight_decay,
                                  consol_mode=args.consol_mode, bf_m=args.bf_m, bf_g=args.bf_g,
                                  ewc_lambda=args.ewc_lambda, bf_start_task=args.bf_start_task,
                                  consol_cfg=cc, seed=seed, device=args.device))

        def xf(a):
            return sum(res["scratch"][-1]["steps"][k] / res[a][-1]["steps"][k] for k in XBLOCK) / len(XBLOCK)
        print(f"[seed {seed}] x-FTSR floor={xf('floor'):.2f} R={xf('replay_only'):.2f} "
              f"C={xf('consol_only'):.2f} RC={xf('replay_plus_consol'):.2f}", file=sys.stderr, flush=True)
    return res


def aggregate(res, p, K, cfg):
    seeds = len(res["floor"])
    equiv = cfg["equiv_log"]

    def raw(arm, ks):
        return [sum(res["scratch"][s]["steps"][k] / res[arm][s]["steps"][k] for k in ks) / len(ks) for s in range(seeds)]

    def lf(arm, ks):
        return [sum(math.log(res["scratch"][s]["steps"][k] / res[arm][s]["steps"][k]) for k in ks) / len(ks) for s in range(seeds)]
    xblock_raw = {a: boot_ci(raw(a, XBLOCK), seed=1) for a in BASE_ARMS}
    xblock_log = {a: boot_ci(lf(a, XBLOCK), seed=3) for a in BASE_ARMS}

    def paired(a, b, bseed):
        fa, fb = lf(a, XBLOCK), lf(b, XBLOCK)
        return boot_ci([fa[s] - fb[s] for s in range(seeds)], seed=bseed)
    d_RC_vs_R = paired("replay_plus_consol", "replay_only", 91)        # consolidation ADDS on top of replay
    d_RC_vs_C = paired("replay_plus_consol", "consol_only", 92)        # replay adds on top of consolidation
    d_C_vs_floor = paired("consol_only", "floor", 93)                  # consolidation alone vs floor
    d_R_vs_floor = paired("replay_only", "floor", 94)                  # replay alone vs floor (anchor ~exp84)
    lfRC, lfR, lfC, lfF = lf("replay_plus_consol", XBLOCK), lf("replay_only", XBLOCK), lf("consol_only", XBLOCK), lf("floor", XBLOCK)
    d_super = boot_ci([lfRC[s] + lfF[s] - lfR[s] - lfC[s] for s in range(seeds)], seed=90)  # superadditivity term

    def endret(a):
        return [sum(res[a][s]["ret_after"][K - 1]) / K for s in range(seeds)]
    ret = {a: boot_ci(endret(a), seed=RET_SEED[a]) for a in BASE_ARMS}
    ret_RC_minus_R = boot_ci([endret("replay_plus_consol")[s] - endret("replay_only")[s] for s in range(seeds)], seed=95)
    perk = {a: {f"k{k}": boot_ci([res["scratch"][s]["steps"][k] / res[a][s]["steps"][k] for s in range(seeds)], seed=k)
                for k in XBLOCK} for a in BASE_ARMS}

    def compounds(a):
        kk = sorted(XBLOCK)
        d = boot_ci([math.log(res["scratch"][s]["steps"][kk[-1]] / res[a][s]["steps"][kk[-1]]) -
                     math.log(res["scratch"][s]["steps"][kk[0]] / res[a][s]["steps"][kk[0]]) for s in range(seeds)], seed=80)
        return {"k_first": kk[0], "k_last": kk[-1], "last_minus_first_logFTSR": d, "compounds": bool(d[1] > 0)}
    compounding = {a: compounds(a) for a in ("replay_plus_consol", "replay_only", "consol_only")}

    beats_R = bool(d_RC_vs_R[1] > 0)
    beats_C = bool(d_RC_vs_C[1] > 0)
    rc_compounds = compounding["replay_plus_consol"]["compounds"]
    ret_held = bool(ret_RC_minus_R[1] > -0.05)
    INTERACTION_PASS = bool(beats_R and beats_C and rc_compounds and ret_held)

    verdict = (
        "exp85 replay×consolidation INTERACTION (n={n}, log-FTSR, EQUIV=±{eq}). "
        "Δ_RC_vs_R={a}[{al:+.2f},{ah:+.2f}]={sa}; Δ_RC_vs_C={b}[{bl:+.2f},{bh:+.2f}]={sb}; "
        "Δ_super={c}[{cl:+.2f},{ch:+.2f}]={sc}. RC compounds={rcc}; retention(RC−R)={rr:+.3f} held={rh}. "
        "Raw x-block FTSR: floor={xf:.2f} R={xr:.2f} C={xc:.2f} RC={xrc:.2f}. "
        "INTERACTION (RC beats BOTH replay-only AND consolidation-only, CI-disjoint) = {P} [ADDITIVE 'beats both parents' gate; NOT super-additivity — see d_super]. "
        "{recipe_note}"
    ).format(n=seeds, eq=equiv,
             a=round(d_RC_vs_R[0], 2), al=d_RC_vs_R[1], ah=d_RC_vs_R[2], sa=_state(d_RC_vs_R, equiv),
             b=round(d_RC_vs_C[0], 2), bl=d_RC_vs_C[1], bh=d_RC_vs_C[2], sb=_state(d_RC_vs_C, equiv),
             c=round(d_super[0], 2), cl=d_super[1], ch=d_super[2], sc=_state(d_super, equiv),
             rcc=rc_compounds, rr=ret_RC_minus_R[0], rh=ret_held,
             xf=xblock_raw["floor"][0], xr=xblock_raw["replay_only"][0], xc=xblock_raw["consol_only"][0],
             xrc=xblock_raw["replay_plus_consol"][0], P=INTERACTION_PASS,
             recipe_note=cfg.get("recipe_note", "RECIPE = <pending workflow wvxj66tdp; NO-OP placeholder>"))

    return {
        "experiment": "85_betb_replay_x_consolidation (Bet B — the thesis-critical interaction test)",
        "charter": "CONTEXT-B.md §5/§8 + Terminology",
        "config": {**cfg, "seeds": seeds, "xblock_tasks": XBLOCK, "wblock_tasks": WBLOCK},
        "HEADLINE_interaction": {"delta_RC_vs_R_LOG": d_RC_vs_R, "delta_RC_vs_C_LOG": d_RC_vs_C,
                                 "delta_superadditivity_LOG": d_super,
                                 "states": {"RC_vs_R": _state(d_RC_vs_R, equiv), "RC_vs_C": _state(d_RC_vs_C, equiv),
                                            "superadditivity": _state(d_super, equiv)},
                                 "INTERACTION_PASS": INTERACTION_PASS},
        "delta_C_vs_floor_LOG": d_C_vs_floor, "delta_R_vs_floor_LOG": d_R_vs_floor,
        "xblock_FTSR_raw": xblock_raw, "xblock_logFTSR": xblock_log,
        "compounding_survival_logscale": compounding,
        "end_retention": ret, "retention_RC_minus_R": ret_RC_minus_R,
        "per_newblock_FTSR_raw_vs_k": perk,
        "anchor_replayOnly_xblock_FTSR_should_reproduce_exp84_replayNoConsol": xblock_raw["replay_only"],
        "equiv_margin_log": equiv,
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=17)
    ap.add_argument("--K", type=int, default=10, dest="K")
    ap.add_argument("--frac", type=float, default=0.7)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=8000, dest="max_steps")
    ap.add_argument("--crit", type=float, default=0.90)
    ap.add_argument("--eval-every", type=int, default=100, dest="eval_every")
    ap.add_argument("--consol-steps", type=int, default=400, dest="consol_steps")   # for future offline recipes
    ap.add_argument("--consol-mode", default="bennafusi", dest="consol_mode")        # bennafusi | ewc (control)
    ap.add_argument("--bf-m", type=int, default=4, dest="bf_m")                      # Benna-Fusi chain length
    ap.add_argument("--bf-g", type=float, default=0.03, dest="bf_g")                 # Benna-Fusi coupling
    ap.add_argument("--ewc-lambda", type=float, default=0.01, dest="ewc_lambda")     # EWC-lite frozen-anchor strength
    ap.add_argument("--bf-start-task", type=int, default=1, dest="bf_start_task")    # engage BF after bootstrap
    ap.add_argument("--replay-frac", type=float, default=0.5, dest="replay_frac")
    ap.add_argument("--equiv-log", type=float, default=0.22, dest="equiv_log")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0, dest="weight_decay")
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--merge", nargs="*", default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    global XBLOCK, WBLOCK
    if args.smoke:
        args.seeds, args.K = 2, 8
    XBLOCK = [k for k in range(args.K) if k % 2 == 0 and k >= 2]
    WBLOCK = [k for k in range(args.K) if k % 2 == 1]
    cfg = {"p": args.p, "K": args.K, "frac": args.frac, "embed": args.embed, "hidden": args.hidden,
           "max_steps": args.max_steps, "crit": args.crit, "consol_steps": args.consol_steps,
           "replay_frac": args.replay_frac, "equiv_log": args.equiv_log, "lr": args.lr,
           "weight_decay": args.weight_decay, "consol_mode": args.consol_mode, "bf_m": args.bf_m,
           "bf_g": args.bf_g, "ewc_lambda": args.ewc_lambda, "bf_start_task": args.bf_start_task,
           "recipe_note": (f"RECIPE = Benna-Fusi (m={args.bf_m}, g={args.bf_g}, engages@task{args.bf_start_task})"
                           if args.consol_mode == "bennafusi" else
                           f"CONTROL = frozen-EWC anchor (lambda={args.ewc_lambda}, engages@task{args.bf_start_task})")}

    if args.merge:
        res = {a: [] for a in ARMS}
        for fp in args.merge:
            sh = json.loads(pathlib.Path(fp).read_text())
            for a in ARMS:
                res[a].extend(sh["per_seed_raw"][a])
        out = aggregate(res, args.p, args.K, cfg)
    else:
        res = collect(args, args.seed_start, args.seeds)
        out = aggregate(res, args.p, args.K, cfg)
        out["per_seed_raw"] = res
        out["config"]["seed_start"] = args.seed_start

    print(json.dumps(out, indent=2))
    print(f"\n=== {out['verdict']}", file=sys.stderr, flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
