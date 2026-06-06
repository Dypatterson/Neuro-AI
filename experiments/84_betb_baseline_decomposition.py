"""experiments/84 — Bet B DRILL-DOWN / CHARACTERIZATION (NOT a graduation): what is the BASELINE's compounding
transfer MADE OF?

CHARTER: CONTEXT-B.md §5/§8. This is the Report-137 NEXT (CONTEXT-B.md:330-335), reframed by the exp84
design-audit (ws9bfc18f) as a CHARACTERIZATION of the baseline — NOT a domain go/no-go. SCOPE (audit blocker #2):
the harness's "consolidation" (exp83.consolidate, full-batch CE over the buffer) is MORE REHEARSAL, NO
restructuring. So this experiment can only decompose the baseline's compounding into REHEARSAL ingredients; it
CANNOT speak to whether a RESTRUCTURING consolidator manufactures structure (that is the genuinely-next
experiment, a restructuring Stage-2 recipe swapped into this same harness — CONTEXT-B.md:161-170, 248-251). It
does NOT license abandoning the modular-arithmetic domain.

THE QUESTION. Report 137 found the on-target positive lives in the BASELINE: plain replay+consolidation
COMPOUNDS (new-alphabet x-block FTSR 1.6 -> 10.4 over the stream, 7/8 seeds). `plain` bundles THREE things:
(1) interleaved REPLAY during fast per-task learning, (2) an OFFLINE full-batch consolidation pass before each
task, and (3) the extra OPTIMIZER COMPUTE that offline pass spends (~consol_steps MLP-trainable steps/task, NOT
counted in the FTSR denominator — audit blocker #1's compute confound). Decompose them.

ARMS (all-trainable; one model; K add/sub-on-rotating-blocks stream; dynamics bit-identical to exp83 via import
so `plain` REPRODUCES 137 as an anchor — audit verified byte-identity empirically):
  scratch             fresh model per task                                     (FTSR denominator)
  plain               interleaved replay + FULL-BATCH offline consolidation    (= exp83 'plain'; 137 baseline; ANCHOR)
  replay_matched      interleaved replay + MINIBATCH offline rehearsal,        (COMPUTE-MATCHED to plain: same #
                      same optimizer-step budget as plain's offline pass        offline optimizer steps, minibatch)
  replay_no_consol    interleaved replay, NO offline pass                       (isolates the whole offline pass)
  no_replay_no_consol NO replay, NO offline pass (pure sequential)              (no-rehearsal floor; ~catastrophic)

ADDITIVE DECOMPOSITION of the baseline's x-block FTSR (new-alphabet tasks k in {2,4,6,8}), all PAIRED within-seed
bootstrap CIs over seeds (n>=24; default 32):
  plain - floor  =  Delta_replay  +  Delta_compute  +  Delta_structure
  Delta_replay    = FTSR(replay_no_consol) - FTSR(floor)            interleaved replay's own contribution
  Delta_compute   = FTSR(replay_matched)   - FTSR(replay_no_consol) effect of equal-compute extra rehearsal
  Delta_structure = FTSR(plain)            - FTSR(replay_matched)   PRIMARY: does the FULL-BATCH OFFLINE pass beat
                                                                    equal-compute minibatch rehearsal? (compute-controlled)
  Delta_consol    = FTSR(plain)            - FTSR(replay_no_consol) = Delta_compute + Delta_structure (offline pass total,
                                                                    compute-CONFOUNDED — reported as a drill-down only)

VERDICT (characterization; THREE-state per delta against a pre-registered equivalence margin EQUIV; audit fix):
  CI-lo > 0                      -> POSITIVE (this ingredient adds x-block transfer)
  CI within [-EQUIV, +EQUIV]     -> EQUIVALENT (genuinely inert at this resolution)
  straddles 0, |CI| > EQUIV      -> INCONCLUSIVE / under-powered (report MDE; add seeds; DO NOT conclude inert)
No PASS/GRADUATES token. No "domain cannot discriminate" claim. No "+ circuit linearly reachable" causal claim.
Drill-downs that can VETO an "inert" read of the offline pass (audit fix #5): does the 1.6->10.4 per-k
compounding SURVIVE in replay_no_consol / replay_matched? is replay_no_consol retention >= plain's?

ANTI-HOMUNCULUS: every arm is a FIXED schedule; replay/rehearsal is uniform/content-blind (no metric-reading
supervisor, no "if king/queen then boost"). Ablating/swapping a fixed offline dynamic is legal.
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

# Import exp83's leaf helpers so the learning dynamics are BIT-IDENTICAL (the `plain` anchor must reproduce 137).
_spec = importlib.util.spec_from_file_location("exp83", REPO / "experiments" / "83_betb_two_timescale.py")
exp83 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(exp83)
build_stream = exp83.build_stream
ContinualNet = exp83.ContinualNet
train_task = exp83.train_task
consolidate = exp83.consolidate          # FULL-BATCH offline rehearsal (used by `plain`; identical to 137)
evaluate = exp83.evaluate
to_tensors = exp83.to_tensors
boot_ci = exp83.boot_ci

ARMS = ["scratch", "plain", "replay_matched", "replay_no_consol", "no_replay_no_consol"]
BASE_ARMS = ["plain", "replay_matched", "replay_no_consol", "no_replay_no_consol"]
RET_SEED = {a: 10 + i for i, a in enumerate(BASE_ARMS)}   # fixed bootstrap seeds (audit: no PYTHONHASHSEED jitter)
XBLOCK = None   # new-alphabet tasks: even >= 2 (cross-block transfer)
WBLOCK = None   # odd tasks (within-block, sanity)


def consolidate_minibatch(model, opt, buf, steps, batch, gen, device):
    """Compute-matched offline rehearsal: SAME optimizer-step budget as exp83.consolidate (full-batch), but as
    MINIBATCH SGD over the union of buffered tasks. Isolates 'extra rehearsal COMPUTE / optimizer-state warmth'
    from 'the full-batch offline-pass STRUCTURE'. NO restructuring (still pure rehearsal)."""
    if not buf or steps <= 0:
        return
    for prm in model.mlp.parameters():
        prm.requires_grad_(True)
    A, B, Y, T = [], [], [], []
    for (rows, t) in buf:
        a, b, y = to_tensors(rows, device)
        A.append(a); B.append(b); Y.append(y); T.append(torch.full((len(rows),), t, device=device, dtype=torch.long))
    A, B, Y, T = torch.cat(A), torch.cat(B), torch.cat(Y), torch.cat(T)
    N = A.shape[0]
    for _ in range(steps):
        idx = torch.randint(N, (min(batch, N),), generator=gen).to(device)
        ta, tb, ty, tt = A[idx], B[idx], Y[idx], T[idx]
        loss = 0.0
        for t in torch.unique(tt).tolist():                  # group by head (each task has its own readout)
            mt = tt == t
            loss = loss + F.cross_entropy(model(ta[mt], tb[mt], t), ty[mt])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


def run_arm(arm, p, K, frac, *, embed, hidden, max_steps, crit, eval_every, consol_steps, consol_batch,
            replay_frac, lr, weight_decay, seed, device):
    """Thin driver over exp83's leaf helpers. The `plain` path is byte-identical to exp83.run_arm('plain', ...)
    (verified empirically by the design audit). Arms differ ONLY in the offline-pass kind and the replay frac."""
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

    m = ContinualNet(vocab, p, K, embed, hidden, seed).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
    buf = []
    offline = {"plain": "full", "replay_matched": "mini"}.get(arm, None)   # which offline pass, if any
    arm_replay = 0.0 if arm == "no_replay_no_consol" else replay_frac      # the floor gets NO interleaved replay
    for k in range(K):
        if offline and buf:
            if offline == "full":
                consolidate(m, opt, buf, consol_steps, device)            # exp83's full-batch pass (= 137)
            else:
                consolidate_minibatch(m, opt, buf, consol_steps, consol_batch, g, device)  # compute-matched
        steps[k] = train_task(m, opt, k, stream[k][0], stream[k][1], freeze_mlp=False, max_steps=max_steps,
                              crit=crit, eval_every=eval_every, buf=buf, replay_frac=arm_replay, gen=g, device=device)
        buf.append((stream[k][0], k))                                     # store real examples (raw_full retention)
        ret_after[k] = [evaluate(m, j, stream[j][1], device) for j in range(k + 1)]
    return {"steps": steps, "ret_after": ret_after}


def collect(args, seed_start, seeds):
    res = {a: [] for a in ARMS}
    for seed in range(seed_start, seed_start + seeds):
        for a in ARMS:
            r = run_arm(a, args.p, args.K, args.frac, embed=args.embed, hidden=args.hidden, max_steps=args.max_steps,
                        crit=args.crit, eval_every=args.eval_every, consol_steps=args.consol_steps,
                        consol_batch=args.consol_batch, replay_frac=args.replay_frac, lr=args.lr,
                        weight_decay=args.weight_decay, seed=seed, device=args.device)
            res[a].append(r)

        def xf(a):
            return sum(res["scratch"][-1]["steps"][k] / res[a][-1]["steps"][k] for k in XBLOCK) / len(XBLOCK)

        def mret(a):
            ra = res[a][-1]["ret_after"]; return sum(ra[args.K - 1]) / len(ra[args.K - 1]) if ra else float("nan")
        print(f"[seed {seed}] x-FTSR plain={xf('plain'):.2f} matched={xf('replay_matched'):.2f} "
              f"r_no_c={xf('replay_no_consol'):.2f} floor={xf('no_replay_no_consol'):.2f} | "
              f"end-ret plain={mret('plain'):.3f} r_no_c={mret('replay_no_consol'):.3f}",
              file=sys.stderr, flush=True)
    return res


def _state(ci, equiv):
    """Three-state classification of a paired delta CI against an equivalence margin."""
    mean, lo, hi = ci
    if lo > 0:
        return "POSITIVE"
    if hi < 0:
        return "NEGATIVE"
    if lo >= -equiv and hi <= equiv:
        return "EQUIVALENT"            # CI fully inside the margin -> genuinely inert at this resolution
    return "INCONCLUSIVE"              # straddles 0 and extends beyond margin -> under-powered


def aggregate(res, p, K, cfg):
    import math
    seeds = len(res["plain"])
    equiv = cfg["equiv_log"]   # equivalence margin in NATURAL-LOG FTSR units (e.g. 0.22 = within a 1.25x ratio)

    # FTSR is a heavy-tailed RATIO (an arm converging in one eval period -> a huge ratio). Speedups are inherently
    # MULTIPLICATIVE, so ALL inferential statistics are done on LOG-FTSR = mean_k log(scratch_steps/arm_steps).
    # Raw FTSR means are still reported for continuity with Report 137 (the 5.44 / 1.6->10.4 anchor).
    def raw(arm, ks):
        return [sum(res["scratch"][s]["steps"][k] / res[arm][s]["steps"][k] for k in ks) / len(ks)
                for s in range(seeds)]

    def lf(arm, ks):
        return [sum(math.log(res["scratch"][s]["steps"][k] / res[arm][s]["steps"][k]) for k in ks) / len(ks)
                for s in range(seeds)]
    xblock_raw = {a: boot_ci(raw(a, XBLOCK), seed=1) for a in BASE_ARMS}        # for 137 continuity / anchor
    xblock_log = {a: boot_ci(lf(a, XBLOCK), seed=3) for a in BASE_ARMS}
    wblock_raw = {a: boot_ci(raw(a, WBLOCK), seed=2) for a in BASE_ARMS}

    def paired(a, b, bseed):                                                    # paired within-seed LOG-FTSR delta
        fa, fb = lf(a, XBLOCK), lf(b, XBLOCK)
        return boot_ci([fa[s] - fb[s] for s in range(seeds)], seed=bseed)
    delta_replay = paired("replay_no_consol", "no_replay_no_consol", 96)        # interleaved replay's contribution
    delta_compute = paired("replay_matched", "replay_no_consol", 97)            # equal-compute extra rehearsal
    delta_structure = paired("plain", "replay_matched", 98)                     # PRIMARY (compute-controlled)
    delta_consol = paired("plain", "replay_no_consol", 99)                      # offline pass total (compute-CONFOUNDED)

    def endret(a):
        return [sum(res[a][s]["ret_after"][K - 1]) / K for s in range(seeds)]
    ret = {a: boot_ci(endret(a), seed=RET_SEED[a]) for a in BASE_ARMS}
    ret_replayNoConsol_minus_plain = boot_ci([endret("replay_no_consol")[s] - endret("plain")[s] for s in range(seeds)], seed=95)
    ret_curve = {a: [boot_ci([sum(res[a][s]["ret_after"][k]) / (k + 1) for s in range(seeds)], seed=k) for k in range(K)]
                 for a in BASE_ARMS}
    # per-new-block RAW FTSR vs k (readability): does the 137 plain 1.6->10.4 pattern SURVIVE without the offline pass?
    perk = {a: {f"k{k}": boot_ci([res["scratch"][s]["steps"][k] / res[a][s]["steps"][k] for s in range(seeds)], seed=k)
                for k in XBLOCK} for a in BASE_ARMS}

    def compounds(a):   # LOG-FTSR at last new-block > LOG-FTSR at first new-block (CI-disjoint) = the 137 signature
        kk = sorted(XBLOCK)
        first = [math.log(res["scratch"][s]["steps"][kk[0]] / res[a][s]["steps"][kk[0]]) for s in range(seeds)]
        last = [math.log(res["scratch"][s]["steps"][kk[-1]] / res[a][s]["steps"][kk[-1]]) for s in range(seeds)]
        d = boot_ci([last[s] - first[s] for s in range(seeds)], seed=80)
        return {"k_first": kk[0], "k_last": kk[-1], "last_minus_first_logFTSR": d, "compounds": bool(d[1] > 0)}
    compounding = {a: compounds(a) for a in ("plain", "replay_matched", "replay_no_consol")}

    states = {"delta_replay": _state(delta_replay, equiv), "delta_compute": _state(delta_compute, equiv),
              "delta_structure": _state(delta_structure, equiv), "delta_consol": _state(delta_consol, equiv)}
    # achieved MDE proxy: half the structure-delta CI width in LOG units (audit fix: report power, never silently null)
    mde_structure = (delta_structure[2] - delta_structure[1]) / 2.0
    survives = {a: compounding[a]["compounds"] for a in compounding}
    ret_held = bool(ret_replayNoConsol_minus_plain[1] > -0.05)

    def rat(ci):   # back-transform a log-delta CI to a geometric-mean RATIO for readability
        return (math.exp(ci[0]), math.exp(ci[1]), math.exp(ci[2]))

    verdict = (
        "CHARACTERIZATION (DRILL-DOWN, NOT a graduation; scope: REHEARSAL recipes only — no restructuring tested, "
        "so this does NOT license abandoning the domain). LOG-FTSR additive decomposition (n={n}, EQUIV=+/-{eq} log "
        "= within {er:.2f}x; deltas shown as log[CI] (=geomean ratio x{rr:.2f})): "
        "Delta_replay={dr:+.2f}[{drl:+.2f},{drh:+.2f}](x{rrr:.2f})={sr}; "
        "Delta_compute={dco:+.2f}[{dcol:+.2f},{dcoh:+.2f}](x{rco:.2f})={sco}; "
        "PRIMARY Delta_structure(full-batch offline vs equal-compute minibatch)={dst:+.2f}[{dstl:+.2f},{dsth:+.2f}]"
        "(x{rst:.2f})={sst} (MDE~{mde:.2f} log). Compounding (137 signature) survives: plain={cp}, "
        "replay_matched={cm}, replay_no_consol={cn}. Retention(replay_no_consol - plain)={rret:+.3f} (held={rh}). "
        "Raw x-block FTSR: plain={xp:.2f} matched={xm:.2f} r_no_c={xr:.2f} floor={xf:.2f}. "
        "NEXT (the real test): swap a RESTRUCTURING Stage-2 consolidator into this harness and re-measure "
        "Delta_structure against plain (CONTEXT-B.md:161-170)."
    ).format(n=seeds, eq=equiv, er=math.exp(equiv), rr=math.exp(delta_structure[0]),
             dr=delta_replay[0], drl=delta_replay[1], drh=delta_replay[2], rrr=rat(delta_replay)[0], sr=states["delta_replay"],
             dco=delta_compute[0], dcol=delta_compute[1], dcoh=delta_compute[2], rco=rat(delta_compute)[0], sco=states["delta_compute"],
             dst=delta_structure[0], dstl=delta_structure[1], dsth=delta_structure[2], rst=rat(delta_structure)[0], sst=states["delta_structure"],
             mde=mde_structure, cp=survives["plain"], cm=survives["replay_matched"], cn=survives["replay_no_consol"],
             rret=ret_replayNoConsol_minus_plain[0], rh=ret_held,
             xp=xblock_raw["plain"][0], xm=xblock_raw["replay_matched"][0], xr=xblock_raw["replay_no_consol"][0],
             xf=xblock_raw["no_replay_no_consol"][0])

    return {
        "experiment": "84_betb_baseline_decomposition (Bet B CHARACTERIZATION / DRILL-DOWN — NOT a graduation)",
        "charter": "CONTEXT-B.md §5/§8 (Report-137 NEXT); scoped per design-audit ws9bfc18f",
        "config": {**cfg, "seeds": seeds, "xblock_tasks": XBLOCK, "wblock_tasks": WBLOCK},
        "stats_scale": "LOG-FTSR (natural log of scratch_steps/arm_steps); raw FTSR reported for 137 continuity",
        "PRIMARY_delta_structure_plain_minus_replayMatched_LOG": delta_structure,
        "PRIMARY_delta_structure_as_geomean_ratio": rat(delta_structure),
        "delta_replay_replayNoConsol_minus_floor_LOG": delta_replay,
        "delta_compute_replayMatched_minus_replayNoConsol_LOG": delta_compute,
        "delta_consol_plain_minus_replayNoConsol_LOG_COMPUTE_CONFOUNDED": delta_consol,
        "states_vs_equiv_margin": states, "equiv_margin_log": equiv, "mde_structure_halfwidth_log": mde_structure,
        "xblock_FTSR_raw": xblock_raw, "xblock_logFTSR": xblock_log, "within_block_FTSR_raw_sanity": wblock_raw,
        "compounding_survival_logscale": compounding,
        "end_retention": ret, "retention_replayNoConsol_minus_plain": ret_replayNoConsol_minus_plain,
        "retention_curve_meanover_0..k": ret_curve,
        "per_newblock_FTSR_raw_vs_k": perk,
        "anchor_plain_xblock_FTSR_raw_should_reproduce_137": xblock_raw["plain"],
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
    ap.add_argument("--consol-steps", type=int, default=400, dest="consol_steps")
    ap.add_argument("--consol-batch", type=int, default=256, dest="consol_batch")
    ap.add_argument("--replay-frac", type=float, default=0.5, dest="replay_frac")
    ap.add_argument("--equiv-log", type=float, default=0.22, dest="equiv_log")   # log-FTSR margin; 0.22 ~ within 1.25x
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
    XBLOCK = [k for k in range(args.K) if k % 2 == 0 and k >= 2]   # new-alphabet tasks (cross-block transfer)
    WBLOCK = [k for k in range(args.K) if k % 2 == 1]              # within-block (sub-after-add), sanity
    cfg = {"p": args.p, "K": args.K, "frac": args.frac, "embed": args.embed, "hidden": args.hidden,
           "max_steps": args.max_steps, "crit": args.crit, "consol_steps": args.consol_steps,
           "consol_batch": args.consol_batch, "replay_frac": args.replay_frac, "equiv_log": args.equiv_log,
           "lr": args.lr, "weight_decay": args.weight_decay}

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
    x = out["xblock_FTSR_raw"]
    print(f"=== raw x-block FTSR: plain={x['plain'][0]:.2f} matched={x['replay_matched'][0]:.2f} "
          f"r_no_c={x['replay_no_consol'][0]:.2f} floor={x['no_replay_no_consol'][0]:.2f} | "
          f"states={out['states_vs_equiv_margin']} | seeds={len(res['plain'])}", file=sys.stderr, flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
