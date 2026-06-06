"""experiments/81 — Bet B DRILL-DOWN: does self-generated "pseudo-sleep" SURVIVE a long stream?

CHARTER: CONTEXT-B.md §8. This is a DRILL-DOWN, not a graduation run. The §8 graduation headline is
FTSR-beats-replay (compounding). Report 134 ADDENDUM found self-generated "pseudo-sleep" (the model
regenerates its own past from its weights, stores ZERO raw examples — pseudo-rehearsal, Robins 1995) beats
a STARVED raw-replay buffer at K=3 tasks (retention +0.45 [0.39,0.53]) — BUT flagged retention is only
PARTIAL (~0.48) because self-labels degrade as the model drifts.

THE QUESTION HERE: over a LONG stream (K tasks), does the pseudo-sleep advantage PERSIST/GROW, or does
generative-replay degradation (the model dreaming increasingly-wrong versions of old tasks — errors
compounding each round) COLLAPSE it? This gates whether plain pseudo-rehearsal is worth pushing vs.
needing the Stage-2 restructuring / two-timescale consolidator (CONTEXT-B §5).

Stream: K linear modular maps on a SHARED alphabet [0,p):  task t -> label = (a + c_t*b) mod p, with the
c_t distinct nonzero coefficients. Every task shares the number-circle (transfer is possible); each is a
genuinely distinct function (real forgetting). Task-IL (task id given at eval), per-task linear head,
shared embedding + MLP. The grokking recipe (AdamW wd=1.0, full-batch) makes each task GENERALIZE.

Arms:
  scratch    fresh model per task              — the FTSR denominator
  plain      sequential, NO replay             — lower bound (expect catastrophic forgetting)
  raw_tight  replay buffer cap=K_CAP/task      — the ADDENDUM baseline that BROKE
  raw_full   replay buffer UNCAPPED            — retention CEILING (stores everything)
  pseudo     self-generated rehearsal          — the lead brain mechanism (stores 0 raw)

Matched offline-consolidation compute + replay_frac across all replay arms; only the buffer SOURCE differs
(stored-real-capped vs stored-real-full vs dreamed) — isolates "what you rehearse." Multi-seed, bootstrap CIs.

Headline (this drill-down): retention-at-length delta  pseudo − raw_tight  after the FULL stream (mean acc
over all K tasks), bootstrap CI over seeds. Direction-positive = the ADDENDUM win survives length.
Drill-downs: retention-vs-k curve (degradation shape), steps-to-crit-vs-k (plasticity), FTSR_k-vs-k (the §8
compounding metric), and pseudo-vs-raw_full gap (how close dreaming gets to storing everything).
Anti-homunculus: pseudo = the model querying itself; no supervisor picks what to rehearse.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

WD = 1.0       # AdamW decoupled weight decay — the grokking recipe (held-out generalization, not memorization)
K_CAP = 2      # raw_tight stored examples per task (the ADDENDUM regime where raw replay broke)
PSEUDO_N = 128 # self-generated rehearsal examples per past task ('dreaming' budget)


# =====================================================================
# Tasks: K modular-arithmetic tasks = alternating add/sub on rotating DISJOINT alphabet blocks.
# Only add (c=1) and sub (c=16) grok within budget on this substrate (varied coefficients do NOT —
# verified); so the stream varies the ALPHABET, not the coefficient, to keep every task grokkable while
# preserving shared structure (within-block embedding + the shared +/- circuit) and real interference.
#   task t:  op = add if t even else sub;  block = t // 2;  base = block*p;
#            tokens = base+i, base+j (disjoint per block);  label = (i op j) mod p  in [0,p)
# This is the faithful K-fold generalization of exp80's T1=add@blk0 / T2=sub@blk0 / T3=add@blk1.
# =====================================================================

def make_task(op, p, base):
    rows = []
    for i in range(p):
        for j in range(p):
            lab = (i + j) % p if op == "add" else (i - j) % p
            rows.append((base + i, base + j, lab))
    return rows


def split_task(rows, frac, gen):
    idx = torch.randperm(len(rows), generator=gen).tolist()
    n = int(frac * len(rows))
    return [rows[k] for k in idx[:n]], [rows[k] for k in idx[n:]]


def to_tensors(rows, device):
    a = torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)
    b = torch.tensor([r[1] for r in rows], dtype=torch.long, device=device)
    y = torch.tensor([r[2] for r in rows], dtype=torch.long, device=device)
    return a, b, y


def build_stream(p, K, frac, gen):
    """K tasks: alternating add/sub on rotating disjoint blocks. Returns ([(tr,te) x K], bases, vocab)."""
    stream, bases = [], []
    for t in range(K):
        op = "add" if t % 2 == 0 else "sub"
        base = (t // 2) * p
        bases.append(base)
        stream.append(split_task(make_task(op, p, base), frac, gen))
    vocab = ((K + 1) // 2) * p                                        # number of blocks * p
    return stream, bases, vocab


# =====================================================================
# Model: shared embedding + shared MLP + per-task linear head (Task-IL)
# =====================================================================

class ContinualNet(nn.Module):
    def __init__(self, vocab, p, K, embed=64, hidden=256, seed=0):
        super().__init__()
        torch.manual_seed(seed * 7919 + 1)
        self.emb = nn.Embedding(vocab, embed)
        self.mlp = nn.Sequential(nn.Linear(2 * embed, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(hidden, p) for _ in range(K)])

    def forward(self, a, b, task):
        h = self.mlp(torch.cat([self.emb(a), self.emb(b)], dim=-1))
        return self.heads[task](h)


def evaluate(model, task_idx, te, device):
    a, b, y = to_tensors(te, device)
    with torch.no_grad():
        return (model(a, b, task_idx).argmax(-1) == y).float().mean().item()


def generate_pseudo(model, task_idx, p, base, n, gen, device):
    """Self-generated rehearsal: random (a,b) in the task's block, labeled by the model's OWN current
    prediction. Stores no raw data — the model 'dreams' its current knowledge of an old task."""
    ai = torch.randint(p, (n,), generator=gen); bi = torch.randint(p, (n,), generator=gen)
    a = (base + ai).to(device); b = (base + bi).to(device)
    with torch.no_grad():
        yhat = model(a, b, task_idx).argmax(-1)
    return list(zip((base + ai).tolist(), (base + bi).tolist(), yhat.cpu().tolist()))


def consolidate(model, opt, buf, steps, device):
    """Offline 'sleep' over the buffer (NO new-task data). FULL-batch, uniform (matched compute across arms)."""
    if not buf or steps <= 0:
        return
    data = [(to_tensors(rows, device), tidx) for (rows, tidx) in buf]
    for _ in range(steps):
        loss = 0.0
        for ((a, b, y), tidx) in data:
            loss = loss + F.cross_entropy(model(a, b, tidx), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


def train_task(model, opt, task_idx, tr, te, *, max_steps, crit, eval_every, buf, replay_frac, gen, device):
    """FULL-BATCH train to criterion (held-out acc>=crit) with interleaved replay. Returns steps_to_crit.
    Hot-loop optimized: buffer is pre-tensorized ONCE per task; replay subset is sampled on-device each step
    (CPU-generated indices -> device), so no per-step Python-list rebuild / host transfer / .tolist() sync."""
    a, b, y = to_tensors(tr, device)
    # pre-tensorize each buffered task once; per-step replay = a small random subset per task (balanced)
    buf_t = [(*to_tensors(rows, device), tidx) for (rows, tidx) in buf] if (buf and replay_frac > 0) else []
    rep_per_task = max(1, int(replay_frac * len(tr) / max(1, len(buf_t)))) if buf_t else 0
    steps_to_crit = max_steps
    for step in range(1, max_steps + 1):
        loss = F.cross_entropy(model(a, b, task_idx), y)
        for (ba, bb, by, tidx) in buf_t:
            n = ba.shape[0]
            m = min(n, rep_per_task)
            idx = torch.randint(n, (m,), generator=gen).to(device)
            loss = loss + F.cross_entropy(model(ba[idx], bb[idx], tidx), by[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % eval_every == 0 and evaluate(model, task_idx, te, device) >= crit:
            steps_to_crit = step
            break
    return steps_to_crit


# =====================================================================
# Arms
# =====================================================================

def run_arm(arm, p, K, frac, *, embed, hidden, max_steps, crit, eval_every, sleep_steps, replay_frac,
            lr, seed, device):
    g = torch.Generator().manual_seed(seed * 104729 + 7)
    stream, bases, vocab = build_stream(p, K, frac, g)
    steps = [None] * K
    # ret_after[k] = list over j<=k of held-out acc on task j, measured right after finishing task k
    ret_after = [None] * K

    if arm == "scratch":                                  # fresh model per task — the FTSR denominator
        for k in range(K):
            m = ContinualNet(vocab, p, K, embed, hidden, seed).to(device)
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=WD)
            steps[k] = train_task(m, opt, k, stream[k][0], stream[k][1], max_steps=max_steps, crit=crit,
                                  eval_every=eval_every, buf=[], replay_frac=0.0, gen=g, device=device)
        return {"steps": steps, "ret_after": None}

    m = ContinualNet(vocab, p, K, embed, hidden, seed).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=WD)
    use_replay = arm in ("raw_tight", "raw_full", "pseudo", "pseudo_snap")
    buf = []
    for k in range(K):
        if arm == "pseudo":                               # CHRONIC re-dream: regenerate the WHOLE past from the
            buf = [(generate_pseudo(m, j, p, bases[j], PSEUDO_N, g, device), j)  # CURRENT (drifted) weights each round
                   for j in range(k)]                     #  -> self-label rot compounds
        if use_replay and buf:
            consolidate(m, opt, buf, sleep_steps, device)
        steps[k] = train_task(m, opt, k, stream[k][0], stream[k][1], max_steps=max_steps, crit=crit,
                              eval_every=eval_every, buf=(buf if use_replay else []),
                              replay_frac=(replay_frac if use_replay else 0.0), gen=g, device=device)
        if arm in ("raw_tight", "raw_full"):              # store (capped or full) real examples
            store = stream[k][0]
            if arm == "raw_tight" and len(store) > K_CAP:
                sidx = torch.randperm(len(store), generator=g)[:K_CAP].tolist()
                store = [store[i] for i in sidx]
            buf.append((store, k))
        elif arm == "pseudo_snap":                        # SNAPSHOT: dream task k ONCE now (fresh, ~crit-correct),
            buf.append((generate_pseudo(m, k, p, bases[k], PSEUDO_N, g, device), k))  # freeze it, never re-dream
        ret_after[k] = [evaluate(m, j, stream[j][1], device) for j in range(k + 1)]
    return {"steps": steps, "ret_after": ret_after}


def boot_ci(vals, n=4000, seed=0):
    t = torch.tensor([v for v in vals if v == v], dtype=torch.float64)
    if t.numel() == 0:
        return (float("nan"), float("nan"), float("nan"))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(t.numel(), (n, t.numel()), generator=g)
    means = t[idx].mean(1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return (float(t.mean()), lo, hi)


ARMS = ["scratch", "plain", "raw_tight", "raw_full", "pseudo", "pseudo_snap"]


def collect(args, seed_start, seeds):
    """Run all arms for seeds [seed_start, seed_start+seeds). Returns res = {arm: [per-seed {steps,ret_after}]}."""
    K = args.K
    res = {arm: [] for arm in ARMS}
    for seed in range(seed_start, seed_start + seeds):
        for arm in ARMS:
            r = run_arm(arm, args.p, K, args.frac, embed=args.embed, hidden=args.hidden, max_steps=args.max_steps,
                        crit=args.crit, eval_every=args.eval_every, sleep_steps=args.sleep_steps,
                        replay_frac=args.replay_frac, lr=args.lr, seed=seed, device=args.device)
            res[arm].append(r)
        def mret(arm):
            ra = res[arm][-1]["ret_after"]
            return sum(ra[K - 1]) / len(ra[K - 1]) if ra else float("nan")
        print(f"[seed {seed}] end-mean-retention  plain={mret('plain'):.3f} raw_tight={mret('raw_tight'):.3f} "
              f"raw_full={mret('raw_full'):.3f} pseudo={mret('pseudo'):.3f}", file=sys.stderr, flush=True)
    return res


def aggregate(res, p, K, cfg):
    """Recompute every headline/drill-down from raw per-seed res (works for a single run OR merged shards)."""
    seeds = len(res["pseudo"])

    def mean_ret_at(arm, k):                               # one scalar per seed = mean acc over tasks 0..k
        return [sum(res[arm][s]["ret_after"][k]) / (k + 1) for s in range(seeds)]
    ret_curve = {arm: [boot_ci(mean_ret_at(arm, k), seed=k) for k in range(K)]
                 for arm in ("plain", "raw_tight", "raw_full", "pseudo", "pseudo_snap")}

    end_pseudo = mean_ret_at("pseudo", K - 1); end_tight = mean_ret_at("raw_tight", K - 1)
    end_full = mean_ret_at("raw_full", K - 1); end_snap = mean_ret_at("pseudo_snap", K - 1)
    headline_delta = boot_ci([end_pseudo[s] - end_tight[s] for s in range(seeds)], seed=99)
    pseudo_vs_full = boot_ci([end_pseudo[s] - end_full[s] for s in range(seeds)], seed=98)  # negative = below ceiling
    snap_vs_tight = boot_ci([end_snap[s] - end_tight[s] for s in range(seeds)], seed=97)
    snap_vs_pseudo = boot_ci([end_snap[s] - end_pseudo[s] for s in range(seeds)], seed=96)   # >0 = snapshot fixes rot
    snap_vs_full = boot_ci([end_snap[s] - end_full[s] for s in range(seeds)], seed=95)        # gap to ceiling

    t1_pseudo = {f"after_T{k+1}": boot_ci([res["pseudo"][s]["ret_after"][k][0] for s in range(seeds)], seed=k)
                 for k in range(K)}
    t1_snap = {f"after_T{k+1}": boot_ci([res["pseudo_snap"][s]["ret_after"][k][0] for s in range(seeds)], seed=k)
               for k in range(K)}
    t1_tight = {f"after_T{k+1}": boot_ci([res["raw_tight"][s]["ret_after"][k][0] for s in range(seeds)], seed=k)
                for k in range(K)}

    def steps_at(arm, k):
        return [res[arm][s]["steps"][k] for s in range(seeds)]
    plasticity = {arm: [boot_ci(steps_at(arm, k), seed=k) for k in range(K)] for arm in ("scratch", "pseudo", "raw_full")}

    def ftsr_curve(arm):
        return [boot_ci([res["scratch"][s]["steps"][k] / res[arm][s]["steps"][k] if res[arm][s]["steps"][k]
                         else float("nan") for s in range(seeds)], seed=k) for k in range(K)]
    ftsr = {arm: ftsr_curve(arm) for arm in ("pseudo", "raw_full")}

    survives = bool(headline_delta[1] > 0)
    near_ceiling = bool(pseudo_vs_full[1] > -0.10)
    verdict = (
        "PSEUDO-SLEEP SURVIVES THE LONG STREAM — its retention advantage over a starved raw buffer holds at "
        "K={} (delta CI-lo>0). The ADDENDUM positive is not a K=3 artifact." if survives else
        "PSEUDO-SLEEP DEGRADES — the K=3 advantage does NOT hold over a long stream (delta CI straddles/ <0): "
        "generative-replay self-label rot. HONEST NULL → motivates the Stage-2 restructuring/two-timescale "
        "consolidator (CONTEXT-B §5), which is the genuinely-new piece anyway.").format(K)

    out = {
        "experiment": "81_betb_longstream_pseudo (Bet B DRILL-DOWN on Report 134 ADDENDUM)",
        "charter": "CONTEXT-B.md §8 (drill-down, NOT a graduation run)",
        "config": {**cfg, "seeds": seeds},
        "HEADLINE_retention_delta_pseudo_minus_rawtight_end": headline_delta,
        "pseudo_minus_rawfull_end": pseudo_vs_full,
        "snapshot_minus_rawtight_end": snap_vs_tight,
        "snapshot_minus_pseudo_end": snap_vs_pseudo,   # >0 = freezing fresh dreams fixes the chronic-redream rot
        "snapshot_minus_rawfull_end": snap_vs_full,
        "end_mean_retention": {"plain": boot_ci(mean_ret_at("plain", K - 1), seed=1),
                               "raw_tight": boot_ci(end_tight, seed=2), "raw_full": boot_ci(end_full, seed=3),
                               "pseudo": boot_ci(end_pseudo, seed=4), "pseudo_snap": boot_ci(end_snap, seed=5)},
        "retention_curve_meanover_0..k": ret_curve,
        "oldest_task_T1_retention_vs_streampos": {"pseudo": t1_pseudo, "pseudo_snap": t1_snap, "raw_tight": t1_tight},
        "plasticity_steps_to_crit_vs_k": plasticity,
        "FTSR_vs_k": ftsr,
        "gates": {"pseudo_survives_long_stream": survives, "pseudo_near_storeverything_ceiling": near_ceiling},
        "verdict": verdict,
    }
    return out, (end_pseudo, end_tight, end_full, headline_delta, survives)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=17)
    ap.add_argument("--K", type=int, default=10, dest="K")
    ap.add_argument("--frac", type=float, default=0.7)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=6000, dest="max_steps")
    ap.add_argument("--crit", type=float, default=0.90)
    ap.add_argument("--eval-every", type=int, default=100, dest="eval_every")
    ap.add_argument("--sleep-steps", type=int, default=400, dest="sleep_steps")
    ap.add_argument("--replay-frac", type=float, default=0.5, dest="replay_frac")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0, dest="weight_decay")
    ap.add_argument("--k-cap", type=int, default=2, dest="k_cap")
    ap.add_argument("--pseudo-n", type=int, default=128, dest="pseudo_n")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--merge", nargs="*", default=None,
                    help="shard json files to merge (each has per_seed_raw); recompute aggregates over all seeds")
    args = ap.parse_args()
    global WD, K_CAP, PSEUDO_N
    WD = args.weight_decay; K_CAP = args.k_cap; PSEUDO_N = args.pseudo_n
    if args.smoke:
        args.seeds, args.K = 2, 5

    cfg = {"p": args.p, "K": args.K, "frac": args.frac, "embed": args.embed, "hidden": args.hidden,
           "max_steps": args.max_steps, "crit": args.crit, "sleep_steps": args.sleep_steps,
           "replay_frac": args.replay_frac, "k_cap": K_CAP, "pseudo_n": PSEUDO_N}

    if args.merge:                                          # combine shards: concat per-seed raw, re-aggregate
        res = {arm: [] for arm in ARMS}
        shard0 = None
        for fp in args.merge:
            sh = json.loads(pathlib.Path(fp).read_text())
            shard0 = shard0 or sh
            for arm in ARMS:
                res[arm].extend(sh["per_seed_raw"][arm])
        p, K = shard0["config"]["p"], shard0["config"]["K"]
        cfg = {k: shard0["config"][k] for k in cfg if k in shard0["config"]}
        out, (ep, et, ef, hd, surv) = aggregate(res, p, K, cfg)
    else:
        res = collect(args, args.seed_start, args.seeds)
        out, (ep, et, ef, hd, surv) = aggregate(res, args.p, args.K, cfg)
        out["per_seed_raw"] = res                           # so this run can be merged with sibling shards
        out["config"]["seed_start"] = args.seed_start

    print(json.dumps(out, indent=2))
    print(f"\n=== {out['verdict']}", file=sys.stderr, flush=True)
    print(f"=== end-retention  pseudo={boot_ci(ep)[0]:.3f}  raw_tight={boot_ci(et)[0]:.3f}  raw_full={boot_ci(ef)[0]:.3f}"
          f" | HEADLINE delta(pseudo-tight)={hd[0]:+.3f} [{hd[1]:+.3f},{hd[2]:+.3f}] | survives={surv} | "
          f"seeds={len(res['pseudo'])}", file=sys.stderr, flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
