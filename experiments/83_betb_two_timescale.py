"""experiments/83 — Bet B STAGE 2 (graduation attempt): does a two-timescale split MANUFACTURE transfer?

CHARTER: CONTEXT-B.md §5/§8. THE graduation question: does protecting the shared "+" circuit during fast
task-learning (and consolidating it gently offline) MANUFACTURE the cross-block forward transfer that a plain
replay buffer cannot — beating plain replay on new-alphabet learning speed, retention held?

WHY this mechanism (Report 136 diagnostic, n=8): a reusable "+" circuit EXISTS in the shared MLP (freeze it
onto a new alphabet -> 3.3x faster than scratch) but ORDINARY TRAINING DESTROYS it (same MLP trainable ~
scratch; freeze beats warm-start 2.91x). The missing transfer is a PROTECTION failure. => protect the circuit.

Two timescales (anti-homunculus: a FIXED schedule, no supervisor reads a metric):
  FAST  — when a new task arrives, FREEZE the MLP; train only embeddings(+head). New embeddings snap onto the
          existing clean circuit (the diagnostic's 3.3x) instead of dragging it off its "+".
  SLOW  — offline, UNFREEZE and consolidate the MLP over replayed tasks, so the circuit keeps IMPROVING
          (the §5 guard: a permanently-frozen circuit is not consolidation; freeze_forever tests that).

Arms (one model, K=10 add/sub-on-rotating-blocks stream; task 0 = bootstrap, MLP trainable, for all arms):
  scratch        fresh model per task (FTSR denominator)
  plain          all-trainable per-task learning + offline consolidation (= Report-135 raw_full; ~0 x-block transfer)
  protect        TWO-TIMESCALE: MLP frozen during per-task learning + offline consolidation (the mechanism)
  freeze_forever MLP frozen during per-task learning, NO offline consolidation (circuit can't improve — drill-down)

HEADLINE: cross-block FTSR on the NEW-alphabet tasks k in {2,4,6,8} (each a fresh alphabet after the circuit
has formed): mean FTSR(protect) - FTSR(plain), bootstrap CI over seeds (n>=8). GRADUATES if protect manufactures
transfer plain does not (delta CI-lo > 0 AND protect FTSR CI-lo > 1) AND end-stream retention(protect) >=
retention(plain) - 0.05. Drill-downs: within-block FTSR (k odd, sanity ~high for all); retention curves;
protect vs freeze_forever (does the slow consolidation add value beyond a static freeze?).
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

WD = 1.0


def make_task(op, p, base):
    return [(base + i, base + j, (i + j) % p if op == "add" else (i - j) % p) for i in range(p) for j in range(p)]


def split_task(rows, frac, gen):
    idx = torch.randperm(len(rows), generator=gen).tolist(); n = int(frac * len(rows))
    return [rows[k] for k in idx[:n]], [rows[k] for k in idx[n:]]


def to_tensors(rows, device):
    return (torch.tensor([r[0] for r in rows], device=device),
            torch.tensor([r[1] for r in rows], device=device),
            torch.tensor([r[2] for r in rows], device=device))


def build_stream(p, K, frac, gen):
    stream, bases = [], []
    for t in range(K):
        op = "add" if t % 2 == 0 else "sub"; base = (t // 2) * p
        bases.append(base); stream.append(split_task(make_task(op, p, base), frac, gen))
    vocab = ((K + 1) // 2) * p
    return stream, bases, vocab


class ContinualNet(nn.Module):
    def __init__(self, vocab, p, K, embed=64, hidden=256, seed=0):
        super().__init__(); torch.manual_seed(seed * 7919 + 1)
        self.emb = nn.Embedding(vocab, embed)
        self.mlp = nn.Sequential(nn.Linear(2 * embed, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(hidden, p) for _ in range(K)])

    def forward(self, a, b, t):
        return self.heads[t](self.mlp(torch.cat([self.emb(a), self.emb(b)], -1)))


def evaluate(model, t, te, device):
    a, b, y = to_tensors(te, device)
    with torch.no_grad():
        return (model(a, b, t).argmax(-1) == y).float().mean().item()


def consolidate(model, opt, buf, steps, device):
    """Offline SLOW pass: MLP TRAINABLE, train over all replayed tasks (full-batch). The circuit improves."""
    if not buf or steps <= 0:
        return
    for prm in model.mlp.parameters():
        prm.requires_grad_(True)
    data = [(to_tensors(rows, device), t) for (rows, t) in buf]
    for _ in range(steps):
        loss = 0.0
        for ((a, b, y), t) in data:
            loss = loss + F.cross_entropy(model(a, b, t), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


def train_task(model, opt, task_idx, tr, te, *, freeze_mlp, max_steps, crit, eval_every, buf, replay_frac,
               gen, device):
    """FAST per-task learning to crit. If freeze_mlp: MLP requires_grad=False (grads None -> AdamW skips it),
    so only embeddings+heads adapt onto the (frozen) circuit. Returns steps_to_crit."""
    for prm in model.mlp.parameters():
        prm.requires_grad_(not freeze_mlp)
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
        if step % eval_every == 0 and evaluate(model, task_idx, te, device) >= crit:
            steps = step; break
    for prm in model.mlp.parameters():
        prm.requires_grad_(True)
    return steps


def run_arm(arm, p, K, frac, *, embed, hidden, max_steps, crit, eval_every, consol_steps, replay_frac,
            lr, seed, device):
    g = torch.Generator().manual_seed(seed * 104729 + 7)
    stream, bases, vocab = build_stream(p, K, frac, g)
    steps = [None] * K
    ret_after = [None] * K

    if arm == "scratch":
        for k in range(K):
            m = ContinualNet(vocab, p, K, embed, hidden, seed).to(device)
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=WD)
            steps[k] = train_task(m, opt, k, stream[k][0], stream[k][1], freeze_mlp=False, max_steps=max_steps,
                                  crit=crit, eval_every=eval_every, buf=[], replay_frac=0.0, gen=g, device=device)
        return {"steps": steps, "ret_after": None}

    m = ContinualNet(vocab, p, K, embed, hidden, seed).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=WD)
    buf = []
    for k in range(K):
        # task 0 is the bootstrap: build the initial circuit (MLP trainable) for EVERY arm
        freeze = (arm in ("protect", "freeze_forever")) and k > 0
        if arm in ("plain", "protect") and buf:                  # offline SLOW consolidation (improve the circuit)
            consolidate(m, opt, buf, consol_steps, device)
        steps[k] = train_task(m, opt, k, stream[k][0], stream[k][1], freeze_mlp=freeze, max_steps=max_steps,
                              crit=crit, eval_every=eval_every, buf=buf, replay_frac=replay_frac, gen=g, device=device)
        buf.append((stream[k][0], k))                            # store real examples (raw_full-style retention)
        ret_after[k] = [evaluate(m, j, stream[j][1], device) for j in range(k + 1)]
    return {"steps": steps, "ret_after": ret_after}


def boot_ci(vals, n=4000, seed=0):
    t = torch.tensor([v for v in vals if v == v], dtype=torch.float64)
    if t.numel() == 0:
        return (float("nan"),) * 3
    gg = torch.Generator().manual_seed(seed)
    idx = torch.randint(t.numel(), (n, t.numel()), generator=gg)
    means = t[idx].mean(1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return (float(t.mean()), lo, hi)


ARMS = ["scratch", "plain", "protect", "freeze_forever"]
XBLOCK = None   # set per K: even tasks >= 2 (new alphabets after the circuit forms)
WBLOCK = None   # odd tasks (within-block, sub-after-add)


def collect(args, seed_start, seeds):
    res = {a: [] for a in ARMS}
    for seed in range(seed_start, seed_start + seeds):
        for a in ARMS:
            r = run_arm(a, args.p, args.K, args.frac, embed=args.embed, hidden=args.hidden, max_steps=args.max_steps,
                        crit=args.crit, eval_every=args.eval_every, consol_steps=args.consol_steps,
                        replay_frac=args.replay_frac, lr=args.lr, seed=seed, device=args.device)
            res[a].append(r)
        def mret(a):
            ra = res[a][-1]["ret_after"]; return sum(ra[args.K - 1]) / len(ra[args.K - 1]) if ra else float("nan")
        def xf(a):
            return sum(res["scratch"][-1]["steps"][k] / res[a][-1]["steps"][k] for k in XBLOCK) / len(XBLOCK)
        print(f"[seed {seed}] x-block FTSR plain={xf('plain'):.2f} protect={xf('protect'):.2f} "
              f"frz={xf('freeze_forever'):.2f} | end-ret plain={mret('plain'):.3f} protect={mret('protect'):.3f}",
              file=sys.stderr, flush=True)
    return res


def aggregate(res, p, K, cfg):
    seeds = len(res["plain"])

    def ftsr(arm, ks):                                            # per-seed mean FTSR over task-set ks
        return [sum(res["scratch"][s]["steps"][k] / res[arm][s]["steps"][k] for k in ks) / len(ks)
                for s in range(seeds)]
    xblock = {a: boot_ci(ftsr(a, XBLOCK), seed=1) for a in ("plain", "protect", "freeze_forever")}
    wblock = {a: boot_ci(ftsr(a, WBLOCK), seed=2) for a in ("plain", "protect", "freeze_forever")}
    headline_delta = boot_ci([ftsr("protect", XBLOCK)[s] - ftsr("plain", XBLOCK)[s] for s in range(seeds)], seed=99)
    protect_vs_frz = boot_ci([ftsr("protect", XBLOCK)[s] - ftsr("freeze_forever", XBLOCK)[s] for s in range(seeds)], seed=98)

    def endret(a):
        return [sum(res[a][s]["ret_after"][K - 1]) / K for s in range(seeds)]
    ret = {a: boot_ci(endret(a), seed=hash(a) % 999) for a in ("plain", "protect", "freeze_forever")}
    ret_delta = boot_ci([endret("protect")[s] - endret("plain")[s] for s in range(seeds)], seed=97)
    ret_curve = {a: [boot_ci([sum(res[a][s]["ret_after"][k]) / (k + 1) for s in range(seeds)], seed=k) for k in range(K)]
                 for a in ("plain", "protect", "freeze_forever")}
    # per-new-block FTSR vs k (does transfer GROW as the circuit consolidates more tasks? = compounding)
    perk = {a: {f"k{k}": boot_ci([res["scratch"][s]["steps"][k] / res[a][s]["steps"][k] for s in range(seeds)], seed=k)
                for k in XBLOCK} for a in ("plain", "protect")}

    manufactures = bool(headline_delta[1] > 0 and xblock["protect"][1] > 1.0)
    retains = bool(ret_delta[1] > -0.05)
    consol_adds = bool(protect_vs_frz[1] > 0)                    # protect beats a static freeze (consolidation helps)
    PASS = bool(manufactures and retains)
    if PASS:
        verdict = ("GRADUATES — the two-timescale split MANUFACTURES cross-block transfer a plain replay buffer "
                   "cannot: protect x-block FTSR {:.2f} vs plain {:.2f} (delta CI-lo {:+.2f}>0, protect CI-lo {:.2f}>1), "
                   "retention held (delta {:+.3f}). Consolidation-adds-value(vs static freeze)={}.").format(
                       xblock["protect"][0], xblock["plain"][0], headline_delta[1], xblock["protect"][1],
                       ret_delta[0], consol_adds)
    else:
        verdict = ("NO GRADUATION — manufactures={} (protect x-block FTSR {:.2f} [{:.2f},] vs plain {:.2f}; delta "
                   "{:+.2f} [{:+.2f},{:+.2f}]), retains={} (delta {:+.3f}). Per the charter: iterate the recipe, "
                   "not a dead end.").format(manufactures, xblock["protect"][0], xblock["protect"][1],
                       xblock["plain"][0], headline_delta[0], headline_delta[1], headline_delta[2], retains, ret_delta[0])

    return {
        "experiment": "83_betb_two_timescale (Bet B STAGE 2 graduation attempt)",
        "charter": "CONTEXT-B.md §5/§8",
        "config": {**cfg, "seeds": seeds, "xblock_tasks": XBLOCK, "wblock_tasks": WBLOCK},
        "HEADLINE_xblock_FTSR_protect_minus_plain": headline_delta,
        "xblock_FTSR": xblock, "within_block_FTSR_sanity": wblock,
        "protect_minus_freezeforever_xblock": protect_vs_frz,
        "end_retention": ret, "retention_protect_minus_plain": ret_delta,
        "retention_curve_meanover_0..k": ret_curve,
        "per_newblock_FTSR_vs_k": perk,
        "gates": {"manufactures_transfer": manufactures, "retains_within_0.05": retains,
                  "consolidation_adds_value": consol_adds},
        "PASS": PASS, "verdict": verdict,
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
    ap.add_argument("--replay-frac", type=float, default=0.5, dest="replay_frac")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0, dest="weight_decay")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--merge", nargs="*", default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    global WD, XBLOCK, WBLOCK
    WD = args.weight_decay
    if args.smoke:
        args.seeds, args.K = 2, 8
    XBLOCK = [k for k in range(args.K) if k % 2 == 0 and k >= 2]   # new-alphabet tasks (cross-block transfer)
    WBLOCK = [k for k in range(args.K) if k % 2 == 1]              # sub-after-add (within-block, sanity)
    cfg = {"p": args.p, "K": args.K, "frac": args.frac, "embed": args.embed, "hidden": args.hidden,
           "max_steps": args.max_steps, "crit": args.crit, "consol_steps": args.consol_steps,
           "replay_frac": args.replay_frac}

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
    x = out["xblock_FTSR"]; h = out["HEADLINE_xblock_FTSR_protect_minus_plain"]; r = out["retention_protect_minus_plain"]
    print(f"=== x-block FTSR: plain={x['plain'][0]:.2f} protect={x['protect'][0]:.2f}[{x['protect'][1]:.2f},] "
          f"frz={out['xblock_FTSR']['freeze_forever'][0]:.2f} | delta(protect-plain)={h[0]:+.2f}[{h[1]:+.2f},{h[2]:+.2f}] "
          f"| ret protect-plain={r[0]:+.3f} | PASS={out['PASS']} | seeds={len(res['plain'])}", file=sys.stderr, flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
