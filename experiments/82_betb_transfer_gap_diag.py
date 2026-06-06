"""experiments/82 — Bet B DIAGNOSTIC: WHY do new-alphabet tasks get ~zero head start? (gap drill-down for Report 135)

CHARTER: CONTEXT-B.md §8 / §5. DIAGNOSTIC, not a graduation run. Report 135 found: over a K=10 add/sub stream,
transfer is purely LOCAL (sub-after-add ~10×; a NEW alphabet block FTSR≈1 = no head start). Stage 2 (a
restructuring/two-timescale consolidator) is supposed to manufacture that missing cross-block transfer — but
ONLY if a reusable shared structure can exist at all. This experiment finds the bottleneck BEFORE we build it.

THE PROBE — a frozen-MLP transfer ladder. The model = shared_embed(token) -> shared_MLP("+/- circuit") ->
per-task head. For a NEW alphabet, only the token->circle map is genuinely new; the MLP COULD transfer.
Learn a held-out new-alphabet add task with the MLP FROZEN from different sources, vs full scratch:
  scratch       all trainable (denominator)
  freeze_rand   MLP frozen at RANDOM init        — controls for the freezing penalty per se
  freeze_plain  MLP frozen from a PLAIN sequential run (no replay)
  freeze_full   MLP frozen from a raw_full sequential run (best-retained stream model)
  freeze_joint  MLP frozen from a JOINT-trained model (all blocks at once = best-case reusable circuit)
(embed for the new tokens + a fresh head are always trainable; only the MLP source/frozen-ness varies.)

HEADLINE (diagnostic): new-block transfer ratio T = steps(scratch) / steps(arm), n>=8, bootstrap CI.
The load-bearing contrast is freeze_joint vs freeze_rand (both frozen; differ ONLY in MLP content):
  - joint MLP speeds the new block (T_joint > T_rand, CI-disjoint)  => a reusable "+" circuit EXISTS.
  - sequential MLP does too (T_full ~ T_joint)                       => the stream KEEPS it (no Stage-2 gap).
  - sequential ~ rand but joint > rand                               => circuit exists, stream DESTROYS it
                                                                          => Stage 2 well-posed, headroom = the gap.
  - joint ~ rand                                                     => NO reusable circuit even in principle
                                                                          => bottleneck is irreducible token-learning;
                                                                             Stage 2 needs a different lever.
Drill-down: cross-block embedding-geometry ALIGNMENT (do per-block circles share a subspace?) explains the headline.
Anti-homunculus: pure measurement; no mechanism added.
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


# ----- tasks (same family as exp81: add/sub on disjoint blocks) -----
def make_task(op, p, base):
    return [(base + i, base + j, (i + j) % p if op == "add" else (i - j) % p) for i in range(p) for j in range(p)]


def split_task(rows, frac, gen):
    idx = torch.randperm(len(rows), generator=gen).tolist(); n = int(frac * len(rows))
    return [rows[k] for k in idx[:n]], [rows[k] for k in idx[n:]]


def to_tensors(rows, device):
    return (torch.tensor([r[0] for r in rows], device=device),
            torch.tensor([r[1] for r in rows], device=device),
            torch.tensor([r[2] for r in rows], device=device))


# ----- models: multi-head (stream/joint) and single-head (probe) share emb+mlp shape -----
class MultiHeadNet(nn.Module):
    def __init__(self, vocab, p, n_tasks, embed=64, hidden=256, seed=0):
        super().__init__(); torch.manual_seed(seed * 7919 + 1)
        self.emb = nn.Embedding(vocab, embed)
        self.mlp = nn.Sequential(nn.Linear(2 * embed, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(hidden, p) for _ in range(n_tasks)])

    def forward(self, a, b, t):
        return self.heads[t](self.mlp(torch.cat([self.emb(a), self.emb(b)], -1)))


class ProbeNet(nn.Module):
    def __init__(self, vocab, p, embed=64, hidden=256, seed=0):
        super().__init__(); torch.manual_seed(seed * 7919 + 1)
        self.emb = nn.Embedding(vocab, embed)
        self.mlp = nn.Sequential(nn.Linear(2 * embed, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, p)

    def forward(self, a, b):
        return self.head(self.mlp(torch.cat([self.emb(a), self.emb(b)], -1)))


def evaluate1(model, te, device):
    a, b, y = to_tensors(te, device)
    with torch.no_grad():
        return (model(a, b).argmax(-1) == y).float().mean().item()


def build_stream(p, K, frac, gen):
    stream, bases = [], []
    for t in range(K):
        op = "add" if t % 2 == 0 else "sub"; base = (t // 2) * p
        bases.append(base); stream.append(split_task(make_task(op, p, base), frac, gen))
    return stream, bases


# ----- stream trainers (produce a shared-MLP source) -----
def train_stream(arm, p, K, frac, gen, *, embed, hidden, max_steps, crit, eval_every, replay_frac, lr, seed, device):
    """Train a MultiHeadNet through the K-task stream. arm='plain'|'raw_full'. Returns the trained model."""
    stream, _ = build_stream(p, K, frac, gen)
    vocab = ((K + 1) // 2 + 1) * p                          # +1 block of room for the held-out probe block
    m = MultiHeadNet(vocab, p, K, embed, hidden, seed).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=WD)
    buf = []
    for k in range(K):
        a, b, y = to_tensors(stream[k][0], device)
        flat = [(r, t) for (rows, t) in buf for r in rows]
        bt = [(*to_tensors(rows, device), t) for (rows, t) in buf] if (buf and arm == "raw_full") else []
        rep = max(1, int(replay_frac * len(stream[k][0]) / max(1, len(bt)))) if bt else 0
        for step in range(1, max_steps + 1):
            loss = F.cross_entropy(m(a, b, k), y)
            for (ba, bb, by, t) in bt:
                idx = torch.randint(ba.shape[0], (min(ba.shape[0], rep),), generator=gen).to(device)
                loss = loss + F.cross_entropy(m(ba[idx], bb[idx], t), by[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step % eval_every == 0:
                te = stream[k][1]; aa, bb2, yy = to_tensors(te, device)
                with torch.no_grad():
                    if (m(aa, bb2, k).argmax(-1) == yy).float().mean().item() >= crit:
                        break
        if arm == "raw_full":
            buf.append((stream[k][0], k))
    return m


def train_joint(p, K, frac, gen, *, embed, hidden, max_steps, lr, seed, device):
    stream, _ = build_stream(p, K, frac, gen)
    vocab = ((K + 1) // 2 + 1) * p
    m = MultiHeadNet(vocab, p, K, embed, hidden, seed).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=WD)
    data = [to_tensors(stream[k][0], device) for k in range(K)]
    for _ in range(max_steps):
        loss = 0.0
        for k in range(K):
            a, b, y = data[k]; loss = loss + F.cross_entropy(m(a, b, k), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return m


# ----- the probe: learn a held-out new-alphabet add task, MLP from `source` (frozen unless source is None) -----
def probe_newblock(mlp_state, p, base, frac, gen, *, embed, hidden, max_steps, crit, eval_every, lr, seed, device,
                   freeze):
    rows = make_task("add", p, base); tr, te = split_task(rows, frac, gen)
    vocab = base + p
    m = ProbeNet(vocab, p, embed, hidden, seed).to(device)
    if mlp_state is not None:
        m.mlp.load_state_dict(mlp_state)
    if freeze:
        for prm in m.mlp.parameters():
            prm.requires_grad_(False)
    params = [prm for prm in m.parameters() if prm.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WD)
    a, b, y = to_tensors(tr, device)
    steps = max_steps
    for step in range(1, max_steps + 1):
        loss = F.cross_entropy(m(a, b), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % eval_every == 0 and evaluate1(m, te, device) >= crit:
            steps = step; break
    return steps


def emb_alignment(model, p, K, device):
    """Drill-down: do per-block circles share a 2D subspace? Mean pairwise principal-subspace overlap across blocks.
    For each block, take its p token embeddings, center, top-2 right singular vectors; overlap = ||V_i^T V_j||_F^2 / 2
    (1 = identical plane, 0 = orthogonal). Returns mean over block pairs."""
    nb = (K + 1) // 2
    subs = []
    W = model.emb.weight.detach()
    for blk in range(nb):
        E = W[blk * p:(blk + 1) * p].to(torch.float32)
        E = E - E.mean(0, keepdim=True)
        V = torch.linalg.svd(E, full_matrices=False).Vh[:2]      # (2, embed)
        subs.append(V)
    if len(subs) < 2:
        return float("nan")
    tot, cnt = 0.0, 0
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            tot += (subs[i] @ subs[j].T).pow(2).sum().item() / 2.0; cnt += 1
    return tot / cnt


def boot_ci(vals, n=4000, seed=0):
    t = torch.tensor([v for v in vals if v == v], dtype=torch.float64)
    if t.numel() == 0:
        return (float("nan"),) * 3
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(t.numel(), (n, t.numel()), generator=g)
    means = t[idx].mean(1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return (float(t.mean()), lo, hi)


def run_seed(seed, args):
    dev = args.device
    g = torch.Generator().manual_seed(seed * 104729 + 7)
    K = args.K
    nb = (K + 1) // 2
    probe_base = nb * args.p                                  # held-out NEW alphabet block (never in the stream)
    kw_stream = dict(embed=args.embed, hidden=args.hidden, max_steps=args.max_steps, crit=args.crit,
                     eval_every=args.eval_every, lr=args.lr, seed=seed, device=dev)

    m_plain = train_stream("plain", args.p, K, args.frac, g, replay_frac=args.replay_frac, **kw_stream)
    m_full = train_stream("raw_full", args.p, K, args.frac, g, replay_frac=args.replay_frac, **kw_stream)
    m_joint = train_joint(args.p, K, args.frac, g, embed=args.embed, hidden=args.hidden,
                          max_steps=args.joint_steps, lr=args.lr, seed=seed, device=dev)   # generous budget = fair best-case

    mlp = lambda mm: {k: v.clone() for k, v in mm.mlp.state_dict().items()}
    kw_probe = dict(embed=args.embed, hidden=args.hidden, max_steps=args.max_steps, crit=args.crit,
                    eval_every=args.eval_every, lr=args.lr, seed=seed, device=dev)
    rand_src = MultiHeadNet(probe_base + args.p, args.p, 1, args.embed, args.hidden, seed + 13).to(dev)  # fresh random MLP
    steps = {
        "scratch":         probe_newblock(None,          args.p, probe_base, args.frac, g, freeze=False, **kw_probe),
        "freeze_rand":     probe_newblock(mlp(rand_src), args.p, probe_base, args.frac, g, freeze=True,  **kw_probe),
        "freeze_plain":    probe_newblock(mlp(m_plain),  args.p, probe_base, args.frac, g, freeze=True,  **kw_probe),
        "freeze_full":     probe_newblock(mlp(m_full),   args.p, probe_base, args.frac, g, freeze=True,  **kw_probe),
        "freeze_joint":    probe_newblock(mlp(m_joint),  args.p, probe_base, args.frac, g, freeze=True,  **kw_probe),
        # WARM-START: same MLP source, but TRAINABLE (replicates the actual stream's new-block dynamics).
        # frozen >> warmstart ~ scratch  =>  the circuit exists but ordinary training destroys it (=> protect it).
        "warmstart_full":  probe_newblock(mlp(m_full),   args.p, probe_base, args.frac, g, freeze=False, **kw_probe),
        "warmstart_joint": probe_newblock(mlp(m_joint),  args.p, probe_base, args.frac, g, freeze=False, **kw_probe),
    }
    align = {"plain": emb_alignment(m_plain, args.p, K, dev), "full": emb_alignment(m_full, args.p, K, dev),
             "joint": emb_alignment(m_joint, args.p, K, dev)}
    return {"steps": steps, "align": align}


def aggregate(res, cfg):
    seeds = len(res)
    arms = ["scratch", "freeze_rand", "freeze_plain", "freeze_full", "freeze_joint", "warmstart_full", "warmstart_joint"]

    def ratio(num, den, sd):
        return boot_ci([res[s]["steps"][num] / res[s]["steps"][den] if res[s]["steps"][den] else float("nan")
                        for s in range(seeds)], seed=sd)
    # transfer ratio T = scratch / arm (per seed); >1 = faster than learning from scratch
    T = {a: ratio("scratch", a, hash(a) % 9999) for a in arms}
    # MLP-content benefit: freeze_rand / arm (both frozen; >1 => the trained MLP carries reusable structure)
    content = {a: ratio("freeze_rand", a, hash("c" + a) % 9999) for a in ("freeze_plain", "freeze_full", "freeze_joint")}
    # PROTECTION benefit: warmstart / freeze (same MLP source; >1 => freezing beats letting it adapt => protect it)
    protect = {src: ratio("warmstart_" + src, "freeze_" + src, hash("p" + src) % 9999) for src in ("full", "joint")}
    align = {src: boot_ci([res[s]["align"][src] for s in range(seeds)], seed=hash(src) % 9999)
             for src in ("plain", "full", "joint")}
    steps_ci = {a: boot_ci([float(res[s]["steps"][a]) for s in range(seeds)], seed=hash("s" + a) % 9999) for a in arms}

    reusable = bool(content["freeze_full"][1] > 1.0 or content["freeze_joint"][1] > 1.0)  # a frozen good MLP transfers
    warmstart_loses = bool(T["warmstart_full"][2] < 1.30)         # warm-start gives ~no transfer (CI-hi below ~1.3)
    protect_helps = bool(protect["full"][1] > 1.0)                # freezing strictly beats warm-start (CI-lo>1)
    if reusable and protect_helps:
        verdict = ("STAGE-2 TARGET CONFIRMED — a reusable '+' circuit EXISTS (frozen full MLP transfers "
                   "{:.2f}x vs random) and ORDINARY TRAINING DESTROYS IT (freeze beats warm-start {:.2f}x; warm-start "
                   "T={:.2f} ~ scratch). The new-block FTSR~1 in Report 135 is a PROTECTION failure, not a missing "
                   "circuit. => Stage 2 = a two-timescale split: protect/slow the shared circuit, fast-adapt only the "
                   "new embeddings. Measurable headroom = freeze ({:.2f}x) over the stream's unfrozen ~1x.").format(
                       content["freeze_full"][0], protect["full"][0], T["warmstart_full"][0], T["freeze_full"][0])
    elif reusable and not protect_helps:
        verdict = ("NUANCE — a reusable circuit exists (frozen full {:.2f}x) but warm-start does NOT clearly lose it "
                   "(protection {:.2f}x, CI includes 1). Re-examine before committing to 'protect the circuit'.").format(
                       content["freeze_full"][0], protect["full"][0])
    else:
        verdict = ("CAUSE 2 — NO reusable '+' circuit even from a frozen trained MLP (full content {:.2f}x, joint "
                   "{:.2f}x ~ random). Bottleneck is the irreducible token->circle learning; a shared MLP can't "
                   "transfer across disjoint alphabets here. Stage 2 needs a different lever (shared embedding "
                   "geometry / circle prior), not circuit protection.").format(
                       content["freeze_full"][0], content["freeze_joint"][0])

    return {
        "experiment": "82_betb_transfer_gap_diag (DIAGNOSTIC for Report 135)",
        "charter": "CONTEXT-B.md §8/§5 (diagnostic, not graduation)",
        "config": {**cfg, "seeds": seeds},
        "steps_to_crit": steps_ci,
        "transfer_ratio_T_scratch_over_arm": T,
        "mlp_content_benefit_rand_over_arm": content,   # >1 (CI-lo) = frozen trained MLP carries reusable structure
        "protection_benefit_warmstart_over_freeze": protect,  # >1 (CI-lo) = freezing beats adapting => protect it
        "cross_block_embedding_alignment": align,        # 1 = blocks share a plane; ~0 = independent circles
        "gates": {"reusable_circuit_exists": reusable, "warmstart_loses_it": warmstart_loses,
                  "protection_helps": protect_helps},
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=17)
    ap.add_argument("--K", type=int, default=10, dest="K")
    ap.add_argument("--frac", type=float, default=0.7)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=6000, dest="max_steps")
    ap.add_argument("--joint-steps", type=int, default=12000, dest="joint_steps")  # generous: 10-task joint needs more
    ap.add_argument("--crit", type=float, default=0.90)
    ap.add_argument("--eval-every", type=int, default=100, dest="eval_every")
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
    global WD
    WD = args.weight_decay
    if args.smoke:
        args.seeds, args.K, args.max_steps, args.joint_steps = 2, 6, 6000, 8000
    cfg = {"p": args.p, "K": args.K, "frac": args.frac, "embed": args.embed, "hidden": args.hidden,
           "max_steps": args.max_steps, "crit": args.crit, "lr": args.lr}

    if args.merge:
        res = []
        for fp in args.merge:
            res.extend(json.loads(pathlib.Path(fp).read_text())["per_seed_raw"])
        out = aggregate(res, cfg)
    else:
        res = []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            res.append(run_seed(seed, args))
            st = res[-1]["steps"]
            print(f"[seed {seed}] scratch={st['scratch']} rand={st['freeze_rand']} fz_full={st['freeze_full']} "
                  f"fz_joint={st['freeze_joint']} ws_full={st['warmstart_full']} ws_joint={st['warmstart_joint']}",
                  file=sys.stderr, flush=True)
        out = aggregate(res, cfg)
        out["per_seed_raw"] = res
        out["config"]["seed_start"] = args.seed_start

    print(json.dumps(out, indent=2))
    print(f"\n=== {out['verdict']}", file=sys.stderr, flush=True)
    c = out["mlp_content_benefit_rand_over_arm"]; T = out["transfer_ratio_T_scratch_over_arm"]; pr = out["protection_benefit_warmstart_over_freeze"]
    print(f"=== reusable(rand/frozen,>1): full={c['freeze_full'][0]:.2f}[{c['freeze_full'][1]:.2f},] "
          f"joint={c['freeze_joint'][0]:.2f}[{c['freeze_joint'][1]:.2f},] | T_freeze_full={T['freeze_full'][0]:.2f} "
          f"T_warmstart_full={T['warmstart_full'][0]:.2f} | protect(ws/fz,>1): full={pr['full'][0]:.2f}[{pr['full'][1]:.2f},]"
          f" | align full={out['cross_block_embedding_alignment']['full'][0]:.2f}", file=sys.stderr, flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
