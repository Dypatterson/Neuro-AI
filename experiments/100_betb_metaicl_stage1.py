"""experiments/100 — Bet B / SCAN-MCD STAGE 1: reproduce the meta-in-context-learning compound-axis lift.

CHARTER: notes/betb-metaICL-consolidation-door-precommit.md. The walls sweep (wf_f8f9906e / _ac82a9e1 /
_e1196790) found ONE grown-not-injected, compound-axis-positive datapoint: meta-ICL (arXiv:2403.11834)
lifts SCAN-MCD by a TRAINING REGIME alone on a same-size from-scratch model. This is the cheap Stage-1
reproduction: does the regime give a CI-disjoint lift over the SAME-architecture vanilla baseline on
MCD2/MCD3 at OUR (reduced) scale? A null here = "didn't reproduce at this scale", not "regime fails".

PREAMBLE (CLAUDE.md):
  Active capability: Bet-B meta-ICL Stage-1 reproduction (the regime-permeability gate for the
    'grow-in-context -> consolidate-into-weights' door).
  Headline per notes/betb-metaICL-consolidation-door-precommit.md §5: Δ = EM(meta-ICL, k=M-1 support)
    - EM(vanilla, k=0), CI-disjoint > 0 on MCD2 AND MCD3, n>=5 seeds, bootstrap CI. Repro target (paper):
    MCD1 21.8->~60-71, MCD2 25.6->~53-75.
  Controls: vanilla = SAME causal-Transformer architecture (isolates the REGIME, not transformer-vs-GRU
    capacity); label-shuffle arm (its MCD3 regression is a fidelity check); k=0 collapse check (Stage-2 crux).
  Why now: the only compound-axis-positive grown lever in the whole walls sweep; load-bearing unknown =
    consolidation = the project's core thesis.

ANTI-HOMUNCULUS: the meta-ICL regime is a training-episode construction + a STANDARD causal-LM next-token
loss + an output-vocab permutation. No supervisor, no if-X-then-Y; "composition" is the loss-minimizing
solution SGD finds (attend to support). PASS (a learning objective, not an arbitration). See note §6.

MODEL: small from-scratch decoder-only causal Transformer (NOT the GRU — meta-ICL needs attention over the
support set). Reduced scale vs the paper's 8L/512d/25.2M; documented. Reuses exp87.load_pairs + mcd_split.
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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_s = importlib.util.spec_from_file_location("exp87", REPO / "experiments" / "87_betb_scan_gap.py")
exp87 = importlib.util.module_from_spec(_s); _s.loader.exec_module(exp87)
MCD = REPO / "data" / "scan" / "mcd_split"

# unified-vocab specials (own ids; independent of exp87's seq2seq specials)
PAD, BOS, IO, SEP, EOS = 0, 1, 2, 3, 4
SPECIALS = ["<pad>", "<bos>", "<io>", "<sep>", "<eos>"]


# ---- vocab ------------------------------------------------------------------------------------------------

def build_unified_vocab(pairs):
    """One vocab over command + action tokens. Returns (stoi, itos, out_id_range) where out_id_range is the
    [lo,hi) span of ACTION-token ids (the only ids label-shuffle permutes)."""
    in_toks = sorted({t for c, _ in pairs for t in c})
    out_toks = sorted({t for _, a in pairs for t in a})
    itos = SPECIALS + in_toks + out_toks
    stoi = {t: i for i, t in enumerate(itos)}
    lo = len(SPECIALS) + len(in_toks)
    hi = lo + len(out_toks)
    return stoi, itos, (lo, hi)


# ---- episode / sequence construction ----------------------------------------------------------------------

def build_sequence(pairs_subset, stoi, out_perm=None):
    """Concatenate [BOS] x1 <io> y1 <sep> ... xM <io> yM <eos>. Returns (tokens, is_output) lists.
    is_output[j]=True iff predicting tokens[j] should be SCORED (action tokens + their SEP/EOS terminator).
    out_perm: optional dict remapping action-token ids (label-shuffle), applied to all y in this episode."""
    toks = [BOS]; is_out = [False]
    n = len(pairs_subset)
    for i, (cmd, act) in enumerate(pairs_subset):
        for t in cmd:
            toks.append(stoi[t]); is_out.append(False)
        toks.append(IO); is_out.append(False)
        for t in act:
            tid = stoi[t]
            if out_perm is not None:
                tid = out_perm[tid]
            toks.append(tid); is_out.append(True)
        term = EOS if i == n - 1 else SEP
        toks.append(term); is_out.append(True)
    return toks, is_out


def sample_out_perm(out_range, gen):
    lo, hi = out_range
    ids = list(range(lo, hi))
    perm = torch.randperm(len(ids), generator=gen).tolist()
    return {ids[i]: ids[perm[i]] for i in range(len(ids))}


def make_episode_batch(pairs, M, batch_size, stoi, out_range, gen, *, shuffle_p=0.0, device):
    """Sample `batch_size` episodes of M random pairs each; build padded (tokens, loss_mask). loss_mask
    marks NEXT-token positions to score (causal LM): target seq[1:], scored where is_output[1:] is True."""
    seqs, masks = [], []
    npairs = len(pairs)
    for _ in range(batch_size):
        idx = torch.randint(0, npairs, (M,), generator=gen).tolist()
        subset = [pairs[i] for i in idx]
        perm = sample_out_perm(out_range, gen) if (shuffle_p > 0 and torch.rand(1, generator=gen).item() < shuffle_p) else None
        toks, is_out = build_sequence(subset, stoi, out_perm=perm)
        seqs.append(toks); masks.append(is_out)
    L = max(len(s) for s in seqs)
    tok = torch.full((batch_size, L), PAD, dtype=torch.long)
    lossm = torch.zeros((batch_size, L), dtype=torch.bool)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        tok[i, :len(s)] = torch.tensor(s)
        lossm[i, :len(m)] = torch.tensor(m)
    return tok.to(device), lossm.to(device)


# ---- model ------------------------------------------------------------------------------------------------

class CausalTransformer(nn.Module):
    def __init__(self, vocab, d_model, nhead, nlayers, dim_ff, max_len, dropout, seed):
        super().__init__()
        torch.manual_seed(seed * 7919 + 1)
        self.tok = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)              # learned absolute positions
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
        self.max_len = max_len

    def forward(self, tok):
        B, L = tok.shape
        pos = torch.arange(L, device=tok.device).unsqueeze(0)
        h = self.tok(tok) + self.pos(pos)
        causal = torch.triu(torch.ones(L, L, device=tok.device, dtype=torch.bool), diagonal=1)
        kpm = (tok == PAD)
        h = self.enc(h, mask=causal, src_key_padding_mask=kpm)
        return self.head(self.ln(h))                          # (B,L,V)


def train(model, pairs, M, stoi, out_range, *, steps, batch_size, lr, shuffle_p, seed, device, log_every=200):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed * 104729 + 7)
    model.train()
    run = 0.0
    for step in range(1, steps + 1):
        tok, lossm = make_episode_batch(pairs, M, batch_size, stoi, out_range, gen, shuffle_p=shuffle_p, device=device)
        logits = model(tok)                                   # (B,L,V)
        # causal LM: predict tok[:,1:] from logits[:,:-1]; score where lossm[:,1:] is True
        lg = logits[:, :-1].reshape(-1, logits.size(-1))
        tgt = tok[:, 1:].reshape(-1)
        msk = lossm[:, 1:].reshape(-1)
        loss = F.cross_entropy(lg[msk], tgt[msk])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run += loss.item()
        if step % log_every == 0:
            print(f"      step {step}/{steps} loss={run / log_every:.3f}", file=sys.stderr, flush=True); run = 0.0


# ---- eval (generation with k support pairs) ---------------------------------------------------------------

@torch.no_grad()
def evaluate(model, test_pairs, train_pairs, k, stoi, itos, out_range, *, max_gen, eval_batch, max_n, seed, device):
    """For each test (query) pair: sample k support pairs from train, build [BOS] sup... <sep> qx <io>,
    generate action tokens until EOS, exact-match vs gold. k=0 => no support (vanilla). Identity output map."""
    model.eval()
    gen = torch.Generator().manual_seed(seed * 1299709 + 3)
    queries = test_pairs if max_n is None or max_n >= len(test_pairs) else \
        [test_pairs[i] for i in torch.randperm(len(test_pairs), generator=gen)[:max_n].tolist()]
    correct = 0
    out_lo, out_hi = out_range
    for b in range(0, len(queries), eval_batch):
        chunk = queries[b:b + eval_batch]
        prefixes, golds = [], []
        for cmd, act in chunk:
            toks = [BOS]
            if k > 0:
                sidx = torch.randint(0, len(train_pairs), (k,), generator=gen).tolist()
                for scmd, sact in (train_pairs[i] for i in sidx):
                    toks += [stoi[t] for t in scmd] + [IO] + [stoi[t] for t in sact] + [SEP]
            toks += [stoi[t] for t in cmd] + [IO]
            prefixes.append(toks)
            golds.append([stoi[t] for t in act])
        # right-pad prefixes; track true lengths; generate by per-seq last-position gather
        Lp = max(len(p) for p in prefixes)
        total = Lp + max_gen
        B = len(prefixes)
        seq = torch.full((B, total), PAD, dtype=torch.long, device=device)
        cur = torch.tensor([len(p) for p in prefixes], device=device)
        for i, p in enumerate(prefixes):
            seq[i, :len(p)] = torch.tensor(p, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        gen_tokens = [[] for _ in range(B)]
        for _ in range(max_gen):
            cmax = int(cur.max().item())
            logits = model(seq[:, :cmax])                      # (B,cmax,V)
            last = logits[torch.arange(B), cur - 1]            # (B,V) logits at each seq's true last pos
            last[:, PAD] = -1e9; last[:, BOS] = -1e9; last[:, IO] = -1e9   # never generate these
            nxt = last.argmax(-1)                              # (B,)
            nxt = nxt.masked_fill(done, PAD)
            for i in range(B):
                if not done[i]:
                    seq[i, cur[i]] = nxt[i]
            cur = cur + (~done).long()
            newly = (nxt == EOS) | (nxt == SEP)
            for i in range(B):
                if not done[i] and not newly[i]:
                    gen_tokens[i].append(int(nxt[i].item()))
            done = done | newly
            if bool(done.all()):
                break
        for i in range(B):
            correct += int(gen_tokens[i] == golds[i])
    return correct / len(queries)


# ---- run --------------------------------------------------------------------------------------------------

def run_seed(args, split, arm, seed):
    tr = exp87.load_pairs(str(MCD / f"tasks_train_{split}.txt"))
    te = exp87.load_pairs(str(MCD / f"tasks_test_{split}.txt"))
    stoi, itos, out_range = build_unified_vocab(tr + te)
    M = 1 if arm == "vanilla" else args.M
    shuffle_p = args.shuffle_p if arm == "metaicl_shuffle" else 0.0
    k_eval = 0 if arm == "vanilla" else M - 1
    # max sequence length the model must support (train episodes + eval prefix+gen)
    max_pair = max(len(c) + len(a) + 2 for c, a in tr + te)
    max_len = M * max_pair + 8
    model = CausalTransformer(len(itos), args.d_model, args.nhead, args.nlayers, args.dim_ff,
                              max_len, args.dropout, seed).to(args.device)
    train(model, tr, M, stoi, out_range, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
          shuffle_p=shuffle_p, seed=seed, device=args.device)
    max_gen = max(len(a) for _, a in te) + 2
    acc = evaluate(model, te, tr, k_eval, stoi, itos, out_range, max_gen=max_gen,
                   eval_batch=args.eval_batch, max_n=args.max_eval, seed=seed, device=args.device)
    # k=0 collapse check for metaicl arms (Stage-2 crux: does composition live in the context?)
    acc_k0 = None
    if arm != "vanilla" and args.k0_check:
        acc_k0 = evaluate(model, te, tr, 0, stoi, itos, out_range, max_gen=max_gen,
                          eval_batch=args.eval_batch, max_n=args.max_eval, seed=seed, device=args.device)
    print(f"  [{split} {arm} s{seed}] EM(k={k_eval})={acc:.4f}" + (f"  EM(k=0)={acc_k0:.4f}" if acc_k0 is not None else ""),
          file=sys.stderr, flush=True)
    return {"acc": acc, "acc_k0": acc_k0, "M": M, "params": sum(p.numel() for p in model.parameters())}


def boot_ci(vals, gen, reps=2000):
    t = torch.tensor(vals, dtype=torch.float)
    n = len(t)
    means = torch.stack([t[torch.randint(0, n, (n,), generator=gen)].mean() for _ in range(reps)])
    return float(t.mean()), float(means.quantile(0.025)), float(means.quantile(0.975))


def paired_delta(a, b, gen, reps=2000):
    """CI on mean(a)-mean(b), paired by seed."""
    da = torch.tensor(a, dtype=torch.float) - torch.tensor(b, dtype=torch.float)
    n = len(da)
    means = torch.stack([da[torch.randint(0, n, (n,), generator=gen)].mean() for _ in range(reps)])
    return float(da.mean()), float(means.quantile(0.025)), float(means.quantile(0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="mcd1,mcd2,mcd3")
    ap.add_argument("--arms", default="vanilla,metaicl,metaicl_shuffle")
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--shuffle-p", type=float, default=1.0, dest="shuffle_p")
    ap.add_argument("--d-model", type=int, default=256, dest="d_model")
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=4)
    ap.add_argument("--dim-ff", type=int, default=512, dest="dim_ff")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-batch", type=int, default=64, dest="eval_batch")
    ap.add_argument("--max-eval", type=int, default=256, dest="max_eval", help="subsample test set for speed; -1=full")
    ap.add_argument("--k0-check", action="store_true", dest="k0_check")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.d_model, args.nhead, args.nlayers, args.dim_ff = 64, 2, 2, 128
        args.steps, args.seeds, args.M, args.max_eval, args.batch_size = 80, 1, 6, 32, 16
        args.splits, args.arms = "mcd1", "vanilla,metaicl"
    if args.max_eval is not None and args.max_eval < 0:
        args.max_eval = None

    splits = args.splits.split(",")
    arms = args.arms.split(",")
    res = {s: {a: [] for a in arms} for s in splits}
    for s in splits:
        for a in arms:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                res[s][a].append(run_seed(args, s, a, seed))

    cig = torch.Generator().manual_seed(20260629)
    summary = {}
    for s in splits:
        summary[s] = {}
        for a in arms:
            accs = [r["acc"] for r in res[s][a]]
            m, lo, hi = boot_ci(accs, cig)
            entry = {"mean": m, "ci": [lo, hi], "per_seed": accs, "params": res[s][a][0]["params"], "M": res[s][a][0]["M"]}
            k0 = [r["acc_k0"] for r in res[s][a] if r["acc_k0"] is not None]
            if k0:
                entry["mean_k0"] = float(torch.tensor(k0).mean()); entry["per_seed_k0"] = k0
            summary[s][a] = entry
        # headline: meta-ICL minus vanilla, CI-disjoint > 0 ?
        if "vanilla" in arms:
            for a in arms:
                if a == "vanilla":
                    continue
                d, dlo, dhi = paired_delta([r["acc"] for r in res[s][a]], [r["acc"] for r in res[s]["vanilla"]], cig)
                summary[s][f"DELTA_{a}_minus_vanilla"] = {"mean": d, "ci": [dlo, dhi], "ci_disjoint_pos": bool(dlo > 0)}

    headline_pass = None
    if all(x in splits for x in ("mcd2", "mcd3")) and "metaicl" in arms:
        headline_pass = bool(summary["mcd2"].get("DELTA_metaicl_minus_vanilla", {}).get("ci_disjoint_pos") and
                             summary["mcd3"].get("DELTA_metaicl_minus_vanilla", {}).get("ci_disjoint_pos"))
    out = {
        "experiment": "100_betb_metaicl_stage1",
        "charter": "notes/betb-metaICL-consolidation-door-precommit.md",
        "config": vars(args),
        "summary": summary,
        "STAGE1_HEADLINE_metaicl_minus_vanilla_CI_disjoint_pos_on_MCD2_AND_MCD3": headline_pass,
        "repro_target_paper": {"mcd1": [21.8, 60.4, 71.2], "mcd2": [25.6, 53.3, 74.8], "mcd3": [19.7, 50.7, 38.7]},
    }
    print(json.dumps(out, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
