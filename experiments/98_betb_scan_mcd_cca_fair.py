"""experiments/98 — Bet B / SCAN-MCD lever 1 RE-DONE FAIRLY: a CAPACITY-MATCHED composition operator.

CHARTER: the interpretation-skeptic verification of Report 148 (2026-06-17) overturned 148-as-stated. Two findings:
  (1) cca - cca_holistic = -0.026 is a COIN-FLIP (4 seeds +, 4 seeds -; one-seed-fragile; leave-one-out -> -0.003)
      = "cannot distinguish", NOT "redundant".
  (2) cca_holistic is NOT a fair baseline: cca conditions on W*[mean-pool(L); mean-pool(R)] (order-DESTROYING),
      cca_holistic conditions on the encoder's full SEQUENTIAL GRU summary h. So 148 confounded "composition"
      with "weak mean-pool summarizer" -- a fresh confound, like CCC's frozen-decoder one.

THE FAIR TEST (this file): hold COMPOSER CAPACITY fixed, vary ONLY clause-boundary-respect. The composer is a
per-clause GRU READOUT (re-encode each clause span with the SHARED encoder GRU; take the hidden at the clause's
last real token -- an order-aware, capacity-matched summary), recombined by W_conj. The matched non-compositional
baseline (fair_nosplit) is the SAME learned W over the WHOLE-command GRU readout (= W_unary(h)). cca_holistic
(cond = raw h) is the reference. So:
  fair_cca      cond = W_conj([gru_readout(clause_L) ; gru_readout(clause_R)])   (clause-structure-aware)
  fair_nosplit  cond = W_unary(gru_readout(whole) = h)                           (same capacity, NO clause split)
  cca_holistic  cond = h                                                          (raw holistic reference)
  fair_random   cond = W_conj over a RANDOM interior split's readouts            (must NULL)
  vanilla_plain exp87 floor
HEADLINE (the skeptic's bar): fair_cca - fair_nosplit AND fair_cca - cca_holistic CI-disjoint > 0, SIGN ROBUST to
leave-one-seed-out, on mcd1 AND mcd2, n=8. Plus PER-EPOCH test-EM trajectory (diagnoses the smoke->full crossover).
If fair_cca still ties/loses capacity-matched, seed-robust, on both splits -> the 'wall' reading is GENUINELY earned.
If it wins -> 148 overturned, the thesis is alive and the lever is composer-fairness.

PREAMBLE (CLAUDE.md): Active capability = Bet-B Stage-1 composition-as-inference, FAIR re-test (146 arena).
Headline = fair_cca vs the capacity-matched fair_nosplit + cca_holistic, CI-disjoint, seed-robust, mcd1+mcd2, n=8.
Controls = fair_random (must null), vanilla floor, per-epoch trajectory. Why now: verify-before-deciding; the
prior lever-1 (exp95) used a confounded weak composer.
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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _load(name, fname):
    s = importlib.util.spec_from_file_location(name, REPO / "experiments" / fname)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


exp87 = _load("exp87", "87_betb_scan_gap.py")
exp94 = _load("exp94", "94_betb_scan_mcd_ccc.py")
load_pairs, build_vocab, to_ids, pad, make_batches = (exp87.load_pairs, exp87.build_vocab, exp87.to_ids,
                                                      exp87.pad, exp87.make_batches)
PAD, SOS, EOS = exp87.PAD, exp87.SOS, exp87.EOS
MCD = REPO / "data" / "scan" / "mcd_split"


class CondAttnDecoder(nn.Module):
    def __init__(self, vocab, embed, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed, padding_idx=PAD)
        self.gru = nn.GRU(embed + hidden, hidden, batch_first=True)
        self.attn_combine = nn.Linear(2 * hidden, hidden)
        self.out = nn.Linear(hidden, vocab)

    def step(self, y_prev, h, enc_out, src_mask, cond):
        o, h = self.gru(torch.cat([self.emb(y_prev), cond], -1).unsqueeze(1), h)
        scores = torch.bmm(o, enc_out.transpose(1, 2)).squeeze(1).masked_fill(~src_mask, float("-inf"))
        ctx = torch.bmm(F.softmax(scores, -1).unsqueeze(1), enc_out).squeeze(1)
        return self.out(torch.tanh(self.attn_combine(torch.cat([o.squeeze(1), ctx], -1)))), h


def _gru_readout(enc, src):
    """Order-aware, capacity-matched clause/whole summary: encode with the SHARED GRU, take the hidden at the
    last REAL (non-PAD) token. src (B,L). Returns (B,H)."""
    out, _ = enc(src)                                   # (B,L,H)
    lens = (src != PAD).sum(1)                          # (B,)
    idx = (lens - 1).clamp(min=0)
    return out.gather(1, idx[:, None, None].expand(-1, 1, out.shape[2])).squeeze(1)


class FairCCASeq2Seq(nn.Module):
    """cond is a capacity-matched per-clause GRU readout (fair_cca / fair_random / fair_nosplit) or the raw
    holistic h (cca_holistic / vanilla-equivalent). The ONLY thing varied across composition arms is whether the
    GRU-readout respects the conjunction boundary."""

    def __init__(self, in_vocab, out_vocab, embed, hidden, seed, *, mode):
        super().__init__()
        torch.manual_seed(seed * 7919 + 1)
        self.enc = exp87.Encoder(in_vocab, embed, hidden)
        self.dec = CondAttnDecoder(out_vocab, embed, hidden)
        self.W_and = nn.Linear(2 * hidden, hidden)
        self.W_after = nn.Linear(2 * hidden, hidden)
        self.W_unary = nn.Linear(hidden, hidden)
        self.out_vocab, self.mode = out_vocab, mode      # mode in {holistic, fair_cca, fair_nosplit, fair_random}
        self.and_id = self.after_id = None

    def _clause_srcs(self, src, split_pos, conj_present):
        """Build left-aligned clause-L (prefix) and clause-R (suffix) src tensors at split_pos."""
        B, S = src.shape; dev = src.device; pos = torch.arange(S, device=dev)
        lengths = (src != PAD).sum(1)
        src_L = src.masked_fill(pos[None, :] >= split_pos[:, None], PAD)          # prefix
        Rstart = split_pos + 1
        gidx = (Rstart[:, None] + pos[None, :]).clamp(max=S - 1)
        src_R = src.gather(1, gidx).masked_fill(pos[None, :] >= (lengths - Rstart).clamp(min=0)[:, None], PAD)
        return src_L, src_R

    def _cond(self, src, src_mask, h, gen):
        if self.mode == "holistic":
            return h.squeeze(0)
        h_whole = _gru_readout(self.enc, src)            # = h.squeeze(0); recompute for symmetry/capacity-match
        if self.mode == "fair_nosplit":
            return self.W_unary(h_whole)
        is_conj = (src == self.and_id) | (src == self.after_id)
        conj_present = is_conj.any(1)
        conj_pos = is_conj.float().argmax(1)
        is_after = src.gather(1, conj_pos.unsqueeze(1)).squeeze(1) == self.after_id
        split_pos = conj_pos
        if self.mode == "fair_random":
            lengths = src_mask.sum(1)
            u = torch.rand(src.shape[0], generator=gen).to(src.device)
            split_pos = ((u * (lengths - 1).clamp(min=1).float()).long() + 1).minimum((lengths - 1).clamp(min=1))
        src_L, src_R = self._clause_srcs(src, split_pos, conj_present)
        cat = torch.cat([_gru_readout(self.enc, src_L), _gru_readout(self.enc, src_R)], -1)
        e = torch.where(is_after.unsqueeze(1), self.W_after(cat), self.W_and(cat))
        return torch.where(conj_present.unsqueeze(1), e, self.W_unary(h_whole))   # single-clause fall-through

    def forward(self, src, src_mask, tgt, tf_ratio, gen):
        enc_out, h = self.enc(src)
        cond = self._cond(src, src_mask, h, gen)
        dh = cond.unsqueeze(0).contiguous(); B, T = tgt.shape
        y = torch.full((B,), SOS, dtype=torch.long, device=src.device); out = []
        for t in range(T):
            lg, dh = self.dec.step(y, dh, enc_out, src_mask, cond); out.append(lg)
            y = tgt[:, t] if torch.rand(1, generator=gen).item() < tf_ratio else lg.argmax(-1)
        return torch.stack(out, 1)

    @torch.no_grad()
    def greedy(self, src, src_mask, max_len, gen):
        enc_out, h = self.enc(src)
        cond = self._cond(src, src_mask, h, gen)
        dh = cond.unsqueeze(0).contiguous(); B = src.shape[0]
        y = torch.full((B,), SOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device); seqs = []
        for _ in range(max_len):
            lg, dh = self.dec.step(y, dh, enc_out, src_mask, cond)
            y = lg.argmax(-1).masked_fill(done, PAD); seqs.append(y.clone()); done = done | (y == EOS)
            if done.all():
                break
        return torch.stack(seqs, 1)


@torch.no_grad()
def exact_match(model, pairs, in_vocab, out_vocab, *, max_len, batch_size, device):
    model.eval(); correct = 0; gen = torch.Generator().manual_seed(0)
    for b in range(0, len(pairs), batch_size):
        chunk = pairs[b:b + batch_size]
        src, _ = pad([to_ids(c[0], in_vocab) for c in chunk], device)
        gold = [to_ids(c[1], out_vocab) for c in chunk]
        pred = model.greedy(src, src != PAD, max_len, gen).tolist()
        for p, g in zip(pred, gold):
            if EOS in p:
                p = p[:p.index(EOS) + 1]
            correct += int(p == g)
    return correct / len(pairs)


def train_traj(model, tr, te, in_vocab, out_vocab, *, epochs, batch_size, lr, tf_ratio, seed, max_len, device, eval_every):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed * 104729 + 7); traj = {}
    for ep in range(epochs):
        model.train()
        for src, tgt in make_batches(tr, in_vocab, out_vocab, batch_size, gen):
            src, _ = pad(src, device); tgt, _ = pad(tgt, device)
            loss = F.cross_entropy(model(src, src != PAD, tgt, tf_ratio, gen).reshape(-1, model.out_vocab),
                                   tgt.reshape(-1), ignore_index=PAD)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if (ep + 1) % eval_every == 0 or ep + 1 == epochs:
            traj[ep + 1] = exact_match(model, te, in_vocab, out_vocab, max_len=max_len, batch_size=batch_size, device=device)
            print(f"    ep{ep + 1} test-EM={traj[ep + 1]:.4f}", file=sys.stderr, flush=True)
    return traj


def load_mcd(split):
    return (load_pairs(str(MCD / f"tasks_train_{split}.txt")), load_pairs(str(MCD / f"tasks_test_{split}.txt")))


def run_seed(args, seed, split, arms):
    tr, te = load_mcd(split); allp = tr + te
    in_vocab, _ = build_vocab(allp, "in"); out_vocab, _ = build_vocab(allp, "out")
    max_len = max(len(p[1]) for p in allp) + 2
    and_id, after_id = in_vocab["and"], in_vocab["after"]
    res = {}
    for arm in arms:
        if arm == "vanilla_plain":
            m = exp87.Seq2Seq(len(in_vocab), len(out_vocab), args.embed, args.hidden, seed).to(args.device)
            exp87.train(m, tr, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                        tf_ratio=args.tf_ratio, seed=seed, device=args.device)
            res[arm] = {"final": exp87.exact_match(m, te, in_vocab, out_vocab, max_len=max_len, batch_size=args.batch_size, device=args.device)}
        else:
            m = FairCCASeq2Seq(len(in_vocab), len(out_vocab), args.embed, args.hidden, seed, mode=arm).to(args.device)
            m.and_id, m.after_id = and_id, after_id
            traj = train_traj(m, tr, te, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size,
                              lr=args.lr, tf_ratio=args.tf_ratio, seed=seed, max_len=max_len, device=args.device,
                              eval_every=args.eval_every)
            res[arm] = {"final": traj[max(traj)], "traj": traj}
        print(f"  [{split} s{seed}] {arm} = {res[arm]['final']:.4f}", file=sys.stderr, flush=True)
    return res


def leave_one_out(a, b):
    import statistics
    d = [a[i] - b[i] for i in range(len(a))]
    loo = [statistics.mean(d[:i] + d[i + 1:]) for i in range(len(d))]
    return {"delta_mean": statistics.mean(d), "per_seed": d, "loo_min": min(loo), "loo_max": max(loo),
            "sign_robust": all((x > 0) == (statistics.mean(d) > 0) for x in loo)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", type=int, default=64); ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--tf-ratio", type=float, default=0.5, dest="tf_ratio")
    ap.add_argument("--eval-every", type=int, default=5, dest="eval_every")
    ap.add_argument("--seeds", type=int, default=8); ap.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    ap.add_argument("--device", default="mps"); ap.add_argument("--splits", default="mcd1,mcd2")
    ap.add_argument("--arms", default="vanilla_plain,cca_holistic,fair_cca,fair_nosplit,fair_random")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs, args.splits, args.eval_every = 1, 10, "mcd1", 5
        args.arms = "cca_holistic,fair_cca,fair_nosplit"

    arms = args.arms.split(",")
    out = {"experiment": "98_betb_scan_mcd_cca_fair (capacity-matched composition, fair lever-1 re-test)",
           "charter": "skeptic verification of Report 148", "config": vars(args), "by_split": {}}
    for split in args.splits.split(","):
        per = {a: [] for a in arms}; trajs = {a: [] for a in arms}
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            r = run_seed(args, seed, split, arms)
            for a in arms:
                per[a].append(r[a]["final"])
                if "traj" in r[a]:
                    trajs[a].append(r[a]["traj"])
        sdict = {a: exp94.summary(per[a]) for a in arms}
        deltas = {}
        for a, b in [("fair_cca", "fair_nosplit"), ("fair_cca", "cca_holistic"), ("fair_cca", "vanilla_plain"),
                     ("fair_cca", "fair_random")]:
            if a in per and b in per and len(per[a]) == len(per[b]) and len(per[a]) > 1:
                deltas[f"{a}_minus_{b}"] = {**exp94.paired_delta(per[a], per[b]), **leave_one_out(per[a], per[b])}
        out["by_split"][split] = {"arms": sdict, "paired_deltas": deltas,
                                  "traj_mean": {a: ({ep: round(sum(t[ep] for t in trajs[a]) / len(trajs[a]), 4)
                                                     for ep in trajs[a][0]} if trajs[a] else None) for a in arms}}
        print(f"\n=== {split}: " + " ".join(f"{a}={sdict[a]['mean']:.3f}" for a in arms), file=sys.stderr, flush=True)
    print(json.dumps(out, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
