"""experiments/87 — Bet B SCAN discriminating regime, STAGE 0: reproduce the documented compositional gap.

CHARTER: notes/betb-scan-discriminating-regime-precommit.md (after the modular-arithmetic regime was shown
NULL — toy substrate too brittle, 2026-06-15). This is the make-or-break VALIDITY gate, NOT a graduation
(CLAUDE.md preamble): a small seq2seq must REPRODUCE the documented SCAN failure —

  random/simple split   -> ~100% exact-match   (the model CAN learn SCAN; sanity ceiling)
  add-primitive-jump    -> ~0-5% exact-match    (jump held out of all train compositions -> the GAP)

If the small model does NOT clearly fail the jump split, the regime isn't discriminating at our scale and we
pick a config that clearly fails (precommit §5) BEFORE any mechanism (Stage 1). Data: data/scan/ (fetch_scan.sh).

MODEL: 1-layer GRU encoder-decoder with Luong attention, teacher forcing. Laptop/MPS-sized. Greedy decode for
exact-match eval. No mechanism here — Stage 0 is the gap-diagnostic only.
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

PAD, SOS, EOS = 0, 1, 2
SPECIAL = ["<pad>", "<sos>", "<eos>"]
DATA = REPO / "data" / "scan"


# ---- data -------------------------------------------------------------------------------------------------

def load_pairs(fp):
    pairs = []
    for line in pathlib.Path(fp).read_text().splitlines():
        if not line.strip():
            continue
        cmd, act = line.split(" OUT: ")
        pairs.append((cmd[4:].split(), act.split()))     # strip "IN: "
    return pairs


def build_vocab(pairs, which):
    toks = sorted({t for p in pairs for t in (p[0] if which == "in" else p[1])})
    itos = SPECIAL + toks
    return {t: i for i, t in enumerate(itos)}, itos


def to_ids(seq, vocab, add_eos=True):
    ids = [vocab[t] for t in seq]
    return ids + ([EOS] if add_eos else [])


def make_batches(pairs, in_vocab, out_vocab, batch_size, gen, shuffle=True):
    idx = torch.randperm(len(pairs), generator=gen).tolist() if shuffle else list(range(len(pairs)))
    for b in range(0, len(idx), batch_size):
        chunk = [pairs[i] for i in idx[b:b + batch_size]]
        src = [to_ids(c[0], in_vocab) for c in chunk]
        tgt = [to_ids(c[1], out_vocab) for c in chunk]
        yield src, tgt


def pad(seqs, device):
    m = max(len(s) for s in seqs)
    t = torch.full((len(seqs), m), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        t[i, :len(s)] = torch.tensor(s)
    return t.to(device), torch.tensor([len(s) for s in seqs], device=device)


# ---- model ------------------------------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, vocab, embed, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed, padding_idx=PAD)
        self.gru = nn.GRU(embed, hidden, batch_first=True)

    def forward(self, src):
        out, h = self.gru(self.emb(src))          # out (B,S,H), h (1,B,H)
        return out, h


class AttnDecoder(nn.Module):
    def __init__(self, vocab, embed, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed, padding_idx=PAD)
        self.gru = nn.GRU(embed, hidden, batch_first=True)
        self.attn_combine = nn.Linear(2 * hidden, hidden)
        self.out = nn.Linear(hidden, vocab)

    def step(self, y_prev, h, enc_out, src_mask):
        # y_prev (B,), h (1,B,H), enc_out (B,S,H), src_mask (B,S) True=valid
        o, h = self.gru(self.emb(y_prev).unsqueeze(1), h)        # o (B,1,H)
        q = o.transpose(0, 1)                                    # (1,B,H) -> use (B,1,H)? keep (B,1,H)=o
        scores = torch.bmm(o, enc_out.transpose(1, 2)).squeeze(1)  # (B,S)
        scores = scores.masked_fill(~src_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)                        # (B,S)
        ctx = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)  # (B,H)
        combined = torch.tanh(self.attn_combine(torch.cat([o.squeeze(1), ctx], -1)))  # (B,H)
        return self.out(combined), h                            # logits (B,V), h


class Seq2Seq(nn.Module):
    def __init__(self, in_vocab, out_vocab, embed=64, hidden=200, seed=0):
        super().__init__()
        torch.manual_seed(seed * 7919 + 1)
        self.enc = Encoder(in_vocab, embed, hidden)
        self.dec = AttnDecoder(out_vocab, embed, hidden)
        self.out_vocab = out_vocab

    def forward(self, src, src_mask, tgt, tf_ratio, gen):
        enc_out, h = self.enc(src)
        B, T = tgt.shape
        y = torch.full((B,), SOS, dtype=torch.long, device=src.device)
        logits = []
        for t in range(T):
            lg, h = self.dec.step(y, h, enc_out, src_mask)
            logits.append(lg)
            teacher = torch.rand(1, generator=gen).item() < tf_ratio
            y = tgt[:, t] if teacher else lg.argmax(-1)
        return torch.stack(logits, 1)                           # (B,T,V)

    @torch.no_grad()
    def greedy(self, src, src_mask, max_len):
        enc_out, h = self.enc(src)
        B = src.shape[0]
        y = torch.full((B,), SOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        seqs = []
        for _ in range(max_len):
            lg, h = self.dec.step(y, h, enc_out, src_mask)
            y = lg.argmax(-1)
            y = y.masked_fill(done, PAD)
            seqs.append(y.clone())
            done = done | (y == EOS)
            if done.all():
                break
        return torch.stack(seqs, 1)                             # (B,L)


# ---- train / eval -----------------------------------------------------------------------------------------

def train(model, pairs, in_vocab, out_vocab, *, epochs, batch_size, lr, tf_ratio, seed, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed * 104729 + 7)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for src, tgt in make_batches(pairs, in_vocab, out_vocab, batch_size, gen):
            src, _ = pad(src, device)
            tgt, _ = pad(tgt, device)
            src_mask = src != PAD
            logits = model(src, src_mask, tgt, tf_ratio, gen)
            loss = F.cross_entropy(logits.reshape(-1, model.out_vocab), tgt.reshape(-1), ignore_index=PAD)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        print(f"    epoch {ep + 1}/{epochs} loss={tot / max(1, len(pairs) // batch_size):.3f}",
              file=sys.stderr, flush=True)


@torch.no_grad()
def exact_match(model, pairs, in_vocab, out_vocab, *, max_len, batch_size, device):
    model.eval()
    correct = 0
    for b in range(0, len(pairs), batch_size):
        chunk = pairs[b:b + batch_size]
        src, _ = pad([to_ids(c[0], in_vocab) for c in chunk], device)
        gold = [to_ids(c[1], out_vocab) for c in chunk]                  # include EOS
        pred = model.greedy(src, src != PAD, max_len).tolist()
        for p, g in zip(pred, gold):
            # truncate prediction at first EOS (inclusive), compare to gold (which ends in EOS)
            if EOS in p:
                p = p[:p.index(EOS) + 1]
            correct += int(p == g)
    return correct / len(pairs)


# ---- run --------------------------------------------------------------------------------------------------

def run_split(name, train_fp, test_fp, args, seed):
    train_pairs = load_pairs(train_fp)
    test_pairs = load_pairs(test_fp)
    # vocab from the UNION so eval never OOVs (jump etc. all appear in train, but be safe)
    allp = train_pairs + test_pairs
    in_vocab, _ = build_vocab(allp, "in")
    out_vocab, _ = build_vocab(allp, "out")
    max_len = max(len(p[1]) for p in allp) + 2
    model = Seq2Seq(len(in_vocab), len(out_vocab), args.embed, args.hidden, seed).to(args.device)
    train(model, train_pairs, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, tf_ratio=args.tf_ratio, seed=seed, device=args.device)
    acc = exact_match(model, test_pairs, in_vocab, out_vocab, max_len=max_len,
                      batch_size=args.batch_size, device=args.device)
    print(f"  [{name} seed {seed}] test exact-match = {acc:.4f}", file=sys.stderr, flush=True)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tf-ratio", type=float, default=0.5, dest="tf_ratio")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--splits", default="simple,jump")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.seeds = 1, 1

    SPLITS = {"simple": (DATA / "tasks_train_simple.txt", DATA / "tasks_test_simple.txt"),
              "jump": (DATA / "tasks_train_addprim_jump.txt", DATA / "tasks_test_addprim_jump.txt")}
    want = [s for s in args.splits.split(",") if s in SPLITS]

    res = {s: [] for s in want}
    for s in want:
        tr, te = SPLITS[s]
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            res[s].append(run_split(s, str(tr), str(te), args, seed))

    def stat(v):
        t = torch.tensor(v)
        return {"mean": float(t.mean()), "min": float(t.min()), "max": float(t.max()), "per_seed": v}
    summary = {s: stat(res[s]) for s in want}

    gap_ok = None
    if "simple" in summary and "jump" in summary:
        gap_ok = bool(summary["simple"]["mean"] >= 0.90 and summary["jump"]["mean"] <= 0.05)
    out = {
        "experiment": "87_betb_scan_gap (Bet B SCAN — Stage 0 gap-diagnostic)",
        "charter": "notes/betb-scan-discriminating-regime-precommit.md",
        "config": vars(args),
        "exact_match_by_split": summary,
        "STAGE0_GATE_random>=0.90_AND_jump<=0.05": gap_ok,
        "verdict": (f"SCAN Stage 0: simple={summary.get('simple', {}).get('mean', float('nan')):.4f} "
                    f"jump={summary.get('jump', {}).get('mean', float('nan')):.4f} -> gap reproduced={gap_ok}. "
                    "PASS => the discriminating regime is real at our scale; Stage 1 (replay x consolidation) "
                    "licensed. FAIL(jump high) => pick a config that clearly fails before any mechanism."),
    }
    print(json.dumps(out, indent=2))
    print(f"\n=== {out['verdict']}", file=sys.stderr, flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
