"""experiments/90 — Bet B / SCAN Stage 1 iterate 3: TWO-SIDED role+filler factorization.

CHARTER: notes/betb-scan-stage1-consolidation-precommit.md + Reports 141/142. Report 142 (exp89) put the
role+filler split on the INPUT (encoder on role, decoder reads identity from filler) -> consolidation became
LOAD-BEARING (jump-split ~2x in 5/5) but PARTIAL, capped at a leaky premise ceiling 0.26. Diagnostics localized
the leak precisely:
  - oracle filler (force ctx_fill = fill_jump): NO change (0.185 -> 0.183) -> NOT the filler channel.
  - CANONICAL decoder feedback (feed a single "a verb was emitted" token instead of the specific I_JUMP):
    single-verb premise 0.18 -> 1.0000. THE LEAK is the decoder's identity-FEEDBACK: I_JUMP never appeared
    mid-sequence in training (jump only as the standalone single-token output), so emitting it breaks the
    autoregressive recurrence.

THE FIX (this file): factor the OUTPUT side too. The decoder RECURS on the ROLE-CLASS of its previous output
(all verb-actions -> one canonical VERB class; turns/EOS keep identity) — identity is READ OUT by the head
(fillsel) but does NOT re-enter the recurrence. So emitting I_JUMP vs I_WALK leaves the decoder state
identical -> jump rides the trained verb-slot dynamics. Input side unchanged from exp89.

Same two decisive outcomes as exp89: (A) baseline already high = architectural fix; (B) baseline low (role_jump
off-manifold), consolidation aligns role_jump and LIFTS to the (now ~1.0) ceiling = CONSOLIDATION IS THE HERO,
thesis graduates. Anti-homunculus: peer set computed-in-code; role-class map is a fixed content-blind grouping
of OUTPUT tokens by type (verbs-as-a-class), not a per-token metric branch. FENCE: iterative gradient.
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

_s = importlib.util.spec_from_file_location("exp87", REPO / "experiments" / "87_betb_scan_gap.py")
exp87 = importlib.util.module_from_spec(_s); _s.loader.exec_module(exp87)
load_pairs, build_vocab, to_ids, pad, make_batches = (exp87.load_pairs, exp87.build_vocab, exp87.to_ids,
                                                      exp87.pad, exp87.make_batches)
PAD, SOS, EOS = exp87.PAD, exp87.SOS, exp87.EOS
DATA = REPO / "data" / "scan"
VERB_ACTIONS = ["I_JUMP", "I_WALK", "I_RUN", "I_LOOK"]


def build_roleclass_map(out_vocab_size, verb_ids):
    """OUTPUT-side role classes: every verb-action collapses to ONE 'VERB' class; all other tokens keep
    identity. Returns rc_map (out_vocab -> rc index) and rc_size. Content-blind grouping by token type."""
    verb_set = set(verb_ids)
    rc_map = [0] * out_vocab_size
    nxt = 0
    non_verb_rc = {}
    for tok in range(out_vocab_size):
        if tok in verb_set:
            continue
        non_verb_rc[tok] = nxt; nxt += 1
    verb_class = nxt; rc_size = nxt + 1
    for tok in range(out_vocab_size):
        rc_map[tok] = verb_class if tok in verb_set else non_verb_rc[tok]
    return torch.tensor(rc_map), rc_size


class Factored2Seq2Seq(nn.Module):
    def __init__(self, in_vocab, out_vocab, verb_ids, role_dim=48, fill_dim=16, hidden=200, seed=0):
        super().__init__()
        torch.manual_seed(seed * 7919 + 1)
        self.r, self.f, self.H = role_dim, fill_dim, hidden
        self.in_emb = nn.Embedding(in_vocab, role_dim + fill_dim, padding_idx=PAD)
        self.enc = nn.GRU(role_dim, hidden, batch_first=True)
        rc_map, rc_size = build_roleclass_map(out_vocab, verb_ids)
        self.register_buffer("rc_map", rc_map)                          # out token -> role-class (for decoder INPUT)
        self.dec_emb = nn.Embedding(rc_size, role_dim)                  # decoder recurs on ROLE-CLASS, not identity
        self.dec = nn.GRU(role_dim, hidden, batch_first=True)
        self.attn_combine = nn.Linear(2 * hidden, hidden)
        self.struct_ids = torch.tensor([i for i in range(out_vocab) if i not in set(verb_ids)])
        self.verb_ids = torch.tensor(list(verb_ids))
        self.n_struct = self.struct_ids.numel()
        self.base = nn.Linear(hidden, self.n_struct + 1)
        self.fillsel = nn.Linear(fill_dim, len(verb_ids))
        self.out_vocab = out_vocab

    def encode(self, src):
        e = self.in_emb(src)
        enc_out, h = self.enc(e[..., :self.r])
        return enc_out, e[..., self.r:], h

    def _logits(self, d, enc_out, fill_mem, src_mask):
        scores = torch.bmm(d, enc_out.transpose(1, 2)).squeeze(1).masked_fill(~src_mask, float("-inf"))
        a = F.softmax(scores, dim=-1)
        ctx_role = torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)
        ctx_fill = torch.bmm(a.unsqueeze(1), fill_mem).squeeze(1)
        h = torch.tanh(self.attn_combine(torch.cat([d.squeeze(1), ctx_role], -1)))
        base = self.base(h); fillsel = self.fillsel(ctx_fill); B = base.shape[0]
        logits = torch.zeros(B, self.out_vocab, device=base.device)
        logits = logits.index_copy(1, self.struct_ids.to(base.device), base[:, :self.n_struct])
        logits = logits.index_copy(1, self.verb_ids.to(base.device), base[:, self.n_struct:self.n_struct + 1] + fillsel)
        return logits

    def _decin(self, y):
        return self.dec_emb(self.rc_map[y])                            # specific token -> role-class -> embedding

    def forward(self, src, src_mask, tgt, tf_ratio, gen):
        enc_out, fill_mem, h = self.encode(src)
        B, T = tgt.shape
        y = torch.full((B,), SOS, dtype=torch.long, device=src.device); out = []
        for t in range(T):
            d, h = self.dec(self._decin(y).unsqueeze(1), h)
            lg = self._logits(d, enc_out, fill_mem, src_mask); out.append(lg)
            teacher = torch.rand(1, generator=gen).item() < tf_ratio
            y = tgt[:, t] if teacher else lg.argmax(-1)
        return torch.stack(out, 1)

    @torch.no_grad()
    def greedy(self, src, src_mask, max_len):
        enc_out, fill_mem, h = self.encode(src); B = src.shape[0]
        y = torch.full((B,), SOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device); seqs = []
        for _ in range(max_len):
            d, h = self.dec(self._decin(y).unsqueeze(1), h)
            y = self._logits(d, enc_out, fill_mem, src_mask).argmax(-1).masked_fill(done, PAD)
            seqs.append(y.clone()); done = done | (y == EOS)
            if done.all():
                break
        return torch.stack(seqs, 1)


def train(model, pairs, in_vocab, out_vocab, *, epochs, batch_size, lr, tf_ratio, seed, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed * 104729 + 7); model.train()
    for ep in range(epochs):
        tot = 0.0
        for src, tgt in make_batches(pairs, in_vocab, out_vocab, batch_size, gen):
            src, _ = pad(src, device); tgt, _ = pad(tgt, device)
            logits = model(src, src != PAD, tgt, tf_ratio, gen)
            loss = F.cross_entropy(logits.reshape(-1, model.out_vocab), tgt.reshape(-1), ignore_index=PAD)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); tot += loss.item()
        print(f"    epoch {ep + 1}/{epochs} loss={tot / max(1, len(pairs) // batch_size):.3f}", file=sys.stderr, flush=True)


@torch.no_grad()
def exact_match(model, pairs, in_vocab, out_vocab, *, max_len, batch_size, device):
    model.eval(); correct = 0
    for b in range(0, len(pairs), batch_size):
        chunk = pairs[b:b + batch_size]
        src, _ = pad([to_ids(c[0], in_vocab) for c in chunk], device)
        gold = [to_ids(c[1], out_vocab) for c in chunk]
        pred = model.greedy(src, src != PAD, max_len).tolist()
        for p, g in zip(pred, gold):
            if EOS in p:
                p = p[:p.index(EOS) + 1]
            correct += int(p == g)
    return correct / len(pairs)


def consolidate_role(model, pairs, in_vocab, out_vocab, peer_ids, *, lam, steps, batch_size, lr, gen, device):
    for p in model.parameters():
        p.requires_grad_(False)
    model.in_emb.weight.requires_grad_(True)
    opt = torch.optim.Adam([model.in_emb.weight], lr=lr); model.train(); bt = None
    for _ in range(steps):
        if bt is None:
            bt = iter(make_batches(pairs, in_vocab, out_vocab, batch_size, gen))
        try:
            src, tgt = next(bt)
        except StopIteration:
            bt = iter(make_batches(pairs, in_vocab, out_vocab, batch_size, gen)); src, tgt = next(bt)
        src, _ = pad(src, device); tgt, _ = pad(tgt, device)
        lg = model(src, src != PAD, tgt, 0.5, gen)
        loss = F.cross_entropy(lg.reshape(-1, model.out_vocab), tgt.reshape(-1), ignore_index=PAD)
        role = model.in_emb.weight[peer_ids, :model.r]
        loss = loss + lam * ((role - role.mean(0, keepdim=True)) ** 2).sum(1).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([model.in_emb.weight], 5.0); opt.step()
    for p in model.parameters():
        p.requires_grad_(True)


def run(args, seed):
    trp = load_pairs(str(DATA / "tasks_train_addprim_jump.txt")); tep = load_pairs(str(DATA / "tasks_test_addprim_jump.txt"))
    allp = trp + tep
    in_vocab, _ = build_vocab(allp, "in"); out_vocab, _ = build_vocab(allp, "out")
    max_len = max(len(p[1]) for p in allp) + 2
    peer = {p[0][0] for p in trp if len(p[0]) == 1}
    assert peer == {"jump", "walk", "run", "look"}, f"peer -> {peer}"
    peer_ids = sorted(in_vocab[t] for t in peer); verb_ids = [out_vocab[t] for t in VERB_ACTIONS]
    g = torch.Generator().manual_seed(seed * 104729 + 7)
    model = Factored2Seq2Seq(len(in_vocab), len(out_vocab), verb_ids, args.role_dim, args.fill_dim, args.hidden, seed).to(args.device)
    train(model, trp, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          tf_ratio=args.tf_ratio, seed=seed, device=args.device)
    em = lambda: exact_match(model, tep, in_vocab, out_vocab, max_len=max_len, batch_size=args.batch_size, device=args.device)
    ab = em(); print(f"  [factored2 baseline seed {seed}] jump-split = {ab:.4f}", file=sys.stderr, flush=True)
    consolidate_role(model, trp, in_vocab, out_vocab, peer_ids, lam=args.lam, steps=args.consol_steps,
                     batch_size=args.batch_size, lr=args.consol_lr, gen=g, device=args.device)
    ac = em(); print(f"  [factored2 consolidation seed {seed}] jump-split = {ac:.4f}", file=sys.stderr, flush=True)
    return ab, ac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role-dim", type=int, default=48, dest="role_dim")
    ap.add_argument("--fill-dim", type=int, default=16, dest="fill_dim")
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tf-ratio", type=float, default=0.5, dest="tf_ratio")
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--consol-steps", type=int, default=800, dest="consol_steps")
    ap.add_argument("--consol-lr", type=float, default=1e-3, dest="consol_lr")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs, args.consol_steps = 1, 12, 200
    base, consol = [], []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        b, c = run(args, seed); base.append(b); consol.append(c)

    def ci(v):
        t = torch.tensor(v); return {"mean": float(t.mean()), "min": float(t.min()), "max": float(t.max()), "per_seed": v}
    out = {"experiment": "90_betb_scan_factored2 (two-sided role+filler factorization)", "config": vars(args),
           "factored2_baseline": ci(base), "factored2_consolidation": ci(consol),
           "delta": ci([consol[i] - base[i] for i in range(len(base))])}
    out["verdict"] = (f"FACTORED2: baseline={out['factored2_baseline']['mean']:.4f} "
                      f"consolidation={out['factored2_consolidation']['mean']:.4f} (Δ={out['delta']['mean']:+.4f}). "
                      "Outcome A=arch fix; Outcome B=consolidation lifts to ~1.0 ceiling = GRADUATION.")
    print(json.dumps(out, indent=2))
    print(f"\n=== {out['verdict']}", file=sys.stderr, flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
