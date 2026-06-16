"""experiments/89 — Bet B / SCAN Stage 1, iterate 2: ARCHITECTURAL role+filler factorization.

CHARTER: notes/betb-scan-stage1-consolidation-precommit.md + Report 141. Report 141 showed a POST-HOC
representation-alignment consolidation cannot manufacture jump-composition: aligning the verb-role breaks
identity because the vanilla model reads identity from the SAME features as structure (L_align frozen at
30.85 even at lam=50 — irreconcilable with replay's identity constraint, absent a factored representation).

THE FIX (this file): build the role/filler split INTO the model so identity is read from a SEPARATE channel
than structure, then re-ask the Report-141 question.
  - embedding e_t = [role_t (r dims) ; fill_t (f dims)].
  - ENCODER runs on ROLE ONLY -> structure is role-driven; verbs with aligned roles are processed identically.
  - DECODER: attention over the role-encoder; structural tokens (turns/EOS) + a single verb-SLOT score come
    from role (d, ctx_role); WHICH verb-action is selected by the attended FILLER (ctx_fill -> 4 verb logits).
    So role decides WHEN/structure, filler decides WHICH verb. Identity lives in filler, architecturally.

THE TWO DECISIVE OUTCOMES:
  (A) the factored architecture ALONE already composes jump (jump-split high without consolidation) ->
      an architectural compositional-generalization fix (known class; equivariance-like). Interesting but
      consolidation NOT the hero -> note it, REDUNDANT-leaning.
  (B) the factored architecture alone does NOT compose jump (role_jump off-manifold, trained only standalone),
      BUT a consolidation aligning role_jump to the peer role-manifold DOES (identity preserved by the filler
      channel) -> CONSOLIDATION IS THE HERO; the Bet-B restructuring thesis SURVIVES (first real positive).

ARMS: baseline_factored (train only) | consolidation_factored (+ offline role-alignment pass). Plus the
post-hoc embedding-swap premise check, now expected to keep identity (role swap, fill kept).
ANTI-HOMUNCULUS: peer set computed-in-code (CONDITION 1); align = uniform variance over the peer ROLE
sub-embeddings; replay content-blind; no metric-read/branch. FENCE: iterative gradient (Bet B legal).
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


class FactoredSeq2Seq(nn.Module):
    """role->structure, filler->identity. The architectural fix for Report 141's entanglement."""

    def __init__(self, in_vocab, out_vocab, verb_ids, role_dim=48, fill_dim=16, hidden=200, seed=0):
        super().__init__()
        torch.manual_seed(seed * 7919 + 1)
        self.r, self.f, self.H = role_dim, fill_dim, hidden
        self.in_emb = nn.Embedding(in_vocab, role_dim + fill_dim, padding_idx=PAD)
        self.enc = nn.GRU(role_dim, hidden, batch_first=True)           # ENCODER ON ROLE ONLY
        self.dec_emb = nn.Embedding(out_vocab, role_dim, padding_idx=PAD)
        self.dec = nn.GRU(role_dim, hidden, batch_first=True)
        self.attn_combine = nn.Linear(2 * hidden, hidden)
        # struct head over the NON-verb tokens + 1 verb-SLOT score (verb-agnostic timing); identity is NOT here
        self.struct_ids = torch.tensor([i for i in range(out_vocab) if i not in set(verb_ids)])
        self.verb_ids = torch.tensor(list(verb_ids))
        self.n_struct = self.struct_ids.numel()
        self.base = nn.Linear(hidden, self.n_struct + 1)               # last logit = "emit a verb here" (role-driven)
        self.fillsel = nn.Linear(fill_dim, len(verb_ids))             # WHICH verb (filler-driven)
        self.out_vocab = out_vocab

    def encode(self, src):
        e = self.in_emb(src)
        role, fill = e[..., :self.r], e[..., self.r:]
        enc_out, h = self.enc(role)                                    # (B,S,H), (1,B,H)
        return enc_out, fill, h

    def _logits(self, d, enc_out, fill_mem, src_mask):
        # attention over role-encoder; ctx_role -> structure, ctx_fill -> identity
        scores = torch.bmm(d, enc_out.transpose(1, 2)).squeeze(1)      # (B,S)
        scores = scores.masked_fill(~src_mask, float("-inf"))
        a = F.softmax(scores, dim=-1)                                  # (B,S)
        ctx_role = torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)       # (B,H)
        ctx_fill = torch.bmm(a.unsqueeze(1), fill_mem).squeeze(1)      # (B,f)
        h = torch.tanh(self.attn_combine(torch.cat([d.squeeze(1), ctx_role], -1)))  # (B,H)
        base = self.base(h)                                            # (B, n_struct+1)
        fillsel = self.fillsel(ctx_fill)                              # (B, n_verbs)
        B = base.shape[0]
        logits = torch.zeros(B, self.out_vocab, device=base.device)
        logits = logits.index_copy(1, self.struct_ids.to(base.device), base[:, :self.n_struct])
        verb_logits = base[:, self.n_struct:self.n_struct + 1] + fillsel   # verb-slot score + which-verb
        logits = logits.index_copy(1, self.verb_ids.to(base.device), verb_logits)
        return logits

    def forward(self, src, src_mask, tgt, tf_ratio, gen):
        enc_out, fill_mem, h = self.encode(src)
        B, T = tgt.shape
        y = torch.full((B,), SOS, dtype=torch.long, device=src.device)
        out = []
        for t in range(T):
            d, h = self.dec(self.dec_emb(y).unsqueeze(1), h)
            lg = self._logits(d, enc_out, fill_mem, src_mask)
            out.append(lg)
            teacher = torch.rand(1, generator=gen).item() < tf_ratio
            y = tgt[:, t] if teacher else lg.argmax(-1)
        return torch.stack(out, 1)

    @torch.no_grad()
    def greedy(self, src, src_mask, max_len):
        enc_out, fill_mem, h = self.encode(src)
        B = src.shape[0]
        y = torch.full((B,), SOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        seqs = []
        for _ in range(max_len):
            d, h = self.dec(self.dec_emb(y).unsqueeze(1), h)
            y = self._logits(d, enc_out, fill_mem, src_mask).argmax(-1)
            y = y.masked_fill(done, PAD)
            seqs.append(y.clone()); done = done | (y == EOS)
            if done.all():
                break
        return torch.stack(seqs, 1)


def train(model, pairs, in_vocab, out_vocab, *, epochs, batch_size, lr, tf_ratio, seed, device, params=None):
    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed * 104729 + 7)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for src, tgt in make_batches(pairs, in_vocab, out_vocab, batch_size, gen):
            src, _ = pad(src, device); tgt, _ = pad(tgt, device)
            logits = model(src, src != PAD, tgt, tf_ratio, gen)
            loss = F.cross_entropy(logits.reshape(-1, model.out_vocab), tgt.reshape(-1), ignore_index=PAD)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        print(f"    epoch {ep + 1}/{epochs} loss={tot:.3f}", file=sys.stderr, flush=True)


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
    """Offline pass: align the ROLE sub-embedding of the peer set (uniform variance) + replay. Identity (fill)
    is protected ARCHITECTURALLY (read from a separate channel), so aligning role should NOT break it."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.in_emb.weight.requires_grad_(True)
    opt = torch.optim.Adam([model.in_emb.weight], lr=lr); model.train()
    bt = None
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
        role = model.in_emb.weight[peer_ids, :model.r]                 # ROLE half of the peer embeddings
        loss = loss + lam * ((role - role.mean(0, keepdim=True)) ** 2).sum(1).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([model.in_emb.weight], 5.0); opt.step()
    for p in model.parameters():
        p.requires_grad_(True)


def run(args, seed):
    train_pairs = load_pairs(str(DATA / "tasks_train_addprim_jump.txt"))
    test_pairs = load_pairs(str(DATA / "tasks_test_addprim_jump.txt"))
    allp = train_pairs + test_pairs
    in_vocab, _ = build_vocab(allp, "in"); out_vocab, _ = build_vocab(allp, "out")
    max_len = max(len(p[1]) for p in allp) + 2
    peer = {p[0][0] for p in train_pairs if len(p[0]) == 1}
    assert peer == {"jump", "walk", "run", "look"}, f"peer -> {peer}"          # CONDITION 1
    peer_ids = sorted(in_vocab[t] for t in peer)
    verb_ids = [out_vocab[t] for t in VERB_ACTIONS]

    g = torch.Generator().manual_seed(seed * 104729 + 7)
    model = FactoredSeq2Seq(len(in_vocab), len(out_vocab), verb_ids, args.role_dim, args.fill_dim,
                            args.hidden, seed).to(args.device)
    train(model, train_pairs, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, tf_ratio=args.tf_ratio, seed=seed, device=args.device)
    em = lambda: exact_match(model, test_pairs, in_vocab, out_vocab, max_len=max_len,
                             batch_size=args.batch_size, device=args.device)
    acc_base = em()
    print(f"  [factored baseline seed {seed}] jump-split = {acc_base:.4f}", file=sys.stderr, flush=True)

    consolidate_role(model, train_pairs, in_vocab, out_vocab, peer_ids, lam=args.lam, steps=args.consol_steps,
                     batch_size=args.batch_size, lr=args.consol_lr, gen=g, device=args.device)
    acc_consol = em()
    print(f"  [factored consolidation seed {seed}] jump-split = {acc_consol:.4f}", file=sys.stderr, flush=True)
    return acc_base, acc_consol


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
    ap.add_argument("--consol-steps", type=int, default=500, dest="consol_steps")
    ap.add_argument("--consol-lr", type=float, default=1e-3, dest="consol_lr")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs, args.consol_steps = 1, 15, 200

    base, consol = [], []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        b, c = run(args, seed); base.append(b); consol.append(c)

    def ci(v):
        t = torch.tensor(v); return {"mean": float(t.mean()), "min": float(t.min()), "max": float(t.max()), "per_seed": v}
    out = {"experiment": "89_betb_scan_factored (Bet B / SCAN Stage 1 iterate 2 — architectural role+filler)",
           "config": vars(args), "factored_baseline": ci(base), "factored_consolidation": ci(consol),
           "delta_consol_minus_base": ci([consol[i] - base[i] for i in range(len(base))])}
    out["verdict"] = (f"FACTORED: baseline={out['factored_baseline']['mean']:.4f} "
                      f"consolidation={out['factored_consolidation']['mean']:.4f} "
                      f"(Δ={out['delta_consol_minus_base']['mean']:+.4f}). "
                      "OUTCOME A (baseline already high) = architectural fix, consolidation not hero. "
                      "OUTCOME B (baseline low, consolidation lifts) = CONSOLIDATION IS THE HERO, thesis survives.")
    print(json.dumps(out, indent=2))
    print(f"\n=== {out['verdict']}", file=sys.stderr, flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
