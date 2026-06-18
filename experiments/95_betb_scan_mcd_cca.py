"""experiments/95 — Bet B / SCAN-MCD Stage-1 LEVER 1: Clause-Compositional ARCHITECTURE (composition-as-INFERENCE).

CHARTER: notes/RETROSPECTIVE-addendum-2026-06-17-first-discriminating-test.md §4 lever 1 +
notes/betb-mcd-stage1-mechanism-precommit.md. RESOLVES the Report-147 (CCC) confound.

THE CCC CONFOUND (Report 147 §verdict + precommit §6): CCC consolidated a TRAINED model by forcing the encoder
state h -> e_comp with the DECODER FROZEN, so (a) the decoder was never trained on the composed state and (b)
the composition was a REGULARIZER TARGET, never the path that produces the answer. The frozen decoder kept
decoding the standard h better, and the consolidation hurt. The composition hypothesis was tested only as a
regularizer-on-top-of-pooling, NOT as the inference mechanism.

THE FIX (this file): make e_comp the INFERENCE PATH. A seq2seq whose decoder is CONDITIONED (initial state +
per-step input concat) on a representation that is EITHER the composed clause representation e_comp (the
mechanism) OR the holistic encoder state h (the matched-architecture baseline). Trained END-TO-END (decoder NOT
frozen). The ONLY difference between cca and cca_holistic is composed-vs-holistic conditioning -> isolates
whether composing the clause structure into the decode path improves compositional generalization. The parser
(content-blind conjunction split) + learned compose operator {W_AND, W_AFTER} are the exp94 ones, anti-homunculus
PASS-WITH-FIXES; here they live in the FORWARD pass (a problem-generic compositional architecture, CONTEXT.md §3
structural-prior boundary) rather than a consolidation regularizer. (Escalation to hierarchical
decode-each-clause-and-concatenate would be MORE structure-injecting and needs a fresh anti-homunculus pass.)

PREAMBLE (CLAUDE.md):
  Active capability: Bet-B Stage-1 mechanism on the GECA-resistant MCD arena (Report 146), resolving the 147 confound.
  Headline per RETROSPECTIVE-addendum §4 lever 1: cca - cca_holistic CI-disjoint > 0 AND cca >= faithful GECA, >=8 seeds.
  Controls: vanilla_plain floor (exp87) ; cca_holistic (matched arch, holistic conditioning) ; cca_random (must NULL) ;
            cca_nosplit (composition-vs-pooling separator) ; vanilla_geca (redundancy decider).
  Last verified: Report 147 (CCC consolidation-as-regularizer NULLS/hurts).
  Why now: user chose 'resolve the confound first' (RETROSPECTIVE-addendum §5 lean). If cca ALSO nulls with the
           architecture confound removed, the 'wall' reading is earned.
HONEST PRIOR: composition-as-init may be ignorable via attention over enc_out -> cca could ~ baseline (null). The
per-step e_comp conditioning is added to give the composition teeth; if it still nulls, that is the decisive datum.
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
exp92 = _load("exp92", "92_betb_scan_compound_stage0.py")
exp94 = _load("exp94", "94_betb_scan_mcd_ccc.py")
load_pairs, build_vocab, to_ids, pad, make_batches = (exp87.load_pairs, exp87.build_vocab, exp87.to_ids,
                                                      exp87.pad, exp87.make_batches)
PAD, SOS, EOS = exp87.PAD, exp87.SOS, exp87.EOS
MCD = REPO / "data" / "scan" / "mcd_split"


# ---- model: decoder conditioned (init + per-step) on a composed/holistic vector --------------------------

class CondAttnDecoder(nn.Module):
    """exp87 AttnDecoder, but the GRU input is [emb(y_prev) ; cond] where cond is the (fixed-per-command)
    conditioning vector (composed clause rep e_comp, or holistic h). Attention over enc_out is unchanged."""

    def __init__(self, vocab, embed, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed, padding_idx=PAD)
        self.gru = nn.GRU(embed + hidden, hidden, batch_first=True)
        self.attn_combine = nn.Linear(2 * hidden, hidden)
        self.out = nn.Linear(hidden, vocab)

    def step(self, y_prev, h, enc_out, src_mask, cond):
        inp = torch.cat([self.emb(y_prev), cond], -1).unsqueeze(1)        # (B,1,embed+H)
        o, h = self.gru(inp, h)
        scores = torch.bmm(o, enc_out.transpose(1, 2)).squeeze(1).masked_fill(~src_mask, float("-inf"))
        a = F.softmax(scores, dim=-1)
        ctx = torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)
        combined = torch.tanh(self.attn_combine(torch.cat([o.squeeze(1), ctx], -1)))
        return self.out(combined), h


class CCASeq2Seq(nn.Module):
    """use_comp=True: condition the decoder on the COMPOSED clause rep e_comp (parse_mode selects conj/random/
    nosplit). use_comp=False: condition on the HOLISTIC encoder state h (the matched-architecture baseline)."""

    def __init__(self, in_vocab, out_vocab, embed, hidden, seed, *, use_comp, parse_mode="conj"):
        super().__init__()
        torch.manual_seed(seed * 7919 + 1)
        self.enc = exp87.Encoder(in_vocab, embed, hidden)
        self.dec = CondAttnDecoder(out_vocab, embed, hidden)
        self.W_and = nn.Linear(2 * hidden, hidden)
        self.W_after = nn.Linear(2 * hidden, hidden)
        self.W_unary = nn.Linear(hidden, hidden)
        self.out_vocab = out_vocab
        self.use_comp, self.parse_mode = use_comp, parse_mode
        self.and_id = self.after_id = None                                # set per-run from vocab

    def _cond(self, enc_out, src, src_mask, gen):
        if not self.use_comp:
            return self._h_holistic                                       # holistic encoder state h
        return exp94.compute_ecomp(self, enc_out, src, src_mask, parse_mode=self.parse_mode,
                                   and_id=self.and_id, after_id=self.after_id, gen=gen)

    def forward(self, src, src_mask, tgt, tf_ratio, gen):
        enc_out, h = self.enc(src)
        self._h_holistic = h.squeeze(0)                                   # cache for holistic conditioning
        cond = self._cond(enc_out, src, src_mask, gen)
        dh = cond.unsqueeze(0).contiguous()                               # decoder init = conditioning vector
        B, T = tgt.shape
        y = torch.full((B,), SOS, dtype=torch.long, device=src.device); out = []
        for t in range(T):
            lg, dh = self.dec.step(y, dh, enc_out, src_mask, cond)
            out.append(lg)
            teacher = torch.rand(1, generator=gen).item() < tf_ratio
            y = tgt[:, t] if teacher else lg.argmax(-1)
        return torch.stack(out, 1)

    @torch.no_grad()
    def greedy(self, src, src_mask, max_len, gen):
        enc_out, h = self.enc(src)
        self._h_holistic = h.squeeze(0)
        cond = self._cond(enc_out, src, src_mask, gen)
        dh = cond.unsqueeze(0).contiguous(); B = src.shape[0]
        y = torch.full((B,), SOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device); seqs = []
        for _ in range(max_len):
            lg, dh = self.dec.step(y, dh, enc_out, src_mask, cond)
            y = lg.argmax(-1).masked_fill(done, PAD)
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
    gen = torch.Generator().manual_seed(0)
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


def load_mcd(split):
    return (load_pairs(str(MCD / f"tasks_train_{split}.txt")), load_pairs(str(MCD / f"tasks_test_{split}.txt")))


def run_seed(args, seed, split, arms):
    tr, te = load_mcd(split)
    allp = tr + te
    in_vocab, _ = build_vocab(allp, "in"); out_vocab, _ = build_vocab(allp, "out")
    max_len = max(len(p[1]) for p in allp) + 2
    and_id, after_id = in_vocab["and"], in_vocab["after"]
    res = {}

    def em(m):
        return exact_match(m, te, in_vocab, out_vocab, max_len=max_len, batch_size=args.batch_size, device=args.device)

    if "vanilla_plain" in arms:
        vm = exp87.Seq2Seq(len(in_vocab), len(out_vocab), args.embed, args.hidden, seed).to(args.device)
        exp87.train(vm, tr, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    tf_ratio=args.tf_ratio, seed=seed, device=args.device)
        res["vanilla_plain"] = exp87.exact_match(vm, te, in_vocab, out_vocab, max_len=max_len, batch_size=args.batch_size, device=args.device)
        print(f"  [{split} s{seed}] vanilla_plain = {res['vanilla_plain']:.4f}", file=sys.stderr, flush=True)

    if "vanilla_geca" in arms:
        tr_geca, _ = exp92.geca_augment(tr, [], args.min_shared_env)
        gm = exp87.Seq2Seq(len(in_vocab), len(out_vocab), args.embed, args.hidden, seed).to(args.device)
        exp87.train(gm, tr_geca, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    tf_ratio=args.tf_ratio, seed=seed, device=args.device)
        res["vanilla_geca"] = exp87.exact_match(gm, te, in_vocab, out_vocab, max_len=max_len, batch_size=args.batch_size, device=args.device)
        print(f"  [{split} s{seed}] vanilla_geca = {res['vanilla_geca']:.4f}", file=sys.stderr, flush=True)

    cca_specs = {"cca_holistic": (False, "conj"), "cca": (True, "conj"),
                 "cca_random": (True, "random"), "cca_nosplit": (True, "nosplit")}
    for arm, (use_comp, pm) in cca_specs.items():
        if arm not in arms:
            continue
        m = CCASeq2Seq(len(in_vocab), len(out_vocab), args.embed, args.hidden, seed, use_comp=use_comp, parse_mode=pm).to(args.device)
        m.and_id, m.after_id = and_id, after_id
        train(m, tr, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
              tf_ratio=args.tf_ratio, seed=seed, device=args.device)
        res[arm] = em(m)
        print(f"  [{split} s{seed}] {arm} = {res[arm]:.4f}", file=sys.stderr, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tf-ratio", type=float, default=0.5, dest="tf_ratio")
    ap.add_argument("--min-shared-env", type=int, default=3, dest="min_shared_env")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--splits", default="mcd1")
    ap.add_argument("--arms", default="vanilla_plain,cca_holistic,cca,cca_random,cca_nosplit,vanilla_geca")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs = 1, 10
        args.arms = "vanilla_plain,cca_holistic,cca"

    arms = args.arms.split(",")
    out = {"experiment": "95_betb_scan_mcd_cca (composition-as-INFERENCE; lever 1, resolves the 147 confound)",
           "charter": "notes/RETROSPECTIVE-addendum-2026-06-17-first-discriminating-test.md §4 lever 1",
           "config": vars(args), "by_split": {}}
    for split in args.splits.split(","):
        per = {a: [] for a in arms}
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            r = run_seed(args, seed, split, arms)
            for a in arms:
                if a in r:
                    per[a].append(r[a])
        sdict = {a: exp94.summary(per[a]) for a in arms if per[a]}
        deltas = {}
        for a, b in [("cca", "cca_holistic"), ("cca", "vanilla_plain"), ("cca", "cca_random"),
                     ("cca", "cca_nosplit"), ("cca", "vanilla_geca")]:
            if per.get(a) and per.get(b) and len(per[a]) == len(per[b]):
                deltas[f"{a}_minus_{b}"] = exp94.paired_delta(per[a], per[b])
        out["by_split"][split] = {"arms": sdict, "paired_deltas": deltas}
        print(f"\n=== {split}: " + " ".join(f"{a}={sdict[a]['mean']:.3f}" for a in sdict), file=sys.stderr, flush=True)

    print(json.dumps(out, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
