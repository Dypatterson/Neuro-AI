"""experiments/99 — Bet B / SCAN-MCD Lever 3: FACTORED SUBSTRATE × clause-composition consolidation.

CHARTER: notes/betb-mcd-lever3-factored-precommit.md (anti-homunculus PASS-WITH-FIXES, 2026-06-21).
The user-chosen Lever 3 (RETROSPECTIVE-addendum-2026-06-17 §4): carry Report 142's substrate-shape
lever to the GECA-resistant MCD arena (146). The LITERAL 142 port is dead (MCD's primitive-filler
axis is saturated; mcd2 = 0% filler-holes; 142's len==1 peer set is EMPTY on mcd1/mcd3 — see
experiments/analyze_mcd_divergence_axis.py). The FAITHFUL reframe (this file): carry 142's factored
substrate (structure ROLE channel ⊥ content FILL channel; encoder on role only; a separate content
head) but match the CONSOLIDATION to MCD's actual axis (clause COMPOSITION = the CCC operation that
NULLED on the holistic substrate, Report 147).

THE 2×2 the retrospective asks for (isolates SUBSTRATE-SHAPE as the lever):
                 - consolidation            + CCC consolidation
  holistic       vanilla_plain (~0.17)      ccc           (147: NULLS, -0.029)
  factored       factored_baseline          factored_ccc  (THE arm)
Thesis-alive = consolidation INERT on holistic (re-confirms 147) AND LOAD-BEARING on factored.

FIXES folded from the anti-homunculus review:
  (1) the content/filler partition is DISCOVERED in-code by a biconditional input<->output predicate
      and asserted (CONDITION 1) — never hardcoded.
  (2) random_partition / *_random / *_nosplit are CO-PRIMARY arms at n=8 + CI (the false-positive
      deciders), not positive-only follow-ups.
  (3) verb_ids/struct_ids are derived from the DISCOVERED predicate (not exp89's 4-verb literal,
      which would mis-route left/right).
  (4) premise gate is pinned: TRAIN-EM >= 0.90 AND factored_baseline test-EM >= 0.10.

HEADLINE: factored_ccc - factored_baseline CI-disjoint > 0 AND factored_ccc >= vanilla_geca, n>=8,
mcd1 (+ mcd2). Controls that must NULL/separate: random_partition, factored_ccc_random,
factored_ccc_nosplit. Honest prior LOW (no emergent MCD winner in the lit; leans structure-injecting);
the high-value most-likely outcome is the interpretable null that retires 142 as the missing lever.
"""
from __future__ import annotations

import argparse
import copy
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


# ---- problem-generic content discovery (FIX 1 / CONDITION 1) ----------------------------------------------

def discover_content(tr, in_vocab, out_vocab):
    """Problem-generic, computed-in-code: an input token t is CONTENT iff some output token o has a
    PERFECT biconditional co-occurrence over the corpus ({i: t in cmd_i} == {i: o in out_i}). No
    standalone-command / hardcoded list needed. On SCAN-MCD this discovers
    {jump,walk,run,look,left,right} <-> {I_JUMP,I_WALK,I_RUN,I_LOOK,I_TURN_LEFT,I_TURN_RIGHT}.
    Returns (sorted in_content_ids, sorted out_content_ids=verb_ids, mapping-for-logging)."""
    in_toks = sorted({t for c, _ in tr for t in c})
    out_toks = sorted({o for _, a in tr for o in a})
    cmd_has = {t: set() for t in in_toks}
    out_has = {o: set() for o in out_toks}
    for i, (c, a) in enumerate(tr):
        for t in set(c):
            cmd_has[t].add(i)
        for o in set(a):
            out_has[o].add(i)
    mapping = {}
    for t in in_toks:
        for o in out_toks:
            if cmd_has[t] and out_has[o] and cmd_has[t] == out_has[o]:
                mapping[t] = o
    assert mapping, "CONDITION 1: biconditional content discovery returned EMPTY — refusing to run"
    in_content_ids = sorted(in_vocab[t] for t in mapping)
    out_content_ids = sorted(out_vocab[o] for o in mapping.values())
    return in_content_ids, out_content_ids, mapping


# ---- factored substrate (ported + generalized from experiments/89) -----------------------------------------

class FactoredSeq2Seq(nn.Module):
    """role -> structure/skeleton (encoder on role only); fill -> which-action (separate content head).
    verb_ids = the DISCOVERED output content tokens (FIX 3). W_* are the clause-composition algebra used
    ONLY by the consolidation loss, so the standard forward path is the matched-capacity factored_baseline."""

    def __init__(self, in_vocab, out_vocab, verb_ids, role_dim=48, fill_dim=16, hidden=200, seed=0):
        super().__init__()
        torch.manual_seed(seed * 7919 + 1)
        self.r, self.f, self.H = role_dim, fill_dim, hidden
        self.in_emb = nn.Embedding(in_vocab, role_dim + fill_dim, padding_idx=PAD)
        self.enc = nn.GRU(role_dim, hidden, batch_first=True)                  # ENCODER ON ROLE ONLY
        self.dec_emb = nn.Embedding(out_vocab, role_dim, padding_idx=PAD)
        self.dec = nn.GRU(role_dim, hidden, batch_first=True)
        self.attn_combine = nn.Linear(2 * hidden, hidden)
        vset = set(verb_ids)
        self.register_buffer("struct_ids", torch.tensor([i for i in range(out_vocab) if i not in vset]))
        self.register_buffer("verb_ids", torch.tensor(list(verb_ids)))
        self.n_struct = self.struct_ids.numel()
        self.base = nn.Linear(hidden, self.n_struct + 1)                       # non-content tokens + 1 "emit-action" slot
        self.fillsel = nn.Linear(fill_dim, len(verb_ids))                      # WHICH action (content head)
        self.out_vocab = out_vocab
        self.W_and = nn.Linear(2 * hidden, hidden)                             # clause-composition algebra (consolidation only)
        self.W_after = nn.Linear(2 * hidden, hidden)
        self.W_unary = nn.Linear(hidden, hidden)

    def encode(self, src):
        e = self.in_emb(src)
        role, fill = e[..., :self.r], e[..., self.r:]
        enc_out, h = self.enc(role)
        return enc_out, fill, h

    def _logits(self, d, enc_out, fill_mem, src_mask):
        scores = torch.bmm(d, enc_out.transpose(1, 2)).squeeze(1).masked_fill(~src_mask, float("-inf"))
        a = F.softmax(scores, dim=-1)
        ctx_role = torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)
        ctx_fill = torch.bmm(a.unsqueeze(1), fill_mem).squeeze(1)
        hcomb = torch.tanh(self.attn_combine(torch.cat([d.squeeze(1), ctx_role], -1)))
        base = self.base(hcomb)
        fillsel = self.fillsel(ctx_fill)
        B = base.shape[0]
        logits = torch.zeros(B, self.out_vocab, device=base.device)
        logits = logits.index_copy(1, self.struct_ids, base[:, :self.n_struct])
        verb_logits = base[:, self.n_struct:self.n_struct + 1] + fillsel
        logits = logits.index_copy(1, self.verb_ids, verb_logits)
        return logits

    def dec_step(self, y_prev, h, enc_out, fill_mem, src_mask):
        d, h = self.dec(self.dec_emb(y_prev).unsqueeze(1), h)
        return self._logits(d, enc_out, fill_mem, src_mask), h

    def forward(self, src, src_mask, tgt, tf_ratio, gen, init_h=None):
        enc_out, fill_mem, h = self.encode(src)
        if init_h is not None:
            h = init_h
        B, T = tgt.shape
        y = torch.full((B,), SOS, dtype=torch.long, device=src.device)
        out = []
        for t in range(T):
            lg, h = self.dec_step(y, h, enc_out, fill_mem, src_mask)
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
            lg, h = self.dec_step(y, h, enc_out, fill_mem, src_mask)
            y = lg.argmax(-1).masked_fill(done, PAD)
            seqs.append(y.clone()); done = done | (y == EOS)
            if done.all():
                break
        return torch.stack(seqs, 1)


# ---- factored clause-composition consolidation (ported from exp94.consolidate_ccc) -------------------------

def _decode_ce_factored(model, enc_out, fill_mem, src_mask, init_h, tgt):
    """Teacher-forced decode from init_h ((1,B,H)) through the FROZEN factored decoder; CE vs gold."""
    B, T = tgt.shape
    h = init_h
    y = torch.full((B,), SOS, dtype=torch.long, device=enc_out.device)
    logits = []
    for t in range(T):
        lg, h = model.dec_step(y, h, enc_out, fill_mem, src_mask)
        logits.append(lg)
        y = tgt[:, t]
    return F.cross_entropy(torch.stack(logits, 1).reshape(-1, model.out_vocab), tgt.reshape(-1), ignore_index=PAD)


def consolidate_ccc_factored(model, pairs, in_vocab, out_vocab, *, parse_mode, and_id, after_id,
                             beta, gamma, delta, steps, batch_size, lr, gen, device):
    """Offline: decoder FROZEN; encoder (in_emb + role GRU) + W_* unfrozen. Force the holistic role-state
    h to be a learned composition of ROLE-channel clause-pools + decode-consistency + anti-collapse.
    Restructures the STRUCTURE/skeleton channel; content preserved in the fill channel. parse_mode in
    {conj, random, nosplit}. Reuses exp94.compute_ecomp over the ROLE encoder output (model has W_*)."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.in_emb.weight.requires_grad_(True)
    for p in model.enc.parameters():
        p.requires_grad_(True)
    for mod in (model.W_and, model.W_after, model.W_unary):
        for p in mod.parameters():
            p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train(); bt = None
    for _ in range(steps):
        if bt is None:
            bt = iter(make_batches(pairs, in_vocab, out_vocab, batch_size, gen))
        try:
            src, tgt = next(bt)
        except StopIteration:
            bt = iter(make_batches(pairs, in_vocab, out_vocab, batch_size, gen)); src, tgt = next(bt)
        src, _ = pad(src, device); tgt, _ = pad(tgt, device); src_mask = src != PAD
        enc_out, fill_mem, h = model.encode(src)                               # role encoder
        ce_whole = _decode_ce_factored(model, enc_out, fill_mem, src_mask, h, tgt)
        e_comp = exp94.compute_ecomp(model, enc_out, src, src_mask, parse_mode=parse_mode,
                                     and_id=and_id, after_id=after_id, gen=gen)
        L_self = ((h.squeeze(0) - e_comp) ** 2).sum(-1).mean()
        ce_comp = _decode_ce_factored(model, enc_out, fill_mem, src_mask, e_comp.unsqueeze(0), tgt)
        perm = torch.randperm(e_comp.shape[0], generator=gen).to(e_comp.device)
        dd = ((e_comp - e_comp[perm]) ** 2).sum(-1)
        L_hinge = F.relu(1.0 - dd).mean()
        loss = ce_whole + beta * L_self + gamma * ce_comp + delta * L_hinge
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0); opt.step()
    for p in model.parameters():
        p.requires_grad_(True)


# ---- helpers ----------------------------------------------------------------------------------------------

def load_mcd(split):
    return (load_pairs(str(MCD / f"tasks_train_{split}.txt")), load_pairs(str(MCD / f"tasks_test_{split}.txt")))


def random_partition_ids(out_vocab_size, n_content, gen):
    """A RANDOM size-matched partition of the NON-pad output tokens into the content (fill) head."""
    cand = [i for i in range(out_vocab_size) if i != PAD]
    perm = torch.randperm(len(cand), generator=gen).tolist()
    return sorted(cand[perm[i]] for i in range(n_content))


def train_em(model, tr, in_vocab, out_vocab, *, max_len, batch_size, device, gen, n=1000):
    """In-distribution competence: exact-match on a random n-sample of TRAIN (premise-gate #1)."""
    idx = torch.randperm(len(tr), generator=gen).tolist()[:n]
    sample = [tr[i] for i in idx]
    return exp87.exact_match(model, sample, in_vocab, out_vocab, max_len=max_len, batch_size=batch_size, device=device)


# ---- run ----------------------------------------------------------------------------------------------------

def run_seed(args, seed, split, arms):
    tr, te = load_mcd(split)
    allp = tr + te
    in_vocab, _ = build_vocab(allp, "in"); out_vocab, _ = build_vocab(allp, "out")
    max_len = max(len(p[1]) for p in allp) + 2
    and_id, after_id = in_vocab["and"], in_vocab["after"]
    in_content, verb_ids, mapping = discover_content(tr, in_vocab, out_vocab)   # CONDITION 1
    res, extra = {}, {}
    extra["discovered_content"] = mapping

    def em(m, greedy_model=None):
        return exp87.exact_match(m, te, in_vocab, out_vocab, max_len=max_len, batch_size=args.batch_size, device=args.device)

    # --- holistic arms (the top row of the 2x2; re-confirms 147 internally with matched seeds) ---
    if "vanilla_plain" in arms:
        vm = exp87.Seq2Seq(len(in_vocab), len(out_vocab), args.embed, args.hidden, seed).to(args.device)
        exp87.train(vm, tr, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    tf_ratio=args.tf_ratio, seed=seed, device=args.device)
        res["vanilla_plain"] = em(vm)
        print(f"  [{split} s{seed}] vanilla_plain = {res['vanilla_plain']:.4f}", file=sys.stderr, flush=True)
    if "vanilla_geca" in arms:
        tr_geca, _ = exp92.geca_augment(tr, [], args.min_shared_env)
        gm = exp87.Seq2Seq(len(in_vocab), len(out_vocab), args.embed, args.hidden, seed).to(args.device)
        exp87.train(gm, tr_geca, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    tf_ratio=args.tf_ratio, seed=seed, device=args.device)
        res["vanilla_geca"] = em(gm)
        print(f"  [{split} s{seed}] vanilla_geca = {res['vanilla_geca']:.4f}", file=sys.stderr, flush=True)
    if "ccc" in arms:                                                         # holistic + CCC consolidation
        hm = exp94.CCCSeq2Seq(len(in_vocab), len(out_vocab), args.embed, args.hidden, seed).to(args.device)
        exp87.train(hm, tr, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    tf_ratio=args.tf_ratio, seed=seed, device=args.device)
        cgen = torch.Generator().manual_seed(seed * 100003 + 11)
        exp94.consolidate_ccc(hm, tr, in_vocab, out_vocab, parse_mode="conj", and_id=and_id, after_id=after_id,
                              beta=args.beta, gamma=args.gamma, delta=args.delta, steps=args.consol_steps,
                              batch_size=args.batch_size, lr=args.consol_lr, gen=cgen, device=args.device)
        res["ccc"] = em(hm)
        print(f"  [{split} s{seed}] ccc(holistic) = {res['ccc']:.4f}", file=sys.stderr, flush=True)

    # --- factored arms (the bottom row) ---
    factored_modes = {"factored_ccc": ("conj", verb_ids), "factored_ccc_random": ("random", verb_ids),
                      "factored_ccc_nosplit": ("nosplit", verb_ids)}
    want_factored = [a for a in arms if a in factored_modes] + (["random_partition"] if "random_partition" in arms else [])
    need_base = "factored_baseline" in arms or want_factored
    if need_base:
        fm = FactoredSeq2Seq(len(in_vocab), len(out_vocab), verb_ids, args.role_dim, args.fill_dim,
                             args.hidden, seed).to(args.device)
        exp87.train(fm, tr, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    tf_ratio=args.tf_ratio, seed=seed, device=args.device)
        if "factored_baseline" in arms or args.premise:
            res["factored_baseline"] = em(fm)
            tgen = torch.Generator().manual_seed(seed * 100003 + 11)
            res["_train_em"] = train_em(fm, tr, in_vocab, out_vocab, max_len=max_len, batch_size=args.batch_size,
                                        device=args.device, gen=tgen)
            print(f"  [{split} s{seed}] factored_baseline = {res['factored_baseline']:.4f} "
                  f"(train-EM={res['_train_em']:.4f})", file=sys.stderr, flush=True)
        snap = copy.deepcopy(fm.state_dict())
        for arm in [a for a in arms if a in factored_modes]:
            mode, vids = factored_modes[arm]
            fm.load_state_dict(snap)
            cgen = torch.Generator().manual_seed(seed * 100003 + 11)
            consolidate_ccc_factored(fm, tr, in_vocab, out_vocab, parse_mode=mode, and_id=and_id, after_id=after_id,
                                     beta=args.beta, gamma=args.gamma, delta=args.delta, steps=args.consol_steps,
                                     batch_size=args.batch_size, lr=args.consol_lr, gen=cgen, device=args.device)
            res[arm] = em(fm)
            print(f"  [{split} s{seed}] {arm} = {res[arm]:.4f}", file=sys.stderr, flush=True)

    # --- random_partition control (FIX 2): factored substrate w/ a RANDOM content routing, + CCC consol ---
    if "random_partition" in arms:
        pgen = torch.Generator().manual_seed(seed * 100003 + 11)
        rverb = random_partition_ids(len(out_vocab), len(verb_ids), pgen)
        rm = FactoredSeq2Seq(len(in_vocab), len(out_vocab), rverb, args.role_dim, args.fill_dim,
                             args.hidden, seed).to(args.device)
        exp87.train(rm, tr, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    tf_ratio=args.tf_ratio, seed=seed, device=args.device)
        cgen = torch.Generator().manual_seed(seed * 100003 + 11)
        consolidate_ccc_factored(rm, tr, in_vocab, out_vocab, parse_mode="conj", and_id=and_id, after_id=after_id,
                                 beta=args.beta, gamma=args.gamma, delta=args.delta, steps=args.consol_steps,
                                 batch_size=args.batch_size, lr=args.consol_lr, gen=cgen, device=args.device)
        res["random_partition"] = em(rm)
        extra["random_partition_verb_ids"] = rverb
        print(f"  [{split} s{seed}] random_partition = {res['random_partition']:.4f}", file=sys.stderr, flush=True)

    res["_extra"] = extra
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role-dim", type=int, default=48, dest="role_dim")
    ap.add_argument("--fill-dim", type=int, default=16, dest="fill_dim")
    ap.add_argument("--embed", type=int, default=64)        # holistic arms' embedding (= role+fill)
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tf-ratio", type=float, default=0.5, dest="tf_ratio")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=3.0)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--consol-steps", type=int, default=800, dest="consol_steps")
    ap.add_argument("--consol-lr", type=float, default=1e-3, dest="consol_lr")
    ap.add_argument("--min-shared-env", type=int, default=3, dest="min_shared_env")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--splits", default="mcd1,mcd2")
    ap.add_argument("--arms", default="vanilla_plain,ccc,factored_baseline,factored_ccc,"
                                      "factored_ccc_random,factored_ccc_nosplit,random_partition,vanilla_geca")
    ap.add_argument("--premise", action="store_true", help="premise-gate only: factored_baseline + train-EM, 1-2 seeds")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.premise:
        args.arms = "factored_baseline"
        if args.seeds > 2:
            args.seeds = 2
    if args.smoke:
        args.seeds, args.epochs, args.consol_steps = 1, 10, 150
        args.splits = "mcd1"
        if not args.premise:
            args.arms = "vanilla_plain,factored_baseline,factored_ccc,factored_ccc_nosplit"

    arms = args.arms.split(",")
    out = {"experiment": "99_betb_scan_mcd_factored (Lever 3: factored substrate x clause-composition consolidation)",
           "charter": "notes/betb-mcd-lever3-factored-precommit.md", "config": vars(args),
           "floor_ref": {"vanilla": "0.17/0.14/0.02 in-house", "GECA": "51/30/12 (Conklin 2021)",
                         "ceiling_structure_injecting": "AuxSeq 99.9/90/98, LeAR 100/100/100"},
           "by_split": {}}
    for split in args.splits.split(","):
        per = {a: [] for a in arms}
        train_ems, disc = [], None
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            r = run_seed(args, seed, split, arms)
            disc = r.get("_extra", {}).get("discovered_content", disc)
            if "_train_em" in r:
                train_ems.append(r["_train_em"])
            for a in arms:
                if a in r:
                    per[a].append(r[a])
        sdict = {a: exp94.summary(per[a]) for a in arms if per[a]}
        deltas = {}
        pairs_to_test = [("factored_ccc", "factored_baseline"), ("factored_ccc", "vanilla_geca"),
                         ("factored_ccc", "factored_ccc_random"), ("factored_ccc", "factored_ccc_nosplit"),
                         ("factored_ccc", "random_partition"), ("ccc", "vanilla_plain")]
        for a, b in pairs_to_test:
            if per.get(a) and per.get(b) and len(per[a]) == len(per[b]) and len(per[a]) > 1:
                deltas[f"{a}_minus_{b}"] = exp94.paired_delta(per[a], per[b])
        block = {"arms": sdict, "paired_deltas": deltas, "discovered_content": disc}
        if train_ems:
            block["factored_train_em"] = {"mean": sum(train_ems) / len(train_ems), "per_seed": train_ems}
            block["PREMISE_GATE"] = {
                "train_em_mean": block["factored_train_em"]["mean"],
                "factored_baseline_mean": sdict.get("factored_baseline", {}).get("mean"),
                "PASS_train>=0.90_AND_base>=0.10": bool(block["factored_train_em"]["mean"] >= 0.90
                                                        and sdict.get("factored_baseline", {}).get("mean", 0) >= 0.10),
            }
        out["by_split"][split] = block
        print(f"\n=== {split}: " + " ".join(f"{a}={sdict[a]['mean']:.3f}" for a in sdict), file=sys.stderr, flush=True)
        if "PREMISE_GATE" in block:
            print(f"    PREMISE_GATE: {block['PREMISE_GATE']}", file=sys.stderr, flush=True)
    print(json.dumps(out, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
