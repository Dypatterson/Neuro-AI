"""experiments/94 — Bet B / SCAN-MCD Stage-1 mechanism: Clause-Compositional Consolidation (CCC).

CHARTER: notes/betb-mcd-stage1-mechanism-precommit.md (anti-homunculus PASS-WITH-FIXES, 2026-06-17) +
Report 146 (MCD Stage-0 PASS → GECA-resistant arena). Mechanism chosen by a 4-angle design panel
(wf_dff3f687) + a decisive data verification: MCD's hardness is the CONFIGURATION/COMBINATION axis (100% of
mcd1-test templates novel; ~all 2-clause and/after conjunctions with novel clause-combinations), NOT the
verb-filler axis (saturated except add-jump-shaped holes). So the mechanism targets clause COMPOSITION, not
filler-invariance.

THE MECHANISM (CCC): a fixed content-blind parser splits each command at its single top-level conjunction
(and/after) into clause_L/clause_R. A learned per-conjunction compose operator forms
  e_comp = W_conj @ [pool(enc_out[L]) ; pool(enc_out[R])]   (single-clause -> e_comp = pool(whole), fall-through).
An offline consolidation (decoder FROZEN; encoder+embedding+W unfrozen) minimizes a fixed local loss:
  L = CE_whole + beta*||h - e_comp||^2 (self-consistency) + gamma*CE(decode_from(e_comp), gold) (decode-consistency)
      + delta*hinge(e_comp apart)  (anti-collapse).
Forcing the holistic encoder state h to be a learned composition of clause-pools (and to decode correctly) puts
novel clause COMBINATIONS on the seen manifold -> the recombination GECA cannot reach (it needs surface-
substitutable compounds; Report 146: ~0% MCD reach). beta=gamma=delta=0 => byte-identical to the baseline.

ARMS (per precommit §4 + the anti-homunculus-mandated controls):
  vanilla_plain   exp87 Seq2Seq floor (in-house ~0.17/0.14/0.02)
  ccc_baseline    CCC model, standard train, NO consolidation (matched-capacity baseline; ~= vanilla)
  ccc             consolidation with the CONJUNCTION split                          (THE MECHANISM)
  ccc_random      consolidation with a RANDOM interior split (same W keyed by real conj) -> MUST NULL
  ccc_nosplit     consolidation with e_comp = W_unary @ pool(whole), NO split        -> must NOT match ccc
  vanilla_geca    vanilla + in-house token-GECA (redundancy decider; published strong GECA 51/30/12 cited)
HEADLINE: ccc - ccc_baseline CI-disjoint > 0 AND ccc >= vanilla_geca, >=8 seeds, on >= mcd1.
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
load_pairs, build_vocab, to_ids, pad, make_batches = (exp87.load_pairs, exp87.build_vocab, exp87.to_ids,
                                                      exp87.pad, exp87.make_batches)
PAD, SOS, EOS = exp87.PAD, exp87.SOS, exp87.EOS
MCD = REPO / "data" / "scan" / "mcd_split"


# ---- model -----------------------------------------------------------------------------------------------

class CCCSeq2Seq(exp87.Seq2Seq):
    """exp87 Seq2Seq + a learned compose algebra used ONLY by the CCC consolidation loss.
    The standard forward/greedy path is inherited unchanged => ccc_baseline (no consolidation) is the matched-
    capacity baseline; the W_* params have zero gradient in standard training (not in the forward graph)."""

    def __init__(self, in_vocab, out_vocab, embed=64, hidden=200, seed=0):
        super().__init__(in_vocab, out_vocab, embed, hidden, seed)   # seeds + builds enc/dec identically to exp87
        self.W_and = nn.Linear(2 * hidden, hidden)
        self.W_after = nn.Linear(2 * hidden, hidden)
        self.W_unary = nn.Linear(hidden, hidden)


def _masked_pool(x, mask):
    # x (B,S,H), mask (B,S) bool -> (B,H) mean over masked positions
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


def compute_ecomp(model, enc_out, src, src_mask, *, parse_mode, and_id, after_id, gen):
    B, S, H = enc_out.shape
    dev = enc_out.device
    pos = torch.arange(S, device=dev)
    is_conj = (src == and_id) | (src == after_id)               # (B,S)
    conj_present = is_conj.any(1)                                # (B,)
    conj_pos = is_conj.float().argmax(1)                         # first conj pos (0 if none)
    is_after = src.gather(1, conj_pos.unsqueeze(1)).squeeze(1) == after_id

    pool_whole = _masked_pool(enc_out, src_mask)
    if parse_mode == "nosplit":
        return model.W_unary(pool_whole)

    split_pos = conj_pos
    if parse_mode == "random":
        lengths = src_mask.sum(1)                                # (B,)
        u = torch.rand(B, generator=gen).to(dev)                # CPU generator -> device (MPS gen unsupported)
        split_pos = (u * (lengths - 1).clamp(min=1).float()).long() + 1   # interior split in [1, len-1]
        split_pos = torch.minimum(split_pos, (lengths - 1).clamp(min=1))

    L_mask = (pos.unsqueeze(0) < split_pos.unsqueeze(1)) & src_mask & conj_present.unsqueeze(1)
    R_mask = (pos.unsqueeze(0) > split_pos.unsqueeze(1)) & src_mask & conj_present.unsqueeze(1)
    cat = torch.cat([_masked_pool(enc_out, L_mask), _masked_pool(enc_out, R_mask)], -1)   # (B,2H)
    e_conj = torch.where(is_after.unsqueeze(1), model.W_after(cat), model.W_and(cat))
    return torch.where(conj_present.unsqueeze(1), e_conj, pool_whole)   # single-clause fall-through


def _decode_ce(model, enc_out, src_mask, init_h, tgt):
    """Teacher-forced decode from init_h (=(1,B,H)); decoder params frozen by caller. Returns CE vs gold."""
    B, T = tgt.shape
    h = init_h
    y = torch.full((B,), SOS, dtype=torch.long, device=enc_out.device)
    logits = []
    for t in range(T):
        lg, h = model.dec.step(y, h, enc_out, src_mask)
        logits.append(lg)
        y = tgt[:, t]
    logits = torch.stack(logits, 1)
    return F.cross_entropy(logits.reshape(-1, model.out_vocab), tgt.reshape(-1), ignore_index=PAD)


def consolidate_ccc(model, pairs, in_vocab, out_vocab, *, parse_mode, and_id, after_id,
                    beta, gamma, delta, steps, batch_size, lr, gen, device):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.enc.parameters():                            # unfreeze encoder (input emb + GRU)
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
        enc_out, h = model.enc(src)                             # h (1,B,H)
        ce_whole = _decode_ce(model, enc_out, src_mask, h, tgt)
        e_comp = compute_ecomp(model, enc_out, src, src_mask, parse_mode=parse_mode,
                               and_id=and_id, after_id=after_id, gen=gen)
        L_self = ((h.squeeze(0) - e_comp) ** 2).sum(-1).mean()
        ce_comp = _decode_ce(model, enc_out, src_mask, e_comp.unsqueeze(0), tgt)
        perm = torch.randperm(e_comp.shape[0], generator=gen).to(e_comp.device)
        d = ((e_comp - e_comp[perm]) ** 2).sum(-1)             # squared pairwise (shuffled) distance
        L_hinge = F.relu(1.0 - d).mean()                       # fixed margin=1.0 anti-collapse floor
        loss = ce_whole + beta * L_self + gamma * ce_comp + delta * L_hinge
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0); opt.step()
    for p in model.parameters():
        p.requires_grad_(True)


# ---- run -------------------------------------------------------------------------------------------------

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
        return exp87.exact_match(m, te, in_vocab, out_vocab, max_len=max_len, batch_size=args.batch_size, device=args.device)

    if "vanilla_plain" in arms:
        vm = exp87.Seq2Seq(len(in_vocab), len(out_vocab), args.embed, args.hidden, seed).to(args.device)
        exp87.train(vm, tr, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, tf_ratio=args.tf_ratio, seed=seed, device=args.device)
        res["vanilla_plain"] = em(vm)
        print(f"  [{split} s{seed}] vanilla_plain = {res['vanilla_plain']:.4f}", file=sys.stderr, flush=True)

    if "vanilla_geca" in arms:
        tr_geca, _ = exp92.geca_augment(tr, [], args.min_shared_env)
        gm = exp87.Seq2Seq(len(in_vocab), len(out_vocab), args.embed, args.hidden, seed).to(args.device)
        exp87.train(gm, tr_geca, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, tf_ratio=args.tf_ratio, seed=seed, device=args.device)
        res["vanilla_geca"] = em(gm)
        print(f"  [{split} s{seed}] vanilla_geca = {res['vanilla_geca']:.4f}", file=sys.stderr, flush=True)

    ccc_modes = [a for a in arms if a in ("ccc", "ccc_random", "ccc_nosplit")]
    if "ccc_baseline" in arms or ccc_modes:
        model = CCCSeq2Seq(len(in_vocab), len(out_vocab), args.embed, args.hidden, seed).to(args.device)
        exp87.train(model, tr, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, tf_ratio=args.tf_ratio, seed=seed, device=args.device)
        if "ccc_baseline" in arms:
            res["ccc_baseline"] = em(model)
            print(f"  [{split} s{seed}] ccc_baseline = {res['ccc_baseline']:.4f}", file=sys.stderr, flush=True)
        snap = copy.deepcopy(model.state_dict())
        mode_map = {"ccc": "conj", "ccc_random": "random", "ccc_nosplit": "nosplit"}
        for arm in ccc_modes:
            model.load_state_dict(snap)
            cgen = torch.Generator().manual_seed(seed * 100003 + 11)   # CPU generator (make_batches + rand/perm)
            consolidate_ccc(model, tr, in_vocab, out_vocab, parse_mode=mode_map[arm], and_id=and_id,
                            after_id=after_id, beta=args.beta, gamma=args.gamma, delta=args.delta,
                            steps=args.consol_steps, batch_size=args.batch_size, lr=args.consol_lr,
                            gen=cgen, device=args.device)
            res[arm] = em(model)
            print(f"  [{split} s{seed}] {arm} = {res[arm]:.4f}", file=sys.stderr, flush=True)
    return res


def summary(v, n_boot=10000, seed=0):
    t = torch.tensor(v, dtype=torch.float64)
    if t.numel() == 1:
        return {"mean": float(t), "min": float(t), "max": float(t), "ci95": [float(t), float(t)], "per_seed": v}
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, t.numel(), (n_boot, t.numel()), generator=g)
    means = t[idx].mean(1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return {"mean": float(t.mean()), "min": float(t.min()), "max": float(t.max()), "ci95": [lo, hi], "per_seed": v}


def paired_delta(a, b, n_boot=10000, seed=0):
    ta, tb = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    d = ta - tb; g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, d.numel(), (n_boot, d.numel()), generator=g)
    dm = d[idx].mean(1)
    lo, hi = torch.quantile(dm, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return {"delta_mean": float(d.mean()), "ci95": [lo, hi], "per_seed": d.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", type=int, default=64)
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
    ap.add_argument("--splits", default="mcd1")
    ap.add_argument("--arms", default="vanilla_plain,ccc_baseline,ccc,ccc_random,ccc_nosplit,vanilla_geca")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs, args.consol_steps = 1, 10, 200
        args.arms = "vanilla_plain,ccc_baseline,ccc"

    arms = args.arms.split(",")
    splits = args.splits.split(",")
    out = {"experiment": "94_betb_scan_mcd_ccc (Clause-Compositional Consolidation, Stage-1)",
           "charter": "notes/betb-mcd-stage1-mechanism-precommit.md", "config": vars(args),
           "published_floor": {"vanilla": "0.17/0.14/0.02 in-house", "GECA": "51/30/12 (Conklin 2021)",
                               "ceiling_structure_injecting": "AuxSeq 99.9/90/98, LeAR 100/100/100"},
           "by_split": {}}
    for split in splits:
        per = {a: [] for a in arms}
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            r = run_seed(args, seed, split, arms)
            for a in arms:
                if a in r:
                    per[a].append(r[a])
        sdict = {a: summary(per[a]) for a in arms if per[a]}
        deltas = {}
        if per.get("ccc") and per.get("ccc_baseline") and len(per["ccc"]) == len(per["ccc_baseline"]):
            deltas["ccc_minus_baseline"] = paired_delta(per["ccc"], per["ccc_baseline"])
        if per.get("ccc") and per.get("ccc_random") and len(per["ccc"]) == len(per["ccc_random"]):
            deltas["ccc_minus_random"] = paired_delta(per["ccc"], per["ccc_random"])
        if per.get("ccc") and per.get("ccc_nosplit") and len(per["ccc"]) == len(per["ccc_nosplit"]):
            deltas["ccc_minus_nosplit"] = paired_delta(per["ccc"], per["ccc_nosplit"])
        if per.get("ccc") and per.get("vanilla_geca") and len(per["ccc"]) == len(per["vanilla_geca"]):
            deltas["ccc_minus_geca"] = paired_delta(per["ccc"], per["vanilla_geca"])
        out["by_split"][split] = {"arms": sdict, "paired_deltas": deltas}
        print(f"\n=== {split}: " + " ".join(f"{a}={sdict[a]['mean']:.3f}" for a in sdict), file=sys.stderr, flush=True)

    print(json.dumps(out, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
