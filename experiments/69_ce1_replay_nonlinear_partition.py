"""experiments/69 — CE-1 ⊗ 127: emergent Replay-interleaving × an online NONLINEAR-PARTITION writer.

FROZEN pre-commit: notes/emergent-codebook/phase-3-ce1x127-replay-nonlinear-partition-precommit.md
(grilled /grill-with-docs Q1-Q6, 2026-06-01; run-log §7). DRILL-DOWN feasibility oracle, NOT graduation.
Rung-1 SUBSTRATE-FREE screen (no build-gate). Floor (055-058) untouched. Reuses exp61/62/63/65/68.

THE ONE QUESTION: does an EMERGENT, codes-derived priority REWEIGHT of WHICH windows get consolidated
(the two-timescale loop) let an online local NONLINEAR-PARTITION writer (k-WTA) reach paradigmatic
structure that UNIFORM consolidation (as-is) misses — i.e. are the 121-127 nulls false negatives of
isolation? (The audit's key discriminator: replay changes the EFFECTIVE OPERATOR, not just window order.)

REHEARSAL #2 design (precommit §7, fixes #1-#3 from rehearsal #1; params FROZEN-by-principle, not tuned):
  * WITHIN-SENTENCE co-occurrence (sentences are the units; NO cross-sentence sliding windows) — kills
    the locality leak that contaminated the rehearsal-#1 negative control.
  * HEADROOM corpus: a strong COMMON-MODE background (bg tokens in most sentences) dominates the operator,
    so the paradigmatic pair-shared-context is a SUBDOMINANT component; a dominant-partition k-WTA on the
    UNIFORM operator is pulled to the common mode (as-is misses the pairs). NMF (subdominant modes) is the
    "signal exists" ceiling. Pairs are also RARE (~9% of sentences).
  * EVOLVING MULTIPLICITY REWEIGHT (the mechanism): each epoch the per-sentence replay weight =
    novelty × surprise, read OFF THE CURRENT CODES (substrate-free, rung-1) and RECOMPUTED each epoch
    (the two-timescale loop). Weight reshapes the EFFECTIVE co-occurrence operator (not just order).
  * Controls: GAUGE (random weights, priority destroyed), STATIC (one-shot 1/sqrt-freq reweight — the
    surprise≈rarity trap; the loop must beat it), UNIFORM as-is (Arm A), NMF ceiling, grow_G linear floor.
  * Headline = within-target LABEL-SHUFFLE B-KILL (hubness-immune: real pair vs partner-shuffled), >=4/5.

PAIRED SMOKE GATE (precommit §6/Q6): KIND — NMF recovers the planted signal; Arm B clears the B-KILL AND
beats Arm A by >=0.05 AND beats STATIC by >=0.02; the GAUGE does NOT reproduce B. TRAP (disjoint contexts,
hubs but no real pairs) — the mechanism NULLs (B-KILL fails). Both required, else SMOKE-FAIL.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import statistics as st
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import torch  # noqa: E402


def _load(modname, fname):
    spec = importlib.util.spec_from_file_location(modname, REPO / "experiments" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

exp61 = _load("exp61", "61_phase3_second_order_growth.py")
exp62 = _load("exp62", "62_exp61_oracles.py")
exp65 = _load("exp65", "65_escape_route_triage.py")
exp68 = _load("exp68", "68_nonlinear_competition_kill_test.py")

from energy_memory.phase2.corpus import build_vocabulary  # noqa: E402
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402

EPS = 1e-12
F64 = torch.float64


def l2rows(M):
    return M / M.norm(dim=1, keepdim=True).clamp(min=EPS)


# =====================================================================
# Planted corpora — sentences (within-sentence cooc), common-mode headroom, multi-pair (B-KILL-able)
# =====================================================================

def _planted(seed, *, paradigmatic, n_pairs=6, n_ctx=5, n_blob=10, blob_size=6, n_bg=5,
             n_pair_per=25, n_collo=1500, bg_prob=0.85):
    """Return (sentences[list[list[token-str]]], pair_names). Tokens match TOKEN_RE [a-z0-9']+
    (lowercase, no underscores). A strong COMMON-MODE background (bg*) is sprinkled into most
    sentences -> the dominant operator mode; the pair-shared-context signal is SUBDOMINANT + RARE.
    KIND: tgta_p/tgtb_p share ctx_p (disjoint across pairs), never co-occur -> pair-specific 2nd-order
    overlap. TRAP: tgta_p uses caa_p, tgtb_p uses cbb_p (all disjoint) -> hubs, but NO pair structure."""
    g = torch.Generator().manual_seed(seed * 31 + 5)

    def rint(hi):
        return int(torch.randint(0, hi, (1,), generator=g).item())

    pair_names = [(f"tgta{p}", f"tgtb{p}") for p in range(n_pairs)]
    bg = [f"bg{i}" for i in range(n_bg)]
    blobs = [[f"blob{b}w{i}" for i in range(blob_size)] for b in range(n_blob)]
    if paradigmatic:
        ctx = {p: [f"ctx{p}q{i}" for i in range(n_ctx)] for p in range(n_pairs)}
    else:
        ctxA = {p: [f"caa{p}q{i}" for i in range(n_ctx)] for p in range(n_pairs)}
        ctxB = {p: [f"cbb{p}q{i}" for i in range(n_ctx)] for p in range(n_pairs)}

    def add_bg(sent):
        if rint(100) < int(bg_prob * 100):
            sent = sent + [bg[rint(n_bg)]]
            if rint(100) < 40:
                sent = sent + [bg[rint(n_bg)]]
        return sent

    sentences = []
    # RARE paradigmatic pair sentences (the subdominant signal)
    for p in range(n_pairs):
        for _ in range(n_pair_per):
            which = rint(2)
            pool = (ctx[p] if paradigmatic else (ctxA[p] if which == 0 else ctxB[p]))
            cs = [pool[rint(n_ctx)] for _ in range(3)]
            tgt = pair_names[p][which]
            sentences.append(add_bg(cs[:1] + [tgt] + cs[1:]))
    # DOMINANT collocational blob sentences (the common/dominant mode)
    for _ in range(n_collo):
        b = rint(n_blob)
        cs = [blobs[b][rint(blob_size)] for _ in range(4)]
        sentences.append(add_bg(cs))
    # global shuffle: sentence ORDER carries no pair identity (within-sentence cooc is the only signal;
    # the replay MULTIPLICITY reweight is the lever, not the base order)
    perm = torch.randperm(len(sentences), generator=g).tolist()
    sentences = [sentences[i] for i in perm]
    return sentences, pair_names


def _encode(sentences, vocab, special):
    sp = set(int(s) for s in special)
    out = []
    for sent in sentences:
        ids = [vocab.token_to_id[t] for t in sent if t in vocab.token_to_id]
        ids = [i for i in ids if i not in sp]
        out.append(ids)
    return out


def _presence(sent_ids, V):
    P = torch.zeros((len(sent_ids), V), dtype=F64)
    for s, ids in enumerate(sent_ids):
        if ids:
            P[s, torch.tensor(ids, dtype=torch.long)] = 1.0
    return P


def weighted_cooc(P, w):
    """Within-sentence weighted co-occurrence: Pi[i,j] = sum_s w_s P[s,i] P[s,j] (symmetric, diag 0)."""
    Pw = P * w[:, None]
    Pi = Pw.T @ P
    Pi.fill_diagonal_(0.0)
    uni = Pw.sum(dim=0)
    return Pi, uni, float(uni.sum().item())


def _sppmi_L(Pi, uni, total, k_sym):
    return l2rows(exp61.build_sppmi(Pi, uni, total, k_sym))


# =====================================================================
# The NONLINEAR-PARTITION writer (k-WTA): shared step; offline (uniform) & online (reweighted) variants
# =====================================================================

def _kwta_step(L, W, cap):
    a = (L @ W.T).clamp(min=0.0)
    cap = min(cap, a.shape[1])
    thr = torch.topk(a, cap, dim=1).values[:, -1:].clamp(min=EPS)
    mask = (a >= thr).to(F64)
    Y = a * mask
    won = mask.T @ L
    cnt = mask.sum(0).clamp(min=1.0).unsqueeze(1)
    return Y, l2rows(W + 0.5 * (won / cnt - W))


def _init_W(V, k, seed):
    g = torch.Generator().manual_seed(seed * 104729 + 7)
    return l2rows(torch.rand((k, V), generator=g, dtype=F64))


def kwta_offline(L, k, cap, epochs, seed):
    W = _init_W(L.shape[0], k, seed)
    Y = None
    for _ in range(epochs):
        Y, W = _kwta_step(L, W, cap)
    return l2rows(Y)


def kwta_stream(P, V, k, cap, k_sym, epochs, inner, seed, weight_fn):
    """Online k-WTA. Each epoch: per-sentence replay weights w = weight_fn(Y, ep) (evolving for the
    priority arm), build the REWEIGHTED effective operator, take `inner` competitive steps carrying W.
    The reweight = which windows get consolidated more = the audit's effective-operator change."""
    W = _init_W(V, k, seed)
    Y = torch.zeros((V, k), dtype=F64)
    n = P.shape[0]
    epoch_codes = []
    for ep in range(epochs):
        w = weight_fn(Y, ep) if weight_fn is not None else torch.ones(n, dtype=F64)
        Pi, uni, total = weighted_cooc(P, w)
        L = _sppmi_L(Pi, uni, total, k_sym)
        for _ in range(inner):
            Y, W = _kwta_step(L, W, cap)
        epoch_codes.append(l2rows(Y).clone())
    return l2rows(Y), epoch_codes


# =====================================================================
# Replay weights: priority (novelty x surprise, evolving) / gauge (random) / static (one-shot rarity)
# =====================================================================

def _sentence_token_index(sent_ids):
    return [torch.tensor(ids, dtype=torch.long) if ids else None for ids in sent_ids]


def priority_weight_fn(sent_idx, mult_scale):
    """fn(Y, ep) -> per-sentence weight = 1 + mult_scale * normalized(novelty * surprise), read OFF the
    CURRENT codes Y (substrate-free) and recomputed each epoch (the two-timescale loop). novelty = mean
    token under-consolidation (1 - max_k Y); surprise = 1 - max_k(mean pooled code). Y=0 -> uniform."""
    def fn(Y, ep):
        n = len(sent_idx)
        if float(Y.abs().max()) < EPS:
            return torch.ones(n, dtype=F64)
        tok_nov = 1.0 - Y.max(dim=1).values
        scores = torch.zeros(n, dtype=F64)
        for s, idx in enumerate(sent_idx):
            if idx is None or idx.numel() == 0:
                continue
            nov = float(tok_nov[idx].mean())
            surprise = float(1.0 - Y[idx].mean(dim=0).max())
            scores[s] = nov * surprise
        lo, hi = float(scores.min()), float(scores.max())
        rng = (hi - lo) or 1.0
        return 1.0 + mult_scale * (scores - lo) / rng
    return fn


def gauge_weight_fn(n, mult_scale, seed):
    """GAUGE: random per-sentence weights in [1, 1+mult_scale], FIXED per epoch (priority CONTENT
    destroyed, multiplicity distribution preserved). Ignores Y."""
    g = torch.Generator().manual_seed(seed * 99991 + 3)
    fixed = 1.0 + mult_scale * torch.rand(n, generator=g, dtype=F64)
    return lambda Y, ep: fixed


def static_weight_fn(P, mult_scale):
    """STATIC control (one-shot, the surprise≈rarity trap): per-sentence weight from token rarity
    1/sqrt(freq), FIXED across epochs. The evolving loop (Arm B) must BEAT this or it's just rarity."""
    freq = P.sum(dim=0).clamp(min=1.0)
    inv = 1.0 / freq.sqrt()
    n = P.shape[0]
    sc = torch.zeros(n, dtype=F64)
    for s in range(n):
        idx = P[s].nonzero(as_tuple=True)[0]
        if idx.numel():
            sc[s] = float(inv[idx].mean())
    lo, hi = float(sc.min()), float(sc.max())
    rng = (hi - lo) or 1.0
    fixed = 1.0 + mult_scale * (sc - lo) / rng
    return lambda Y, ep: fixed


# =====================================================================

def _pairs(vocab, pair_names, V, special, C):
    def tid(t):
        return vocab.token_to_id.get(t)
    para, targs = [], []
    for a, b in pair_names:
        ai, bi = tid(a), tid(b)
        if ai is not None and bi is not None and ai != bi:
            para.append((min(ai, bi), max(ai, bi)))
            targs += [ai, bi]
    # hubness-matched rand = cross-pair A-target pairs (same token class as para, never paired)
    a_ids = [tid(a) for (a, _b) in pair_names if tid(a) is not None]
    rand = []
    for i in range(len(a_ids)):
        for j in range(i + 1, len(a_ids)):
            rand.append((min(a_ids[i], a_ids[j]), max(a_ids[i], a_ids[j])))
    collo = []
    for p, (a, _b) in enumerate(pair_names):
        ai = tid(a)
        ci = tid(f"ctx{p}q0")
        ci = ci if ci is not None else tid(f"caa{p}q0")
        if ai is not None and ci is not None and ai != ci:
            collo.append((min(ai, ci), max(ai, ci)))
    shuf = exp68.derange_partners(para, seed=7)   # within-target label shuffle = the B-KILL
    return para, collo, rand, shuf


def _kill(Y, para, rand, collo, shuf, n_boot, seed):
    """Return (kill_mean, kill_lo, spec_mean) — kill = para_cos - shuffled_cos (hubness-immune B-KILL)."""
    r = exp68.read_specificity(Y, para, rand, collo, shuf, n_boot, seed)
    return r["kill_para_minus_shuf"][0], r["kill_para_minus_shuf"][1], r["spec_para_minus_rand"][0]


def run_corpus(label, paradigmatic, args):
    sentences, pair_names = _planted(args.smoke_seed, paradigmatic=paradigmatic,
                                     n_pairs=args.n_pairs, n_collo=args.n_collo)
    flat = " ".join(t for s in sentences for t in s)
    vocab = build_vocabulary([flat], max_vocab=args.max_vocab)
    V = len(vocab.id_to_token)
    special = {vocab.unk_id, vocab.mask_id}
    sent_ids = _encode(sentences, vocab, special)
    P = _presence(sent_ids, V)
    sent_idx = _sentence_token_index(sent_ids)
    # k_sym + uniform operator (the fixed structural knob; from the UNIFORM cooc)
    Cu, uniu, totu = weighted_cooc(P, torch.ones(P.shape[0], dtype=F64))
    k_sym, dens, _ = exp61.pick_k_by_density(Cu, uniu, totu, 0.3, 0.5)
    k_sym = k_sym or 1
    sppmi_u = exp61.build_sppmi(Cu, uniu, totu, k_sym)
    para, collo, rand, shuf = _pairs(vocab, pair_names, V, special, Cu)
    print(f"[{label}] V={V} sentences={len(sentences)} k_sym={k_sym} dens={dens:.3f} "
          f"n_para={len(para)} n_rand={len(rand)} n_shuf={len(shuf)}", file=sys.stderr, flush=True)

    acc = {n: [] for n in ("nmf", "offline", "floor", "A", "B", "gauge", "static")}
    bkill_lo = {n: [] for n in ("A", "B", "gauge", "static")}
    traj = {"A": [], "B": []}
    for seed in range(args.seeds):
        def kill(Y):
            return _kill(Y, para, rand, collo, shuf, args.n_boot, args.boot_seed)
        # NMF ceiling (subdominant modes) — "signal exists"
        Ynmf = exp65.nmf_slots(exp61.build_S(sppmi_u), args.k, seed=seed)
        # offline k-WTA on UNIFORM operator (dominant partition — the as-is reference at full budget)
        Yoff = kwta_offline(l2rows(sppmi_u), args.k, args.cap, args.offline_epochs, seed)
        # linear floor (grow_G on uniform build_S)
        sub = TorchFHRR(dim=args.D, seed=seed, device="cpu", alpha_anti=1.0)
        G0 = sub.random_vectors(V); deff0 = exp61.d_eff(sub, G0)
        cooc_x = torch.log1p(exp61.cooc_counts_for_pairs(Cu, torch.tensor(para, dtype=torch.long))) if para else None
        fr = exp65.faithful_read(sub, G0, deff0, exp61.build_S(sppmi_u), para, rand, cooc_x,
                                 [0.0, 0.1], args.floor_epochs, 0.3, 0.9, args.n_boot, args.boot_seed)
        floor = fr["best"]["spec"] if fr["best"] else float("nan")
        # online arms
        YA, cA = kwta_stream(P, V, args.k, args.cap, k_sym, args.epochs, args.inner, seed, None)
        YB, cB = kwta_stream(P, V, args.k, args.cap, k_sym, args.epochs, args.inner, seed,
                             priority_weight_fn(sent_idx, args.mult_scale))
        YG, _ = kwta_stream(P, V, args.k, args.cap, k_sym, args.epochs, args.inner, seed,
                            gauge_weight_fn(P.shape[0], args.mult_scale, seed))
        YS, _ = kwta_stream(P, V, args.k, args.cap, k_sym, args.epochs, args.inner, seed,
                            static_weight_fn(P, args.mult_scale))
        for name, Y in (("nmf", Ynmf), ("offline", Yoff), ("A", YA), ("B", YB), ("gauge", YG), ("static", YS)):
            km, klo, _sp = kill(Y)
            acc[name].append(km)
            if name in bkill_lo:
                bkill_lo[name].append(klo)
        acc["floor"].append(floor)
        traj["A"].append([kill(c)[0] for c in cA])
        traj["B"].append([kill(c)[0] for c in cB])
        print(f"  [{label} s{seed}] nmf={acc['nmf'][-1]:+.3f} off={acc['offline'][-1]:+.3f} "
              f"floor={floor:+.3f} A={acc['A'][-1]:+.3f} B={acc['B'][-1]:+.3f} "
              f"gauge={acc['gauge'][-1]:+.3f} static={acc['static'][-1]:+.3f} | Bkill_lo={bkill_lo['B'][-1]:+.3f}",
              file=sys.stderr, flush=True)

    def m(k):
        xs = [x for x in acc[k] if x == x]
        return st.mean(xs) if xs else float("nan")
    bpass = sum(1 for x in bkill_lo["B"] if x == x and x > 0)
    out = {
        "nmf_ceiling": m("nmf"), "offline_uniform": m("offline"), "linear_floor": m("floor"),
        "armA_asis": m("A"), "armB_replay": m("B"), "gauge_random": m("gauge"), "static_reweight": m("static"),
        "B_minus_A": m("B") - m("A"), "B_minus_static": m("B") - m("static"),
        "gauge_minus_A": m("gauge") - m("A"), "B_KILL_seeds": f"{bpass}/{args.seeds}",
        "metric": "kill = para_cos - within-target-shuffle_cos (hubness-immune B-KILL)",
        "traj_A": traj["A"], "traj_B": traj["B"],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=6, dest="n_pairs")
    ap.add_argument("--n-collo", type=int, default=1500, dest="n_collo")
    ap.add_argument("--max-vocab", type=int, default=400, dest="max_vocab")
    ap.add_argument("--D", type=int, default=2048)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--cap", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--offline-epochs", type=int, default=60, dest="offline_epochs")
    ap.add_argument("--floor-epochs", type=int, default=20, dest="floor_epochs")
    ap.add_argument("--mult-scale", type=float, default=4.0, dest="mult_scale")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
    ap.add_argument("--boot-seed", type=int, default=7, dest="boot_seed")
    ap.add_argument("--smoke-seed", type=int, default=0, dest="smoke_seed")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    kind = run_corpus("KIND", True, args)
    trap = run_corpus("TRAP", False, args)

    kind_signal = kind["nmf_ceiling"] > 0.05                       # NMF (subdominant) recovers the planted signal
    kind_B = (int(kind["B_KILL_seeds"].split("/")[0]) >= max(1, args.seeds - 1)
              and kind["B_minus_A"] >= 0.05 and kind["B_minus_static"] >= 0.02)
    kind_gauge_ok = kind["gauge_minus_A"] < 0.02                   # gauge does NOT reproduce B
    trap_null = int(trap["B_KILL_seeds"].split("/")[0]) <= max(0, args.seeds - 2)
    smoke_pass = bool(kind_signal and kind_B and kind_gauge_ok and trap_null)
    verdict = ("SMOKE-PASS — kind: NMF recovers the signal, Arm B clears the B-KILL + beats A & static, "
               "gauge does NOT reproduce; trap: NULLs. The mechanism CAN surface subdominant paradigmatic "
               "structure via evolving replay-reweight where as-is misses it → authorize the WikiText "
               "rung-1 screen." if smoke_pass else
               "SMOKE-FAIL — paired gate not cleared (see smoke_gate flags). Do NOT proceed to WikiText. "
               "Report honestly; a persistent B≈A is itself evidence replay-reweight is a weak lever for a "
               "nonlinear-partition writer (the partition does the work).")
    out = {
        "experiment": "69_ce1_replay_nonlinear_partition (RUNG-1 PLANTED SMOKE rehearsal #2; substrate-free; NOT graduation)",
        "precommit": "notes/emergent-codebook/phase-3-ce1x127-replay-nonlinear-partition-precommit.md (§7 run-log)",
        "config": vars(args), "kind": kind, "trap": trap,
        "smoke_gate": {"kind_NMF_recovers_signal": kind_signal, "kind_B_pass": kind_B,
                       "kind_gauge_discriminates": kind_gauge_ok, "trap_nulls": trap_null,
                       "SMOKE_PASS": smoke_pass},
        "VERDICT": verdict,
    }
    print(json.dumps(out, indent=2, default=float))
    print(f"\n[VERDICT] {verdict}", file=sys.stderr, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
