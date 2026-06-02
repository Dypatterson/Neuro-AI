"""experiments/76 — TEM local-reachability oracle (Fork B; DRILL-DOWN oracle, NOT graduation).

Frozen pre-commit: notes/emergent-codebook/phase-3-tem-local-reachability-precommit.md.

THE ONE question: can a FIXED-RANDOM-SLOT, online-Hebbian, NO-global-optimization writer reach the
paradigmatic (king/queen) structure that Oracle E's GLOBAL NMF recovers (+0.124/+0.176/+0.188,
Report 125 §3)? Oracle E's positive came from NMF jointly+globally optimizing the slot ASSIGNMENT
(the suspected load-bearing step). The faithful local writer FREEZES the slot basis to fixed-random
BEFORE seeing king/queen and accumulates token->slot occupancy by local Hebbian binding only:
  W = OP @ S_rand   (OP = fixed batch-offline co-occurrence statistic; S_rand = frozen V×k nonneg;
                     the matmul is the batched form of per-window accumulation — NO eig/NMF/SVD/backprop).
Read = STATIC slot-vector cosine (exp65:311 / exp68.read_specificity), NOT grow_G (grow_G is the
damped power iteration exp61:348-377 that re-imposes the 121-125 dominant-mode bound).

Writers (frozen-slot, static-read), k∈{8,16,32}, over OP∈{M_trans (E's op), build_S (SPPMI floor op),
M_SR (path-integration: tem_frozen on M_SR == tem_frozen_sr)}:
  tem_frozen       = l2rows(OP⁺ @ S_rand)                       [immediate-neighbour / path-integration on M_SR]
  tem_frozen_comp  = local per-token k-WTA (cap=k//4) on tem_frozen  [LOCAL competition, NOT global k-means]
References (reuse): linear_floor = flat-SPPMI grow_G (exp65.faithful_read, g2 baseline);
  E_nmf = global NMF Oracle-E ceiling (exp65.nmf_slots); rand_nonneg = pure random codes (exp68, g4 anti-inflation).

Headline = within-paradigmatic-SET label-shuffle B-KILL (Report-126 hubness discriminator).
MANDATORY §3 spectral diagnosis EVERY run: where does paradigmatic structure live in slot-occupancy?
  (low para_centroid_rank => DOMINANT/reachable modes; high => SUBDOMINANT) — converts an ambiguous
  null into the two pre-registered sub-cases (a) basis-change-not-spectrum / (b) frozen-assignment-was-load-bearing.

Substrate-free rung-1; FHRR-port rung-2 deferred to a PASS. Reuses exp61/62/63/65/68 verbatim.
Floor (055-058) untouched (separate operators, never read at recall).
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
exp63 = _load("exp63", "63_directional_successor_oracle.py")
exp65 = _load("exp65", "65_escape_route_triage.py")
exp68 = _load("exp68", "68_nonlinear_competition_kill_test.py")

from energy_memory.phase2.corpus import build_vocabulary, encode_texts, make_windows  # noqa: E402
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402

EPS = 1e-12


def l2rows(M):
    return M / M.norm(dim=1, keepdim=True).clamp(min=EPS)


# =====================================================================
# The frozen-slot local writer (the only new mechanism) + local competition + spectral diagnosis
# =====================================================================

def tem_frozen_write(OP, S_rand):
    """Online Hebbian accumulation of frozen neighbour slot-signatures: W = OP⁺ @ S_rand, nonneg,
    ℓ2-rows. OP = fixed batch-offline co-occurrence statistic; S_rand = frozen V×k nonneg slots.
    NO eig/NMF/SVD/backprop — the matmul is the batched form of per-window Hebbian binding."""
    W = OP.clamp(min=0.0).to(torch.float64) @ S_rand
    return l2rows(W)


def local_kwta(W, cap):
    """LOCAL per-token competition: each token keeps its top-`cap` of its OWN k slots (a fixed
    rank-threshold dynamic, not a metric-read branch). NOT the global k-means over tokens (Report 127)."""
    cap = max(1, min(cap, W.shape[1]))
    thr = torch.topk(W, cap, dim=1).values[:, -1:].clamp(min=EPS)
    return l2rows(W * (W >= thr).to(W.dtype))


def spectral_diagnosis(W, para, rand, top_q=0.25):
    """Where does paradigmatic structure live in the slot-occupancy? Static-cosine Gram W Wᵀ =
    Σ_m s_m² u_m u_mᵀ; per-mode para-discrimination align[m]=mean_para(u_m[a]u_m[b])−mean_rand(...),
    mass w[m]=s_m²·max(align,0). Returns para_centroid_rank∈[0,1] (low=DOMINANT/reachable modes,
    high=SUBDOMINANT) and dominant_frac (mass in top-⌈top_q·k⌉ modes)."""
    Wc = (W - W.mean(dim=0, keepdim=True)).to(torch.float64)
    U, S, _ = torch.linalg.svd(Wc, full_matrices=False)
    r = S.shape[0]
    if not para or not rand or r < 2:
        return {"para_centroid_rank": float("nan"), "dominant_frac": float("nan"), "n_modes": int(r)}
    pa = torch.tensor(para, dtype=torch.long)
    ra = torch.tensor(rand, dtype=torch.long)

    def align(col):
        return float((col[pa[:, 0]] * col[pa[:, 1]]).mean() - (col[ra[:, 0]] * col[ra[:, 1]]).mean())

    al = torch.tensor([align(U[:, m]) for m in range(r)], dtype=torch.float64)
    w = (S ** 2) * al.clamp(min=0.0)
    tot = w.sum().clamp(min=EPS)
    ranks = torch.arange(r, dtype=torch.float64)
    centroid = float((ranks * w).sum() / tot) / max(r - 1, 1)
    top_n = max(1, int(round(top_q * r)))
    return {"para_centroid_rank": centroid, "dominant_frac": float(w[:top_n].sum() / tot),
            "n_modes": int(r), "top_n": top_n, "align_first5": [round(float(x), 5) for x in al[:5]]}


# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-source", default="wikitext", dest="corpus_source")
    ap.add_argument("--wikitext-name", default="wikitext-2-raw-v1", dest="wikitext_name")
    ap.add_argument("--D", type=int, default=4096)
    ap.add_argument("--W", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--max-vocab", type=int, default=2000, dest="max_vocab")
    ap.add_argument("--paradigmatic-max-cooc", type=int, default=2, dest="paradigmatic_max_cooc")
    ap.add_argument("--simlex-min-sim", type=float, default=5.0, dest="simlex_min_sim")
    ap.add_argument("--K-sr", type=int, default=10, dest="K_sr")
    ap.add_argument("--k-slots", default="8,16,32", dest="k_slots")
    ap.add_argument("--operators", default="M_trans,build_S,M_SR")
    ap.add_argument("--svd-rank", type=int, default=300, dest="svd_rank")
    ap.add_argument("--eta-grid", default="0,0.05,0.1,0.2", dest="eta_grid")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--alpha0", type=float, default=0.3)
    ap.add_argument("--alpha-decay", type=float, default=0.9, dest="alpha_decay")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--bkill-min", type=int, default=8, dest="bkill_min")
    ap.add_argument("--n-boot", type=int, default=4000, dest="n_boot")
    ap.add_argument("--boot-seed", type=int, default=7, dest="boot_seed")
    ap.add_argument("--planted-seed", type=int, default=0, dest="planted_seed")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    k_slots = [int(x) for x in args.k_slots.split(",") if x != ""]
    eta_grid = [float(x) for x in args.eta_grid.split(",") if x != ""]
    ops_want = [o for o in args.operators.split(",") if o]

    # ---- corpus / vocab / windows / operators (seed-independent) ----
    splits = exp61.load_corpus(args.corpus_source, args, args.planted_seed)
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    V = len(vocab.id_to_token)
    special = {vocab.unk_id, vocab.mask_id}
    train_ids = encode_texts(splits["train"], vocab)
    windows = make_windows(train_ids, args.W)
    C, uni, total = exp61.build_cooccurrence(windows, V, special, device="cpu")
    k_sym, _, _ = exp61.pick_k_by_density(C, uni, total, 0.3, 0.5)
    if k_sym is None:
        k_sym = 1
    sppmi = exp61.build_sppmi(C, uni, total, k_sym)
    para, collo, rand, kq, sha, src = exp63.select_pairs(args, vocab, C, V, special)
    shuf = exp68.derange_partners([(a, b) for (a, b) in para], args.boot_seed)
    cooc_x = (torch.log1p(exp61.cooc_counts_for_pairs(C, torch.tensor(para, dtype=torch.long)))
              if para else None)

    operators = {}
    Fs = exp65.build_distance_coocs(windows, V, special, args.W, args.gamma)
    F = sum(Fs)
    if "M_trans" in ops_want:
        operators["M_trans"] = exp65.build_transition_operator(F)      # E's operator (+0.19)
    if "build_S" in ops_want:
        operators["build_S"] = exp61.build_S(sppmi)                    # SPPMI 2nd-order floor op (+0.021)
    if "M_SR" in ops_want:
        operators["M_SR"] = exp65.build_SR_operator(F, args.gamma, args.K_sr)  # path-integration (tem_frozen on M_SR == _sr)
    M_floor = exp61.build_S(sppmi)                                     # flat-SPPMI floor operator for grow_G

    # ---- calibration anchor (once) ----
    anchor = exp63.raw_sppmi_svd_anchor(sppmi, para, rand, collo, kq, args.svd_rank,
                                        args.n_boot, args.boot_seed)
    a_spec = anchor["specificity_para_minus_random"]["mean"]
    a_lo, a_hi = anchor["specificity_para_minus_random"]["ci"]
    hw = (a_hi - a_lo) / 2 if a_hi == a_hi else 0.0
    calib_ok = bool((0.082 - hw) <= a_spec <= (0.137 + hw)
                    and abs(anchor["king_queen_cos"] - 0.222) <= 0.04)
    print(f"V={V} windows={len(windows)} n_para={len(para)} ops={list(operators)} src={src} "
          f"[anchor] spec={a_spec:+.4f} kq={anchor['king_queen_cos']:.3f} calib_ok={calib_ok}",
          file=sys.stderr, flush=True)

    def rd(Y, seed):
        return exp68.read_specificity(Y, para, rand, collo, shuf, args.n_boot, seed)

    def corr_hi(Y):
        if cooc_x is None or not para:
            return float("nan")
        y = exp62._cos_real(Y, para)
        return exp61.corr_bootstrap_ci(cooc_x, y, n_boot=args.n_boot, seed=args.boot_seed)[2]

    # ---- linear floor (grow_G on flat-SPPMI) + rand_nonneg (dense + sparsity-matched), per seed ----
    floor_seed = []
    randnn_seed = {k: [] for k in k_slots}          # dense random codes — g4 control for tem_frozen
    randnn_comp_seed = {k: [] for k in k_slots}     # k-WTA'd random codes — sparsity-matched g4 control for tem_frozen_comp
    for seed in range(args.seeds):
        sub = TorchFHRR(dim=args.D, seed=seed, device=args.device, alpha_anti=1.0)
        G0 = sub.random_vectors(V)
        deff0 = exp61.d_eff(sub, G0)
        fr = exp65.faithful_read(sub, G0, deff0, M_floor, para, rand, cooc_x, eta_grid,
                                 args.epochs, args.alpha0, args.alpha_decay, args.n_boot, args.boot_seed)
        floor_seed.append(fr["best"]["spec"] if fr["best"] else float("nan"))
        for k in k_slots:
            rn = exp68.random_nonneg(V, k, seed)
            randnn_seed[k].append(rd(rn, seed)["spec_para_minus_rand"][0])
            randnn_comp_seed[k].append(rd(local_kwta(rn, k // 4), seed)["spec_para_minus_rand"][0])
    lin_floor = st.mean([x for x in floor_seed if x == x]) if any(x == x for x in floor_seed) else float("nan")
    print(f"[linear floor] grow_G flat-SPPMI spec={lin_floor:+.4f}", file=sys.stderr, flush=True)

    results = {}
    for opname, OP in operators.items():
        per = {"tem_frozen": {k: [] for k in k_slots}, "tem_frozen_comp": {k: [] for k in k_slots},
               "E_nmf": {k: [] for k in k_slots},
               "diag_tem": {k: [] for k in k_slots}, "diag_E": {k: [] for k in k_slots}}
        for seed in range(args.seeds):
            for k in k_slots:
                S_rand = exp68.random_nonneg(V, k, seed)              # FROZEN before pairs are read at this k
                Wt = tem_frozen_write(OP, S_rand)
                Wc = local_kwta(Wt, k // 4)
                We = exp65.nmf_slots(OP, k, seed=seed)
                per["tem_frozen"][k].append({**rd(Wt, seed), "corr_hi": corr_hi(Wt)})
                per["tem_frozen_comp"][k].append({**rd(Wc, seed), "corr_hi": corr_hi(Wc)})
                per["E_nmf"][k].append({**rd(We, seed), "corr_hi": corr_hi(We)})
                per["diag_tem"][k].append(spectral_diagnosis(Wt, para, rand))
                per["diag_E"][k].append(spectral_diagnosis(We, para, rand))
            print(f"[{opname} seed {seed}] tem(k{k_slots[-1]})spec="
                  f"{per['tem_frozen'][k_slots[-1]][-1]['spec_para_minus_rand'][0]:+.4f} "
                  f"E(k{k_slots[-1]})spec={per['E_nmf'][k_slots[-1]][-1]['spec_para_minus_rand'][0]:+.4f} "
                  f"tem_centroid={per['diag_tem'][k_slots[-1]][-1]['para_centroid_rank']:.3f} "
                  f"E_centroid={per['diag_E'][k_slots[-1]][-1]['para_centroid_rank']:.3f}",
                  file=sys.stderr, flush=True)

        e_ceiling = max(st.mean([c["spec_para_minus_rand"][0] for c in per["E_nmf"][k]]) for k in k_slots)

        def verdict(name):
            cells = {}
            for k in k_slots:
                rows = per[name][k]
                spec_mean = st.mean([c["spec_para_minus_rand"][0] for c in rows])
                spec_lo = st.mean([c["spec_para_minus_rand"][1] for c in rows])
                kill_lo = [c["kill_para_minus_shuf"][1] for c in rows]
                ch = st.mean([c["corr_hi"] for c in rows if c["corr_hi"] == c["corr_hi"]]) if any(
                    c["corr_hi"] == c["corr_hi"] for c in rows) else float("nan")
                rn = st.mean(randnn_comp_seed[k] if name == "tem_frozen_comp" else randnn_seed[k])
                bkill = sum(1 for x in kill_lo if x == x and x > 0)
                g1 = bool(spec_lo > 0.04)
                g2 = bool(spec_mean - lin_floor >= 0.02)
                g3 = bool(ch == ch and ch < 0.15)
                g4 = bool(spec_mean - rn >= 0.02)
                cells[k] = {"spec_mean": spec_mean, "spec_ci_lo_mean": spec_lo, "corr_hi_mean": ch,
                            "kill_lo_per_seed": [round(x, 4) for x in kill_lo],
                            "rand_nonneg_spec": rn, "B_KILL_seeds": f"{bkill}/{len(rows)}",
                            "recover_frac_of_E": spec_mean / e_ceiling if e_ceiling > 1e-9 else float("nan"),
                            "para_centroid_rank": st.mean([d["para_centroid_rank"] for d in per[
                                "diag_tem" if name == "tem_frozen" else (
                                    "diag_E" if name == "E_nmf" else "diag_tem")][k]
                                if d["para_centroid_rank"] == d["para_centroid_rank"]] or [float("nan")]),
                            "g1": g1, "g2_beats_floor": g2, "g3_not_collocational": g3,
                            "g4_beats_randcodes": g4,
                            "PASS": bool(g1 and g2 and g3 and g4 and bkill >= args.bkill_min)}
            best_k = max(k_slots, key=lambda k: cells[k]["spec_mean"])
            return {"cells": cells, "best_k": best_k, "PASS": any(cells[k]["PASS"] for k in k_slots)}

        results[opname] = {
            "linear_floor": lin_floor, "E_ceiling_bestk": e_ceiling,
            "tem_frozen": verdict("tem_frozen"), "tem_frozen_comp": verdict("tem_frozen_comp"),
            "E_nmf": verdict("E_nmf"),
            "_per_seed": {kk: per[kk] for kk in ("tem_frozen", "tem_frozen_comp", "E_nmf",
                                                 "diag_tem", "diag_E")},
        }
        for w in ("tem_frozen", "tem_frozen_comp", "E_nmf"):
            bk = results[opname][w]["best_k"]
            c = results[opname][w]["cells"][bk]
            print(f"[{opname} {w}] best_k={bk} spec={c['spec_mean']:+.4f} (CIlo {c['spec_ci_lo_mean']:+.4f}) "
                  f"recoverE={c['recover_frac_of_E']:.2f} corr_hi={c['corr_hi_mean']:+.3f} "
                  f"centroid={c['para_centroid_rank']:.3f} B_KILL={c['B_KILL_seeds']} "
                  f"PASS={results[opname][w]['PASS']}", file=sys.stderr, flush=True)

    any_tem_pass = any(results[o][w]["PASS"] for o in results for w in ("tem_frozen", "tem_frozen_comp"))

    # ---- null sub-case selection from the §3 diagnosis (pre-registered) ----
    def subcase():
        if any_tem_pass:
            return "PASS"
        tem_centroids, e_centroids, e_recover = [], [], []
        for o in results:
            for k in k_slots:
                tem_centroids += [d["para_centroid_rank"] for d in results[o]["_per_seed"]["diag_tem"][k]
                                  if d["para_centroid_rank"] == d["para_centroid_rank"]]
                e_centroids += [d["para_centroid_rank"] for d in results[o]["_per_seed"]["diag_E"][k]
                                if d["para_centroid_rank"] == d["para_centroid_rank"]]
            e_recover.append(results[o]["E_nmf"]["cells"][results[o]["E_nmf"]["best_k"]]["recover_frac_of_E"])
        tem_c = st.mean(tem_centroids) if tem_centroids else float("nan")
        e_c = st.mean(e_centroids) if e_centroids else float("nan")
        # (b) if E concentrates para in dominant modes (low centroid) AND E beats the floor, but tem nulls.
        # FIX (2026-06-02, post adversarial verification): gate (b) on E BEATING THE FLOOR (g2) per
        # precommit §4, NOT on E's full five-gate PASS. E_nmf can NEVER PASS because g3 (corr<0.15) is
        # DEAD at n=40 (the +0.109 SVD anchor also fails g3; Report 125 §6 recommended removing g3 here).
        # The old `any(E_nmf['PASS'])` made the (b) branch dead code → falsely emitted (a). B-KILL is the
        # real collocational/pair-specificity arbiter, decoupled from g3.
        e_beats_floor = any(
            results[o]["E_nmf"]["cells"][results[o]["E_nmf"]["best_k"]]["g2_beats_floor"]
            for o in results)
        if e_beats_floor and e_c == e_c and e_c < 0.5 and tem_c == tem_c and tem_c >= e_c + 0.1:
            return ("NULL (b) frozen-assignment-was-load-bearing — global slot-ASSIGNMENT carried "
                    f"E's positive (E centroid {e_c:.3f} dominant; tem centroid {tem_c:.3f} subdominant). "
                    "TEM inherits the bound for any non-backprop local writer; AMBIGUOUS for 'substrate "
                    "can't help' — do NOT over-bank. Next: the global slot-optimization is the lever "
                    "(needs backprop, banned) OR multi-layer hierarchy.")
        return ("NULL (a) basis-change-not-spectrum — paradigmatic structure is STILL subdominant in "
                f"slot-occupancy (tem centroid {tem_c:.3f}); the nonneg factorization changed the BASIS "
                "not the spectrum's shape. The flat-code bound GENERALIZES to nonneg-factorized "
                "single-layer codes; substrate-invariant for single-layer local writers. Next: genuine "
                "MULTI-LAYER hierarchy or the Codebook-growth⇄Replay combination. NOT more operator knobs.")

    verdict = ("INVALID — calibration anchor missed +0.109/0.222; fix harness, re-run." if not calib_ok
               else (subcase() if not any_tem_pass else
                     "PASS — a FIXED-RANDOM-SLOT local writer cleared g1∧g2∧g3∧g4∧B-KILL≥%d/%d → the "
                     "FIRST local writer to break the locality trap. Escalate to the rung-2 FHRR D=4096 "
                     "port (exp74 pattern) BEFORE banking; magnitude = 'reaches the factorization target', "
                     "NOT 'beats SVD/NMF' (Report 125 §3 caveat-1). The TEM substrate build is the user's "
                     "gate, with this PASS its precondition." % (args.bkill_min, args.seeds)))

    out = {"experiment": "76_tem_local_reachability_oracle (Fork B; DRILL-DOWN, NOT graduation)",
           "precommit": "notes/emergent-codebook/phase-3-tem-local-reachability-precommit.md",
           "config": {"D": args.D, "V": V, "max_vocab": args.max_vocab, "k_slots": k_slots,
                      "operators": list(operators), "seeds": args.seeds, "K_sr": args.K_sr,
                      "bkill_min": args.bkill_min, "corpus_source": args.corpus_source,
                      "simlex": f"{src}; sha={sha[:16] if sha and sha != 'PLANTED' else sha}"},
           "anchor": {"spec": a_spec, "ci": [a_lo, a_hi], "kq": anchor["king_queen_cos"], "calib_ok": calib_ok},
           "linear_floor": lin_floor, "results": results, "VERDICT": verdict}
    print(json.dumps(out, indent=2, default=float))
    print(f"[VERDICT] {verdict[:130]}", file=sys.stderr, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
