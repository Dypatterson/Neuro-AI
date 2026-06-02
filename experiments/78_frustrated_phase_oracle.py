"""experiments/78 — Frustrated-phase oracle (LEAD 1; DRILL-DOWN, NOT graduation).

Frozen pre-commit: notes/emergent-codebook/phase-3-frustrated-phase-precommit.md.

The ONE question: does a LOCAL frustrated-XY phase dynamic on the FHRR phase channel — the affordance
the entire 121-129 arc left idle — reach the SUBDOMINANT (king/queen) structure by INVERTING the
target (frustration drives co-occurring/collocational pairs ANTI-phase, expelling the dominant mode
from the phase channel so the paradigmatic signal is the phase-coherent survivor)?

Per dimension d, offline replay-pass settling (A = build_S rows, the SAME stat grow_G uses):
    theta_i[d] += eta * SIGN * Σ_j A_ij sin(theta_j[d] - theta_i[d])
  SIGN = -1 (frustrated, the +pi lag) | +1 (attractive control, must NULL).
Read = within-para-set label-shuffle B-KILL on the complex cosine of the settled phasors exp(i*theta)
  (stack [Re, Im] -> read_specificity does row-cosine = mean_d cos(theta_i - theta_j) = phase coherence).

MANDATORY KILL-GATE (§2 precommit): the spectral-reduction probe — does the settled-phase read reach
MORE than the bottom-k / top-k eigenvector embeddings of build_S? If it merely recovers them it is a
SPECTRAL FLASHLIGHT (demote, as Report 124 §4 demoted the SVD +0.531 artifact) and the bound holds in
the phase channel too (frontier = multi-layer). Controls: attractive-must-null, eta=0-null,
collocational<paradigmatic, the inverse-frequency phase arm (Report 128 §8 — must beat it), CEILING
(reach-NOT-exceed global NMF/SVD). Substrate-free screen (real angles mod 2pi); FHRR D=4096 port = rung-2/fenced.
Reuses exp61/63/65/68 verbatim. Floor (055-058) untouched.
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

EPS = 1e-12


# =====================================================================
# The frustrated-XY dynamic (the only new mechanism) + reads + the spectral kill-gate
# =====================================================================

def frustrated_xy(A, D, eta, iters, sign, seed):
    """Settle V phase-rows theta (V x D) under coupling A (V x V, real, nonneg). Per step:
       theta += eta * sign * [cos(theta)⊙(A@sin theta) − sin(theta)⊙(A@cos theta)]   (= Σ_j A_ij sin(θ_j−θ_i))
    sign=-1 ⇒ frustrated (+π lag, co-occurring pairs DE-synchronize); sign=+1 ⇒ attractive control.
    Complex matmul avoided (MPS-safe): split into real A @ {cos,sin}. Returns settled (cos, sin)."""
    V = A.shape[0]
    g = torch.Generator().manual_seed(seed * 2654435761 % (2**31))
    theta = 2 * torch.pi * torch.rand((V, D), generator=g, dtype=torch.float64)
    Af = A.to(torch.float64)
    for _ in range(iters):
        c, s = torch.cos(theta), torch.sin(theta)
        torque = c * (Af @ s) - s * (Af @ c)            # Σ_j A_ij sin(θ_j − θ_i), per dim
        theta = theta + eta * sign * torque
    return torch.cos(theta), torch.sin(theta)


def phase_codes(c, s):
    """Stack [Re, Im] of exp(i·theta) → V×2D real; row-cosine == mean_d cos(θ_i−θ_j) (phase coherence)."""
    return torch.cat([c, s], dim=1)


def spectral_embed(A, k, which):
    """Bottom-k or top-k eigenvector embedding of symmetric A (the spectral-reduction reference)."""
    w, V = torch.linalg.eigh(A.to(torch.float64))       # ascending eigenvalues
    idx = list(range(k)) if which == "bottom" else list(range(A.shape[0] - k, A.shape[0]))
    return V[:, idx]                                     # V×k real


def read(Y, para, rand, collo, shuf, n_boot, seed):
    return exp68.read_specificity(Y, para, rand, collo, shuf, n_boot, seed)


# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-source", default="wikitext", dest="corpus_source")
    ap.add_argument("--wikitext-name", default="wikitext-2-raw-v1", dest="wikitext_name")
    ap.add_argument("--W", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--max-vocab", type=int, default=2000, dest="max_vocab")
    ap.add_argument("--paradigmatic-max-cooc", type=int, default=2, dest="paradigmatic_max_cooc")
    ap.add_argument("--simlex-min-sim", type=float, default=5.0, dest="simlex_min_sim")
    ap.add_argument("--D-grid", default="64,128,256", dest="D_grid")
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--probe-k", type=int, default=32, dest="probe_k")
    ap.add_argument("--nmf-k", type=int, default=32, dest="nmf_k")
    ap.add_argument("--svd-rank", type=int, default=300, dest="svd_rank")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--bkill-min", type=int, default=8, dest="bkill_min")
    ap.add_argument("--n-boot", type=int, default=4000, dest="n_boot")
    ap.add_argument("--boot-seed", type=int, default=7, dest="boot_seed")
    ap.add_argument("--planted-seed", type=int, default=0, dest="planted_seed")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    D_grid = [int(x) for x in args.D_grid.split(",") if x != ""]

    # ---- corpus / build_S / pairs / anchor (seed-independent) ----
    splits = exp61.load_corpus(args.corpus_source, args, args.planted_seed)
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    V = len(vocab.id_to_token)
    special = {vocab.unk_id, vocab.mask_id}
    windows = make_windows(encode_texts(splits["train"], vocab), args.W)
    C, uni, total = exp61.build_cooccurrence(windows, V, special, device="cpu")
    k_sym, _, _ = exp61.pick_k_by_density(C, uni, total, 0.3, 0.5)
    if k_sym is None:
        k_sym = 1
    sppmi = exp61.build_sppmi(C, uni, total, k_sym)
    build_S = exp61.build_S(sppmi)                                   # the operator A (nonneg PSD Gram)
    A = build_S / build_S.abs().max().clamp(min=EPS)                 # scale for stable XY dynamics
    para, collo, rand, kq, sha, src = exp63.select_pairs(args, vocab, C, V, special)
    shuf = exp68.derange_partners([(a, b) for (a, b) in para], args.boot_seed)
    # codes-independent inverse-frequency coupling (Report-128 §8 control): rank-1, marginal freq only
    gfreq = (1.0 / (uni.to(torch.float64) + 1.0)).sqrt()
    A_freq = torch.outer(gfreq, gfreq)
    A_freq = A_freq / A_freq.abs().max().clamp(min=EPS)
    A_freq.fill_diagonal_(0.0)

    anchor = exp63.raw_sppmi_svd_anchor(sppmi, para, rand, collo, kq, args.svd_rank,
                                        args.n_boot, args.boot_seed)
    a_spec = anchor["specificity_para_minus_random"]["mean"]
    a_lo, a_hi = anchor["specificity_para_minus_random"]["ci"]
    hw = (a_hi - a_lo) / 2 if a_hi == a_hi else 0.0
    calib_ok = bool((0.082 - hw) <= a_spec <= (0.137 + hw) and abs(anchor["king_queen_cos"] - 0.222) <= 0.04)
    print(f"V={V} windows={len(windows)} n_para={len(para)} src={src} "
          f"[anchor] spec={a_spec:+.4f} kq={anchor['king_queen_cos']:.3f} calib_ok={calib_ok}",
          file=sys.stderr, flush=True)

    # ---- global ceiling (NMF) + spectral-reduction reference reads (seed-independent) ----
    nmf_spec = st.mean([read(exp65.nmf_slots(build_S, args.nmf_k, seed=s), para, rand, collo, shuf,
                             args.n_boot, args.boot_seed)["spec_para_minus_rand"][0] for s in range(3)])
    bottomk = read(spectral_embed(build_S, args.probe_k, "bottom"), para, rand, collo, shuf, args.n_boot, args.boot_seed)
    topk = read(spectral_embed(build_S, args.probe_k, "top"), para, rand, collo, shuf, args.n_boot, args.boot_seed)
    spectral_ref = max(bottomk["spec_para_minus_rand"][0], topk["spec_para_minus_rand"][0])
    print(f"[ceiling] nmf(k{args.nmf_k})={nmf_spec:+.4f} svd_anchor={a_spec:+.4f} | "
          f"[spectral-ref] bottom{args.probe_k}={bottomk['spec_para_minus_rand'][0]:+.4f} "
          f"top{args.probe_k}={topk['spec_para_minus_rand'][0]:+.4f}", file=sys.stderr, flush=True)

    results = {}
    for D in D_grid:
        per = {"frustrated": [], "attractive": [], "eta0": [], "freq": [], "collo_frustrated": []}
        for seed in range(args.seeds):
            cF, sF = frustrated_xy(A, D, args.eta, args.iters, sign=-1.0, seed=seed)       # frustrated (+π)
            cA, sA = frustrated_xy(A, D, args.eta, args.iters, sign=+1.0, seed=seed)       # attractive (control)
            c0, s0 = frustrated_xy(A, D, 0.0, args.iters, sign=-1.0, seed=seed)            # eta=0 (random phases)
            cQ, sQ = frustrated_xy(A_freq, D, args.eta, args.iters, sign=-1.0, seed=seed)  # inverse-freq arm
            rF = read(phase_codes(cF, sF), para, rand, collo, shuf, args.n_boot, seed)
            per["frustrated"].append(rF)
            per["attractive"].append(read(phase_codes(cA, sA), para, rand, collo, shuf, args.n_boot, seed))
            per["eta0"].append(read(phase_codes(c0, s0), para, rand, collo, shuf, args.n_boot, seed))
            per["freq"].append(read(phase_codes(cQ, sQ), para, rand, collo, shuf, args.n_boot, seed))
            per["collo_frustrated"].append(rF["collo_cos"])                                # collocational must score LOWER
            print(f"[D={D} seed {seed}] frust spec={rF['spec_para_minus_rand'][0]:+.4f} "
                  f"kill_lo={rF['kill_para_minus_shuf'][1]:+.4f} attr="
                  f"{per['attractive'][-1]['spec_para_minus_rand'][0]:+.4f} "
                  f"freq={per['freq'][-1]['spec_para_minus_rand'][0]:+.4f}", file=sys.stderr, flush=True)

        def agg(name):
            rows = per[name]
            return {"spec_mean": st.mean([r["spec_para_minus_rand"][0] for r in rows]),
                    "spec_ci_lo": st.mean([r["spec_para_minus_rand"][1] for r in rows]),
                    "para_cos": st.mean([r["para_cos"] for r in rows]),
                    "kill_lo_per_seed": [round(r["kill_para_minus_shuf"][1], 4) for r in rows],
                    "bkill_seeds": sum(1 for r in rows if r["kill_para_minus_shuf"][1] > 0)}

        F, At, E0, Fq = agg("frustrated"), agg("attractive"), agg("eta0"), agg("freq")
        collo_mean = st.mean(per["collo_frustrated"])
        # gates (frozen §3)
        g_bkill = F["bkill_seeds"] >= args.bkill_min
        g_spec = F["spec_ci_lo"] > 0.0
        g_attr_null = At["spec_ci_lo"] <= 0.0 or (F["spec_mean"] - At["spec_mean"]) >= 0.02
        g_eta0_null = E0["spec_ci_lo"] <= 0.02
        g_beats_freq = (F["spec_mean"] - Fq["spec_mean"]) >= 0.02
        g_collo_lower = collo_mean < F["para_cos"]
        g_ceiling = F["spec_mean"] <= max(nmf_spec, a_spec) + 1e-9         # reach-NOT-exceed
        g_killgate = (F["spec_mean"] - spectral_ref) >= 0.02              # reaches MORE than bottom/top-k spectral
        PASS = bool(g_bkill and g_spec and g_attr_null and g_eta0_null and g_beats_freq
                    and g_collo_lower and g_ceiling and g_killgate)
        results[f"D={D}"] = {
            "frustrated": F, "attractive": At, "eta0": E0, "freq_arm": Fq, "collo_cos": collo_mean,
            "gates": {"g_bkill": g_bkill, "g_spec_cilo": g_spec, "g_attractive_nulls": g_attr_null,
                      "g_eta0_nulls": g_eta0_null, "g_beats_freq": g_beats_freq,
                      "g_collo_lower": g_collo_lower, "g_ceiling_reach_not_exceed": g_ceiling,
                      "g_killgate_beats_spectral": g_killgate, "PASS": PASS}}
        print(f"[D={D}] frust spec={F['spec_mean']:+.4f}(CIlo {F['spec_ci_lo']:+.4f}) B-KILL={F['bkill_seeds']}/{args.seeds} "
              f"| attr={At['spec_mean']:+.4f} freq={Fq['spec_mean']:+.4f} spectral_ref={spectral_ref:+.4f} "
              f"nmf={nmf_spec:+.4f} | PASS={PASS}", file=sys.stderr, flush=True)

    any_pass = any(results[d]["gates"]["PASS"] for d in results)
    verdict = ("INVALID — anchor missed +0.109/0.222." if not calib_ok else
               ("PASS — frustrated-phase reaches the subdominant structure by a NEW channel, beats the "
                "spectral-reduction reference (NOT a flashlight), attractive/eta0/freq all null, "
                "reach-not-exceed ceiling. FIRST local writer past the single-layer bound via phase → "
                "escalate to the rung-2 FHRR D=4096 port BEFORE banking; 'local route to the global "
                "code', NOT 'beats SVD'." if any_pass else
                "NULL — frustrated-phase does not clear the gate. Sub-case from the diagnostics: "
                "(i) hubness if label-shuffle B-KILL fails at positive spec; (ii) deflates-to-PMI if "
                "freq-arm matches; (iii) SPECTRAL-FLASHLIGHT if it merely recovers the bottom/top-k "
                "eigenvectors (g_killgate False) → the bound holds in the phase channel too → frontier "
                "is a genuine LOCAL MULTI-LAYER writer (Report 129 §56). A spec-tightening result, banked."))
    out = {"experiment": "78_frustrated_phase_oracle (LEAD 1; DRILL-DOWN, NOT graduation)",
           "precommit": "notes/emergent-codebook/phase-3-frustrated-phase-precommit.md",
           "config": {"D_grid": D_grid, "V": V, "eta": args.eta, "iters": args.iters,
                      "probe_k": args.probe_k, "seeds": args.seeds, "corpus_source": args.corpus_source,
                      "simlex": f"{src}; sha={sha[:16] if sha and sha != 'PLANTED' else sha}"},
           "anchor": {"spec": a_spec, "kq": anchor["king_queen_cos"], "calib_ok": calib_ok},
           "ceiling": {"nmf": nmf_spec, "svd_anchor": a_spec, "spectral_ref": spectral_ref,
                       "bottomk": bottomk["spec_para_minus_rand"][0], "topk": topk["spec_para_minus_rand"][0]},
           "results": results, "VERDICT": verdict}
    print(json.dumps(out, indent=2, default=float))
    print(f"[VERDICT] {verdict[:120]}", file=sys.stderr, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
