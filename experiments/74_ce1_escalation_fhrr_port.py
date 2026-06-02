"""experiments/74 — CE-1 escalation Phase A: rung-2 FHRR port (D-sweep) + heavier-budget g-static re-test.

Pre-registered: notes/emergent-codebook/phase-3-ce1x127-replay-nonlinear-partition-precommit.md §8
(frozen 2026-06-02, user-approved "escalate the working mechanism"). Escalates the exp73/Report-128
working gap-closer (emergent LOCAL surprise-targeted PAIR-replay) up the fidelity ladder + re-tests the one
open gate condition. RUNG-2, substrate-aware (NOT graduation). Reuses exp61/63/65/68/70/72/73 verbatim.

WHY (Report 128 = NULL-BUT-RISING): Arm B closes 102% of the +0.045 locality gap (10/10, gauge-
discriminated → it's the *targeting*, not replay mass), but does NOT robustly beat the one-shot
surprise-reweight (g-static), so the two-timescale LOOP is unproven at n=10/20-epoch. Two escalation axes:

  (1) RUNG-2 FHRR SINGLE-SHOT PORT (the §4.5 substrate-reality headline). The rung-1 reads partition the
      EXACT float64 operator; rung-2 asks whether the REAL FHRR substrate (1/√D crosstalk = audit bound
      family #2) still expresses the structure. Faithful context-bundle port: each arm's FINAL 1st-order
      SPPMI rows are bundled into FHRR vectors G_i = normalize(Σ_c sppmi[i,c]·E_c), E = random FHRR atoms;
      the 2nd-order build_S structure EMERGES as the FHRR-cosine Gram L_fhrr[i,j]=Re<G_i,G_j>/D + O(1/√D);
      re-run the SAME k-WTA partition on l2rows(L_fhrr), read the hubness-immune B-KILL. D-SWEEP
      {512,1024,2048,4096} + a Dinf anchor (exact, no FHRR = the rung-1 read) so the §4.5 DIVERGENCE
      PROTOCOL is built in: degrade-at-low-D-but-hold-at-high-D = the 1/√D floor (a finding); uniform NULL
      = the substrate genuinely does not express it. Verdict-level (NOT bit-identical) agreement at D=4096:
      reproduce (a) the SIGN + ≥4/5-seed B-KILL of Arm B, (b) (B−A) across-seed CI-lo>0 surviving contraction.

  (2) HEAVIER BUDGET (the NULL-BUT-RISING g-static re-test): re-run the rung-1 exact-operator arms at
      online-epochs=40 (FROZEN, doubled) — more epochs → more Arm-A contraction → larger loop advantage IF
      the loop is real. A vs B vs static vs gauge at the SAME budget; does (B−static) cross at 40?

Floor (055-058) untouched. Magnitudes partition-inflated (NOT "beats SVD/NMF"). Phase B (2nd dataset,
generality) is deferred + user-gated (§8) — run only if this rung-2 port does NOT null.
"""
from __future__ import annotations

import argparse, importlib.util, json, pathlib, statistics as st, sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import torch  # noqa: E402


def _load(m, f):
    spec = importlib.util.spec_from_file_location(m, REPO / "experiments" / f)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

exp61 = _load("exp61", "61_phase3_second_order_growth.py")
exp63 = _load("exp63", "63_directional_successor_oracle.py")
exp65 = _load("exp65", "65_escape_route_triage.py")
exp68 = _load("exp68", "68_nonlinear_competition_kill_test.py")
exp70 = _load("exp70", "70_wikitext_partition_headtohead.py")
exp72 = _load("exp72", "72_online_local_on_build_s.py")
exp73 = _load("exp73", "73_ce1_emergent_replay_gapcloser.py")

from energy_memory.phase2.corpus import build_vocabulary, encode_texts, make_windows  # noqa: E402
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402
EPS = 1e-12; F64 = torch.float64
l2rows = exp70.l2rows


def fhrr_gram(sppmi, sub, V):
    """Faithful FHRR context-bundle port: G_i = normalize(Σ_c sppmi[i,c]·E_c); return the substrate-noisy
    FHRR-cosine Gram (V×V real) ≈ build_S(sppmi) + O(1/√D). E = random FHRR atoms at the substrate's D."""
    E = sub.random_vectors(V)                                   # (V,D) complex unit-modulus
    M = sppmi.to(dtype=E.real.dtype)
    G = sub.normalize(torch.matmul(M, E.real) + 1j * torch.matmul(M, E.imag))   # bundle contexts -> phases
    D = G.shape[1]
    gram = torch.matmul(G, G.conj().transpose(0, 1)).real / D   # fcos Gram (V×V), 1/√D crosstalk
    return gram.to(F64)


def port_read(sppmi, D, V, k, cap, epochs, seed, para, rand, collo, shuf, n_boot, bseed):
    """Port arm's sppmi at dim D (Dinf = exact, no FHRR = the rung-1 read), partition, read B-KILL."""
    if D == "inf":
        L = l2rows(exp61.build_S(sppmi))                        # exact operator (rung-1 anchor)
    else:
        sub = TorchFHRR(dim=int(D), seed=seed * 131 + 7, device="cpu", alpha_anti=0.0)
        L = l2rows(fhrr_gram(sppmi, sub, V))
    Y = exp70.kwta_offline(L, k, cap, epochs, seed)
    return exp68.read_specificity(Y, para, rand, collo, shuf, n_boot, bseed)["kill_para_minus_shuf"]


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
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--offline-epochs", type=int, default=60, dest="offline_epochs")
    ap.add_argument("--online-epochs", type=int, default=20, dest="online_epochs")   # rung-1/rung-2 budget (= Report 128)
    ap.add_argument("--heavy-epochs", type=int, default=40, dest="heavy_epochs")     # FROZEN heavier budget
    ap.add_argument("--inner", type=int, default=6, dest="inner")
    ap.add_argument("--online-decay", type=float, default=0.7, dest="online_decay")
    ap.add_argument("--replay-strength", type=float, default=1.0, dest="replay_strength")
    ap.add_argument("--chunk", type=int, default=30000)
    ap.add_argument("--d-sweep", default="512,1024,2048,4096", dest="d_sweep")
    ap.add_argument("--svd-rank", type=int, default=300, dest="svd_rank")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=4000, dest="n_boot")
    ap.add_argument("--boot-seed", type=int, default=7, dest="boot_seed")
    ap.add_argument("--planted-seed", type=int, default=0, dest="planted_seed")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    Dsweep = [s.strip() for s in args.d_sweep.split(",") if s.strip()]
    if args.smoke:
        args.seeds = min(args.seeds, 2); args.online_epochs = min(args.online_epochs, 6)
        args.heavy_epochs = min(args.heavy_epochs, 8); Dsweep = ["1024", "4096"]

    splits = exp61.load_corpus(args.corpus_source, args, args.planted_seed)
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    V = len(vocab.id_to_token); special = {vocab.unk_id, vocab.mask_id}
    windows = make_windows(encode_texts(splits["train"], vocab), args.W)
    C, uni, total = exp61.build_cooccurrence(windows, V, special, device="cpu")
    k_sym, dens, _ = exp61.pick_k_by_density(C, uni, total, 0.3, 0.5); k_sym = k_sym or 1
    sppmi_full = exp61.build_sppmi(C, uni, total, k_sym)
    para, collo, rand, kq, sha, src = exp63.select_pairs(args, vocab, C, V, special)
    shuf = exp68.derange_partners(para, args.boot_seed)
    anchor = exp63.raw_sppmi_svd_anchor(sppmi_full, para, rand, collo, kq, args.svd_rank, args.n_boot, args.boot_seed)
    a_spec = anchor["specificity_para_minus_random"]["mean"]
    calib_ok = bool(0.082 <= a_spec <= 0.137 and abs(anchor["king_queen_cos"] - 0.222) <= 0.04)
    S_dec, m_chunks = exp73.stream_S(windows, V, special, args.chunk, args.online_decay)
    d_epoch = args.online_decay ** m_chunks
    print(f"V={V} windows={len(windows)} chunks={m_chunks} k_sym={k_sym} n_para={len(para)} "
          f"D-sweep={Dsweep} [anchor] spec={a_spec:+.4f} kq={anchor['king_queen_cos']:.3f} calib_ok={calib_ok}",
          file=sys.stderr, flush=True)

    def tci(vals):
        t = torch.tensor([v for v in vals if v == v], dtype=F64)
        return exp61.flat_bootstrap_ci(t, 4000, 0) if t.numel() else (float("nan"),) * 3

    def run_writer(mode, epochs, seed, want_op=False):
        return exp73.replay_writer(S_dec, d_epoch, C, V, args.k, args.cap, k_sym, epochs, args.inner,
                                   seed, mode, args.replay_strength, return_op=want_op)

    # ---- Part 1: heavier-budget (40-epoch) g-static re-test (rung-1, exact operator) ----
    heavy = {n: {"kill": [], "lo": []} for n in ("A", "B", "gauge", "static")}
    # ---- Part 2: rung-2 FHRR port D-sweep (+ Dinf anchor) ----
    port = {arm: {D: {"kill": [], "lo": []} for D in (["inf"] + Dsweep)} for arm in ("ceiling", "A", "B", "gauge")}

    for seed in range(args.seeds):
        bseed = args.boot_seed + seed * 101
        # Part 1
        for mode, nm in (("none", "A"), ("priority", "B"), ("gauge", "gauge"), ("static", "static")):
            Yh, _ = run_writer(mode, args.heavy_epochs, seed)
            kl = exp68.read_specificity(Yh, para, rand, collo, shuf, args.n_boot, bseed)["kill_para_minus_shuf"]
            heavy[nm]["kill"].append(kl[0]); heavy[nm]["lo"].append(kl[1])
        # Part 2: arms' final sppmi at the Report-128 budget, then port across D
        arm_sppmi = {"ceiling": sppmi_full}
        for mode, nm in (("none", "A"), ("priority", "B"), ("gauge", "gauge")):
            _, _, sp = run_writer(mode, args.online_epochs, seed, want_op=True)
            arm_sppmi[nm] = sp
        for arm, sp in arm_sppmi.items():
            for D in (["inf"] + Dsweep):
                kl = port_read(sp, D, V, args.k, args.cap, args.offline_epochs, seed,
                               para, rand, collo, shuf, args.n_boot, bseed)
                port[arm][D]["kill"].append(kl[0]); port[arm][D]["lo"].append(kl[1])
        msg = " ".join(f"{D}:{st.mean(port['B'][D]['kill']):+.3f}/{st.mean(port['A'][D]['kill']):+.3f}"
                       for D in (["inf"] + Dsweep))
        print(f"  [seed {seed}] HEAVY A={heavy['A']['kill'][-1]:+.3f} B={heavy['B']['kill'][-1]:+.3f} "
              f"static={heavy['static']['kill'][-1]:+.3f} gauge={heavy['gauge']['kill'][-1]:+.3f} | "
              f"PORT B/A @D {msg}", file=sys.stderr, flush=True)

    def m(d):
        xs = [x for x in d["kill"] if x == x]; return st.mean(xs) if xs else float("nan")

    # Part 1 verdict (g-static at 40 epochs)
    bs_h = [b - s for b, s in zip(heavy["B"]["kill"], heavy["static"]["kill"])]
    ba_h = [b - a for b, a in zip(heavy["B"]["kill"], heavy["A"]["kill"])]
    bs_h_ci, ba_h_ci = tci(bs_h), tci(ba_h)
    g_static_heavy = bs_h_ci[1] > 0

    # Part 2 verdict (rung-2 FHRR port)
    port_summary = {}
    for D in (["inf"] + Dsweep):
        ba = [b - a for b, a in zip(port["B"][D]["kill"], port["A"][D]["kill"])]
        ba_ci = tci(ba)
        b_lo_pass = sum(1 for x in port["B"][D]["lo"] if x == x and x > 0)
        port_summary[str(D)] = {
            "B_kill": m(port["B"][D]), "A_kill": m(port["A"][D]), "ceiling_kill": m(port["ceiling"][D]),
            "gauge_kill": m(port["gauge"][D]), "B_minus_A_ci": list(ba_ci), "B_lo_pass": b_lo_pass,
        }
    need = max(4, -(-4 * args.seeds // 5))
    p4096 = port_summary["4096"]
    rung2_pass = bool(p4096["B_minus_A_ci"][1] > 0 and p4096["B_lo_pass"] >= need)
    # D-sweep shape: does (B−A) hold across D, or degrade at low D (1/√D floor)?
    ba_by_D = [port_summary[str(D)]["B_minus_A_ci"][0] for D in Dsweep]
    monotone_floor = bool(ba_by_D[0] < ba_by_D[-1] - 1e-3)   # weaker at low D -> crosstalk floor

    if not calib_ok:
        verdict = "INVALID — anchor missed +0.109/0.222."
    elif rung2_pass:
        flo = " (B−A weakens at low D = the 1/√D crosstalk floor, §4.5 divergence protocol)" if monotone_floor else ""
        verdict = (f"RUNG-2 PASS (n={args.seeds}) — the FHRR substrate EXPRESSES the gap-close: at D=4096 ported "
                   f"Arm B {p4096['B_kill']:+.3f} vs A {p4096['A_kill']:+.3f}, (B−A) CI {p4096['B_minus_A_ci'][1:]}, "
                   f"B-KILL lo>0 {p4096['B_lo_pass']}/{args.seeds}{flo}. Verdict-level agreement with rung-1 "
                   f"holds. Heavier budget (40ep) g-static: (B−static) CI [{bs_h_ci[1]:+.3f},{bs_h_ci[2]:+.3f}] "
                   f"→ {'CROSSES (loop strengthens with budget → SCREEN-PASS candidate)' if g_static_heavy else 'STILL FAILS → bank the one-shot surprise-reweight as the operative LOCAL lever'}. "
                   f"NOT graduation; magnitudes partition-inflated.")
    else:
        verdict = (f"RUNG-2 NULL (n={args.seeds}) — at D=4096 the ported gap-close does NOT reproduce rung-1 "
                   f"((B−A) CI {p4096['B_minus_A_ci'][1:]}, B-KILL lo>0 {p4096['B_lo_pass']}/{args.seeds}); "
                   f"D-sweep {ba_by_D}. {'Weakens at low D = 1/√D crosstalk floor (a finding, not a kill — '
                   'higher D / lower-noise substrate is the indicated move, §4.5).' if monotone_floor else 'Uniform across D → the substrate genuinely does not express the idealized-Euclidean gap-close → bank the rung-2 kill; the bound stands.'} "
                   f"Phase B (2nd dataset) MOOT until rung-2 passes.")
    out = {
        "experiment": "74_ce1_escalation_fhrr_port (Phase A: rung-2 FHRR port D-sweep + heavier-budget g-static; NOT graduation)",
        "precommit": "notes/emergent-codebook/phase-3-ce1x127-replay-nonlinear-partition-precommit.md (§8 escalation)",
        "config": {**vars(args), "V": V, "windows": len(windows), "chunks": m_chunks, "d_epoch": d_epoch,
                   "k_sym": k_sym, "D_sweep": Dsweep, "simlex": f"{src}; sha={sha[:16] if sha and sha != 'PLANTED' else sha}"},
        "anchor": {"spec": a_spec, "kq": anchor["king_queen_cos"], "calib_ok": calib_ok},
        "metric": "kill = para_cos - within-set-shuffle_cos (hubness-immune B-KILL)",
        "heavy_budget_40ep": {n: {"kill_mean": m(heavy[n]), "per_seed_kill": heavy[n]["kill"]} for n in heavy},
        "heavy_B_minus_static_ci": list(bs_h_ci), "heavy_B_minus_A_ci": list(ba_h_ci),
        "g_static_at_40ep": g_static_heavy,
        "rung2_port": port_summary, "rung2_pass": rung2_pass, "dsweep_floor": monotone_floor,
        "VERDICT": verdict,
    }
    print(json.dumps(out, indent=2, default=float))
    print(f"\n[VERDICT] {verdict}", file=sys.stderr, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
