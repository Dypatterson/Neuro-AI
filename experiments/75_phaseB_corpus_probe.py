"""experiments/75 — Phase-B corpus PROBE (throwaway): which different-domain 2nd corpus can support the
CE-1 generality test? (precommit §8 Phase B). NOT a graduation experiment — a feasibility screen.

A valid generality test needs the 2nd corpus to INDEPENDENTLY carry the paradigmatic signal: enough SimLex
pairs surviving in its top-`max_vocab` vocab (n_para) AND its OWN raw-SPPMI-SVD calibration anchor showing a
real para-vs-random specificity (the WikiText +0.109/0.222 band is WikiText-SPECIFIC — each corpus derives
its own). This probe reports (n_para, anchor spec, kq, para>rand sanity) per candidate so the corpus choice
is DATA-DRIVEN, not speculative. We pick the corpus with a valid anchor + enough pairs, then run exp74 on it.

Candidates = genuinely DIFFERENT domains from encyclopedic WikiText (NOT wikitext-103 = same domain):
  ptb (WSJ news/financial), ag_news (news), TinyStories (simple narrative — best concrete-noun coverage).
Loads each directly via `datasets` (no core corpus.py edit until one is chosen). Reuses exp61/63 verbatim.
"""
from __future__ import annotations

import argparse, importlib.util, itertools, pathlib, sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import torch  # noqa: E402


def _load(m, f):
    spec = importlib.util.spec_from_file_location(m, REPO / "experiments" / f)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

exp61 = _load("exp61", "61_phase3_second_order_growth.py")
exp63 = _load("exp63", "63_directional_successor_oracle.py")

from energy_memory.phase2.corpus import build_vocabulary, encode_texts, make_windows  # noqa: E402
from types import SimpleNamespace  # noqa: E402


def load_texts(name, n_docs):
    """Return List[str] train texts for a candidate, or raise. Different-domain corpora only."""
    from datasets import load_dataset
    if name == "ptb":
        ds = load_dataset("ptb_text_only", "penn_treebank", split="train", trust_remote_code=True)
        return [r["sentence"] for r in ds]
    if name == "ag_news":
        ds = load_dataset("fancyzhx/ag_news", split=f"train[:{n_docs}]")
        return [r["text"] for r in ds]
    if name == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        return [r["text"] for r in itertools.islice(ds, n_docs)]
    raise ValueError(name)


def probe(name, texts, args):
    vocab = build_vocabulary(texts, max_vocab=args.max_vocab)
    V = len(vocab.id_to_token); special = {vocab.unk_id, vocab.mask_id}
    windows = make_windows(encode_texts(texts, vocab), args.W)
    C, uni, total = exp61.build_cooccurrence(windows, V, special, device="cpu")
    k_sym, dens, _ = exp61.pick_k_by_density(C, uni, total, 0.3, 0.5); k_sym = k_sym or 1
    sppmi = exp61.build_sppmi(C, uni, total, k_sym)
    pargs = SimpleNamespace(corpus_source=name, simlex_min_sim=args.simlex_min_sim,
                            paradigmatic_max_cooc=args.paradigmatic_max_cooc)
    para, collo, rand, kq, sha, src = exp63.select_pairs(pargs, vocab, C, V, special)
    if not para:
        return {"name": name, "V": V, "windows": len(windows), "n_para": 0, "valid": False,
                "note": "0 SimLex pairs survive in top-vocab → cannot support the test"}
    anchor = exp63.raw_sppmi_svd_anchor(sppmi, para, rand, collo, kq, args.svd_rank, args.n_boot, args.boot_seed)
    spec = anchor["specificity_para_minus_random"]["mean"]
    lo, hi = anchor["specificity_para_minus_random"]["ci"]
    kq_cos = float(anchor.get("king_queen_cos", float("nan")))   # the COSINE (kq from select_pairs = the pair indices)
    # VALID = enough surviving pairs + a real para-vs-random signal (CI-lo>0). king/queen cos is a soft sanity
    # only (king/queen specifically may be absent from a different-domain corpus; the para-SET anchor is the test).
    valid = bool(len(para) >= args.min_pairs and lo > 0)
    return {"name": name, "V": V, "windows": len(windows), "n_para": len(para), "n_collo": len(collo),
            "anchor_spec": spec, "anchor_ci": [lo, hi], "kq": kq_cos, "valid": valid,
            "note": "VALID — own anchor carries the paradigmatic signal" if valid else
                    f"WEAK — n_para={len(para)} (need ≥{args.min_pairs}), spec_lo={lo:+.3f}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="ptb,ag_news,tinystories")
    ap.add_argument("--n-docs", type=int, default=40000, dest="n_docs")
    ap.add_argument("--max-vocab", type=int, default=2000, dest="max_vocab")
    ap.add_argument("--W", type=int, default=6)
    ap.add_argument("--paradigmatic-max-cooc", type=int, default=2, dest="paradigmatic_max_cooc")
    ap.add_argument("--simlex-min-sim", type=float, default=5.0, dest="simlex_min_sim")
    ap.add_argument("--svd-rank", type=int, default=300, dest="svd_rank")
    ap.add_argument("--min-pairs", type=int, default=15, dest="min_pairs")
    ap.add_argument("--n-boot", type=int, default=4000, dest="n_boot")
    ap.add_argument("--boot-seed", type=int, default=7, dest="boot_seed")
    args = ap.parse_args()
    results = []
    for name in [c.strip() for c in args.candidates.split(",") if c.strip()]:
        try:
            texts = load_texts(name, args.n_docs)
            r = probe(name, texts, args)
        except Exception as e:  # noqa: BLE001 — a download/gating failure shouldn't kill the other probes
            r = {"name": name, "valid": False, "note": f"LOAD FAILED: {type(e).__name__}: {str(e)[:160]}"}
        results.append(r)
        print(f"[{name}] " + (f"V={r.get('V')} windows={r.get('windows')} n_para={r.get('n_para')} "
              f"spec={r.get('anchor_spec', float('nan')):+.4f} ci={r.get('anchor_ci')} kq={r.get('kq', float('nan')):.3f} "
              f"VALID={r['valid']} — {r['note']}" if "V" in r else f"VALID={r['valid']} — {r['note']}"),
              file=sys.stderr, flush=True)
    valid = [r for r in results if r.get("valid")]
    print("\n=== PROBE SUMMARY ===", file=sys.stderr)
    if valid:
        best = max(valid, key=lambda r: r["n_para"])
        print(f"RECOMMEND: '{best['name']}' (n_para={best['n_para']}, anchor spec={best['anchor_spec']:+.4f} "
              f"ci={best['anchor_ci']}, kq={best['kq']:.3f}) → wire into load_corpus + run exp74.", file=sys.stderr)
    else:
        print("NO valid candidate — none carries the paradigmatic signal at this scale/vocab. Options: "
              "raise --n-docs / --max-vocab, try a books/web corpus, or run on Colab with a larger slice.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
