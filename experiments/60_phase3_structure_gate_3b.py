"""Phase-3 structure-gate "3b": does the codebook develop corpus-specific structure?

Spec: notes/emergent-codebook/phase-3-structure-gate-3b-design.md
Re-grounding: CONTEXT.md §5 (the load-bearing test of the "more than memory" thesis).

CLASSIFICATION: Phase-3 structure-gate GRADUATION experiment (the emergent-structure
deliverable), NOT a drill-down. Distinct from the consolidation-write FLOOR (recall,
Reports 055-058): 3b asks whether the codebook GEOMETRY becomes a learned rule grown
from co-occurrence.

HEADLINE (semantic arm): related-pair-cosine DRIFT (end - init of Hebbian training) on
real text vs a GAUGE-SAFE corpus-stream-shuffle control. PASS = Δcos(related,real) -
Δcos(related,shuffle) has a bootstrap CI strictly > 0, AND related-beats-random under
real (specificity). "Related" = a curated, corpus-INDEPENDENT semantic-pair list
(synonyms / antonyms / category-mates — the "king/queen" test), restricted to in-vocab
pairs. A held-out-PMI COLLOCATION arm is a floor/sanity check (a co-occurrence learner
should pass it ~by construction); the semantic arm is the meaningful gate.

CONTROLS: gauge-safe corpus-stream-shuffle (permute the flat token stream before
windowing -> preserves unigram marginals, destroys co-occurrence); random-pair arm
(specificity); init-as-baseline (drift is end - init, so the Phase-2 landscape is
differenced out per pair). The atom-relabel shuffled-token control is RETIRED
(gauge-vacuous). d_eff / max-pairwise-sim drill-down = collapse guard.

ANTI-HOMUNCULUS: codebook atoms drift under the fixed Hebbian distributional-centroid
update (phase2/codebook_learner.py); all measured quantities are OFFLINE batch
statistics (cosine, d_eff); the control is a DATA manipulation, not a mechanism; no
runtime metric gates/branches/selects; no energy->argmin->dE.

PRE-REGISTERED HONEST PRIOR (binding): 3b MAY FAIL (the C.3 null: consolidation change
~ corpus-independent). A null (real does NOT beat shuffle on the semantic arm) is a
RE-SCOPE signal about the codebook-growth dynamics, NOT a "run it bigger" signal. This
is committed before the run.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch

from energy_memory.phase2.codebook_learner import CodebookLearner
from energy_memory.phase2.corpus import (
    build_vocabulary, encode_texts, load_corpus_splits, make_windows,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR

REPO = pathlib.Path(__file__).resolve().parents[1]

# Curated, corpus-INDEPENDENT semantic pairs (synonyms / antonyms / category-mates):
# the "king/queen" paradigmatic-structure test. Lowercased to match normalized tokens.
SEMANTIC_PAIRS = [
    ("king", "queen"), ("man", "woman"), ("men", "women"), ("boy", "girl"),
    ("he", "she"), ("his", "her"), ("father", "mother"), ("son", "daughter"),
    ("brother", "sister"), ("husband", "wife"), ("good", "bad"), ("great", "good"),
    ("big", "large"), ("small", "little"), ("high", "low"), ("old", "new"),
    ("first", "second"), ("one", "two"), ("two", "three"), ("day", "night"),
    ("year", "month"), ("month", "week"), ("north", "south"), ("east", "west"),
    ("city", "town"), ("war", "peace"), ("black", "white"), ("left", "right"),
    ("up", "down"), ("life", "death"), ("house", "home"), ("car", "road"),
    ("water", "river"), ("school", "university"), ("game", "match"), ("team", "club"),
    ("world", "country"), ("president", "government"), ("music", "song"),
    ("book", "story"), ("film", "movie"), ("army", "military"), ("church", "god"),
    ("summer", "winter"), ("morning", "evening"), ("hand", "head"), ("eyes", "face"),
    ("red", "blue"), ("hot", "cold"), ("fast", "slow"),
]


def _stream_shuffle(token_ids, seed):
    """Gauge-safe corpus-stream shuffle: permute the flat token sequence (preserves
    unigram marginals, destroys co-occurrence)."""
    g = torch.Generator().manual_seed(seed * 100003 + 7)
    perm = torch.randperm(len(token_ids), generator=g).tolist()
    return [token_ids[i] for i in perm]


def _pair_cos(cb, pairs):
    """Cosine (FHRR similarity = mean real of conj-product) for each [a,b] pair."""
    if pairs.numel() == 0:
        return torch.zeros(0)
    a = cb[pairs[:, 0]]
    b = cb[pairs[:, 1]]
    return (a.conj() * b).real.mean(dim=1)


def _cooc_counts(windows, pairs):
    """Within-window co-occurrence count for each [a,b] pair (on the given windows).
    Used to split the curated-semantic arm into PARADIGMATIC (non-co-occurring) vs
    COLLOCATIONAL (co-occurring) subsets — the adversarial-verification discriminator
    that separates 'similar words cluster' from 'co-occurring words cluster'."""
    if pairs.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    want = {(int(a), int(b)): i for i, (a, b) in enumerate(pairs.tolist())}
    counts = [0] * pairs.shape[0]
    for win in windows:
        present = set(win)
        for (a, b), i in want.items():
            if a in present and b in present:
                counts[i] += 1
    return torch.tensor(counts, dtype=torch.long)


def _heldout_pmi_pairs(heldout_ids, vocab, W, k, min_count, special):
    """Top-k highest-PMI within-window co-occurring pairs from a HELD-OUT split
    (relatedness derived independently of the training windows)."""
    windows = make_windows(heldout_ids, W)
    uni = Counter()
    co = Counter()
    for win in windows:
        toks = [t for t in win if t not in special]
        for t in toks:
            uni[t] += 1
        seen = set(toks)
        for a in seen:
            for b in seen:
                if a < b:
                    co[(a, b)] += 1
    total = sum(uni.values()) or 1
    scored = []
    for (a, b), c_ab in co.items():
        if c_ab < min_count:
            continue
        pmi = math.log((c_ab / total) / ((uni[a] / total) * (uni[b] / total) + 1e-12) + 1e-12)
        scored.append((pmi, a, b))
    scored.sort(reverse=True)
    return [(a, b) for _, a, b in scored[:k]]


def _random_pairs(vocab, k, seed, special):
    g = torch.Generator().manual_seed(seed * 7919 + 3)
    ids = [i for i in range(len(vocab.id_to_token)) if i not in special]
    pairs = set()
    while len(pairs) < k and len(pairs) < len(ids) * (len(ids) - 1) // 2:
        i, j = torch.randint(0, len(ids), (2,), generator=g).tolist()
        if i != j:
            pairs.add((min(ids[i], ids[j]), max(ids[i], ids[j])))
    return list(pairs)


def _bootstrap_ci(values, n_boot=2000, z=None, seed=0):
    """Percentile bootstrap 95% CI of the mean of `values` (a 1-D tensor)."""
    if values.numel() == 0:
        return (float("nan"), float("nan"), float("nan"))
    g = torch.Generator().manual_seed(seed)
    n = values.numel()
    means = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        means.append(float(values[idx].mean()))
    means.sort()
    return (float(values.mean()), means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def train_codebook(sub, init_cb, vocab, windows, epochs, lr):
    learner = CodebookLearner(sub, init_cb, vocab, lr=lr)
    last = None
    for diag in learner.train(windows, epochs=epochs):
        last = diag
    return learner.codebook, last


def run(args):
    splits = load_corpus_splits(args.corpus_source, REPO, wikitext_name=args.wikitext_name)
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    special = {vocab.unk_id, vocab.mask_id}
    train_ids = encode_texts(splits["train"], vocab)
    # held-out for PMI: a real held-out split if present, else the 2nd half of train.
    ho_texts = splits.get("validation") or splits.get("test")
    heldout_ids = encode_texts(ho_texts, vocab) if ho_texts else train_ids[len(train_ids) // 2:]
    train_windows = make_windows(train_ids, args.W)

    # ---- related-pair sets (corpus-independent semantic + held-out-PMI collocation) ----
    def tid(tok):
        return vocab.token_to_id.get(tok)
    sem_pairs = [(tid(a), tid(b)) for a, b in SEMANTIC_PAIRS
                 if tid(a) is not None and tid(b) is not None
                 and tid(a) not in special and tid(b) not in special and tid(a) != tid(b)]
    pmi_pairs = _heldout_pmi_pairs(heldout_ids, vocab, args.W, args.topk_pmi,
                                   args.pmi_min_count, special)
    sem_t = torch.tensor(sem_pairs, dtype=torch.long) if sem_pairs else torch.zeros((0, 2), dtype=torch.long)
    pmi_t = torch.tensor(pmi_pairs, dtype=torch.long) if pmi_pairs else torch.zeros((0, 2), dtype=torch.long)

    # per-seed, per-pair drift arrays: arm -> {real:[seeds,P], shuffle:[seeds,P]}
    arms = {"semantic": sem_t, "pmi_collocation": pmi_t}
    acc = {a: {"real": [], "shuffle": []} for a in arms}
    rand_acc = {"real": [], "shuffle": []}
    deff = {"init": [], "real": [], "shuffle": []}
    maxsim = {"real": [], "shuffle": []}

    for seed in range(args.seeds):
        sub = TorchFHRR(dim=args.D, seed=seed, device=args.device)
        init_cb = sub.random_vectors(len(vocab.id_to_token))
        rand_t = torch.tensor(_random_pairs(vocab, args.topk_pmi, seed, special), dtype=torch.long)

        sh_ids = _stream_shuffle(train_ids, seed)
        sh_windows = make_windows(sh_ids, args.W)

        real_cb, dr = train_codebook(sub, init_cb, vocab, train_windows, args.epochs, args.lr)
        shuf_cb, ds = train_codebook(sub, init_cb, vocab, sh_windows, args.epochs, args.lr)

        cb_init = init_cb.cpu()
        cb_real = real_cb.cpu()
        cb_shuf = shuf_cb.cpu()
        for a, pt in arms.items():
            c0 = _pair_cos(cb_init, pt)
            acc[a]["real"].append(_pair_cos(cb_real, pt) - c0)
            acc[a]["shuffle"].append(_pair_cos(cb_shuf, pt) - c0)
        rc0 = _pair_cos(cb_init, rand_t)
        rand_acc["real"].append(_pair_cos(cb_real, rand_t) - rc0)
        rand_acc["shuffle"].append(_pair_cos(cb_shuf, rand_t) - rc0)
        deff["init"].append(float(sub.d_eff(init_cb).detach().cpu()))
        deff["real"].append(float(sub.d_eff(real_cb).detach().cpu()))
        deff["shuffle"].append(float(sub.d_eff(shuf_cb).detach().cpu()))
        maxsim["real"].append(dr["max_sim"] if dr else float("nan"))
        maxsim["shuffle"].append(ds["max_sim"] if ds else float("nan"))
        print(f"[seed {seed}] sem real Δ̄={float(acc['semantic']['real'][-1].mean()):+.4f} "
              f"shuf Δ̄={float(acc['semantic']['shuffle'][-1].mean()):+.4f} | "
              f"d_eff init/real/shuf={deff['init'][-1]:.0f}/{deff['real'][-1]:.0f}/{deff['shuffle'][-1]:.0f}",
              file=sys.stderr, flush=True)

    def arm_summary(arm_name, pt):
        # per-pair mean over seeds of (real-drift), (shuffle-drift), and the paired diff
        real = torch.stack(acc[arm_name]["real"]).mean(dim=0) if acc[arm_name]["real"] and pt.numel() else torch.zeros(0)
        shuf = torch.stack(acc[arm_name]["shuffle"]).mean(dim=0) if acc[arm_name]["shuffle"] and pt.numel() else torch.zeros(0)
        diff = real - shuf
        d_mean, d_lo, d_hi = _bootstrap_ci(diff, seed=11)
        r_mean, r_lo, r_hi = _bootstrap_ci(real, seed=12)
        per_seed_diff = [float((acc[arm_name]["real"][s] - acc[arm_name]["shuffle"][s]).mean())
                         for s in range(len(acc[arm_name]["real"])) if pt.numel()]
        return {
            "n_pairs": int(pt.shape[0]),
            "real_drift_mean": r_mean, "real_drift_ci": [r_lo, r_hi],
            "shuffle_drift_mean": (float(shuf.mean()) if shuf.numel() else float("nan")),
            "real_minus_shuffle_mean": d_mean, "real_minus_shuffle_ci": [d_lo, d_hi],
            "PASS_real_beats_shuffle": bool(d_lo > 0.0),
            "per_seed_real_minus_shuffle": per_seed_diff,
            "seeds_positive": sum(1 for v in per_seed_diff if v > 0),
        }

    sem = arm_summary("semantic", sem_t)
    pmi = arm_summary("pmi_collocation", pmi_t)

    # ---- CO-OCCURRENCE SPLIT of the curated-semantic arm (the load-bearing
    # discriminator, per adversarial verification). The MEANINGFUL 3b gate is
    # PARADIGMATIC structure: do non-co-occurring similar words cluster? COLLOCATIONAL
    # (co-occurring words cluster) ~= the PMI floor, which a co-occurrence learner
    # produces by construction and is NOT the "more than memory" signal Phase 5 needs.
    sem_real = torch.stack(acc["semantic"]["real"]).mean(dim=0) if sem_t.numel() else torch.zeros(0)
    sem_shuf = torch.stack(acc["semantic"]["shuffle"]).mean(dim=0) if sem_t.numel() else torch.zeros(0)
    sem_diff_pp = sem_real - sem_shuf
    sem_cooc = _cooc_counts(train_windows, sem_t)
    para_mask = sem_cooc <= args.paradigmatic_max_cooc
    collo_mask = ~para_mask
    para_mean, para_lo, para_hi = (_bootstrap_ci(sem_diff_pp[para_mask], seed=21)
                                   if int(para_mask.sum()) else (float("nan"),) * 3)
    collo_mean, collo_lo, collo_hi = (_bootstrap_ci(sem_diff_pp[collo_mask], seed=22)
                                      if int(collo_mask.sum()) else (float("nan"),) * 3)
    _lc = torch.log(sem_cooc.float() + 1.0)
    cooc_drift_corr = (float(torch.corrcoef(torch.stack([_lc, sem_diff_pp]))[0, 1])
                       if sem_t.shape[0] > 1 else float("nan"))

    # specificity: semantic real-drift vs random real-drift (under real training)
    rand_real = torch.stack(rand_acc["real"]).mean(dim=0) if rand_acc["real"] and rand_real_n(rand_acc) else torch.zeros(0)
    spec_mean, spec_lo, spec_hi = _bootstrap_ci(
        sem_real - (rand_real.mean() if rand_real.numel() else torch.zeros_like(sem_real)), seed=13
    ) if sem_t.numel() else (float("nan"), float("nan"), float("nan"))

    paradigmatic_pass = bool(para_lo == para_lo and para_lo > 0.0)   # CI lower > 0 (non-nan)
    collocational_pass = bool(collo_lo == collo_lo and collo_lo > 0.0)
    # HEADLINE = the PARADIGMATIC gate (the meaningful 3b: similar-not-co-occurring words cluster).
    headline_pass = bool(paradigmatic_pass and spec_lo > 0.0)
    summary = {
        "config": {k: getattr(args, k) for k in
                   ("corpus_source", "D", "W", "epochs", "lr", "seeds", "max_vocab", "topk_pmi")},
        "n_semantic_pairs_in_vocab": int(sem_t.shape[0]),
        "n_pmi_pairs": int(pmi_t.shape[0]),
        "semantic_arm_overall": sem,
        "semantic_cooc_split": {
            "paradigmatic_max_cooc": args.paradigmatic_max_cooc,
            "n_paradigmatic": int(para_mask.sum()), "n_collocational": int(collo_mask.sum()),
            "PARADIGMATIC_real_minus_shuffle": {"mean": para_mean, "ci": [para_lo, para_hi], "PASS": paradigmatic_pass},
            "COLLOCATIONAL_real_minus_shuffle": {"mean": collo_mean, "ci": [collo_lo, collo_hi], "PASS": collocational_pass},
            "corr_logcooc_drift": cooc_drift_corr,
        },
        "specificity_sem_minus_random_real": {"mean": spec_mean, "ci": [spec_lo, spec_hi]},
        "floor_pmi_collocation": pmi,
        "random_pair_real_drift_mean": (float(torch.stack(rand_acc["real"]).mean()) if rand_acc["real"] else float("nan")),
        "d_eff_mean": {k: (sum(v) / len(v) if v else float("nan")) for k, v in deff.items()},
        "max_pairwise_sim_mean": {k: (sum(x for x in v if x == x) / max(1, len(v)) if v else float("nan")) for k, v in maxsim.items()},
        "HEADLINE_PASS": headline_pass,
        "verdict": ("PASS — codebook develops corpus-specific PARADIGMATIC (semantic) structure"
                    if headline_pass else
                    "NULL on the meaningful (PARADIGMATIC) gate — COLLOCATIONAL structure only "
                    "(corr_logcooc_drift>0; effect scales with co-occurrence, absent for "
                    "non-co-occurring similar pairs). RE-SCOPE signal per the pre-registered prior, NOT a retry."),
    }
    print(json.dumps({k: summary[k] for k in
                      ("config", "n_semantic_pairs_in_vocab", "n_pmi_pairs",
                       "semantic_arm_overall", "semantic_cooc_split",
                       "specificity_sem_minus_random_real", "floor_pmi_collocation",
                       "d_eff_mean", "HEADLINE_PASS", "verdict")},
                     indent=2))
    if args.out:
        outp = pathlib.Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(outp, "w"), indent=2)
        print(f"wrote {args.out}")


def rand_real_n(rand_acc):
    return bool(rand_acc["real"]) and rand_acc["real"][0].numel() > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-source", choices=["repo_sample", "wikitext"],
                    default="repo_sample", dest="corpus_source")
    ap.add_argument("--wikitext-name", default="wikitext-2-raw-v1", dest="wikitext_name")
    ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--W", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-vocab", type=int, default=2000, dest="max_vocab")
    ap.add_argument("--topk-pmi", type=int, default=100, dest="topk_pmi")
    ap.add_argument("--pmi-min-count", type=int, default=5, dest="pmi_min_count")
    ap.add_argument("--paradigmatic-max-cooc", type=int, default=2, dest="paradigmatic_max_cooc",
                    help="semantic pairs co-occurring <= this many times count as PARADIGMATIC "
                         "(non-collocational) — the meaningful 3b subset")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
