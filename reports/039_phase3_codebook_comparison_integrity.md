# Report 039 — Phase 3 codebook-comparison data integrity: investigation and disposition

**Date:** 2026-05-16
**Active phase:** Phase 4 graduated (report 038); investigating Phase 3 documentation issue flagged by 2026-05-14 audit
**Why this experiment now:** STATUS.md blocker #7 — the 2026-05-14 repo audit flagged `phase3_comparison.json` for HIGH-severity data integrity (byte-identical per-seed results across condition pairs). Investigation brief at [notes/notes/2026-05-15-phase3-codebook-comparison-data-integrity.md](../notes/notes/2026-05-15-phase3-codebook-comparison-data-integrity.md). Phase 4 has now graduated using the `phase3c_reconstruction` codebook; this blocker is the last unresolved Phase 3-era data-integrity claim and either invalidates or doesn't invalidate the Phase 4 starting state.

---

## Headline finding

**The audit's HIGH-severity flag describes a labeling bug, not a data-integrity bug, and Phase 4 is unaffected.**

There are **3 distinct codebook tensors** on disk that get labeled as **6 conditions** in `phase3_comparison.json`. The duplicates are *intentional copies* of phase3a's outputs that phase3b and phase3c each save into their own output directories for self-containment (lines 488–550 in `experiments/03_phase3b_error_driven.py` and `experiments/04_phase3c_reconstruction.py`). The comparison script `experiments/31_phase3_comparison.py` enumerates these copies as if they were independent experimental conditions, producing the "byte-identical pair" pattern the audit flagged.

**Phase 4 uses `phase3c_codebook_reconstruction.pt`, confirmed genuinely unique** (md5 of tensor bytes `68df5fbc…`, distinct from all other Phase 3-era codebooks). The graduation result in [report 038](038_phase4_d1_graduation.md) is not affected by this finding.

---

## The data

### Tensor-byte equivalence groups

Loading all 7 codebook artifacts on disk and hashing the underlying tensor bytes (the file-level md5s differ because `torch.save` writes slightly different wrapper metadata per write, even when the tensor is identical):

| Group MD5 | Members |
|---|---|
| `d4e77fd5…` | `phase2_baseline`, `phase3b_random`, `phase3c_random` (all 3 are the same random codebook) |
| `1f9f90cc…` | `phase3b_hebbian`, `phase3c_hebbian` (both are the same trained-Hebbian codebook) |
| `c5394a7c…` | `phase3b_error_driven` (unique) |
| `68df5fbc…` | `phase3c_reconstruction` (unique; **Phase 4 input**) |

So 7 files on disk, 4 distinct tensors. The audit observed 3 byte-identical pairs in the comparison results; the verified breakdown is:

- `random_baseline_matrix` ≡ `random_phase3c`: **TRUE duplicate** (same tensor).
- `hebbian_phase3c` ≡ `hebbian_phase3b`: **TRUE duplicate** (same tensor).
- `reconstruction` ≡ `error_driven`: **NOT a duplicate** — 5.9M of 8.4M tensor elements differ, max abs diff 2.0. The audit overclaimed on this pair. Identical row only at seed 17 (a coincidence under small-n eval where 0.205 = 9/44 is a frequent draw across many codebooks).

### Where the duplicates come from

`experiments/03_phase3b_error_driven.py` lines 477, 488, 549–550:
```python
random_codebook_path = phase3a_dir / "phase3a_codebook_random.pt"
hebbian_codebook_path = phase3a_dir / "phase3a_codebook_learned.pt"
…
random_codebook = load_codebook(random_codebook_path, …)
hebbian_codebook = load_codebook(hebbian_codebook_path, …)
…
save_codebook(random_codebook, output_dir / "phase3b_codebook_random.pt")
save_codebook(hebbian_codebook, output_dir / "phase3b_codebook_hebbian.pt")
```

`experiments/04_phase3c_reconstruction.py` lines 476, 488, 545–547: same pattern. Both scripts deliberately re-save phase3a's `random` and `learned` (Hebbian) artifacts into their own output dirs, then train a new objective on top (error_driven for phase3b; reconstruction for phase3c). The only *new* artifacts each phase produces are:

- `phase3b_codebook_error_driven.pt` (genuinely new training)
- `phase3c_codebook_reconstruction.pt` (genuinely new training)

This is intentional design for per-phase reproducibility. **The bug is not in the training scripts.** The bug is that `experiments/31_phase3_comparison.py` enumerates these copies as if they were independent experimental conditions, then writes a comparison CSV/JSON with rows for `random_baseline_matrix`, `random_phase3c`, `hebbian_phase3c`, `hebbian_phase3b` as if they represented four independent samples of two underlying conditions. They don't.

---

## What this means for the existing analysis

### Report 017's headline conclusion stands

[Report 017](017_phase3_codebook_comparison.md) concluded:
- The Phase 2 0.205 "baseline" number was single-seed; the 6-seed pooled baseline is **0.089 [0.060, 0.125]**.
- Hebbian codebooks score 0.104 [0.071, 0.141] — nominally above baseline but CI overlaps.
- Reconstruction and error-driven codebooks score below baseline (0.054 / 0.055).

These conclusions are **unaffected** by the duplicate-artifact finding because:
- The Hebbian comparison is genuine (the Hebbian artifact is a real trained codebook from phase3a, regardless of which path it's loaded from).
- The reconstruction vs random comparison is genuine (the reconstruction artifact is genuinely distinct).
- The error_driven vs random comparison is genuine.
- The duplicate listings just add redundant rows that average to the same mean. Removing them would shrink the table to 4 unique conditions but wouldn't change any pooled CI or any verdict.

### Report 017's *interpretation* of the random-pair duplicate was wrong

[Report 017 §3 "Random codebooks reproduce identically across the two random artifacts"](017_phase3_codebook_comparison.md):

> The two artifacts are different random codebooks but produce the same Recall@1 because both encode the same windows with algebraically equivalent role-filler bundles — the *which* random codebook doesn't matter at this metric, only that it's random.

This explanation is wrong. The two artifacts are *literally the same tensor*. The duplicate result is from artifact reuse, not from content-blindness of random codebooks. A correction note has been added to report 017 pointing to this report.

### Phase 4 graduation is unaffected

[Report 038](038_phase4_d1_graduation.md) and [report 026](026_phase4_verification_design_spec.md) use `phase3c_codebook_reconstruction.pt`, which we verified is byte-distinct from every other Phase 3-era codebook. The Phase 4 result does not inherit this issue.

---

## Disposition

### Required cleanup (low priority, documentation-only)

1. `experiments/31_phase3_comparison.py` should be patched to deduplicate by tensor hash before enumerating conditions — or to label the copies explicitly as "same artifact as phase3a." Otherwise future re-runs reproduce the same misleading framing.
2. `phase3_comparison.json` and `phase3_comparison_6seed.json` should be left in place as historical artifacts but read as **4 unique conditions evaluated under 6 labels**, not 6 independent conditions.

### Not required

- Re-running the Phase 3 comparison with genuinely-independent random codebook draws is not load-bearing. The comparison already establishes that no Phase 3 learning objective beats the random baseline at the Phase 2 envelope with CI-disjoint signal; that conclusion doesn't change with independent draws.
- Re-running the Hebbian training with different seeds is not load-bearing either. The Hebbian result is single-seed-trained (phase3a) and that's a known property of the artifact; multi-seed-trained Hebbian is a different experiment, not a fix.

### Closes blocker #7

STATUS.md blocker #7 is closed by this report. The bug is a labeling artifact in `31_phase3_comparison.py`, not a data integrity issue affecting downstream work.

---

## Anti-homunculus check

Pure measurement / forensic work. No mechanism added. Passes trivially.

---

## Raw data

- This investigation: this report.
- Original audit flag: `audit-report-2026-05-14.md §2.3, §Appendix A`.
- Investigation brief (pre-investigation): [`notes/notes/2026-05-15-phase3-codebook-comparison-data-integrity.md`](../notes/notes/2026-05-15-phase3-codebook-comparison-data-integrity.md).
- Comparison data: [`reports/phase3_comparison.json`](phase3_comparison.json), [`reports/phase3_comparison_6seed.json`](phase3_comparison_6seed.json).
- Earlier framing report: [`reports/017_phase3_codebook_comparison.md`](017_phase3_codebook_comparison.md) (gets a correction note).
- Source scripts: [`experiments/03_phase3b_error_driven.py`](../experiments/03_phase3b_error_driven.py), [`experiments/04_phase3c_reconstruction.py`](../experiments/04_phase3c_reconstruction.py), [`experiments/31_phase3_comparison.py`](../experiments/31_phase3_comparison.py).
