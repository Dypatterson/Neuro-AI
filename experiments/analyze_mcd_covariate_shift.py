"""PART 1 (Report 151) — covariate-shift / compound-divergence characterization of SCAN-MCD.

NO TRAINING. Reads data only. Bank-respecting characterization that (a) names MCD's hardness as
COMPOUND covariate-shift (not atom OOV), (b) bounds the headroom any non-leaky atom/clause prior
(incl. the held exp100 entropy soft-prior) could reach above the clause-aug floor, and (c) quantifies
the LEAKAGE in Report 149's 0.297 "atomic floor" (clause-aug is drawn from train_simple + test_simple).

This is the today-feasible salvage of "re-ground the bar" (ReCOGS/SLOG = 3-5 day build, no data here).
It addresses the COVARIATE-SHIFT seam only; the ReCOGS decoder-artifact seam does not exist on SCAN's
flat-action exact-match. See notes/betb-bar-regrounding-part1-precommit.md.

Convention (matches experiments/analyze_mcd_divergence_axis.py): primitive verbs = {jump,walk,look,run};
a clause is templatized by abstracting its primitive verb to "V"; commands split at the single top-level
conjunction (and/after).
"""
from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
SCAN = REPO / "data" / "scan"
MCD = SCAN / "mcd_split"

PRIMS = {"jump", "walk", "look", "run"}
CONJ = ("and", "after")


def load_pairs(fp):
    pairs = []
    for line in pathlib.Path(fp).read_text().splitlines():
        if not line.strip():
            continue
        cmd, act = line.split(" OUT: ")
        pairs.append((tuple(cmd[4:].split()), tuple(act.split())))
    return pairs


def split_clauses(cmd):
    """Split at first top-level conjunction. Returns list of clause token-tuples (1 or 2)."""
    for i, t in enumerate(cmd):
        if t in CONJ:
            return [tuple(cmd[:i]), tuple(cmd[i + 1:])]
    return [tuple(cmd)]


def conj_of(cmd):
    for t in cmd:
        if t in CONJ:
            return t
    return None


def templatize(clause):
    return tuple("V" if t in PRIMS else t for t in clause)


def prim_of(clause):
    ps = [t for t in clause if t in PRIMS]
    return ps[0] if ps else None


def whole_template(cmd):
    """Filler-abstracted whole-command template: (L-template, conj, R-template)."""
    cls = split_clauses(cmd)
    conj = conj_of(cmd)
    return tuple(templatize(c) for c in cls), conj


def tv_distance(train_items, test_items):
    """Total-variation distance between the empirical distributions of two multisets of hashable keys,
    plus the fraction of TEST mass on keys never seen in TRAIN (the coverage gap)."""
    from collections import Counter
    ctr, cte = Counter(train_items), Counter(test_items)
    ntr, nte = sum(ctr.values()), sum(cte.values())
    keys = set(ctr) | set(cte)
    tv = 0.5 * sum(abs(ctr[k] / ntr - cte[k] / nte) for k in keys)
    novel_mass = sum(cte[k] for k in cte if k not in ctr) / nte
    novel_types = sum(1 for k in cte if k not in ctr)
    return tv, novel_mass, novel_types, len(set(cte))


def load_clause_aug(source):
    """Replicate exp96.load_single_clause_aug. source='train+test' is the LEAKED version exp96 uses;
    'train' is the non-leaked version PART 2's redesign should use. Returns set of (cmd_tuple, act_tuple)."""
    files = {"train+test": ["tasks_train_simple.txt", "tasks_test_simple.txt"],
             "train": ["tasks_train_simple.txt"]}[source]
    pairs = []
    for f in files:
        pairs += load_pairs(str(SCAN / f))
    return [(c, a) for c, a in pairs if "and" not in c and "after" not in c]


def analyze(split):
    tr = load_mcd_train = load_pairs(str(MCD / f"tasks_train_{split}.txt"))
    te = load_pairs(str(MCD / f"tasks_test_{split}.txt"))
    tr_cmds = [c for c, _ in tr]
    te_cmds = [c for c, _ in te]
    n = len(te_cmds)

    # ---------- 1. COVARIATE-SHIFT SIGNATURE: atom (low) vs compound (high) divergence ----------
    def atoms(cmds):
        return [t for c in cmds for t in c]

    def bigrams(cmds):
        return [(c[i], c[i + 1]) for c in cmds for i in range(len(c) - 1)]

    def clause_templates(cmds):
        return [templatize(cl) for c in cmds for cl in split_clauses(c)]

    def prim_template_pairs(cmds):
        out = []
        for c in cmds:
            for cl in split_clauses(c):
                p = prim_of(cl)
                if p is not None:
                    out.append((p, templatize(cl)))
        return out

    def whole_templates(cmds):
        return [whole_template(c) for c in cmds]

    shift = {}
    for name, fn in [("atom_unigram", atoms), ("input_bigram", bigrams),
                     ("clause_template", clause_templates), ("prim_clause_template", prim_template_pairs),
                     ("whole_command_compound", whole_templates)]:
        tv, novel_mass, novel_types, n_test_types = tv_distance(fn(tr_cmds), fn(te_cmds))
        shift[name] = {"tv": round(tv, 4), "test_novel_mass": round(novel_mass, 4),
                       "test_novel_types": novel_types, "test_distinct_types": n_test_types}

    # ---------- 2. CLAUSE-AUG ATOMIC-COVERAGE (the headroom bound) ----------
    aug = load_clause_aug("train+test")                  # the set exp96 --clause-aug actually injects
    aug_cmd_set = {c for c, _ in aug}                    # exact single-clause inputs it provides
    aug_template_set = {templatize(c) for c, _ in aug}   # filler-abstracted clause templates it provides
    n_aug = len(aug)

    both_clauses_covered = 0                              # test cmd where every clause is EXACT-covered by aug
    both_templates_covered = 0
    per_clause_total = per_clause_exact = per_clause_template = 0
    multi = 0
    for c in te_cmds:
        cls = split_clauses(c)
        if len(cls) == 2:
            multi += 1
        all_exact = all_tmpl = True
        for cl in cls:
            per_clause_total += 1
            if cl in aug_cmd_set:
                per_clause_exact += 1
            else:
                all_exact = False
            if templatize(cl) in aug_template_set:
                per_clause_template += 1
            else:
                all_tmpl = False
        both_clauses_covered += int(all_exact)
        both_templates_covered += int(all_tmpl)

    # ---------- 3. LEAKAGE of the 0.297 floor (clause-aug sourced from test_simple too) ----------
    aug_train_only = {c for c, _ in load_clause_aug("train")}
    leaked_forms = aug_cmd_set - aug_train_only          # single-clause forms present ONLY because of test_simple
    te_cmd_set = set(te_cmds)
    # MCD test commands that are themselves single-clause AND handed verbatim by clause-aug:
    verbatim_test_leaks = sum(1 for c in te_cmds if conj_of(c) is None and c in aug_cmd_set)
    # per-clause gold-decode handed over (clause appears as a clause-aug command):
    # (already = per_clause_exact above)

    # ---------- 4. TEMPLATE-LEVEL LEAKAGE INVARIANT (pre-registration check) ----------
    te_whole_templates = {whole_template(c) for c in te_cmds}
    # an aug example's "whole template" is a single clause (no conj); does it match any TEST whole-template?
    aug_whole_templates = {((templatize(c),), None) for c in aug_cmd_set}
    aug_template_violations = len(aug_whole_templates & te_whole_templates)

    res = {
        "split": split, "n_train": len(tr_cmds), "n_test": n,
        "n_test_multiclause": multi, "pct_multiclause": round(100 * multi / n, 1),
        "covariate_shift": shift,
        "clause_aug": {
            "n_aug_forms": n_aug, "n_aug_distinct_cmds": len(aug_cmd_set),
            "n_aug_distinct_templates": len(aug_template_set),
            "test_cmds_all_clauses_exact_covered_pct": round(100 * both_clauses_covered / n, 1),
            "test_cmds_all_clauses_template_covered_pct": round(100 * both_templates_covered / n, 1),
            "per_clause_exact_covered_pct": round(100 * per_clause_exact / per_clause_total, 1),
            "per_clause_template_covered_pct": round(100 * per_clause_template / per_clause_total, 1),
        },
        "leakage_0297_floor": {
            "n_aug_forms_only_from_test_simple": len(leaked_forms),
            "mcd_test_cmds_handed_verbatim_by_aug": verbatim_test_leaks,
            "per_clause_gold_decode_handed_over_pct": round(100 * per_clause_exact / per_clause_total, 1),
        },
        "template_leakage_invariant": {
            "aug_whole_templates_matching_a_test_whole_template": aug_template_violations,
            "note": "0 = clause-aug is whole-command-template-safe (single-clause vs multi-clause test); "
                    "the leak is CLAUSE-level gold-decode, not whole-command-template.",
        },
    }
    return res


def fmt(res):
    s = res["split"]
    cs = res["covariate_shift"]
    ca = res["clause_aug"]
    lk = res["leakage_0297_floor"]
    out = []
    out.append(f"\n===== {s}  (train={res['n_train']} test={res['n_test']}, "
               f"{res['pct_multiclause']}% 2-clause) =====")
    out.append("  [1] COVARIATE-SHIFT SIGNATURE  (TV train->test ; test-novel-type count)")
    for k, v in cs.items():
        out.append(f"      {k:24s} TV={v['tv']:.3f}  novel-types={v['test_novel_types']:>4d}"
                   f" / {v['test_distinct_types']:<4d}  novel-mass={v['test_novel_mass']:.3f}")
    out.append("  [2] CLAUSE-AUG HEADROOM BOUND  (how much an atom/clause prior could add above the floor)")
    out.append(f"      test cmds w/ ALL clauses EXACT-covered by clause-aug : {ca['test_cmds_all_clauses_exact_covered_pct']}%")
    out.append(f"      test cmds w/ ALL clause TEMPLATES covered            : {ca['test_cmds_all_clauses_template_covered_pct']}%")
    out.append(f"      per-clause exact-covered / template-covered          : {ca['per_clause_exact_covered_pct']}% / {ca['per_clause_template_covered_pct']}%")
    out.append("  [3] LEAKAGE in Report-149's 0.297 floor  (clause-aug drawn from test_simple too)")
    out.append(f"      clause-aug forms present ONLY via test_simple         : {lk['n_aug_forms_only_from_test_simple']}")
    out.append(f"      MCD test cmds handed VERBATIM by clause-aug           : {lk['mcd_test_cmds_handed_verbatim_by_aug']}")
    out.append(f"      per-clause gold-decode handed over                   : {lk['per_clause_gold_decode_handed_over_pct']}%")
    out.append("  [4] TEMPLATE-LEVEL LEAKAGE INVARIANT (pre-registration)")
    out.append(f"      aug whole-templates matching a test whole-template   : "
               f"{res['template_leakage_invariant']['aug_whole_templates_matching_a_test_whole_template']}  (want 0)")
    return "\n".join(out)


if __name__ == "__main__":
    allres = {}
    for s in ("mcd1", "mcd2", "mcd3"):
        try:
            r = analyze(s)
            allres[s] = r
            print(fmt(r))
        except FileNotFoundError as e:
            print(f"{s}: missing ({e})")
    (REPO / "experiments" / "analyze_mcd_covariate_shift.json").write_text(json.dumps(allres, indent=2))
    print("\nwrote experiments/analyze_mcd_covariate_shift.json")
