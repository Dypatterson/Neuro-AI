"""Throwaway analysis: characterize MCD's divergence axis — is mcd1 test novel at the
PRIMITIVE-FILLER level (a primitive in a local clause-context never seen with it in train)
or only at the CLAUSE-COMBINATION level (every clause seen, only their co-occurrence novel)?

This decides whether a literal Report-142 port (role/filler factorization + align the 4
primitive verbs' role-halves) has any purchase on MCD, or would null for a target-mismatch
reason. Reads data only; reuses exp87.load_pairs.
"""
from __future__ import annotations

import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[0].parent if (pathlib.Path(__file__).resolve().parents[0].name != "Neuro-AI") else pathlib.Path(__file__).resolve().parents[0]
REPO = pathlib.Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("exp87", REPO / "experiments" / "87_betb_scan_gap.py")
exp87 = importlib.util.module_from_spec(_s); _s.loader.exec_module(exp87)
MCD = REPO / "data" / "scan" / "mcd_split"

PRIMS = {"jump", "walk", "look", "run"}          # the 4 primitive verbs (filler axis)
CONJ = {"and", "after"}


def split_clauses(cmd):
    """Split at first top-level conjunction (and/after). Returns list of clause token-tuples."""
    for i, t in enumerate(cmd):
        if t in CONJ:
            return [tuple(cmd[:i]), tuple(cmd[i + 1:])]
    return [tuple(cmd)]


def templatize(clause):
    """Abstract the primitive verb to V (filler-agnostic clause template)."""
    return tuple("V" if t in PRIMS else t for t in clause)


def prim_of(clause):
    ps = [t for t in clause if t in PRIMS]
    return ps[0] if ps else None


def analyze(split):
    tr = exp87.load_pairs(str(MCD / f"tasks_train_{split}.txt"))
    te = exp87.load_pairs(str(MCD / f"tasks_test_{split}.txt"))
    tr_cmds = [c for c, _ in tr]
    te_cmds = [c for c, _ in te]

    # ---- clause inventories from TRAIN
    train_clauses = set()           # exact clause token-tuples
    train_templates = set()         # filler-abstracted clause templates
    train_prim_template = set()     # (primitive, template) pairs actually seen
    train_full_templates = set()    # whole-command template (filler-abstracted)
    for cmd in tr_cmds:
        train_full_templates.add(templatize(tuple(cmd)))
        for cl in split_clauses(cmd):
            train_clauses.add(cl)
            t = templatize(cl)
            train_templates.add(t)
            p = prim_of(cl)
            if p is not None:
                train_prim_template.add((p, t))

    n = len(te_cmds)
    full_template_novel = 0          # whole-command template never in train (exp94's "100% novel" claim)
    all_clauses_seen = 0             # every clause (exact) of the test cmd was seen as a train clause
    has_filler_hole = 0             # >=1 clause is filler-novel: template seen, but NOT with THIS primitive
    n_clauses = 0
    clause_seen = 0                  # exact clause seen in train
    clause_template_seen = 0         # clause template seen in train
    clause_filler_novel = 0          # template seen in train but (this prim, template) NOT seen
    two_clause = 0

    for cmd in te_cmds:
        if templatize(tuple(cmd)) not in train_full_templates:
            full_template_novel += 1
        cls = split_clauses(cmd)
        if len(cls) == 2:
            two_clause += 1
        all_seen = True
        filler_hole = False
        for cl in cls:
            n_clauses += 1
            t = templatize(cl)
            if cl in train_clauses:
                clause_seen += 1
            else:
                all_seen = False
            if t in train_templates:
                clause_template_seen += 1
            p = prim_of(cl)
            if p is not None and t in train_templates and (p, t) not in train_prim_template:
                clause_filler_novel += 1
                filler_hole = True
        if all_seen:
            all_clauses_seen += 1
        if filler_hole:
            has_filler_hole += 1

    print(f"\n===== {split} =====")
    print(f"train cmds={len(tr_cmds)}  test cmds={n}  (test 2-clause={two_clause}, {100*two_clause/n:.1f}%)")
    print(f"train: {len(train_clauses)} distinct clauses, {len(train_templates)} distinct clause-templates, "
          f"{len(train_prim_template)} (prim,template) pairs")
    print("--- WHOLE-COMMAND axis")
    print(f"  whole-command template NOVEL (not in train): {full_template_novel}/{n} = {100*full_template_novel/n:.1f}%   "
          f"[exp94 claim: ~100%]")
    print("--- CLAUSE-COMBINATION axis (is each test clause individually familiar?)")
    print(f"  test cmds where EVERY clause (exact) seen in train: {all_clauses_seen}/{n} = {100*all_clauses_seen/n:.1f}%")
    print(f"  per-clause: exact clause seen        {clause_seen}/{n_clauses} = {100*clause_seen/n_clauses:.1f}%")
    print(f"  per-clause: clause TEMPLATE seen     {clause_template_seen}/{n_clauses} = {100*clause_template_seen/n_clauses:.1f}%")
    print("--- PRIMITIVE-FILLER axis (is the divergence about WHICH primitive fills a seen template?)")
    print(f"  test cmds with >=1 FILLER-HOLE clause (template seen, but NOT with this primitive): "
          f"{has_filler_hole}/{n} = {100*has_filler_hole/n:.1f}%")
    print(f"  per-clause filler-novel (template seen, (prim,template) NOT seen): "
          f"{clause_filler_novel}/{n_clauses} = {100*clause_filler_novel/n_clauses:.1f}%")


if __name__ == "__main__":
    for s in ("mcd1", "mcd2", "mcd3"):
        try:
            analyze(s)
        except FileNotFoundError as e:
            print(f"{s}: missing ({e})")
