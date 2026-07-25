#!/usr/bin/env python3
"""Generate `reports/INDEX.md` — one row per numbered report.

**Why.** `CLAUDE.md` makes "grep before re-running a mechanism by name" a binding
rule, but the prescribed scope (`reports/ notes/emergent-codebook/`) returns
**zero hits on its own canonical example** (`alpha_freq_lambda`, which
`reports/040_freq_weighted_alpha_sweep.md` resolved). On live-line mechanisms it
finds 0/2 for `bennafusi` and 1/4 for `d_super` — every miss sitting in
`CONTEXT-B.md` or the retrospective, both outside the scope. A silent-wrong-answer
grep is worse than no grep: it returns "never built" and the next session rebuilds
something that already exists with tests. That is exactly how Benna-Fusi got
reimplemented as a 27-line class in an experiment script while a 965-line, 26-test
implementation sat in `phase4/consolidation.py`.

Note the schema constraint: report entries are a mix of flat `NNN_*.md` files and
`NNN_*/report.md` directories, several numbers collide (005 three ways; 048-058
each used once as a flat file and once as an unrelated directory), and nine
different directories each contain a file named `02_phase2_retrieval_baseline.md`.
So the index keys on the **full path**, never the basename.

Usage: `python scripts/build_reports_index.py [--check]`
       `--check` exits non-zero if the index is stale (for CI).
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"
INDEX = REPORTS / "INDEX.md"

# Ordered: a negative verdict must win over an incidental "PASS" elsewhere in the
# same line. Report 132's status reads "NULL (depth doesn't help...)" while its
# body discusses passing a ceiling-guard; scoring PASS there would invert it.
VERDICT_PATTERNS = [
    (re.compile(r"\bREDUNDANT\b", re.I), "REDUNDANT"),
    (re.compile(r"\bNO GRADUATION\b|\bNULL[- ]MONOTONE\b|\bNOT a graduation\b", re.I), "NO GRADUATION"),
    (re.compile(r"\bFALSIFIED\b", re.I), "FALSIFIED"),
    (re.compile(r"\bOVERTURN(?:S|ED)?\b", re.I), "OVERTURNED"),
    (re.compile(r"\bNULL\b", re.I), "NULL"),
    (re.compile(r"\bINERT\b", re.I), "INERT"),
    (re.compile(r"\bPARTIAL\b", re.I), "PARTIAL"),
    (re.compile(r"\bFIRST PASS\b|\bGRADUATES?\b|\bPASSES\b|\bPASS\b", re.I), "PASS"),
    (re.compile(r"\bCLEARED\b|\bINTEGRATED\b|\bBANKED\b", re.I), "CLEARED"),
]

# Mechanisms worth finding by name before rebuilding one. Aliases matter: the
# canonical grep example `alpha_freq_lambda` appears nowhere in reports/ because
# Report 040 calls it "freq-weighted α" — the knob name and the prose name differ,
# which is precisely how a scoped grep returns a false "never built".
KEYWORD_ALIASES = {
    "benna-fusi": [r"benna"],
    "ewc": [r"\bEWC\b", r"weight[- ]anchor", r"frozen[- ]L2"],
    "replay": [r"replay"],
    "consolidation": [r"consolidat"],
    "pseudo-rehearsal": [r"pseudo[- ]?(?:sleep|rehearsal)"],
    "two-timescale": [r"two[- ]timescale", r"multi[- ]timescale"],
    "FTSR": [r"\bFTSR\b"],
    "super-additivity": [r"d_super", r"super[- ]addit"],
    "SGNS": [r"\bSGNS\b"],
    "NMF": [r"\bNMF\b"],
    "SVD": [r"\bSVD\b"],
    "k-WTA": [r"k-WTA"],
    "k-means": [r"k-means"],
    "hopfield": [r"hopfield"],
    "FHRR": [r"\bFHRR\b"],
    "codebook": [r"codebook"],
    "decorrelator": [r"decorrelat"],
    "hetero-write": [r"hetero"],
    "HAM": [r"\bHAM\b"],
    "metastability": [r"metastab"],
    "cap-coverage": [r"cap-coverage", r"coverage_lambda"],
    "freq-weighted-alpha": [r"alpha_freq_lambda", r"freq(?:uency)?[- ]weighted"],
    "grow_G": [r"grow_G"],
    "TEM": [r"\bTEM\b"],
    "successor-repr": [r"successor", r"\bSR\b"],
    "SFA": [r"\bSFA\b"],
    "habituation": [r"habituation"],
    "frustrated-phase": [r"frustrated"],
    "capacity": [r"capacity"],
    "B-KILL": [r"B-KILL"],
    "scramble": [r"scrambl"],
    "transfer": [r"\btransfer\b"],
    "compositional": [r"compositional"],
    "bundle-first": [r"bundle-first"],
}


def bet_of(num: int, text: str) -> str:
    if num >= 133:
        return "B"
    if "CONTEXT-B" in text and num >= 130:
        return "B"
    return "A"


def entries():
    seen = []
    for p in sorted(REPORTS.rglob("*.md")):
        rel = p.relative_to(REPORTS)
        m = re.match(r"^(\d{3})[_-]", rel.parts[0])
        if not m:
            continue
        # a directory report is canonically its report.md
        if len(rel.parts) > 1 and rel.name != "report.md":
            continue
        seen.append((int(m.group(1)), p))
    return sorted(seen, key=lambda t: (t[0], str(t[1])))


def summarize(path: pathlib.Path):
    text = path.read_text(errors="replace")
    head = text[:4000]
    title = ""
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    if not title:
        title = path.stem.replace("_", " ")
    title = re.sub(r"\s+", " ", title)[:110]

    # Probe in order of authority: the explicit Status/Verdict line, then the
    # title (which by convention carries the verdict), then the opening prose.
    probes = []
    for field in (r"\*\*Status:?\*\*", r"\*\*Verdict:?\*\*"):
        m2 = re.search(field + r"\s*(.{0,200})", head)
        if m2:
            probes.append(m2.group(1))
    probes.append(title)
    probes.append(head)

    verdict = ""
    for probe in probes:
        for pat, label in VERDICT_PATTERNS:
            if pat.search(probe):
                verdict = label
                break
        if verdict:
            break

    kws = sorted(
        name for name, pats in KEYWORD_ALIASES.items()
        if any(re.search(p, text, re.I) for p in pats)
    )
    return title, verdict or "—", kws[:8], text


def build() -> str:
    rows, counts = [], {}
    for num, path in entries():
        title, verdict, kws, text = summarize(path)
        counts[num] = counts.get(num, 0) + 1
        rel = path.relative_to(REPORTS)
        rows.append((num, str(rel), title, bet_of(num, text), verdict, ", ".join(kws)))

    dupes = sorted(n for n, c in counts.items() if c > 1)
    out = [
        "# Reports index",
        "",
        "*Generated by `scripts/build_reports_index.py` — do not hand-edit.*",
        "",
        "Read this **before** re-running any mechanism by name (`CLAUDE.md`",
        "§measurement rule 5). Keyed on full path: report numbers collide and nine",
        "directories share a basename, so basename lookups are unreliable.",
        "",
        f"{len(rows)} entries.",
    ]
    if dupes:
        out += ["", f"**Colliding numbers** (same number, unrelated work): "
                    f"{', '.join(f'{d:03d}' for d in dupes)}."]
    out += [
        "",
        "| # | Path | Bet | Verdict | Mechanisms | Title |",
        "|---|------|-----|---------|------------|-------|",
    ]
    for num, rel, title, bet, verdict, kws in rows:
        t = title.replace("|", "\\|")
        out.append(f"| {num:03d} | [`{rel}`]({rel}) | {bet} | {verdict} | {kws} | {t} |")
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the index is stale")
    args = ap.parse_args()

    content = build()
    if args.check:
        current = INDEX.read_text() if INDEX.exists() else ""
        if current.strip() != content.strip():
            print("reports/INDEX.md is stale — run scripts/build_reports_index.py", file=sys.stderr)
            return 1
        print("reports/INDEX.md is current")
        return 0
    INDEX.write_text(content)
    print(f"wrote {INDEX.relative_to(REPO)} ({content.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
