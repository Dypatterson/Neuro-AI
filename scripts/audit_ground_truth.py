#!/usr/bin/env python3
"""Read-only audit for docs/ground-truth.

Checks:
- JSONL validity
- duplicate source IDs
- stale PDF local paths and checksums
- expected stale PDF coverage
- card path existence
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = ROOT / "docs" / "ground-truth"
MANIFEST = GROUND_TRUTH / "source_manifest.jsonl"
STALE_RESEARCH = Path("/Users/dypatterson/Desktop/Neuro-AI/research")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    failures: list[str] = []
    if not MANIFEST.exists():
        print(f"missing manifest: {MANIFEST}")
        return 1

    rows = []
    seen: set[str] = set()
    for lineno, line in enumerate(MANIFEST.read_text().splitlines(), 1):
        if not line.strip():
            failures.append(f"blank line at {MANIFEST}:{lineno}")
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(f"invalid JSON at {MANIFEST}:{lineno}: {exc}")
            continue
        source_id = row.get("source_id")
        if not source_id:
            failures.append(f"missing source_id at {MANIFEST}:{lineno}")
            continue
        if source_id in seen:
            failures.append(f"duplicate source_id: {source_id}")
        seen.add(source_id)
        rows.append(row)

    stale_rows = [row for row in rows if row.get("status") == "stale_pdf"]
    link_rows = [row for row in rows if row.get("status") == "link_only"]

    for row in rows:
        card_path = row.get("card_path")
        if not card_path:
            failures.append(f"{row.get('source_id')}: missing card_path")
            continue
        if not (ROOT / card_path).exists():
            failures.append(f"{row.get('source_id')}: missing card path {card_path}")

    for row in stale_rows:
        source_id = row["source_id"]
        local_pdf_path = row.get("local_pdf_path")
        expected_sha = row.get("checksum_sha256")
        if not local_pdf_path:
            failures.append(f"{source_id}: missing local_pdf_path")
            continue
        pdf = Path(local_pdf_path)
        if not pdf.exists():
            failures.append(f"{source_id}: missing PDF {pdf}")
            continue
        if expected_sha and sha256(pdf) != expected_sha:
            failures.append(f"{source_id}: checksum mismatch for {pdf}")

    if STALE_RESEARCH.exists():
        pdfs = {str(p) for p in STALE_RESEARCH.glob("*.pdf")}
        indexed = {row.get("local_pdf_path") for row in stale_rows}
        missing = sorted(pdfs - indexed)
        extra = sorted(indexed - pdfs)
        for path in missing:
            failures.append(f"stale PDF not indexed: {path}")
        for path in extra:
            failures.append(f"indexed stale PDF not in folder: {path}")

    print(f"manifest rows: {len(rows)}")
    print(f"stale_pdf rows: {len(stale_rows)}")
    print(f"link_only rows: {len(link_rows)}")

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
