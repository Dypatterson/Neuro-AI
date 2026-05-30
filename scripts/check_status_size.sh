#!/usr/bin/env bash
# Guard: keep STATUS.md a bookmark, not a log.
#
# STATUS.md regrew to 93 KB / 234 lines by 2026-05-29 — past the 87 KB that
# forced the 2026-05-24 restructure — because "Recent updates" had a per-entry
# cap but no entry-COUNT cap, and "Current state" had no size cap at all. This
# hook is the mechanical backstop: it fails the commit if STATUS.md exceeds the
# budget, so the file physically cannot silently regrow again.
#
# If this fires: move the oldest Recent-updates entries into
# notes/status-log/<month>.md and trim "Current state" back to its five fixed
# fields, then re-commit. See STATUS.md §"Maintaining this file".
#
# Install (once, per clone):  ln -sf ../../scripts/check_status_size.sh .git/hooks/pre-commit
# (this repo's .git/hooks/pre-commit already delegates here).
set -euo pipefail

MAX_LINES=250
MAX_BYTES=20480   # 20 KB

ROOT="$(git rev-parse --show-toplevel)"
STATUS="${ROOT}/STATUS.md"

[ -f "$STATUS" ] || exit 0   # nothing to check

lines=$(wc -l < "$STATUS" | tr -d ' ')
bytes=$(wc -c < "$STATUS" | tr -d ' ')

if [ "$lines" -ge "$MAX_LINES" ] || [ "$bytes" -ge "$MAX_BYTES" ]; then
  echo "ERROR: STATUS.md is over budget (${lines} lines / ${bytes} bytes; cap ${MAX_LINES} lines / ${MAX_BYTES} bytes)." >&2
  echo "       Move the oldest Recent-updates entries to notes/status-log/ and trim Current state" >&2
  echo "       to its five fixed fields before committing. See STATUS.md §\"Maintaining this file\"." >&2
  exit 1
fi

exit 0
