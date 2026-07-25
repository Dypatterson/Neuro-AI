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
#
# CORRECTION 2026-07-25: this file used to claim ".git/hooks/pre-commit already
# delegates here." It does not, and cannot — **git hooks are never cloned**.
# `find .git/hooks -type f -not -name '*.sample'` returned 0 on a fresh checkout
# and `core.hooksPath` was unset, so the guard this repo believed it had has
# never run anywhere but the machine that installed it by hand. It now runs in
# CI (.github/workflows/tests.yml), which is the only place enforcement survives
# a clone.
#
# The byte cap is a backstop against the real 93 KB regrowth, not the primary
# rule — a total-size cap is satisfiable by line-length gaming (the longest
# single line in STATUS.md reached 2,011 characters under the old cap). The
# per-entry cap below is the one that actually keeps the file readable.
set -euo pipefail

MAX_LINES=250
MAX_BYTES=24576   # 24 KB backstop
MAX_ENTRY_CHARS=900   # a "Recent updates" bullet longer than this belongs in a report

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

# The rule that actually matters: a Recent-updates bullet longer than
# MAX_ENTRY_CHARS is narrative, and narrative belongs in the numbered report.
#
# Scoped to *newly added* lines only. Banked entries from 2026-06 exceed the cap,
# and rewriting settled history to satisfy a guard would destroy record detail for
# no gain — the point is to stop regrowth, not to relitigate the log. The five
# fixed "Current state" fields are exempt for the same reason: they are a
# structured block, not log entries.
added=$(git -C "$ROOT" diff --cached -U0 -- STATUS.md 2>/dev/null | grep '^+' | sed 's/^+//' || true)
[ -n "$added" ] || added=$(git -C "$ROOT" diff -U0 -- STATUS.md 2>/dev/null | grep '^+' | sed 's/^+//' || true)

long=$(printf '%s\n' "$added" | awk -v cap="$MAX_ENTRY_CHARS" '
  /^- \*\*[0-9]{4}-[0-9]{2}-[0-9]{2}/ {
    if (length($0) > cap) { printf "  %d chars: %.90s...\n", length($0), $0 }
  }
')

if [ -n "$long" ]; then
  echo "ERROR: new STATUS.md Recent-updates entries over ${MAX_ENTRY_CHARS} chars:" >&2
  echo "$long" >&2
  echo "       Summarize in one line and link the report. See CLAUDE.md §Session start." >&2
  exit 1
fi

exit 0
