"""Generates notebooks/058_integration_wiring_d4096_colab.ipynb (run locally; output committed).

Report 058, graduation-scale convincer: the END-TO-END WIRING CHECK at D=4096 WikiText-2
on GPU. Drives identical masked-encoding data through BOTH the standalone exp-56 harness
(the code behind Reports 055/056) AND the integrated production path
(OnlineCodebookUpdater.observe(cue=) -> consolidate_hetero() -> recall_hetero()), and proves
they produce BIT-IDENTICAL H + basin indices at the real substrate dimension. This is the
D=4096 confirmation of the local CPU result (Report 058: 24/24 cells bit-identical at
D=1024/2048).

Self-contained: the notebook clones the pushed branch for src/ + exp-56 (already on origin),
and base64-materializes experiments/57_e2e_integration_wiring_check.py if it is not yet on the
branch — so it runs whether or not this session's harness has been pushed.
"""
import base64
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXP57 = ROOT / "experiments" / "57_e2e_integration_wiring_check.py"
EXP57_B64 = base64.b64encode(EXP57.read_bytes()).decode()


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


cells = []

cells.append(md(
"""# 🔌 Integration wiring certificate — the graduated mechanism, *through the real consolidation path* @ **D=4096** WikiText-2

**Branch:** `consolidation/role-structure` · FHRR + Modern Hopfield + emergent codebook · _Report 058_

Reports 055/056 graduated a surgical consolidation write (**heteroassociative delta-rule write + L2 cue-space decorrelator**) and proved it is a **role-selective associative memory**. Report 057 folded it into the production orchestrator `OnlineCodebookUpdater` (default-off → byte-identical). Report 058 asked the obvious next question: *does driving real data through the integrated public API reproduce the validated mechanism?* — and answered **yes, bit-identically**, locally at D=1024/2048.

**This notebook is the D=4096 graduation-scale confirmation, on GPU.** It runs a **head-to-head** on the SAME masked-encoding windows:

| | Path A (reference) | Path B (integrated) |
|---|---|---|
| write | `exp56.write_H` | `OnlineCodebookUpdater.observe(cue=)×N` → `consolidate_hetero()` |
| read | `exp56.write_read` | `recall_hetero(cue)` |
| code | the standalone code behind 055/056 | the production fold-in (Report 057) |

Both paths terminate in the **same** leaf functions with **identical** hyper-parameters (`lr=0.5, epochs=20, ridge=1e-5, β=10, max_iter=12`), so a **`torch.equal`** match on the dense `H` *and* the basin indices proves the fold-in introduced **zero computational drift**. The wiring witness is **A == B bit-identical**; the role-Selectivity-Δ is anchored to the published Report-056 D=4096 numbers (obs 1/2/3 = 0.32 / 0.73 / 0.91).

In-sample only (a wiring check, by construction): memorization-recall is the **Phase-3 floor** (contextual-completion, not prediction). Held-out compositional generalization ≈ chance is the *predicted* Phase-3 floor (a Phase-5 deliverable, deferred by design — **not** the project ceiling; see `CONTEXT.md`).

> ⏱️ ~5–15 min on a T4."""))

cells.append(md("## 1 · GPU + fetch the code\n_Clones the pushed branch (src/ substrate + exp-56). If `experiments/57_…` isn't on the branch yet, it is materialized from an embedded copy so the notebook is self-contained._"))

_setup = '''!nvidia-smi -L 2>/dev/null || echo "⚠️  set Runtime ▸ Change runtime type ▸ GPU (T4)"
import os, subprocess, base64
from getpass import getpass
REPO, BRANCH, DEST = "Dypatterson/Neuro-AI", "consolidation/role-structure", "/content/Neuro-AI"
if not os.path.isdir(DEST):
    tok = getpass("GitHub token (press Enter if the repo is public): ").strip()
    url = f"https://{tok}@github.com/{REPO}.git" if tok else f"https://github.com/{REPO}.git"
    subprocess.run(["git", "clone", "--branch", BRANCH, "--depth", "1", url, DEST], check=True)
else:
    subprocess.run(["git", "-C", DEST, "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", DEST, "reset", "--hard", f"origin/{BRANCH}"], check=True)
subprocess.run(["pip", "-q", "install", "datasets"], check=True)
# Self-contained: materialize the exp-57 head-to-head harness if it is not yet on the branch.
HARNESS = f"{DEST}/experiments/57_e2e_integration_wiring_check.py"
if not os.path.exists(HARNESS):
    open(HARNESS, "wb").write(base64.b64decode(__EXP57_B64__))
    print("ℹ️  materialized exp-57 (not yet pushed to the branch)")
else:
    print("✓ exp-57 present on branch")
print("HEAD:", subprocess.check_output(["git", "-C", DEST, "log", "-1", "--oneline"]).decode().strip())
'''
cells.append(code('EXP57_B64 = "%s"\n' % EXP57_B64 + _setup.replace("__EXP57_B64__", "EXP57_B64")))

cells.append(md("## 2 · Run the head-to-head @ D=4096 WikiText-2\n_A single **CUDA-isolated subprocess** (the parent notebook never touches CUDA). The `wikitext` preset = Report-055/056 config: D=4096, N=1000, max_vocab=2000, W=6, 3 seeds, obs 1/2/3, in-sample._"))
cells.append(code(
'''import os, subprocess, time
os.makedirs("/content/results", exist_ok=True)
OUT = "/content/results/wiring_wikitext_D4096.json"
env = {**os.environ, "PYTHONPATH": "src"}
if not os.path.exists(OUT):
    t = time.time()
    cmd = ["python", "experiments/57_e2e_integration_wiring_check.py",
           "--corpora", "wikitext", "--device", "cuda", "--out", OUT]
    r = subprocess.run(cmd, cwd="/content/Neuro-AI", env=env, capture_output=True, text=True)
    print(r.stdout[-2500:])
    if r.returncode != 0:
        print("STDERR:\\n", r.stderr[-3500:]); raise RuntimeError("wiring check failed")
    print(f"\\n✓ {time.time()-t:.0f}s")
else:
    print("cached:", OUT)
'''))

cells.append(md("## 3 · The wiring certificate — bit-identical A == B + role-Selectivity-Δ vs the 055/056 anchor"))
cells.append(code(
'''import json, pandas as pd
R = json.load(open(OUT)); P = R["pooled"]
OBS = sorted({int(k.split("obs")[1]) for k in P})
rows = []
for o in OBS:
    c = P[f"wikitext_obs{o}"]; b = c["integrated_B"]; a = c["standalone_A"]
    rows.append(dict(
        obs=o,
        role_delta_B=round(b["delta"], 3),
        role_delta_A=round(a["delta"], 3),
        B_eq_A=c["B_equals_A"],
        true_rate_B=round(b["true_rate"], 3),
        deranged=round(b["shuffled_rate"], 3),
        two_floor=b["two_floor_pass"],
        anchor_056=c["report_role_delta"],
        A_minus_report=(round(c["A_minus_report"], 3) if c["A_minus_report"] is not None else None),
        random_cb=round(c["rand_rate"], 4),
        chance=round(b["chance"], 4),
        n=c["n_pooled"]))
df = pd.DataFrame(rows); display(df)
print("ALL basin indices bit-identical (A == B), every cell:", R["all_basin_indices_bit_identical"])
'''))

cells.append(md("## 4 · The money plot — the integrated path lands exactly on the standalone reference 📈\n_The standalone-A dashed line sits **under** the integrated-B line (they coincide — bit-identical), and both hit the Report-056 ⭐ anchors. The deranged + random-codebook arms collapse._"))
cells.append(code(
'''import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
rdB = [r["role_delta_B"] for r in rows]; rdA = [r["role_delta_A"] for r in rows]
anc = [r["anchor_056"] for r in rows]; der = [r["deranged"] for r in rows]
rnd = [r["random_cb"] for r in rows]; tr = [r["true_rate_B"] for r in rows]
fig, ax = plt.subplots(figsize=(8.8, 5.4))
ax.plot(OBS, tr, "o-", color="#2a9d8f", lw=2.9, label="write+L2 true recall (integrated B) ≈ ceiling")
ax.plot(OBS, rdA, "--", color="#e9c46a", lw=4.0, alpha=.95, label="role-Δ (standalone A)")
ax.plot(OBS, rdB, "o-", color="#264653", lw=2.2, label="role-Δ (integrated B) — overlies A")
ax.scatter(OBS, anc, marker="*", s=260, color="#c1121f", zorder=6, label="Report-056 anchor")
ax.plot(OBS, der, "k:", alpha=.6, label="position-deranged floor")
ax.plot(OBS, rnd, "o:", color="#999", label="random-codebook control → chance")
ax.set_xlabel("context positions in cue  (sparse → rich)"); ax.set_ylabel("Recall@1 / role-Selectivity-Δ")
ax.set_title("Integration wiring · D=4096 · WikiText-2 · integrated path == standalone (bit-identical)")
ax.set_xticks(OBS); ax.set_ylim(-0.02, 1.04); ax.legend(loc="center right", fontsize=8.3); ax.grid(alpha=.3)
nbit = sum(1 for r in rows if r["B_eq_A"])
ax.text(0.02, 0.97, f"A == B bit-identical: {nbit}/{len(rows)} cells",
        transform=ax.transAxes, fontsize=12, fontweight="bold", va="top",
        bbox=dict(boxstyle="round", fc="#e9f5ec", ec="#2a9d8f"))
plt.tight_layout(); plt.savefig("/content/results/wiring_certificate.png", bbox_inches="tight"); plt.show()
'''))

cells.append(md("## 5 · 🔌 The certificate"))
cells.append(code(
'''ok = R["all_basin_indices_bit_identical"]
print("Integration wiring certificate — D=4096 WikiText-2 (in-sample memorization):\\n")
print("ALL basin indices bit-identical (A == B):", ok, "\\n")
for r in rows:
    am = f"{r['A_minus_report']:+.3f}" if r["A_minus_report"] is not None else " n/a "
    print(f"  obs={r['obs']}: integrated role-Δ {r['role_delta_B']:.3f}  ==  standalone {r['role_delta_A']:.3f}  "
          f"(B==A {r['B_eq_A']}) | true {r['true_rate_B']:.3f} | two_floor {r['two_floor']} | "
          f"056-anchor {r['anchor_056']} (A−rep {am}) | rand {r['random_cb']:.4f} / chance {r['chance']:.4f}")
print()
print("🔌 WIRING CONFIRMED at D=4096 — the integrated OnlineCodebookUpdater path reproduces the\\n"
      "   graduated mechanism BIT-IDENTICALLY at the real substrate dimension ✅" if ok
      else "❌ NOT bit-identical — the fold-in altered the computation; investigate before trusting it.")
'''))

cells.append(md("## 6 · Save to Drive (optional)"))
cells.append(code(
'''from google.colab import drive; drive.mount("/content/drive")
import shutil, os, datetime
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
dest = f"/content/drive/MyDrive/neuro-ai/results/wiring_certificate_{stamp}"; os.makedirs(dest, exist_ok=True)
df.to_csv(f"{dest}/wiring_table.csv", index=False)
for f in os.listdir("/content/results"): shutil.copy(f"/content/results/{f}", dest)
print("Saved →", dest)
'''))

cells.append(md(
"""---
Paste me the **§5 certificate** (or the table + plot). If `A == B` is bit-identical across all cells at D=4096, the integration is confirmed **behaviorally equivalent to the graduated mechanism at the real substrate dimension** — completing the Report-058 wiring claim end-to-end on GPU.

_Harness: `experiments/57_e2e_integration_wiring_check.py --corpora wikitext --device cuda`. Mechanism: `phase4/{hetero_write,decorrelator}.py`; integration: `phase34/online_codebook.py` (`observe(cue=)`/`consolidate_hetero()`/`recall_hetero()`). The key is the RAW masked cue — H bypasses the scene-MHN+unbind that corrupts slot_query at sparse cues._"""))

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "toc_visible": True},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT = ROOT / "notebooks" / "058_integration_wiring_d4096_colab.ipynb"
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {OUT} with {len(cells)} cells (exp-57 embedded: {len(EXP57_B64)} b64 chars)")
