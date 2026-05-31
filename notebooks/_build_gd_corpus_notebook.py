"""Generates notebooks/gd_corpus_d4096_colab.ipynb (run locally; output committed).

Phase-3 G-D, Phase-2 convincer: the SAME role-Selectivity-Δ control panel as the
local synthetic + repo_sample runs (experiments/56_gd_selectivity_panel.py), now on
real WikiText-2 at D=4096. Proves the graduated surgical mechanism (heteroassociative
write + L2 decorrelator) deposits a ROLE-selective recoverable basin at the
information ceiling on real text, with the full control panel + the element-wise
null-space ablation that only fully separates at D=4096."""
import json

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}

cells = []

cells.append(md(
"""# 🧭 G-D role-Selectivity certificate — graduated mechanism @ **D=4096** on WikiText-2

**Branch:** `consolidation/role-structure` · FHRR + Modern Hopfield + emergent codebook

The corpus graduation (Report 055) showed write+decorr tracks the information ceiling at sparse cues. This notebook adds what 055 lacked: the **role-Selectivity-Δ control panel** that proves the recovery is a **ROLE** result, not content-overlap or key-memorization — on real text at the real substrate dimension.

The panel (all on the SAME windows):
- **write+L2** = the graduated mechanism (heteroassociative write + L2 cue-space decorrelator);
- **role-Selectivity-Δ** = `recoverability(true-position cue) − recoverability(position-deranged cue)`, two-floor Wilson rule;
- **write-marginal** = write+L2 beats store-as-is (FHRR binding is already role-selective, so raw Δ is confounded);
- **controls:** random-codebook (→chance), content-matched bag (→collapse = role result), no-decorr (decorr is the active ingredient), **element-wise renorm** (the Report-053 null-space bug — should separate hardest at D=4096), perfect-cue upper bound, C.3 identity-byte-identical E-arm.

Local (CPU) it already passes: synthetic topic-corpus (closed-form ceiling) + repo_sample (every obs passes, write-marginal +0.45/+0.77/+0.64). This is the **D=4096 WikiText scale confirmation**.

> ⏱️ ~20–35 min on a T4."""))

cells.append(md("## 1 · GPU + fetch the code (pulls the G-D harness, exp 56)"))
cells.append(code(
"""!nvidia-smi -L 2>/dev/null || echo "⚠️  set Runtime ▸ Change runtime type ▸ GPU"
import os, subprocess
from getpass import getpass
REPO, BRANCH, DEST = "Dypatterson/Neuro-AI", "consolidation/role-structure", "/content/Neuro-AI"
if not os.path.isdir(DEST):
    tok = getpass("GitHub token (Enter if public): ").strip()
    url = f"https://{tok}@github.com/{REPO}.git" if tok else f"https://github.com/{REPO}.git"
    subprocess.run(["git", "clone", "--branch", BRANCH, "--depth", "1", url, DEST], check=True)
else:
    subprocess.run(["git", "-C", DEST, "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", DEST, "reset", "--hard", f"origin/{BRANCH}"], check=True)
subprocess.run(["pip", "-q", "install", "datasets"], check=True)
print("HEAD:", subprocess.check_output(["git", "-C", DEST, "log", "-1", "--oneline"]).decode().strip())
"""))

cells.append(md("## 2 · Run the G-D panel on WikiText-2 @ D=4096\n_Two subprocesses (CUDA-isolated from this parent): the **in-sample memorization** headline (the contextual-completion capability) and the **held-out** secondary (memory-not-learner check). Full control panel, obs sweep 1/2/3._"))
cells.append(code(
"""import os, subprocess, time
os.makedirs("/content/results", exist_ok=True)
env = {**os.environ, "PYTHONPATH": "src"}
def run(split, out):
    if os.path.exists(out): return out
    t = time.time()
    cmd = ["python", "experiments/56_gd_selectivity_panel.py",
           "--corpus-source", "wikitext", "--wikitext-name", "wikitext-2-raw-v1",
           "--D", "4096", "--W", "6", "--max-vocab", "2000", "--split", split,
           "--observed", "1", "2", "3", "--N", "1000", "--seeds", "3",
           "--epochs", "20", "--device", "cuda", "--out", out]
    r = subprocess.run(cmd, cwd="/content/Neuro-AI", env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:\\n", r.stderr[-3000:]); raise RuntimeError(f"G-D panel ({split}) failed")
    print(f"  {split}: {time.time()-t:.0f}s"); return out
OUT = run("insample", "/content/results/gd_wikitext_D4096_insample.json")   # the headline
OUT_HO = run("heldout",  "/content/results/gd_wikitext_D4096_heldout.json")  # memory-not-learner check
print("✓ done")
"""))

cells.append(md("## 3 · The G-D certificate — in-sample memorization role-Selectivity-Δ + write-marginal\n_The memory recalls role→target bindings role-selectively from sparse cues; the contextual-completion capability._"))
cells.append(code(
"""import json, math, pandas as pd
P = json.load(open(OUT))["pooled"]
OBS = sorted(int(o) for o in P)
rows = []
for o in OBS:
    c = P[str(o)]; w = c["write_l2"]; s = c["store_as_is"]
    rows.append(dict(
        obs=o, chance=round(c["chance"], 4), ceiling=round(c["ceiling_mean"], 3),
        write_L2=round(w["true_rate"], 3), write_ci=f"[{w['true_ci'][0]:.3f},{w['true_ci'][1]:.3f}]",
        store=round(s["true_rate"], 3),
        role_delta=round(w["delta"], 3), two_floor=w["two_floor_pass"],
        write_marginal=round(w["true_rate"] - s["true_rate"], 3),
        wmarg_ci=f"[{c['write_minus_store_ci'][0]:+.3f},{c['write_minus_store_ci'][1]:+.3f}]",
        no_decorr=round(c["write_no_decorr_true"], 3),
        elementwise=round(c["write_elementwise_true"], 3),
        random_cb=round(c["random_codebook_true"], 4),
        content_bag=round(c["content_matched_bag_true"], 3),
        PASS=c["HEADLINE_PASS"]))
df = pd.DataFrame(rows); display(df)
"""))

cells.append(md("## 4 · The money plot — write+L2 tracks the ceiling; store collapses; the role-shuffle + content-bag fall to the floor 📈"))
cells.append(code(
"""import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
wl2 = [r["write_L2"] for r in rows]; st = [r["store"] for r in rows]
ceil = [r["ceiling"] for r in rows]; bag = [r["content_bag"] for r in rows]
shuf = [P[str(r["obs"])]["write_l2"]["shuffled_rate"] for r in rows]
elem = [r["elementwise"] for r in rows]
fig, ax = plt.subplots(figsize=(8.8, 5.4))
ax.fill_between(OBS, shuf, ceil, color="#dfe7ea", label="recoverable headroom (role-floor→ceiling)")
ax.plot(OBS, ceil, "o-", color="#264653", lw=2.2, label="information ceiling (Bayes-optimal)")
ax.plot(OBS, wl2, "o-", color="#2a9d8f", lw=2.9, label="write+L2 (graduated mechanism)")
ax.plot(OBS, elem, "X--", color="#c1121f", lw=1.6, ms=9, label="write+element-wise (the D-null-space bug)")
ax.plot(OBS, st, "o-", color="#777", lw=2, label="store-as-is (scene-MHN)")
ax.plot(OBS, bag, "s:", color="#9b59b6", lw=1.6, label="content-matched bag (no role)")
ax.plot(OBS, shuf, "k:", alpha=.6, label="role-shuffled floor (position-deranged)")
ax.set_xlabel("context positions in cue  (sparse → rich)"); ax.set_ylabel("Recall@1 (top_index_hits)")
ax.set_title("G-D · D=4096 · WikiText-2 · write+L2 is role-selective at the information ceiling")
ax.set_xticks(OBS); ax.set_ylim(-0.02, 1.04); ax.legend(loc="center right", fontsize=8.3); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("/content/results/gd_certificate.png", bbox_inches="tight"); plt.show()
"""))

cells.append(md("## 5 · 🧭 The certificate"))
cells.append(code(
"""passed = [r for r in rows if r["PASS"]]
print("G-D role-Selectivity headline (two-floor Wilson + write-marginal beats store-as-is):")
for r in rows:
    near = abs(r["write_L2"] - r["ceiling"]) < 0.06
    print(f"  obs={r['obs']}: write+L2 {r['write_L2']:.3f} {r['write_ci']} {'·@ceiling' if near else ''} | "
          f"role-Δ {r['role_delta']:.3f} two_floor={r['two_floor']} | "
          f"write-marginal {r['write_marginal']:+.3f} {r['wmarg_ci']} | {'✅ PASS' if r['PASS'] else '—'}")
print()
print("Controls (should: random→chance, content-bag→collapse, no-decorr<<write, element-wise<<L2 at D=4096):")
for r in rows:
    print(f"  obs={r['obs']}: random_cb={r['random_cb']:.4f}  content_bag={r['content_bag']:.3f}  "
          f"no_decorr={r['no_decorr']:.3f}  element-wise={r['elementwise']:.3f}  (L2={r['write_L2']:.3f})")
print()
print(f"🧭 G-D PASSES at {len(passed)}/{len(rows)} cue-richness cells ✅" if passed else "❌ no cell passes")
"""))

cells.append(md("## 5b · Held-out secondary — memory, not learner\n_On real text, held-out recall should be ≈ chance (sparse-cue completion is memorization; no low-rank role→target rule to generalize). This is the expected signature of a memory, and confirms the in-sample PASS is recall, not leakage._"))
cells.append(code(
"""HO = json.load(open(OUT_HO))["pooled"]
print("held-out (train→test) recall — expected ≈ chance on real text:")
for o in OBS:
    h = HO[str(o)]; w = h["write_l2"]
    print(f"  obs={o}: write+L2 held-out {w['true_rate']:.3f} (chance {h['chance']:.4f}, frac_seen {h['frac_seen_mean']:.2f})  "
          f"role-Δ {w['delta']:+.3f}  store {h['store_as_is']['true_rate']:.3f}")
print("\\n→ in-sample PASS = role-selective MEMORY recall; held-out ≈ chance = no generalizable rule (correct for contextual-completion).")
"""))

cells.append(md("## 6 · Save to Drive"))
cells.append(code(
"""from google.colab import drive; drive.mount("/content/drive")
import shutil, os, datetime
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
dest = f"/content/drive/MyDrive/neuro-ai/results/gd_certificate_{stamp}"; os.makedirs(dest, exist_ok=True)
df.to_csv(f"{dest}/gd_certificate_table.csv", index=False)
for f in os.listdir("/content/results"): shutil.copy(f"/content/results/{f}", dest)
print("Saved →", dest)
"""))

cells.append(md(
"""---
Paste me the **§5 certificate** (or the table + plot). If it passes at D=4096, the
G-D gate is fully discharged on real text at the real substrate dimension: the
graduated mechanism deposits a **role-selective** recoverable basin at the information
ceiling, and the element-wise null-space ablation separates hardest here (the
corpus-specific Fork-3 pathology, completing Reports 053/054/055).

_Harness: `experiments/56_gd_selectivity_panel.py --corpus-source wikitext`. Mechanism:
`phase4/{hetero_write,decorrelator}.py` (L2-renorm in `decorrelator.apply()`)._"""))

nb = {"cells": cells, "metadata": {"accelerator": "GPU",
      "colab": {"provenance": [], "toc_visible": True},
      "kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
with open("notebooks/gd_corpus_d4096_colab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote notebooks/gd_corpus_d4096_colab.ipynb with", len(cells), "cells")
