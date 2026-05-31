"""Generates notebooks/graduation_d4096_colab.ipynb (run locally; output committed).

The graduation CERTIFICATE: floor-gated done-gate + information-ceiling overlay +
the one-line-fix ablation (element-wise vs L2 renorm). Re-run this and it pulls the
Report-054 fix and certifies the surgical mechanism at D=4096 on WikiText-2."""
import json

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}

cells = []

cells.append(md(
"""# 🎓 Graduation Certificate — surgical mechanism @ **D=4096** on WikiText-2

**Branch:** `consolidation/role-structure` · FHRR + Modern Hopfield + emergent codebook

This is the formal, floor-gated confirmation that the consolidation-write fix works at the real substrate dimension. The story it certifies:

- the mechanism = **heteroassociative write + cue-space decorrelator**;
- the D=4096 graduation first **failed** (Report 052) — but the autopsy (Report 053) showed it was **mechanism-limited, not task-limited**, and the killer was **D-scale, not corpus**;
- the cause was a **one-line renorm bug** (Report 054): `apply()` renormalized **element-wise** (FHRR phasor convention), filling the rank-deficient null space with noise (~93% at D=4096). **L2-renorm** fixes it.

**This notebook certifies the fix** with three things the first run lacked:
1. the **correct done-gate** — clear the **shuffled-key floor** with disjoint Wilson CIs (not "beat store-as-is", which can be sub-floor);
2. the **information ceiling** overlaid (the Bayes-optimal Recall@1 — no model can beat it);
3. a **smoking-gun ablation** — toggle the one line (element-wise ↔ L2) and watch obs=1 collapse to the floor and snap back to the ceiling.

> ⏱️ ~20–30 min on a T4."""))

cells.append(md("## 1 · GPU + fetch the code (pulls the Report-054 fix)"))
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

cells.append(md("## 2 · Pre-warm WikiText-2 + compute the information ceiling\n_The Bayes-optimal Recall@1 (best any cue→target predictor can do), on matched windows._"))
cells.append(code(
"""import sys; sys.path.insert(0, "/content/Neuro-AI/src")
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from energy_memory.phase2.corpus import (build_vocabulary, encode_texts, load_corpus_splits,
                                          make_windows, sample_windows)
from energy_memory.phase2.encoding import mask_positions

def ceiling_by_obs(source="wikitext", max_vocab=2000, window=6, N=1000, seeds=(0,1,2), OBS=(1,2,3,5)):
    sp = load_corpus_splits(source, Path("/content/Neuro-AI"), wikitext_name="wikitext-2-raw-v1")
    v = build_vocabulary(sp["train"], max_vocab=max_vocab); ids = encode_texts(sp["train"], v)
    mpos = mask_positions(window, 1, "center")[0]; allw = make_windows(ids, window)
    out = {o: [] for o in OBS}
    for s in seeds:
        w = [x for x in sample_windows(allw, min(N, len(allw)), seed=s + 7)
             if x[mpos] != v.unk_id and x[mpos] != v.mask_id]
        for o in OBS:
            g = defaultdict(Counter)
            for x in w:
                ctx = [p for p in range(window) if p != mpos][:max(1, o)]
                g[tuple(x[p] for p in ctx)][x[mpos]] += 1
            hits = sum(c.most_common(1)[0][1] for c in g.values()); n = sum(sum(c.values()) for c in g.values())
            out[o].append(hits / n)
    return {o: float(np.mean(out[o])) for o in OBS}

CEIL = ceiling_by_obs()
print("information ceiling by cue richness:", {o: round(c, 3) for o, c in CEIL.items()})
"""))

cells.append(md("## 3 · Run the graduation sweep — the fix (L2) across cue richness + the bug (element-wise) at obs=1\n_The L2 sweep is the certificate; the obs=1 element-wise run is the ablation control._"))
cells.append(code(
"""import os, subprocess, time
os.makedirs("/content/results", exist_ok=True)
CFG = dict(D=4096, max_vocab=2000, window=6, N=1000, seeds=3, epochs=20)
OBS = [1, 2, 3, 5]
env = {**os.environ, "PYTHONPATH": "src"}

def run(obs, renorm):
    out = f"/content/results/grad_obs{obs}_{renorm}.json"
    if os.path.exists(out): return out
    t = time.time()
    cmd = ["python", "experiments/50_corpus_hetero_write.py",
           "--D", str(CFG["D"]), "--corpus-source", "wikitext", "--max-vocab", str(CFG["max_vocab"]),
           "--window-size", str(CFG["window"]), "--N", str(CFG["N"]), "--observed", str(obs),
           "--seeds", str(CFG["seeds"]), "--epochs", str(CFG["epochs"]),
           "--decorr-renorm", renorm, "--device", "cuda", "--out", out]
    r = subprocess.run(cmd, cwd="/content/Neuro-AI", env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:\\n", r.stderr[-2500:]); raise RuntimeError(f"obs={obs} {renorm} failed")
    print(f"  obs={obs} renorm={renorm}: {time.time()-t:.0f}s"); return out

for obs in OBS: run(obs, "l2")          # the fix — full sweep
run(1, "elementwise")                   # the bug — obs=1 ablation control
print("\\n✓ sweep complete")
"""))

cells.append(md("## 4 · Floor-gated done-gate (Wilson CIs) — the certificate"))
cells.append(code(
"""import json, math, pandas as pd
def wilson(s, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = s / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (p, max(0.0, c-h), min(1.0, c+h))
def pooled(path, arm):
    d = json.load(open(path)); s = sum(round(ps[arm]*ps["N"]) for ps in d["per_seed"]); n = sum(ps["N"] for ps in d["per_seed"])
    return wilson(s, n)
rows = []
for o in OBS:
    p = f"/content/results/grad_obs{o}_l2.json"
    wd, wd_lo, wd_hi = pooled(p, "hetero_whiten")
    st = pooled(p, "store_as_is"); fl = pooled(p, "shuffled_key_control")
    clears = wd_lo > fl[2]                  # CORRECT gate: write+decorr lower-CI > floor upper-CI
    rows.append(dict(obs=o, store=round(st[0],3), write_decorr=round(wd,3),
                     wd_ci=f"[{wd_lo:.3f},{wd_hi:.3f}]", floor=round(fl[0],3),
                     ceiling=round(CEIL[o],3), clears_floor=clears))
df = pd.DataFrame(rows); display(df)
"""))

cells.append(md("## 5 · The money plot — write+decorr tracks the information ceiling 📈"))
cells.append(code(
"""import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
wd = [r["write_decorr"] for r in rows]; st = [r["store"] for r in rows]
fl = [r["floor"] for r in rows]; cl = [r["ceiling"] for r in rows]
abl1 = json.load(open("/content/results/grad_obs1_elementwise.json"))["summary"]["hetero_whiten"]
fig, ax = plt.subplots(figsize=(8.6, 5.3))
ax.fill_between(OBS, fl, cl, color="#dfe7ea", label="recoverable headroom (floor→ceiling)")
ax.plot(OBS, cl, "o-", color="#264653", lw=2.2, label="information ceiling (Bayes-optimal)")
ax.plot(OBS, wd, "o-", color="#2a9d8f", lw=2.8, label="write+decorr — L2 (the fix)")
ax.scatter([1], [abl1], marker="X", s=120, color="#c1121f", zorder=5,
           label=f"write+decorr — element-wise BUG (obs=1) = {abl1:.2f}")
ax.plot(OBS, st, "o-", color="#777", lw=2, label="store-as-is")
ax.plot(OBS, fl, "k:", alpha=.6, label="floor (shuffled key)")
ax.set_xlabel("context positions in cue  (sparse → rich)"); ax.set_ylabel("Recall@1 (top_index_hits)")
ax.set_title("GRADUATION · D=4096 · WikiText-2 · the fix tracks the information ceiling")
ax.set_xticks(OBS); ax.set_ylim(-0.02, 1.04); ax.legend(loc="center right", fontsize=8.5); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("/content/results/graduation_certificate.png", bbox_inches="tight"); plt.show()
"""))

cells.append(md("## 6 · 🎓 The certificate"))
cells.append(code(
"""passed = all(r["clears_floor"] for r in rows)
abl = json.load(open("/content/results/grad_obs1_elementwise.json"))["summary"]
print("Floor-gated graduation (write+decorr clears the shuffled-key floor, disjoint Wilson CIs):")
for r in rows:
    near = abs(r["write_decorr"] - r["ceiling"]) < 0.06
    print(f"  obs={r['obs']}: {r['write_decorr']:.3f} {r['wd_ci']} vs floor {r['floor']:.3f}  "
          f"{'✅ clears' if r['clears_floor'] else '❌ ties/overlaps'}  "
          f"{'· at ceiling ('+str(r['ceiling'])+')' if near else ''}")
print()
print("Smoking-gun ablation (obs=1, D=4096): the one line is the whole story —")
print(f"  element-wise renorm (the bug): {abl['hetero_whiten']:.3f} ≈ floor {abl['shuffled_key_control']:.3f}")
print(f"  L2 renorm (the fix):           {rows[0]['write_decorr']:.3f}  ({rows[0]['write_decorr']/max(1e-9,abl['hetero_whiten']):.1f}× the bug)")
print()
print("🎓 GRADUATES ✅" if passed else "❌ does not clear the floor at every cell")
"""))

cells.append(md("## 7 · Save to Drive"))
cells.append(code(
"""from google.colab import drive; drive.mount("/content/drive")
import shutil, os, datetime
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
dest = f"/content/drive/MyDrive/neuro-ai/results/graduation_certificate_{stamp}"; os.makedirs(dest, exist_ok=True)
df.to_csv(f"{dest}/certificate_table.csv", index=False)
for f in os.listdir("/content/results"): shutil.copy(f"/content/results/{f}", dest)
print("Saved →", dest)
"""))

cells.append(md(
"""---
Paste me the **§6 certificate** (or the table + plot). If it graduates, the surgical
mechanism is confirmed at the real substrate dimension and we move to the next phase
(`entropy`/`margin` drill-downs, then Phase-4 integration). The Report-052 numbers stand;
the fix (Report 054) is the resolution.

_Mechanism: `phase4/{hetero_write,decorrelator}.py`. The fix: L2-renorm in `decorrelator.apply()`._"""))

nb = {"cells": cells, "metadata": {"accelerator": "GPU",
      "colab": {"provenance": [], "toc_visible": True},
      "kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
with open("notebooks/graduation_d4096_colab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote notebooks/graduation_d4096_colab.ipynb (certificate) with", len(cells), "cells")
