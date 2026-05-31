"""Generates notebooks/graduation_d4096_colab.ipynb (run locally; output committed)."""
import json

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}

cells = []

cells.append(md(
"""# 🧠 Phase-4 Surgical Mechanism — Graduation Run @ **D=4096** on WikiText-2

**Branch:** `consolidation/role-structure` · **Substrate:** FHRR + Modern Hopfield + emergent codebook

---

### The story this run tests

A multi-stage investigation localized the project's role-binding failure to the **consolidation write**, then validated a surgical fix on toy + small-corpus data:

| step | finding |
|---|---|
| fork | **NOT rebuild** — the substrate holds associations |
| write alone | rescues synthetic key-only recall, but **collapses under real key correlation** |
| the lever | **cue-space decorrelation** rescues the collapse, and **transfers** to real sparse-cue data (0.59 vs 0.024 where store-as-is fails) |

This notebook runs the **scaled graduation test** at the real substrate dimension **D=4096** on **WikiText-2**:
the project's current **store-as-is** path vs the surgical **heteroassociative write + cue-space decorrelator**,
swept across **cue richness** (sparse → rich), with Wilson-CI'd controls.

**The headline question:** in the *sparse-cue* regime where store-as-is fails, does **write + decorrelation** beat it with non-overlapping CIs — at full scale?

> ⏱️ ~15–30 min on a T4. Results + plots are saved to your Drive at the end."""))

cells.append(md("## 1 · GPU check\n_(the parent process never touches CUDA — every experiment runs as an isolated subprocess, per the project's Colab worker rule)_"))
cells.append(code(
"""!nvidia-smi -L || echo "⚠️  No GPU detected — set Runtime ▸ Change runtime type ▸ GPU (T4 is plenty)"
"""))

cells.append(md("## 2 · Fetch the code\nClones the branch with the surgical-mechanism modules. If your repo is private, paste a GitHub token when prompted (else just press Enter)."))
cells.append(code(
"""import os, subprocess
from getpass import getpass

REPO, BRANCH, DEST = "Dypatterson/Neuro-AI", "consolidation/role-structure", "/content/Neuro-AI"

if not os.path.isdir(DEST):
    tok = getpass("GitHub token (Enter if public): ").strip()
    url = f"https://{tok}@github.com/{REPO}.git" if tok else f"https://github.com/{REPO}.git"
    subprocess.run(["git", "clone", "--branch", BRANCH, "--depth", "1", url, DEST], check=True)
else:
    subprocess.run(["git", "-C", DEST, "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", DEST, "reset", "--hard", f"origin/{BRANCH}"], check=True)

print("HEAD:", subprocess.check_output(["git", "-C", DEST, "log", "-1", "--oneline"]).decode().strip())
subprocess.run(["pip", "-q", "install", "datasets"], check=True)
print("datasets installed ✓")
"""))

cells.append(md("## 3 · Pre-warm the WikiText-2 cache\n_(downloads the corpus once so the timed runs don't pay for it)_"))
cells.append(code(
"""import sys; sys.path.insert(0, "/content/Neuro-AI/src")
from pathlib import Path
from energy_memory.phase2.corpus import load_corpus_splits
splits = load_corpus_splits("wikitext", Path("/content/Neuro-AI"), wikitext_name="wikitext-2-raw-v1")
print({k: len(v) for k, v in splits.items()}, "lines/split — WikiText-2 cached ✓")
"""))

cells.append(md(
"""## 4 · The graduation sweep

For each cue richness (`observed` = context positions in the cue, **1 = sparse … 5 = rich**), at **D=4096, N=1000, 3 seeds**, we measure **Recall@1 via `top_index_hits`** for:

| arm | what it is |
|---|---|
| **store-as-is** | the project's current path (full-window MHN → unbind → cleanup) |
| **write** | heteroassociative write, **no** decorrelation |
| **write + decorrelation** | the surgical mechanism (write + cue-space ZCA decorrelator) |
| _floor_ | shuffled-key control |
| _random codebook_ | readout sanity (→ chance) |

Each run is an isolated `--device cuda` subprocess."""))
cells.append(code(
"""import os, subprocess, time
os.makedirs("/content/results", exist_ok=True)
CFG = dict(D=4096, max_vocab=2000, window=6, N=1000, seeds=3, epochs=20)
OBSERVED = [1, 2, 3, 5]                       # sparse → rich cue
env = {**os.environ, "PYTHONPATH": "src"}

for obs in OBSERVED:
    out = f"/content/results/grad_obs{obs}.json"
    if os.path.exists(out):
        print(f"observed={obs}: cached"); continue
    t = time.time()
    cmd = ["python", "experiments/50_corpus_hetero_write.py",
           "--D", str(CFG["D"]), "--corpus-source", "wikitext", "--max-vocab", str(CFG["max_vocab"]),
           "--window-size", str(CFG["window"]), "--N", str(CFG["N"]), "--observed", str(obs),
           "--seeds", str(CFG["seeds"]), "--epochs", str(CFG["epochs"]), "--device", "cuda", "--out", out]
    r = subprocess.run(cmd, cwd="/content/Neuro-AI", env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR tail:\\n", r.stderr[-2500:]); raise RuntimeError(f"observed={obs} failed")
    print(f"observed={obs}: done in {time.time()-t:.0f}s")
print("\\n✓ sweep complete")
"""))

cells.append(md("## 5 · Aggregate with Wilson CIs"))
cells.append(code(
"""import json, math, pandas as pd

def wilson(s, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = s / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (p, max(0.0, c-h), min(1.0, c+h))

ARMS = {"store_as_is": "store-as-is (current)", "hetero_delta": "write (no decorr)",
        "hetero_whiten": "write + decorrelation", "shuffled_key_control": "floor (shuffled key)",
        "random_codebook_control": "random codebook"}
rows = []
for obs in [1, 2, 3, 5]:
    d = json.load(open(f"/content/results/grad_obs{obs}.json"))
    for arm, label in ARMS.items():
        s = sum(round(ps[arm] * ps["N"]) for ps in d["per_seed"])   # pooled hits
        n = sum(ps["N"] for ps in d["per_seed"])
        p, lo, hi = wilson(s, n)
        rows.append(dict(observed=obs, arm=label, rate=p, lo=lo, hi=hi, n=n))
df = pd.DataFrame(rows)
chance = json.load(open("/content/results/grad_obs1.json"))["chance"]
print(f"Recall@1 (top_index_hits) by cue richness · chance≈{chance:.4f}\\n")
display(df.pivot(index="observed", columns="arm", values="rate").round(3))
"""))

cells.append(md("## 6 · The money plot 📈"))
cells.append(code(
"""import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
fig, ax = plt.subplots(figsize=(8.2, 5))
style = {"store-as-is (current)": ("#777", "o", "-"),
         "write (no decorr)": ("#e08a3c", "s", "--"),
         "write + decorrelation": ("#2a9d8f", "o", "-")}
for arm, (col, mk, ls) in style.items():
    sub = df[df.arm == arm].sort_values("observed")
    ax.errorbar(sub.observed, sub.rate, yerr=[sub.rate - sub.lo, sub.hi - sub.rate],
                marker=mk, ls=ls, lw=2.2, capsize=4, color=col, label=arm)
flo = df[df.arm == "floor (shuffled key)"].sort_values("observed")
ax.plot(flo.observed, flo.rate, "k:", alpha=.5, label="floor (shuffled)")
ax.set_xlabel("context positions in cue   (sparse  →  rich)")
ax.set_ylabel("Recall@1   (top_index_hits)")
ax.set_title("D=4096 · WikiText-2 · store-as-is  vs  write + cue-space decorrelation")
ax.set_xticks([1, 2, 3, 5]); ax.set_ylim(-0.02, 1.02); ax.legend(loc="center right"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("/content/results/graduation_plot.png", bbox_inches="tight"); plt.show()
"""))

cells.append(md("## 7 · Headline verdict (done-gate)\nAt the **sparse cue** (where store-as-is fails), does **write + decorrelation** beat it with **non-overlapping Wilson CIs**?"))
cells.append(code(
"""o = 1
sa = df[(df.observed == o) & (df.arm == "store-as-is (current)")].iloc[0]
wd = df[(df.observed == o) & (df.arm == "write + decorrelation")].iloc[0]
print(f"Sparse cue (observed={o}):")
print(f"  store-as-is:          {sa.rate:.3f}   CI[{sa.lo:.3f}, {sa.hi:.3f}]")
print(f"  write + decorrelation:{wd.rate:.3f}   CI[{wd.lo:.3f}, {wd.hi:.3f}]")
beats = wd.lo > sa.hi
print("\\n" + ("✅ HEADLINE PASS — write+decorrelation > store-as-is, CIs disjoint"
              if beats else "❌ no separation — CIs overlap"))
print("(store-as-is should also WIN at the rich cue — that crossover is the whole story.)")
"""))

cells.append(md("## 8 · Save to Drive\n_Colab writes to Drive, not the repo — recover this folder into `reports/` afterwards._"))
cells.append(code(
"""from google.colab import drive; drive.mount("/content/drive")
import shutil, os, datetime
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
dest = f"/content/drive/MyDrive/neuro-ai/results/graduation_d4096_{stamp}"
os.makedirs(dest, exist_ok=True)
for f in os.listdir("/content/results"):
    shutil.copy(f"/content/results/{f}", dest)
df.to_csv(f"{dest}/graduation_table.csv", index=False)
print("Saved →", dest)
"""))

cells.append(md(
"""## 9 · (optional) Capacity curve — sparse cue, sweep N

The decorrelation rescue is **rank-bounded** (helps to N≈D). This sweeps N at the sparse cue to show where it holds and where it rank-decays — the MESH-scaffold follow-up targets the tail."""))
cells.append(code(
"""import os, subprocess, json, math, time
env = {**os.environ, "PYTHONPATH": "src"}
NS = [500, 1000, 2000, 4000, 8000]
for N in NS:
    out = f"/content/results/cap_N{N}.json"
    if os.path.exists(out): continue
    t = time.time()
    cmd = ["python", "experiments/50_corpus_hetero_write.py", "--D", "4096", "--corpus-source", "wikitext",
           "--max-vocab", "4000", "--window-size", "6", "--N", str(N), "--observed", "1",
           "--seeds", "2", "--epochs", "20", "--device", "cuda", "--out", out]
    r = subprocess.run(cmd, cwd="/content/Neuro-AI", env=env, capture_output=True, text=True)
    print(f"N={N}: {'ok' if r.returncode==0 else 'FAIL'} ({time.time()-t:.0f}s)")
    if r.returncode != 0: print(r.stderr[-1500:])

import matplotlib.pyplot as plt
rows = []
for N in NS:
    p = f"/content/results/cap_N{N}.json"
    if not os.path.exists(p): continue
    d = json.load(open(p)); s = d["summary"]
    rows.append((s["N"], s["N"]/4096, s["store_as_is"], s["hetero_whiten"], s["shuffled_key_control"]))
import pandas as pd; cap = pd.DataFrame(rows, columns=["N","N/D","store","write+decorr","floor"])
display(cap.round(3))
fig, ax = plt.subplots(figsize=(8,4.5))
ax.plot(cap["N/D"], cap["write+decorr"], "o-", color="#2a9d8f", lw=2.2, label="write + decorrelation")
ax.plot(cap["N/D"], cap["store"], "o-", color="#777", lw=2, label="store-as-is")
ax.plot(cap["N/D"], cap["floor"], "k:", alpha=.5, label="floor")
ax.set_xlabel("N / D  (load)"); ax.set_ylabel("Recall@1"); ax.set_title("Capacity curve · sparse cue · D=4096")
ax.legend(); ax.grid(alpha=.3); plt.tight_layout(); plt.savefig("/content/results/capacity_plot.png"); plt.show()
"""))

cells.append(md(
"""---
### After the run
1. **Recover artifacts:** copy the Drive folder into the repo `reports/` (e.g. `reports/052_graduation_d4096/`) and commit — Colab writes to Drive, not the repo.
2. **Read the verdict:** the headline is the sparse-cue separation (§7) + the crossover in the §6 plot (store-as-is wins rich, write+decorrelation wins sparse).
3. **Next:** the MESH fixed-scaffold form (`pdf:mesh-2022`) to lift the rank-bounded capacity past N≈D.

_Mechanism: `src/energy_memory/phase4/{hetero_write,decorrelator}.py`. Design: `notes/emergent-codebook/phase-4-heteroassociative-write-design.md`. Reports 048–051._"""))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
with open("notebooks/graduation_d4096_colab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote notebooks/graduation_d4096_colab.ipynb with", len(cells), "cells")
