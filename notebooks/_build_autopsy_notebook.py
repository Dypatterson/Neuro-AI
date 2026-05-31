"""Generates notebooks/autopsy_pivot_colab.ipynb (run locally; output committed)."""
import json

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}

cells = []

cells.append(md(
"""# 🔬 Autopsy — was the mechanism failing, or was the *task* uninformative?

The D=4096 graduation (Report 052) found write+decorrelation **ties the floor** at the sparse cue (obs=1) on WikiText prose, while it gave a 12.6× rescue on source code. Before we pivot, one question decides **where** we pivot:

> **Is obs=1 task-limited or mechanism-limited?**

We measure the **information ceiling** — the Recall@1 of the *Bayes-optimal* predictor (for each cue, always guess its most-frequent target). No model can beat it. Then we overlay the actual mechanism.

| if … | then … | pivot |
|---|---|---|
| ceiling ≈ floor at obs=1 | the single-token cue carries **no information** on prose — *nothing* could recover it | **task-limited** → the sparse-cue role-binding framing is wrong for natural language; pivot to richer cues / a different binding signal |
| ceiling ≫ mechanism at obs=1 | real recoverable structure exists that the mechanism **leaves on the table** | **mechanism-limited** → one more (stronger) write is worth a shot before abandoning |

Mostly CPU (the ceiling is pure corpus statistics) — fast. An optional GPU cell isolates the killer variable (corpus × D)."""))

cells.append(md("## 1 · Setup"))
cells.append(code(
"""!nvidia-smi -L 2>/dev/null || echo "(no GPU — fine, the core autopsy is CPU-only)"
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

cells.append(md("## 2 · The information ceiling (Bayes-optimal Recall@1) by cue richness\n_Matched to the Report-052 data prep (same corpus, vocab, windows, N), so it is directly comparable to the mechanism._"))
cells.append(code(
"""import sys; sys.path.insert(0, "/content/Neuro-AI/src")
from pathlib import Path
from collections import defaultdict, Counter
from energy_memory.phase2.corpus import (build_vocabulary, encode_texts, load_corpus_splits,
                                          make_windows, sample_windows)
from energy_memory.phase2.encoding import mask_positions

def prep_windows(source, *, max_vocab=2000, window=6, N=1000, seeds=(0,1,2)):
    splits = load_corpus_splits(source, Path("/content/Neuro-AI"),
                                wikitext_name="wikitext-2-raw-v1")
    vocab = build_vocabulary(splits["train"], max_vocab=max_vocab)
    ids = encode_texts(splits["train"], vocab)
    mpos = mask_positions(window, 1, "center")[0]
    allw = make_windows(ids, window)
    per_seed = []
    for s in seeds:
        w = sample_windows(allw, min(N, len(allw)), seed=s + 7)
        w = [x for x in w if x[mpos] != vocab.unk_id and x[mpos] != vocab.mask_id]
        per_seed.append(w)
    return per_seed, mpos

def bayes_ceiling(windows, mpos, observed):
    # cue = the first `observed` context tokens; predict the most-frequent target.
    g = defaultdict(Counter)
    for w in windows:
        ctx = [p for p in range(len(w)) if p != mpos][:max(1, observed)]
        g[tuple(w[p] for p in ctx)][w[mpos]] += 1
    hits = sum(c.most_common(1)[0][1] for c in g.values())
    n = sum(sum(c.values()) for c in g.values())
    uniq = sum(1 for c in g.values() if sum(c.values()) == 1) / max(1, len(g))  # frac one-shot cues
    return hits / n, uniq, len(g)

import numpy as np
OBS = [1, 2, 3, 5]
ceil = {}
for src in ["wikitext", "repo_sample"]:
    per_seed, mpos = prep_windows(src)
    ceil[src] = {o: float(np.mean([bayes_ceiling(w, mpos, o)[0] for w in per_seed])) for o in OBS}
    uniq1 = np.mean([bayes_ceiling(w, mpos, 1)[1] for w in per_seed])
    print(f"{src:12s} ceiling by obs: " + " ".join(f"{o}:{ceil[src][o]:.3f}" for o in OBS)
          + f"   (obs=1 one-shot-cue fraction = {uniq1:.2f})")
"""))

cells.append(md("## 3 · The reframe plot — mechanism vs the information ceiling 📈\n_Mechanism numbers from Report 052 (D=4096 WikiText, pooled). Does write+decorrelation track the ceiling, or fall short of it?_"))
cells.append(code(
"""import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
# Report 052 pooled (WikiText, D=4096):  obs -> (store, write+decorr, floor)
R052 = {1:(0.036,0.098,0.096), 2:(0.035,0.114,0.096), 3:(0.062,0.150,0.096), 5:(0.997,0.263,0.096)}
obs = OBS
store=[R052[o][0] for o in obs]; wd=[R052[o][1] for o in obs]; floor=[R052[o][2] for o in obs]
cl=[ceil["wikitext"][o] for o in obs]
fig, ax = plt.subplots(figsize=(8.4,5.2))
ax.fill_between(obs, floor, cl, color="#dfe7ea", label="recoverable headroom (floor→ceiling)")
ax.plot(obs, cl, "o-", color="#264653", lw=2.4, label="information ceiling (Bayes-optimal)")
ax.plot(obs, wd, "o-", color="#2a9d8f", lw=2.4, label="write + decorrelation (mechanism)")
ax.plot(obs, store, "o-", color="#777", lw=2, label="store-as-is")
ax.plot(obs, floor, "k:", alpha=.6, label="floor (shuffled key)")
ax.set_xlabel("context positions in cue  (sparse → rich)"); ax.set_ylabel("Recall@1")
ax.set_title("WikiText-2 · D=4096 · mechanism vs the information ceiling")
ax.set_xticks(obs); ax.set_ylim(-0.02, 1.02); ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("/content/autopsy_reframe.png", bbox_inches="tight"); plt.show()
"""))

cells.append(md("## 4 · Prose vs code — why the rescue was corpus-specific"))
cells.append(code(
"""fig, ax = plt.subplots(figsize=(7,4.4))
x = np.arange(len(OBS)); wdt=0.38
ax.bar(x-wdt/2, [ceil["wikitext"][o] for o in OBS], wdt, color="#2a9d8f", label="WikiText (prose)")
ax.bar(x+wdt/2, [ceil["repo_sample"][o] for o in OBS], wdt, color="#e9c46a", label="repo_sample (source code)")
ax.axhline(0.096, color="k", ls=":", alpha=.6, label="floor")
ax.set_xticks(x); ax.set_xticklabels(OBS); ax.set_xlabel("context positions in cue")
ax.set_ylabel("information ceiling (Bayes-optimal Recall@1)")
ax.set_title("Why obs=1 transferred on code but not prose")
ax.legend(); ax.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig("/content/autopsy_prose_vs_code.png", bbox_inches="tight"); plt.show()
print(f"obs=1 ceiling:  WikiText {ceil['wikitext'][1]:.3f}  vs  repo_sample {ceil['repo_sample'][1]:.3f}  (floor 0.096)")
"""))

cells.append(md("## 5 · 🧭 The pivot verdict"))
cells.append(code(
"""c1 = ceil["wikitext"][1]; wd1 = 0.098; fl = 0.096
headroom = c1 - wd1            # information the mechanism leaves on the table at obs=1
ceiling_above_floor = c1 - fl  # is the sparse cue informative at all on prose?
print(f"obs=1 (WikiText):  floor {fl:.3f}  |  mechanism {wd1:.3f}  |  information ceiling {c1:.3f}")
print(f"  ceiling above floor: {ceiling_above_floor:+.3f}   |   mechanism below ceiling: {headroom:+.3f}\\n")
if ceiling_above_floor < 0.03:
    print("VERDICT: TASK-LIMITED. The single-token cue carries ~no information on prose")
    print("(ceiling ≈ floor) — no mechanism could recover it. The sparse-cue role-binding")
    print("framing is wrong for natural language. → PIVOT THE TASK (richer cues / different")
    print("binding signal), not the write.")
elif headroom > 0.08:
    print("VERDICT: MECHANISM-LIMITED. Real recoverable structure exists at obs=1")
    print(f"(ceiling {c1:.3f} ≫ mechanism {wd1:.3f}) that the mechanism leaves on the table.")
    print("→ ONE MORE WRITE is justified (stronger orthogonalization / better readout)")
    print("before abandoning.")
else:
    print("VERDICT: MIXED. The sparse cue is weakly informative and the mechanism is near")
    print("the ceiling — marginal headroom. Lean pivot, but a cheap stronger-write probe is")
    print("defensible.")
"""))

cells.append(md("## 6 · (optional, GPU) Killer-variable isolation — corpus × D at obs=1\n_Confirms whether the transfer failure tracks corpus (correlation/informativeness) or dimensionality._"))
cells.append(code(
"""import os, subprocess, json, time
env = {**os.environ, "PYTHONPATH": "src"}; os.makedirs("/content/results", exist_ok=True)
GRID = [("repo_sample", 512), ("repo_sample", 4096), ("wikitext", 512), ("wikitext", 4096)]
rows = []
for src, D in GRID:
    out = f"/content/results/kv_{src}_D{D}.json"
    if not os.path.exists(out):
        cmd = ["python","experiments/50_corpus_hetero_write.py","--D",str(D),"--corpus-source",src,
               "--max-vocab","2000","--window-size","6","--N","800","--observed","1","--seeds","2",
               "--epochs","20","--device","cuda","--out",out]
        t=time.time(); r=subprocess.run(cmd, cwd="/content/Neuro-AI", env=env, capture_output=True, text=True)
        print(f"{src} D={D}: {'ok' if r.returncode==0 else 'FAIL'} ({time.time()-t:.0f}s)")
        if r.returncode!=0: print(r.stderr[-1200:]); continue
    d=json.load(open(out)); s=d["summary"]
    rows.append((src, D, s["key_cos"], s["store_as_is"], s["hetero_whiten"], s["shuffled_key_control"]))
import pandas as pd
kv=pd.DataFrame(rows, columns=["corpus","D","key_cos","store","write+decorr","floor"]).round(3)
display(kv)
print("\\nRead: write+decorr − floor > 0 ⇒ mechanism clears its floor at obs=1.")
"""))

cells.append(md("## 7 · Save to Drive"))
cells.append(code(
"""from google.colab import drive; drive.mount("/content/drive")
import shutil, os, datetime, json
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
dest = f"/content/drive/MyDrive/neuro-ai/results/autopsy_{stamp}"; os.makedirs(dest, exist_ok=True)
json.dump({"ceiling": ceil, "report052_mechanism": R052}, open(f"{dest}/autopsy_ceilings.json","w"), indent=2)
for f in ["autopsy_reframe.png","autopsy_prose_vs_code.png"]:
    if os.path.exists(f"/content/{f}"): shutil.copy(f"/content/{f}", dest)
if os.path.isdir("/content/results"):
    for f in os.listdir("/content/results"): shutil.copy(f"/content/results/{f}", dest)
print("Saved →", dest)
"""))

cells.append(md(
"""---
### After the run
Paste me the **§5 verdict** + the **§2 ceilings** (and the §6 table if you ran it). Then:
- **task-limited** → we pivot the role-binding formulation (cues that actually carry information; or a non-sparse binding probe). The whole "sparse single-cue = the role-binding null" analogy is the thing to drop.
- **mechanism-limited** → one final stronger-write probe at obs=1 (then abandon if it stays at floor, per the Report-052 pre-commit).

_The mechanism numbers overlaid here are Report 052 (committed); the ceiling is computed fresh on matched windows, so the comparison is apples-to-apples. Modules: `phase4/{hetero_write,decorrelator}.py`._"""))

nb = {"cells": cells, "metadata": {"accelerator": "GPU",
      "colab": {"provenance": [], "toc_visible": True},
      "kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
with open("notebooks/autopsy_pivot_colab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote notebooks/autopsy_pivot_colab.ipynb with", len(cells), "cells")
