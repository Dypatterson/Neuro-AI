"""Real-corpus validation of the heteroassociative consolidation write.

Design: notes/emergent-codebook/phase-4-heteroassociative-write-design.md

Wires the Phase-4 heteroassociative write into the masked-token contextual-
completion path on a real corpus (repo_sample / wikitext) and compares it against
the store-as-is baseline the project currently uses, as N (number of consolidated
windows) grows. The write runs as a SEPARATE batch-offline pass over a CLOSED,
seed-fixed buffer (AH condition 3) -- not inside any streaming loop.

Masked-token mapping:
  key   = encode of the masked context window (target position = mask)
  value = the masked target token's atom (index into the token codebook)

Reads (basin-membership top_index_hits over the token codebook; chance = 1/|decode|):
  store_as_is : store full-window encodings in an MHN; cue=masked encoding;
                retrieve -> unbind at the masked position -> cleanup over the
                token codebook. (The project's current structure.)
  hetero_delta / hetero_contrastive : recover = cleanup(H key) over the codebook.

Controls (same windows): random-codebook (-> chance); shuffled-key (recover with a
deranged context key -> chance). Headline = Recall@1 (top_index_hits) of the write
vs store-as-is on the SAME consolidated windows (memorization write-then-read).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from energy_memory.phase2.corpus import (
    build_vocabulary, encode_texts, load_corpus_splits, make_windows, sample_windows,
)
from energy_memory.phase2.encoding import (
    build_position_vectors, encode_window, masked_window, mask_positions,
)
from energy_memory.phase3.basin_readout import top_index_hits
from energy_memory.phase4.hetero_write import (
    HeteroConsolidationBuffer, batched_hopfield_topindex,
    heteroassociative_write, recall_top_index,
)
from energy_memory.phase4.decorrelator import CueDecorrelator
from energy_memory.substrate.torch_fhrr import TorchFHRR


def _deranged(n, seed, device):
    g = torch.Generator().manual_seed(seed * 7919 + 23)
    for _ in range(2000):
        p = torch.randperm(n, generator=g)
        if bool((p != torch.arange(n)).all()):
            return p.to(device)
    return ((torch.arange(n) + 1) % n).to(device)


def _whiten(keys):
    """ZCA-whiten the keys (decorrelation upper bound; see Report 049 §4)."""
    N, D = keys.shape
    cov = (keys.conj().transpose(0, 1) @ keys) / N
    evals, evecs = torch.linalg.eigh(cov)
    inv_sqrt = torch.where(evals > 1e-6, evals.clamp_min(1e-6) ** -0.5, torch.zeros_like(evals))
    w_zca = (evecs * inv_sqrt.to(evecs.dtype)) @ evecs.conj().transpose(0, 1)
    out = keys @ w_zca
    return out / out.abs().clamp_min(1e-12)


def _sparse_cue(w, mpos, observed, window_size, mask_id):
    """Cue keeps only the first `observed` context positions (target + the rest
    masked). Fewer observed -> sparser cue -> store-as-is degrades into its
    failing regime, where a write (+ decorrelation) could matter."""
    ctx = [p for p in range(window_size) if p != mpos]
    keep = set(ctx[:max(1, observed)])
    return tuple(w[p] if p in keep else mask_id for p in range(window_size))


def run(args):
    repo_root = Path(__file__).resolve().parents[1]
    splits = load_corpus_splits(args.corpus_source, repo_root, wikitext_name=args.wikitext_name)
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    train_ids = encode_texts(splits["train"], vocab)
    decode_ids = [i for i, t in enumerate(vocab.id_to_token)
                  if t not in {vocab.unk_token, vocab.mask_token}]
    chance = 1.0 / len(decode_ids)

    out = {"config": vars(args), "vocab_size": len(vocab.id_to_token),
           "n_decode": len(decode_ids), "chance": chance, "per_seed": []}

    for seed in range(args.seeds):
        sub = TorchFHRR(dim=args.D, seed=seed, device=args.device)
        codebook = sub.random_vectors(len(vocab.id_to_token))   # static random (Phase-2 baseline)
        positions = build_position_vectors(sub, args.window_size)
        mpos = mask_positions(args.window_size, 1, "center")[0]

        all_windows = make_windows(train_ids, args.window_size)
        windows = sample_windows(all_windows, min(args.N, len(all_windows)), seed=seed + 7)
        # keep only windows whose masked target is a real (decodable) token
        windows = [w for w in windows if w[mpos] != vocab.unk_id and w[mpos] != vocab.mask_id]
        N = len(windows)
        target_ids = torch.tensor([w[mpos] for w in windows], dtype=torch.long, device=args.device)

        # full + masked encodings
        full_enc = torch.stack([encode_window(sub, positions, codebook, w) for w in windows])
        masked_enc = torch.stack([
            encode_window(sub, positions, codebook,
                          _sparse_cue(w, mpos, args.observed, args.window_size, vocab.mask_id))
            for w in windows])

        # ---- store-as-is masked-token recall (the project's current path) ----
        # retrieve the full window from the masked cue, unbind at the masked
        # position, clean up over the token codebook.
        scene_state, _, _, _ = _retrieve_state(sub, full_enc, masked_enc, args.beta, args.max_iter)
        slot = sub.normalize(sub.unbind(scene_state, positions[mpos]))
        ti_sa, _, _ = batched_hopfield_topindex(sub, codebook, slot, beta=args.beta, max_iter=args.max_iter)
        sa_rate = top_index_hits(ti_sa, target_ids) / N

        # ---- heteroassociative write (closed, seed-fixed, batch-offline) ----
        buf = HeteroConsolidationBuffer(dim=args.D, device=args.device)
        for i in range(N):
            buf.add(masked_enc[i], int(target_ids[i]))   # key = masked context, value = target token
        buf.freeze()
        H = heteroassociative_write(buf, codebook, lr=args.lr, epochs=args.epochs)
        ti_h, _, _ = recall_top_index(sub, H, masked_enc, codebook, beta=args.beta, max_iter=args.max_iter)
        h_rate = top_index_hits(ti_h, target_ids) / N

        # decorrelation arm: whiten the (correlated) real context keys, then write
        wkeys = CueDecorrelator(args.D, renorm=args.decorr_renorm).fit(masked_enc).apply(masked_enc)
        bufw = HeteroConsolidationBuffer(dim=args.D, device=args.device)
        for i in range(N):
            bufw.add(wkeys[i], int(target_ids[i]))
        bufw.freeze()
        Hw = heteroassociative_write(bufw, codebook, lr=args.lr, epochs=args.epochs)
        ti_hw, _, _ = recall_top_index(sub, Hw, wkeys, codebook, beta=args.beta, max_iter=args.max_iter)
        hw_rate = top_index_hits(ti_hw, target_ids) / N
        # mean pairwise key correlation (diagnostic: how correlated are real keys?)
        with torch.no_grad():
            kc = (masked_enc @ masked_enc.conj().T).real.abs() / args.D
            key_cos = float(kc[~torch.eye(N, dtype=torch.bool, device=args.device)].mean())

        # ---- controls ----
        der = _deranged(N, seed, args.device)
        ti_shuf, _, _ = recall_top_index(sub, H, masked_enc[der], codebook, beta=args.beta, max_iter=args.max_iter)
        shuf_rate = top_index_hits(ti_shuf, target_ids) / N      # wrong-context cue -> chance
        rand_cb = sub.random_vectors(len(vocab.id_to_token))
        ti_rand, _, _ = batched_hopfield_topindex(
            sub, rand_cb, (masked_enc @ H.transpose(0, 1)) / args.D, beta=args.beta, max_iter=args.max_iter)
        rand_rate = top_index_hits(ti_rand, target_ids) / N      # random codebook -> chance

        out["per_seed"].append({
            "seed": seed, "N": N, "key_cos": key_cos, "store_as_is": sa_rate,
            "hetero_delta": h_rate, "hetero_whiten": hw_rate,
            "shuffled_key_control": shuf_rate, "random_codebook_control": rand_rate,
        })

    def mean(k):
        return sum(s[k] for s in out["per_seed"]) / len(out["per_seed"])
    out["summary"] = {k: mean(k) for k in
                      ("N", "key_cos", "store_as_is", "hetero_delta", "hetero_whiten",
                       "shuffled_key_control", "random_codebook_control")}
    print(json.dumps({"chance": chance, "n_decode": len(decode_ids), **out["summary"]}, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}")
    return out


def _retrieve_state(sub, patterns, queries, beta, max_iter):
    """Settle queries against patterns; return the settled state (for the
    store-as-is unbind path). Thin wrapper around the module's retrieve."""
    import math
    state = sub.normalize(queries)
    d = patterns.shape[1]
    for _ in range(max_iter):
        scores = (state @ patterns.conj().T).real / d
        w = torch.softmax(beta * scores, dim=1)
        state = sub.normalize(w.to(patterns.dtype) @ patterns)
    return state, None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=2048)
    ap.add_argument("--corpus-source", choices=["repo_sample", "wikitext"], default="repo_sample", dest="corpus_source")
    ap.add_argument("--wikitext-name", default="wikitext-2-raw-v1", dest="wikitext_name")
    ap.add_argument("--max-vocab", type=int, default=512, dest="max_vocab")
    ap.add_argument("--window-size", type=int, default=6, dest="window_size")
    ap.add_argument("--observed", type=int, default=99,
                    help="context positions in the cue (default all; small=sparse, store-as-is fails)")
    ap.add_argument("--decorr-renorm", choices=["l2","elementwise"], default="l2", dest="decorr_renorm",
                    help="decorrelator renorm: l2 (the fix) or elementwise (the Report-053 bug; ablation)")
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--max-iter", type=int, default=12, dest="max_iter")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--lr-push", type=float, default=0.1, dest="lr_push")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
