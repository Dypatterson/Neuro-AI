"""G-D: role-Selectivity-Δ of the graduated surgical mechanism (Phase 3).

Spec: notes/emergent-codebook/phase-3-consolidation-write-design.md
      §G-D substrate amendment (2026-05-31, user-approved).
Builds on: Report 055 (corpus-transfer graduation), Report 048 smoke (the
write-marginal anchor + "target the bottleneck" refinements).

WHY THIS SUBSTRATE (not the clean 4-role rule-toy). A 7-agent design panel whose
judge RAN CODE proved a clean abstract rule-toy cannot reproduce the graduated
mechanism's signature: a cue-recoverable rule lets store-as-is win too (Δ≈0); a
random-hash rule defeats the linear write too (Δ≈0). The signature is intrinsic to
the masked-encoding regime (scene-MHN blend-corruption at sparse cues + low-rank
token structure). So G-D is grounded on the masked-encoding substrate, instantiated
as a POSITION-DEPENDENT TOPIC-CORPUS with a closed-form Bayes ceiling.

HEADLINE (spec §G-D amendment): the role-Selectivity-Δ of write+L2,
  Δ = top_index_hits(true-position cue) − top_index_hits(position-deranged cue)
over the value (target) codebook, two-floor Wilson rule, chance = 1/L; PLUS the
write-marginal anchor Δ(write+L2) − Δ(store-as-is) (FHRR binding is already
role-selective, so raw Δ>0 is confounded — smoke refinement 1).

ANTI-HOMUNCULUS. Every write is a single batch-offline pass over a FROZEN buffer
(hetero_write.freeze gate); the headline write is pull-only delta-rule (no negative;
Report 055:55 the decorrelator is the sole active ingredient); the decorrelator is a
batch ZCA statistic of the cue covariance; every read terminates in a top_index
equality count (never an energy / min-over-branches / ΔE — the Phase-5' fence). The
toy generator and the Bayes ceiling are offline data properties. No runtime metric
is read to gate, branch, or select.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from energy_memory.phase2.encoding import build_position_vectors, encode_window
from energy_memory.phase3.basin_readout import (
    selectivity_delta, top_index_hits, newcombe_diff_ci,
)
from energy_memory.phase4.decorrelator import CueDecorrelator
from energy_memory.phase4.hetero_write import (
    HeteroConsolidationBuffer, batched_hopfield_topindex,
    heteroassociative_write, recall_top_index,
)
from energy_memory.phase5.natural_source_protocol import (
    role_derangement, role_permutation,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR


# --------------------------------------------------------------------------- #
# Toy: a position-dependent topic-corpus (offline data property).
# --------------------------------------------------------------------------- #
class TopicCorpus:
    """N windows over a vocab of L target atoms + Vc context atoms + 1 mask atom.

    Each scene draws a latent topic z~U(L); the masked target token = z with prob
    (1-eps) else U(L) (eps sets the Bayes ceiling < 1). Each context position p is
    filled from a POSITION-DEPENDENT topic affinity A[z, p, ·] over context atoms
    (``posdep=False`` shares one affinity across positions -> a bag/content-only
    structure, the structure-ablation control). Context atoms are SHARED across
    topics, so a sparse cue under-determines the scene (cue collision) but its
    majority target is predictable -> store-as-is's scene-MHN blend-corrupts while
    an aggregate key->value write recovers the majority."""

    def __init__(self, *, seed, W, L, Vc, N, eps, posdep=True):
        g = torch.Generator().manual_seed(seed * 7919 + 1)
        self.W, self.L, self.Vc, self.N, self.eps, self.posdep = W, L, Vc, N, eps, posdep
        self.mpos = W // 2
        self.ctxp = [p for p in range(W) if p != self.mpos]
        self.mask_id = L + Vc                      # last codebook row
        self.vocab = L + Vc + 1
        n_fav = max(2, Vc // 4)
        A = torch.zeros(L, W, Vc)
        for z in range(L):
            shared = torch.randperm(Vc, generator=g)[:n_fav]   # used iff not posdep
            for p in self.ctxp:
                idx = torch.randperm(Vc, generator=g)[:n_fav] if posdep else shared
                A[z, p, idx] = 1.0
        A = A / A.sum(-1, keepdim=True).clamp_min(1e-9)
        self.topics = torch.randint(0, L, (N,), generator=g)
        self.noise = torch.rand(N, generator=g) < eps          # rule vs noise scene
        self.targets = torch.where(
            self.noise, torch.randint(0, L, (N,), generator=g), self.topics)
        windows = []
        for s in range(N):
            z = int(self.topics[s]); row = [0] * W
            for p in self.ctxp:
                row[p] = L + int(torch.multinomial(A[z, p], 1, generator=g))
            row[self.mpos] = int(self.targets[s])
            windows.append(row)
        self.windows = windows

    def observed_positions(self, observed):
        """The first ``observed`` context positions (fixed order; fewer == sparser)."""
        return self.ctxp[:max(1, min(observed, len(self.ctxp)))]

    def bayes_ceiling(self, observed, *, ref=None):
        """Bayes-optimal recoverability of the target from the observed cue:
        the majority target per observed-cue-class. ``ref`` (train) supplies the
        majority map for a held-out read; default = self (memorization ceiling =
        the cue-collision limit, the corpus analogue, Report 055:24)."""
        op = self.observed_positions(observed)
        src = ref if ref is not None else self
        grp = defaultdict(list)
        for s, w in enumerate(src.windows):
            grp[tuple(w[p] for p in op)].append(int(src.targets[s]))
        maj = {k: Counter(v).most_common(1)[0][0] for k, v in grp.items()}
        hits = 0
        for s, w in enumerate(self.windows):
            key = tuple(w[p] for p in op)
            hits += int(maj.get(key, -1) == int(self.targets[s]))
        return hits / self.N

    def frac_seen(self, observed, ref):
        op = self.observed_positions(observed)
        seen = {tuple(w[p] for p in op) for w in ref.windows}
        return sum(tuple(w[p] for p in op) in seen for w in self.windows) / self.N

    def codebook_targets(self, sub):
        """Substrate codebook + value (read) codebook + target indices into it +
        chance. For the topic-corpus the value codebook is the L target atoms."""
        codebook = sub.random_vectors(self.vocab)
        value_cb = codebook[:self.L]
        tgt = torch.tensor([int(t) for t in self.targets], device=value_cb.device)
        return codebook, value_cb, tgt, 1.0 / self.L


class CorpusWindows:
    """Real-corpus window source with the TopicCorpus interface, for the Phase-2
    convincer (the SAME G-D panel on repo_sample / wikitext). Masked-center windows;
    value codebook = the decodable tokens (chance = 1/n_decode); no rule/noise split
    (``noise`` all False -> the entropy rule/noise split degenerates, harmless)."""

    # offline-statistic methods shared with TopicCorpus (depend only on the interface)
    observed_positions = TopicCorpus.observed_positions
    bayes_ceiling = TopicCorpus.bayes_ceiling
    frac_seen = TopicCorpus.frac_seen

    def __init__(self, *, seed, W, N, corpus_source, wikitext_name, max_vocab, repo_root):
        from energy_memory.phase2.corpus import (
            build_vocabulary, encode_texts, load_corpus_splits, make_windows, sample_windows)
        from energy_memory.phase2.encoding import mask_positions
        splits = load_corpus_splits(corpus_source, repo_root, wikitext_name=wikitext_name)
        self._vocab = build_vocabulary(splits["train"], max_vocab=max_vocab)
        ids = encode_texts(splits["train"], self._vocab)
        self._decode_ids = [i for i, t in enumerate(self._vocab.id_to_token)
                            if t not in {self._vocab.unk_token, self._vocab.mask_token}]
        self.W = W
        self.mpos = mask_positions(W, 1, "center")[0]
        self.ctxp = [p for p in range(W) if p != self.mpos]
        self.mask_id = self._vocab.mask_id
        self.vocab = len(self._vocab.id_to_token)
        allw = make_windows(ids, W)
        ws = sample_windows(allw, min(N, len(allw)), seed=seed + 7)
        ws = [w for w in ws
              if w[self.mpos] != self._vocab.unk_id and w[self.mpos] != self._vocab.mask_id]
        self.windows = ws
        self.N = len(ws)
        self.targets = torch.tensor([w[self.mpos] for w in ws], dtype=torch.long)
        self.noise = torch.zeros(self.N, dtype=torch.bool)
        self.L = len(self._decode_ids)         # chance reference (1/n_decode)

    def codebook_targets(self, sub):
        codebook = sub.random_vectors(self.vocab)
        value_cb = codebook[self._decode_ids]
        remap = {tid: i for i, tid in enumerate(self._decode_ids)}
        tgt = torch.tensor([remap[int(t)] for t in self.targets], device=value_cb.device)
        return codebook, value_cb, tgt, 1.0 / len(self._decode_ids)


# --------------------------------------------------------------------------- #
# Cue encoding (masked sparse cue; read-time position transform only).
# --------------------------------------------------------------------------- #
def encode_cue(sub, positions, codebook, w, op, mpos, W, mask_id, *, mode="true", perm=None):
    """Bundle the observed (position, token) bindings; mask the rest + the target.

    mode:
      true    : observed tokens bound to their own positions (the headline cue).
      derange : observed tokens bound to a fixed-point-free permutation of the
                observed positions (Control 3, role-pairing shuffle / Selectivity-Δ).
      randperm: observed tokens bound to a possibly-fixed-point permutation
                (the C.3 E-arm above-chance scatter arm).
      identity: identity permutation (C.3 E-arm; MUST be byte-identical to true).
      bag     : all observed tokens bound to ONE shared position (Control 4,
                content-matched NON-positional: same content multiset, no role).
    """
    keep = set(op)
    toks = [w[p] if p in keep else mask_id for p in range(W)]
    toks[mpos] = mask_id
    if mode == "bag":
        anchor = positions[op[0]]
        terms = [sub.bind(anchor, codebook[toks[p]]) if p in keep
                 else sub.bind(positions[p], codebook[toks[p]]) for p in range(W)]
        return sub.bundle(terms)
    rp = list(range(W))
    if mode in ("derange", "randperm", "identity"):
        if len(op) >= 2:
            if mode == "identity":
                pp = list(range(len(op)))
            elif mode == "derange":
                pp = role_derangement(0, len(op))
            else:
                pp = role_permutation(0, len(op))
            for i, p in enumerate(op):
                rp[p] = op[pp[i]]
        elif mode != "identity":
            # obs==1: cannot derange a single observed slot; bind the lone observed
            # token to a WRONG position (a fixed-point-free remap onto an unobserved
            # context-or-target position) so the selectivity control is non-trivial.
            wrong = (op[0] + 1) % W
            if wrong == op[0]:
                wrong = (op[0] + 2) % W
            rp[op[0]] = wrong
    if perm is not None:                            # explicit per-seed perm override
        rp = perm
    return sub.bundle([sub.bind(positions[rp[p]], codebook[toks[p]]) for p in range(W)])


def _retrieve_state(sub, patterns, queries, beta, mi):
    state = sub.normalize(queries); d = patterns.shape[1]
    for _ in range(mi):
        scores = (state @ patterns.conj().T).real / d
        w = torch.softmax(beta * scores, dim=1)
        state = sub.normalize(w.to(patterns.dtype) @ patterns)
    return state


# --------------------------------------------------------------------------- #
# Read paths. value_codebook = the L target atoms (chance = 1/L, the clean read).
# --------------------------------------------------------------------------- #
def store_read(sub, full_enc, cue_keys, positions, value_cb, mpos, beta, mi):
    """The project's current path: scene-MHN(cue) -> unbind(mask) -> cleanup."""
    state = _retrieve_state(sub, full_enc, cue_keys, beta, mi)
    slot = sub.normalize(sub.unbind(state, positions[mpos]))
    return batched_hopfield_topindex(sub, value_cb, slot, beta=beta, max_iter=mi)


def write_H(keys_train, vidx_train, value_cb, *, decorr, dim, lr, epochs):
    """Fit the (optional) decorrelator on TRAIN keys, write H over the FROZEN
    train buffer. Returns (H, decorrelator_or_None)."""
    dec = None
    kt = keys_train
    if decorr is not None:
        dec = CueDecorrelator(dim, renorm=decorr).fit(keys_train)
        kt = dec.apply(keys_train)
    buf = HeteroConsolidationBuffer(dim, kt.device)
    for i in range(kt.shape[0]):
        buf.add(kt[i], int(vidx_train[i]))
    buf.freeze()
    H = heteroassociative_write(buf, value_cb, lr=lr, epochs=epochs)
    return H, dec


def write_read(sub, H, dec, keys_read, value_cb, beta, mi):
    kr = dec.apply(keys_read) if dec is not None else keys_read
    return recall_top_index(sub, H, kr, value_cb, beta=beta, max_iter=mi)


# --------------------------------------------------------------------------- #
# One (seed, observed) cell: the full panel.
# --------------------------------------------------------------------------- #
def run_cell(args, seed, observed):
    dev = args.device
    if args.corpus_source == "synthetic":
        src = TopicCorpus(seed=seed, W=args.W, L=args.L, Vc=args.Vc, N=args.N,
                          eps=args.eps, posdep=True)
    else:
        import pathlib
        src = CorpusWindows(seed=seed, W=args.W, N=args.N, corpus_source=args.corpus_source,
                            wikitext_name=args.wikitext_name, max_vocab=args.max_vocab,
                            repo_root=pathlib.Path(__file__).resolve().parents[1])
    op = src.observed_positions(observed)
    mpos, mask_id, W = src.mpos, src.mask_id, src.W
    sub = TorchFHRR(dim=args.D, seed=seed, device=dev)
    codebook, value_cb, tgt, chance = src.codebook_targets(sub)
    tgt = tgt.to(dev)
    positions = build_position_vectors(sub, W)
    rule_mask = (~src.noise).to(dev)                 # rule scenes (for entropy split)

    # ---- HELD-OUT SPLIT (spec R1 :57 "on a held-out split"): fit the write H + the
    # decorrelator + the store-as-is scene-MHN on TRAIN, READ on a DISJOINT TEST half.
    # The legacy in-sample memorization read is kept under --split insample for reference.
    half = src.N // 2
    if args.split == "insample":
        tr, te = list(range(src.N)), list(range(src.N))
    else:
        tr, te = list(range(half)), list(range(half, src.N))
    tr_t = torch.tensor(tr, device=dev)
    te_t = torch.tensor(te, device=dev)
    n_te = len(te)
    tgt_te = tgt[te_t]
    rule_te = rule_mask[te_t]

    def cue_set(mode):
        return torch.stack([encode_cue(sub, positions, codebook, w, op, mpos, W,
                                       mask_id, mode=mode) for w in src.windows])

    K_true = cue_set("true")

    # Role-shuffle (Control 3) = a PER-SCENE fixed-point-free position derangement.
    # Per-scene (not one uniform swap) so the destroyed-structure arm SCATTERS to
    # at/above chance instead of landing systematically sub-chance (the anti-overlap
    # artifact the two-floor rule rejects; spec control-3 note + the C.3 E-arm
    # "scatters around 0 from above chance"). Gauge-safe: every observed slot maps to
    # a different slot. Seed-fixed -> reproducible.
    def deranged_cue_set():
        keys = []
        for si, w in enumerate(src.windows):
            g = torch.Generator().manual_seed(seed * 100003 + si * 131 + 17)
            rp = list(range(W))
            if len(op) >= 2:
                while True:
                    pp = torch.randperm(len(op), generator=g).tolist()
                    if all(i != p for i, p in enumerate(pp)):
                        break
                for i, p in enumerate(op):
                    rp[p] = op[pp[i]]
            else:
                cands = [p for p in range(W) if p != op[0]]
                rp[op[0]] = cands[int(torch.randint(0, len(cands), (1,), generator=g))]
            keys.append(encode_cue(sub, positions, codebook, w, op, mpos, W,
                                   mask_id, mode="true", perm=rp))
        return torch.stack(keys)
    K_der = deranged_cue_set()
    K_bag = cue_set("bag")
    K_randperm = cue_set("randperm")
    full_enc = torch.stack([encode_window(sub, positions, codebook, w) for w in src.windows])
    key_cos = float(((K_true[tr_t] @ K_true[tr_t].conj().T).real.abs() / args.D)
                    [~torch.eye(len(tr), dtype=torch.bool, device=dev)].mean())

    # --- C.3 E-arm: identity permutation MUST be byte-identical to the true cue.
    e_arm_identity_byte_identical = bool(torch.equal(K_true, cue_set("identity")))

    # Bayes ceilings: held-out (train-majority -> test) is the honest generalization
    # ceiling; in-sample (test-majority -> test) is the memorization ceiling (for ref).
    def maj_ceiling(idx_fit, idx_eval):
        grp = defaultdict(Counter)
        for i in idx_fit:
            grp[tuple(src.windows[i][p] for p in op)][int(src.targets[i])] += 1
        maj = {k: c.most_common(1)[0][0] for k, c in grp.items()}
        return sum(maj.get(tuple(src.windows[i][p] for p in op), -1) == int(src.targets[i])
                   for i in idx_eval) / len(idx_eval)
    seen = {tuple(src.windows[i][p] for p in op) for i in tr}
    frac_seen = sum(tuple(src.windows[i][p] for p in op) in seen for i in te) / n_te

    cell = {"seed": seed, "observed": observed, "chance": chance, "split": args.split,
            "ceiling": maj_ceiling(tr, te), "ceiling_insample": maj_ceiling(te, te),
            "frac_seen": frac_seen, "key_cos": key_cos, "n_test": n_te,
            "n_rule": int(rule_te.sum()), "n_noise": int((~rule_te).sum()),
            "e_arm_identity_byte_identical": e_arm_identity_byte_identical, "arms": {}}

    def hits_split(ti):                              # ti is test-length
        h = top_index_hits(ti, tgt_te)
        hr = top_index_hits(ti[rule_te], tgt_te[rule_te]) if int(rule_te.sum()) else 0
        hn = top_index_hits(ti[~rule_te], tgt_te[~rule_te]) if int((~rule_te).sum()) else 0
        return h, hr, hn

    def record(name, ti_true, ti_shuf, ent, marg):
        h_t, hr, hn = hits_split(ti_true)
        h_s = top_index_hits(ti_shuf, tgt_te)
        sd = selectivity_delta(true_hits=h_t, true_n=n_te,
                               shuffled_hits=h_s, shuffled_n=n_te, chance=chance)
        d = sd.as_dict()
        d.update({"rule_rate": hr / max(1, int(rule_te.sum())),
                  "noise_rate": hn / max(1, int((~rule_te).sum())),
                  "mean_entropy": float(ent.mean()), "mean_margin": float(marg.mean()),
                  "entropy_rule": float(ent[rule_te].mean()) if int(rule_te.sum()) else 0.0,
                  "entropy_noise": float(ent[~rule_te].mean()) if int((~rule_te).sum()) else 0.0,
                  "true_hits": h_t, "shuffled_hits": h_s, "n": n_te})
        cell["arms"][name] = d
        return d

    # ---------------- store-as-is (the anchor): scene-MHN on TRAIN, read TEST cues ----------------
    ti_sa, ent_sa, mg_sa = store_read(sub, full_enc[tr_t], K_true[te_t], positions, value_cb, mpos, args.beta, args.mi)
    ti_sa_s, _, _ = store_read(sub, full_enc[tr_t], K_der[te_t], positions, value_cb, mpos, args.beta, args.mi)
    sa = record("store_as_is", ti_sa, ti_sa_s, ent_sa, mg_sa)

    # ---------------- write + L2 decorr (the HEADLINE mechanism): fit TRAIN, read TEST ----------------
    H, dec = write_H(K_true[tr_t], tgt[tr_t], value_cb, decorr="l2", dim=args.D, lr=args.lr, epochs=args.epochs)
    ti_w, ent_w, mg_w = write_read(sub, H, dec, K_true[te_t], value_cb, args.beta, args.mi)
    ti_w_s, _, _ = write_read(sub, H, dec, K_der[te_t], value_cb, args.beta, args.mi)
    wl2 = record("write_l2", ti_w, ti_w_s, ent_w, mg_w)

    # write-marginal anchor: Δ(write+L2) − Δ(store-as-is), Newcombe diff of the true arms
    d_lo, d_hi = newcombe_diff_ci(wl2["true_hits"], n_te, sa["true_hits"], n_te)
    cell["write_marginal"] = {
        "write_selectivity_delta": wl2["delta"], "store_selectivity_delta": sa["delta"],
        "delta_of_deltas": wl2["delta"] - sa["delta"],
        "write_minus_store_true": wl2["true_rate"] - sa["true_rate"],
        "write_minus_store_ci": [d_lo, d_hi], "write_beats_store": d_lo > 0.0,
        "headline_pass": bool(wl2["two_floor_pass"] and d_lo > 0.0),
    }

    # ---------------- Control 2: no-decorr (write alone), fit TRAIN read TEST ----------------
    Hnd, _ = write_H(K_true[tr_t], tgt[tr_t], value_cb, decorr=None, dim=args.D, lr=args.lr, epochs=args.epochs)
    ti_nd, ent_nd, mg_nd = write_read(sub, Hnd, None, K_true[te_t], value_cb, args.beta, args.mi)
    ti_nd_s, _, _ = write_read(sub, Hnd, None, K_der[te_t], value_cb, args.beta, args.mi)
    record("write_no_decorr", ti_nd, ti_nd_s, ent_nd, mg_nd)

    # ---------------- Fork-3 ablation: element-wise renorm (the Report-053 bug) ----------------
    He, dece = write_H(K_true[tr_t], tgt[tr_t], value_cb, decorr="elementwise", dim=args.D, lr=args.lr, epochs=args.epochs)
    ti_e, ent_e, mg_e = write_read(sub, He, dece, K_true[te_t], value_cb, args.beta, args.mi)
    ti_e_s, _, _ = write_read(sub, He, dece, K_der[te_t], value_cb, args.beta, args.mi)
    record("write_elementwise", ti_e, ti_e_s, ent_e, mg_e)

    # ---------------- Control 1: random-codebook (readout-leak test): read H-output (TEST) vs random cb ----------------
    rand_cb = sub.random_vectors(value_cb.shape[0])
    recalled = (dec.apply(K_true[te_t]) @ H.transpose(0, 1)) / args.D
    ti_r, ent_r, mg_r = batched_hopfield_topindex(sub, rand_cb, recalled, beta=args.beta, max_iter=args.mi)
    recalled_s = (dec.apply(K_der[te_t]) @ H.transpose(0, 1)) / args.D
    ti_r_s, _, _ = batched_hopfield_topindex(sub, rand_cb, recalled_s, beta=args.beta, max_iter=args.mi)
    record("random_codebook", ti_r, ti_r_s, ent_r, mg_r)

    # ---------------- Control 4: content-matched NON-positional (read HEADLINE H with a bag TEST cue) ----------------
    ti_b, ent_b, mg_b = write_read(sub, H, dec, K_bag[te_t], value_cb, args.beta, args.mi)
    record("content_matched_bag", ti_b, ti_b, ent_b, mg_b)

    # ---------------- Control 5: perfect-cue upper bound (TEST) ----------------
    ti_pc, _, _ = batched_hopfield_topindex(sub, value_cb, value_cb[tgt_te], beta=args.beta, max_iter=args.mi)
    cell["perfect_cue_rate"] = top_index_hits(ti_pc, tgt_te) / n_te

    # ---------------- C.3 E-arm: random-perm scatter (above chance, not sub-chance), TEST ----------------
    ti_w_rp, _, _ = write_read(sub, H, dec, K_randperm[te_t], value_cb, args.beta, args.mi)
    cell["e_arm_randperm_rate"] = top_index_hits(ti_w_rp, tgt_te) / n_te

    # ---------------- structure-ablation diagnostic: a BAG (position-independent) toy, held-out ----------------
    if args.structure_ablation and args.corpus_source == "synthetic":
        tcb = TopicCorpus(seed=seed, W=args.W, L=args.L, Vc=args.Vc, N=args.N, eps=args.eps, posdep=False)
        codb = sub.random_vectors(tcb.vocab); vcb = codb[:args.L]
        opb = tcb.observed_positions(observed)
        tgb = torch.tensor([int(t) for t in tcb.targets], device=dev)
        Kb = torch.stack([encode_cue(sub, positions, codb, w, opb, mpos, W, tcb.mask_id, mode="true") for w in tcb.windows])
        Kb_s = torch.stack([encode_cue(sub, positions, codb, w, opb, mpos, W, tcb.mask_id, mode="derange") for w in tcb.windows])
        hb = tcb.N // 2
        trb = torch.arange(hb, device=dev) if args.split != "insample" else torch.arange(tcb.N, device=dev)
        teb = torch.arange(hb, tcb.N, device=dev) if args.split != "insample" else torch.arange(tcb.N, device=dev)
        Hbb, decbb = write_H(Kb[trb], tgb[trb], vcb, decorr="l2", dim=args.D, lr=args.lr, epochs=args.epochs)
        ti_bt, _, _ = write_read(sub, Hbb, decbb, Kb[teb], vcb, args.beta, args.mi)
        ti_bs, _, _ = write_read(sub, Hbb, decbb, Kb_s[teb], vcb, args.beta, args.mi)
        sdb = selectivity_delta(true_hits=top_index_hits(ti_bt, tgb[teb]), true_n=len(teb),
                                shuffled_hits=top_index_hits(ti_bs, tgb[teb]), shuffled_n=len(teb), chance=chance)
        cell["structure_ablation_bag_toy"] = {
            "write_true_rate": top_index_hits(ti_bt, tgb[teb]) / len(teb),
            "write_selectivity_delta": sdb.delta, "two_floor_pass": sdb.two_floor_pass}
    return cell


def aggregate(cells):
    """Pool hits across seeds at each observed level for the pooled two-floor read."""
    by_obs = defaultdict(list)
    for c in cells:
        by_obs[c["observed"]].append(c)
    out = {}
    for obs, cs in sorted(by_obs.items()):
        L_chance = cs[0]["chance"]
        def pool(arm, field):
            return sum(c["arms"][arm][field] for c in cs)
        n = pool("write_l2", "n")
        wl2 = selectivity_delta(true_hits=pool("write_l2", "true_hits"), true_n=n,
                                shuffled_hits=pool("write_l2", "shuffled_hits"), shuffled_n=n,
                                chance=L_chance)
        sa = selectivity_delta(true_hits=pool("store_as_is", "true_hits"), true_n=n,
                               shuffled_hits=pool("store_as_is", "shuffled_hits"), shuffled_n=n,
                               chance=L_chance)
        dlo, dhi = newcombe_diff_ci(pool("write_l2", "true_hits"), n,
                                    pool("store_as_is", "true_hits"), n)
        # per-seed cluster-honest disclosure: each seed = a distinct corpus + codebook.
        n_seeds = len(cs)
        seeds_two_floor = sum(c["arms"]["write_l2"]["two_floor_pass"] for c in cs)
        seeds_beats_store = sum(c["write_marginal"]["write_beats_store"] for c in cs)
        out[str(obs)] = {
            "n_pooled": n, "n_seeds": n_seeds, "chance": L_chance, "split": cs[0].get("split"),
            "ceiling_mean": sum(c["ceiling"] for c in cs) / n_seeds,                 # held-out ceiling
            "ceiling_insample_mean": sum(c["ceiling_insample"] for c in cs) / n_seeds,
            "frac_seen_mean": sum(c["frac_seen"] for c in cs) / n_seeds,
            "perfect_cue_mean": sum(c["perfect_cue_rate"] for c in cs) / n_seeds,
            "write_l2": wl2.as_dict(), "store_as_is": sa.as_dict(),
            "write_no_decorr_true": sum(c["arms"]["write_no_decorr"]["true_rate"] for c in cs) / n_seeds,
            "write_elementwise_true": sum(c["arms"]["write_elementwise"]["true_rate"] for c in cs) / n_seeds,
            "random_codebook_true": sum(c["arms"]["random_codebook"]["true_rate"] for c in cs) / n_seeds,
            "content_matched_bag_true": sum(c["arms"]["content_matched_bag"]["true_rate"] for c in cs) / n_seeds,
            "delta_of_deltas": wl2.delta - sa.delta,
            "write_minus_store_ci": [dlo, dhi],
            "ROLE_DELTA_PASS": bool(wl2.two_floor_pass),                             # the spec R1 headline
            "WRITE_MARGINAL_PASS": bool(dlo > 0.0),                                  # smoke-refinement-1 add-on
            "HEADLINE_PASS": bool(wl2.two_floor_pass and dlo > 0.0),
            "seeds_two_floor_pass": f"{seeds_two_floor}/{n_seeds}",
            "seeds_write_beats_store": f"{seeds_beats_store}/{n_seeds}",
            "e_arm_identity_byte_identical_all": all(c["e_arm_identity_byte_identical"] for c in cs),
            "e_arm_randperm_rate_mean": sum(c["e_arm_randperm_rate"] for c in cs) / n_seeds,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=2048)
    ap.add_argument("--corpus-source", choices=["synthetic", "repo_sample", "wikitext"],
                    default="synthetic", dest="corpus_source")
    ap.add_argument("--wikitext-name", default="wikitext-2-raw-v1", dest="wikitext_name")
    ap.add_argument("--max-vocab", type=int, default=512, dest="max_vocab")
    ap.add_argument("--W", type=int, default=6)
    ap.add_argument("--L", type=int, default=8, help="topics == target-codebook size; chance=1/L")
    ap.add_argument("--Vc", type=int, default=16, help="context-vocab size")
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--eps", type=float, default=0.25, help="noise frac; sets the Bayes ceiling < 1")
    ap.add_argument("--observed", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--split", choices=["heldout", "insample"], default="heldout",
                    help="heldout (spec R1): fit on train half, read disjoint test half; "
                         "insample: legacy memorization read (fit==read), for reference")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--mi", type=int, default=12, dest="mi")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--structure-ablation", action="store_true", dest="structure_ablation")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    cells = []
    for seed in range(args.seeds):
        for obs in args.observed:
            cells.append(run_cell(args, seed, obs))
    summary = {"config": vars(args), "pooled": aggregate(cells), "cells": cells}

    print(json.dumps({"config": {k: vars(args)[k] for k in ("D", "L", "Vc", "N", "eps", "seeds")},
                      "pooled": summary["pooled"]}, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
