"""Experiment 16: Autoregressive generation diagnostic.

Tests whether the multi-scale architecture produces coherent sequences
when chained autoregressively, or just bigram-quality output.

For each of N prompts (seeded from validation), generates 50 tokens
using:
  - Multi-scale FHRR retrieval (reconstruction codebook)
  - Multi-scale FHRR retrieval (random codebook, for ablation)
  - Bigram greedy baseline
  - Bigram sampled baseline

Both greedy (argmax) and temperature-sampled (T=0.5) variants for
each FHRR config. Measures distinct-N, repetition rate, and
perplexity-under-training-bigrams for each output. Also saves the
generated text for qualitative inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.phase2.corpus import (
    build_vocabulary,
    encode_texts,
    load_corpus_splits,
    make_windows,
    sample_windows,
)
from energy_memory.phase2.encoding import (
    build_position_vectors,
    decode_position,
    encode_window,
)
from energy_memory.phase2.persistence import load_codebook
from energy_memory.substrate.torch_fhrr import TorchFHRR


class ScaleSlot:
    def __init__(
        self,
        substrate: TorchFHRR,
        train_windows: Sequence[tuple[int, ...]],
        window_size: int,
        landscape_size: int,
        codebook: torch.Tensor,
        seed: int,
    ):
        self.substrate = substrate
        self.window_size = window_size
        self.codebook = codebook
        self.positions = build_position_vectors(substrate, window_size)
        actual_l = min(landscape_size, len(train_windows))
        landscape = sample_windows(train_windows, actual_l, seed=seed)
        self.memory: TorchHopfieldMemory[str] = TorchHopfieldMemory(substrate)
        for idx, w in enumerate(landscape):
            self.memory.store(
                encode_window(substrate, self.positions, codebook, w),
                label=f"w_{idx}",
            )


def predict_next_multiscale(
    context: List[int],
    scale_slots: Dict[int, ScaleSlot],
    decode_ids: List[int],
    mask_id: int,
    beta: float,
    decode_k: int,
) -> Dict[int, float]:
    """Return aggregated score per token id (multi-scale combined)."""
    combined: Dict[int, float] = defaultdict(float)
    for scale, slot in scale_slots.items():
        if len(context) < scale - 1:
            continue
        sub_context = context[-(scale - 1):]
        cue_window = list(sub_context) + [mask_id]
        cue = encode_window(slot.substrate, slot.positions, slot.codebook, cue_window)
        result = slot.memory.retrieve(cue, beta=beta, max_iter=12)
        decoded = decode_position(
            slot.substrate, result.state, slot.positions[scale - 1],
            slot.codebook, decode_ids, top_k=decode_k,
        )
        for tok_id, score in decoded:
            combined[tok_id] += score
    return dict(combined)


def sample_from_scores(
    scores: Dict[int, float],
    temperature: float,
    rng: random.Random,
) -> int:
    if not scores:
        return -1
    if temperature <= 1e-9:
        return max(scores, key=scores.get)
    tokens = list(scores.keys())
    raw = [scores[t] / temperature for t in tokens]
    m = max(raw)
    exp_vals = [math.exp(v - m) for v in raw]
    total = sum(exp_vals)
    probs = [v / total for v in exp_vals]
    r = rng.random()
    cum = 0.0
    for tok, p in zip(tokens, probs):
        cum += p
        if r <= cum:
            return tok
    return tokens[-1]


def generate_multiscale(
    seed_tokens: List[int],
    n_new: int,
    scale_slots: Dict[int, ScaleSlot],
    decode_ids: List[int],
    mask_id: int,
    unk_id: int,
    beta: float,
    decode_k: int,
    temperature: float,
    rng: random.Random,
) -> List[int]:
    output = list(seed_tokens)
    for _ in range(n_new):
        scores = predict_next_multiscale(
            output, scale_slots, decode_ids, mask_id, beta, decode_k,
        )
        scores.pop(mask_id, None)
        scores.pop(unk_id, None)
        if not scores:
            break
        next_tok = sample_from_scores(scores, temperature, rng)
        if next_tok < 0:
            break
        output.append(next_tok)
    return output[len(seed_tokens):]


def generate_bigram(
    seed_tokens: List[int],
    n_new: int,
    forward_counts: Dict[int, Counter],
    unigram_best: int,
    unk_id: int,
    temperature: float,
    rng: random.Random,
) -> List[int]:
    output = list(seed_tokens)
    for _ in range(n_new):
        last = output[-1]
        counts = forward_counts.get(last)
        if not counts:
            next_tok = unigram_best
        elif temperature <= 1e-9:
            next_tok = counts.most_common(1)[0][0]
        else:
            tokens = list(counts.keys())
            weights = [counts[t] ** (1.0 / temperature) for t in tokens]
            total = sum(weights)
            probs = [w / total for w in weights]
            r = rng.random()
            cum = 0.0
            chosen = tokens[-1]
            for tok, p in zip(tokens, probs):
                cum += p
                if r <= cum:
                    chosen = tok
                    break
            next_tok = chosen
        if next_tok == unk_id:
            continue
        output.append(next_tok)
    return output[len(seed_tokens):]


def distinct_n(tokens: List[int], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def repetition_rate(tokens: List[int], window: int = 8) -> float:
    if not tokens:
        return 0.0
    repeats = 0
    for i, tok in enumerate(tokens):
        recent = tokens[max(0, i - window):i]
        if tok in recent:
            repeats += 1
    return repeats / len(tokens)


def perplexity_under_bigrams(
    tokens: List[int],
    forward_counts: Dict[int, Counter],
    vocab_size: int,
    smoothing: float = 0.01,
) -> float:
    if len(tokens) < 2:
        return float("inf")
    log_sum = 0.0
    n = 0
    for i in range(1, len(tokens)):
        prev, cur = tokens[i - 1], tokens[i]
        counts = forward_counts.get(prev, {})
        total = sum(counts.values()) + smoothing * vocab_size
        prob = (counts.get(cur, 0) + smoothing) / total
        log_sum += -math.log(prob)
        n += 1
    return math.exp(log_sum / n) if n > 0 else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-vocab", type=int, default=2048)
    parser.add_argument("--corpus-source", default="wikitext")
    parser.add_argument("--wikitext-name", default="wikitext-2-raw-v1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--codebook-path",
        default="reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt",
    )
    parser.add_argument("--scale-landscape", default="2:4096,3:2048,4:1024")
    parser.add_argument("--beta", type=float, default=30.0)
    parser.add_argument("--decode-k", type=int, default=20)
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--seed-len", type=int, default=3)
    parser.add_argument("--gen-len", type=int, default=50)
    parser.add_argument("--temperatures", default="0.0,0.5")
    parser.add_argument("--output-dir", default="reports/phase4_generation")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_ls = {}
    for item in args.scale_landscape.split(","):
        s, l = item.split(":")
        scale_ls[int(s)] = int(l)
    scales = sorted(scale_ls.keys())
    temperatures = [float(x) for x in args.temperatures.split(",")]

    print("loading corpus...", flush=True)
    splits = load_corpus_splits(
        args.corpus_source, repo_root, wikitext_name=args.wikitext_name,
    )
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    train_ids = encode_texts(splits["train"], vocab)
    validation_ids = encode_texts(splits["validation"], vocab)
    print(f"  vocab: {len(vocab.id_to_token)} tokens", flush=True)

    substrate = TorchFHRR(dim=args.dim, seed=args.seed, device=args.device)
    recon_codebook = load_codebook(
        Path(args.codebook_path), device=str(substrate.device),
    )
    random_codebook = substrate.random_vectors(len(vocab.id_to_token))
    print(f"  codebooks loaded: {recon_codebook.shape}", flush=True)

    decode_ids = [
        i for i, t in enumerate(vocab.id_to_token)
        if t not in {vocab.unk_token, vocab.mask_token}
    ]

    print("  building bigram counts...", flush=True)
    forward_counts: Dict[int, Counter] = defaultdict(Counter)
    for i in range(len(train_ids) - 1):
        if train_ids[i] != vocab.unk_id and train_ids[i + 1] != vocab.unk_id:
            forward_counts[train_ids[i]][train_ids[i + 1]] += 1
    unigram_counts = Counter(t for t in train_ids if t != vocab.unk_id)
    unigram_best = unigram_counts.most_common(1)[0][0]

    def build_slots(codebook: torch.Tensor) -> Dict[int, ScaleSlot]:
        slots: Dict[int, ScaleSlot] = {}
        for s in scales:
            train_windows_s = make_windows(train_ids, s)
            slots[s] = ScaleSlot(
                substrate=substrate,
                train_windows=train_windows_s,
                window_size=s,
                landscape_size=scale_ls[s],
                codebook=codebook,
                seed=args.seed + s * 100,
            )
        return slots

    print("  building scale slots (recon)...", flush=True)
    recon_slots = build_slots(recon_codebook)
    print("  building scale slots (random)...", flush=True)
    random_slots = build_slots(random_codebook)

    # Sample prompts from validation (oversample to filter out UNK-bearing ones)
    val_windows = make_windows(validation_ids, max(scales))
    candidate_pool = sample_windows(
        val_windows,
        min(args.n_prompts * 20, len(val_windows)),
        seed=args.seed + 5000,
    )
    prompts = []
    for w in candidate_pool:
        ctx = list(w[: args.seed_len])
        if any(t == vocab.unk_id for t in ctx):
            continue
        prompts.append(ctx)
        if len(prompts) >= args.n_prompts:
            break
    print(f"  using {len(prompts)} prompts (from {len(candidate_pool)} candidates)", flush=True)

    rng = random.Random(args.seed)
    results = []
    qualitative_lines: List[str] = []
    vocab_size = len(vocab.id_to_token)

    for prompt_idx, prompt in enumerate(prompts):
        prompt_text = " ".join(vocab.decode_token(t) for t in prompt)
        qualitative_lines.append(f"\n=== Prompt {prompt_idx + 1}: {prompt_text!r} ===")

        for temp in temperatures:
            temp_label = f"T={temp:.1f}"
            for config_name, slots in [
                ("recon", recon_slots),
                ("random", random_slots),
            ]:
                gen = generate_multiscale(
                    seed_tokens=prompt,
                    n_new=args.gen_len,
                    scale_slots=slots,
                    decode_ids=decode_ids,
                    mask_id=vocab.mask_id,
                    unk_id=vocab.unk_id,
                    beta=args.beta,
                    decode_k=args.decode_k,
                    temperature=temp,
                    rng=rng,
                )
                results.append({
                    "prompt_idx": prompt_idx,
                    "method": f"multiscale_{config_name}",
                    "temperature": temp,
                    "distinct1": distinct_n(gen, 1),
                    "distinct2": distinct_n(gen, 2),
                    "distinct3": distinct_n(gen, 3),
                    "repetition": repetition_rate(gen),
                    "ppl_bigram": perplexity_under_bigrams(
                        gen, forward_counts, vocab_size,
                    ),
                    "length": len(gen),
                })
                gen_text = " ".join(vocab.decode_token(t) for t in gen)
                qualitative_lines.append(
                    f"  [multiscale_{config_name} {temp_label}] {gen_text}"
                )

            # Bigram baseline at same temperature
            gen_bg = generate_bigram(
                seed_tokens=prompt,
                n_new=args.gen_len,
                forward_counts=forward_counts,
                unigram_best=unigram_best,
                unk_id=vocab.unk_id,
                temperature=temp,
                rng=rng,
            )
            results.append({
                "prompt_idx": prompt_idx,
                "method": "bigram",
                "temperature": temp,
                "distinct1": distinct_n(gen_bg, 1),
                "distinct2": distinct_n(gen_bg, 2),
                "distinct3": distinct_n(gen_bg, 3),
                "repetition": repetition_rate(gen_bg),
                "ppl_bigram": perplexity_under_bigrams(
                    gen_bg, forward_counts, vocab_size,
                ),
                "length": len(gen_bg),
            })
            gen_bg_text = " ".join(vocab.decode_token(t) for t in gen_bg)
            qualitative_lines.append(f"  [bigram         {temp_label}] {gen_bg_text}")

    # Aggregate metrics
    def agg(method: str, temp: float, key: str) -> Tuple[float, float]:
        vals = [r[key] for r in results if r["method"] == method and r["temperature"] == temp and math.isfinite(r[key])]
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
        return mean, math.sqrt(var)

    print("\n" + "=" * 70)
    print("Aggregated metrics (mean ± std across prompts)")
    print("=" * 70)
    methods = ["multiscale_recon", "multiscale_random", "bigram"]
    for temp in temperatures:
        print(f"\n  Temperature = {temp}")
        print(
            f"  {'Method':<22} {'distinct1':>10} {'distinct2':>10} "
            f"{'distinct3':>10} {'repetition':>11} {'ppl_bigram':>12}"
        )
        for method in methods:
            d1_m, d1_s = agg(method, temp, "distinct1")
            d2_m, d2_s = agg(method, temp, "distinct2")
            d3_m, d3_s = agg(method, temp, "distinct3")
            rep_m, rep_s = agg(method, temp, "repetition")
            ppl_m, ppl_s = agg(method, temp, "ppl_bigram")
            print(
                f"  {method:<22} "
                f"{d1_m:>6.3f}±{d1_s:.3f} "
                f"{d2_m:>6.3f}±{d2_s:.3f} "
                f"{d3_m:>6.3f}±{d3_s:.3f} "
                f"{rep_m:>7.3f}±{rep_s:.3f} "
                f"{ppl_m:>8.1f}±{ppl_s:.1f}"
            )

    # Save outputs
    json_path = output_dir / "generation_metrics.json"
    with open(json_path, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)

    text_path = output_dir / "generation_samples.txt"
    with open(text_path, "w") as f:
        f.write("\n".join(qualitative_lines))

    print(f"\n  metrics: {json_path}", flush=True)
    print(f"  samples: {text_path}", flush=True)


if __name__ == "__main__":
    main()
