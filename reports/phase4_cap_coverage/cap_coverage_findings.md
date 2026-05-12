# Cap-Coverage Diagnostic — Findings

Re-measured Phase 4 results through the project's standing diagnostics
(cap-coverage error, meta-stable-state rate, softmax entropy) across
β ∈ {5, 10, 30}, three seeds, N=500 per condition.

## Headline

The cap-coverage reframing **does not substantially reinterpret** our earlier
results. The architecture is operating as a contextual-completion system within
its theoretical capacity. Top-K Recall was already capturing the neighborhood
signal the cap-coverage formalism makes explicit.

The earlier finding — multi-scale FHRR gives 2-2.5x bigram lift on contextual
completion — stands. It is neither dramatically larger (the architecture isn't
hiding compositional structure we missed) nor dramatically smaller (it isn't
just a measurement artifact).

## β regime sweep (recon codebook, W=4 evaluation)

| β | mode | entropy | meta-rate | top_score | top-1 | top-10 | cap_t@0.5 |
|---|---|---|---|---|---|---|---|
| 5 | feature | 0.92 | 73% | 0.86 | 9.7% | 33.6% | 0.0% |
| 10 | mixed | 0.72 | 26% | 0.97 | 10.9% | 33.3% | 3.0% |
| 30 | prototype | 0.32 | 5% | 0.99 | 10.6% | 29.9% | 10.2% |

For W=8 evaluation, β=10 has a small but real edge (top-1 = 13.6% vs 13.4% at β=30) — feature-mode discrimination working slightly better than prototype lock-in.

## Six findings

**1. We've been operating in prototype mode at β=30.** Entropy 0.02-0.39,
meta-stable rate near zero, top_score ≈ 1.0. Every retrieval is converging to a
single stored pattern, winner-take-all.

**2. Top-K Recall is roughly β-invariant for the recon codebook.** Going from
feature mode (β=5) to prototype mode (β=30) barely changes top-10 accuracy.
The architecture's neighborhood capacity is structural, not regime-dependent.

**3. Random codebook collapses in feature mode.** At β=5, random W=2 gets
top-1 = 0.1%. At β=30, it gets 4.0%. The reconstruction codebook is doing real
work specifically when retrieval is blended — without learned structure, feature
mode is just noise.

**4. cap_t@0.3 ≈ top-K Recall.** The neighborhood metric at low confidence is
essentially the same as Recall@10. The cap-coverage formalism doesn't reveal a
hidden performance level the identity metric was hiding.

**5. cap_t@0.5 is the metric that combines identity + confidence.** It rises
with β because prototype mode gives high-confidence top-1. At β=30 with recon
W=2 it equals 10.2% — meaning the system gives confident-and-correct answers
about 10% of the time. This is the most architecturally aligned single metric:
"did we get the right answer, with the system knowing it?"

**6. Meta-stable rate diagnoses landscape over-saturation.** W=4 at β=5 has
meta-rate = 100% — the system literally never converges. With 1024 patterns
each containing 4 atoms, the bundle interference is too high at low β. This is
a real capacity ceiling, not a tuning issue.

## What this tells us

The architecture is doing pattern completion within its theoretical capacity.
Identity at ~13% top-1 and neighborhood at ~33% top-10, beating bigrams 2-2.5x.
These numbers are stable across β regimes (for the recon codebook) and survive
the cap-coverage reframing.

The architecture is **not** a hidden compositional system we underestimated.
The cap-coverage / neighborhood interpretation shifts the *narrative* (we're
not failing at sequence prediction, we're doing contextual completion at the
level we claim) but not the *numbers*.

The bottleneck is structural, not measurement. Three potentially-real
constraints surfaced by this diagnostic:

- **Bundle SNR** at high W (W=4 hits meta-rate = 100% at β=5)
- **Codebook discriminability** in feature mode (random codebook collapses;
  reconstruction codebook holds)
- **No top-down feedback between scales** — our multi-scale aggregation
  averages scores but doesn't let longer-context scales constrain shorter-
  context predictions

The first two suggest substrate improvements (qFHRR, better codebook learning).
The third suggests Krotov's HAM-style bidirectional energy convergence is the
next architectural move once we have replay/consolidation in place.
