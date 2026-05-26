# Literature Matrix

This matrix routes source IDs to project mechanisms. It is intentionally compact:
open the manifest and primary source before making a claim.

| Mechanism / Question | Primary Source IDs | Use In Neuro-AI | Caution |
|---|---|---|---|
| FHRR/HRR binding fidelity | `pdf:hrr-learning-2021`, `pmc:PMC9759586`, `arxiv:2405.09689`, `arxiv:2412.00488`, `arxiv:2512.14709` | Binding metrics, unbinding controls, temporal/role encoding | Do not infer Phase 5 success from algebraic decode alone; require retrieval-side controls. |
| Hopfield / MHN retrieval | `pdf:dense-associative-memory-2016`, `pdf:hen-2024`, `pdf:ham-2021`, `arxiv:2008.02217`, `arxiv:2411.08590`, `arxiv:2506.10801`, `arxiv:2603.20115` | Energy memory, beta regimes, basin shaping, hierarchy | Separate storage success from basin retrieval; inspect entropy and top-index hits. |
| Codebook diagnostics | `pdf:geometry-consolidation-2026`, `pdf:neural-collapse-2020`, `arxiv:2505.22749`, `pmc:PMC1963505`, `arxiv:2510.16039` | Phase 3 diagnostics, regime classifier, anti-collapse, slowness | Diagnostics do not become actuators unless expressed as local dynamics. |
| Replay and consolidation | `pdf:benna-fusi-2016`, `pdf:replay-review-2021`, `pdf:mir-2019`, `pdf:infors-2022`, `doi:10.1038/s41467-025-68042-3`, `pmc:PMC6203620`, `pmc:PMC6794196` | Replay selection, consolidation, slow variables, trajectory credit | Avoid metric-triggered replay schedulers; use fixed local priority dynamics. |
| Predictive association / JEPA | `pdf:pam-2026`, `pdf:pam2-2026`, `pdf:llm-jepa-2025`, `pdf:ami-2022`, `arxiv:2604.20850` | Association beyond similarity, concept discovery, language-side JEPA | Keep the LLM as voice/interface, not the self. |
| Active inference / FEP | `pii:S037015732300203X`, `arxiv:2006.04120`, `arxiv:2504.14898`, `arxiv:2505.19867`, `pubmed:40422982`, `github:infer-actively/pymdp` | Anti-homunculus framing, planning as inference, gradient dynamics | Do not use FEP vocabulary to smuggle in a controller. |
| Structural binding alternatives | `pdf:mesh-2022`, `arxiv:2208.12880`, `arxiv:2311.04872`, `arxiv:2403.13218`, `arxiv:2406.18808`, `openreview:fnrzd3ls1d`, `github:ibm/in-memory-factorizer` | Role/filler separation, resonator modules, scaffold baselines, alternate algebras | Substrate swaps need branch isolation and matched protocol evidence. |
| Evaluation and external sanity checks | `arxiv:2312.04927`, `arxiv:2507.11393`, `arxiv:2503.23390`, `arxiv:2406.03980`, `pdf:sqhn-2024` | MQAR, continual-learning baselines, negative result discipline | External evals are sanity checks unless promoted into a design-spec headline. |
| LLM-as-voice interface | `arxiv:2502.00592`, `arxiv:2412.06769`, `arxiv:2405.14831`, `arxiv:2505.22101`, `arxiv:2604.08206`, `ssrn:5377250` | Cross-attention memory tokens, latent thought, anti-RAG comparison | Do not let the LLM become persistence, source selection, or identity. |
| Sleep / temporal structure | `pmc:PMC10659301`, `pii:S0896-6273(25)00756-1`, `pubmed:39322671`, `arxiv:2601.02845`, `arxiv:2604.20943` | Trace tagging, SWR amplitude analogs, temporal hierarchy, sleep phases | Biology analogies need operational substrate metrics before implementation. |

## Promotion Rule

If a `link_only` source becomes central to a design, write a fuller source card
from the primary source first, then update its manifest `status` or notes.
