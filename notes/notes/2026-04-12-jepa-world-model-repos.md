---
date: 2026-04-12
project: personal-ai
tags:
  - source
  - subject/personal-ai
  - subject/cognitive-architecture
  - project/personal-ai
---

# JEPA World Model Repos (Balestriero / LeCun group)

Stumbled onto these while exploring more of LeCun's work. Three-tier architecture — thin paper repo on top of a domain library on top of a generic SSL engine. Potentially relevant to the single-substrate horn of the memory/world-model fork from the 2026-04-10 session.

## LeWM — https://github.com/lucas-maes/le-wm

Official code for **LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels** (Maes, Le Lidec, Scieur, LeCun, Balestriero, 2026). Thin research repo — ~5 files, just the model (`jepa.py`), train/eval scripts, and Hydra configs. The first JEPA that trains stably end-to-end from raw pixels using only **two loss terms**: a next-embedding prediction loss and a Gaussian regularizer on the latents. Cuts tunable loss hyperparameters from six to one vs. prior end-to-end alternatives. ~15M params, single GPU, a few hours of training, plans up to 48× faster than foundation-model-based world models. Relevant because the two-term loss is structurally close to the A↔B loop — prediction and representation sharing one latent space is an existence proof that the memory/world-model collapse is viable.

## stable-worldmodel — https://github.com/galilai-group/stable-worldmodel

Domain library LeWM builds on. Provides Gym/DMC/OGBench environments (PushT, TwoRoom, Humanoid, Cheetah, Walker, Reacher, etc.) with visual and physical factors of variation, HDF5/MP4 trajectory datasets, and a `World` object that unifies data collection → training → evaluation. Includes MPC solvers (CEM, iCEM, MPPI, SGD/Adam, PGD, Augmented Lagrangian) and implemented baselines (DINO-WM, PLDM, LeWM, GCBC, GCIVL, GCIQL). CLI (`swm`) for inspecting datasets/envs/checkpoints without code. The closest analog to the workspace-and-question architecture is its planners — but those are degenerate, externally-specified-question cases. No diffuse mode, no consolidation continuum.

## stable-pretraining — https://github.com/galilai-group/stable-pretraining

Generic PyTorch + Lightning framework for self-supervised and multimodal pretraining. The SSL engine that `stable-worldmodel` uses when models need training. **The interesting piece for the personal-AI project is the observability layer** — callbacks like `OnlineProbe` (live linear-probe on frozen features), `OnlineKNN`, and `RankMe` (representation-rank metric that catches collapse early). Data flows as named dictionaries through components, so any intermediate value is inspectable. Worth stealing the pattern: live metrics on landscape topology rather than post-hoc ones. Same structural risk exists in the trajectory-consolidation story — silent landscape flattening is the cognitive analog of SSL representation collapse.

## Why these matter for the project

- LeWM is the cleanest existence proof that prediction and representation can share one latent space, which is the collapsed horn of the memory-vs-world-model fork.
- The three-tier repo layering (thin paper repo → domain library → generic engine) is a template worth stealing regardless of subject.
- The observability tooling from `stable-pretraining` maps directly onto the need to detect landscape degeneration during trajectory-consolidation.
- None of these have an analog of the question-as-shared-currency or the diffuse/focused/reactive continuum — that's still where this project has something theirs doesn't.

## Open thread for next session

Re-open the JEPA question (flagged as unaddressed in the 2026-04-10 wrap-up). JEPA now reframes as: "does this family of architectures validate or falsify the single-substrate horn of the fork?" rather than "how does this relate?"
