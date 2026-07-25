"""experiments/86 — Bet B: is the compositional regime actually DISCRIMINATING?

## Preamble (per CLAUDE.md §Experiment preamble)

- **Active capability:** Bet B continual compounding-transfer (`CONTEXT-B.md`
  §"THE RE-GROUNDED TARGET").
- **Headline metric** per `notes/RETROSPECTIVE-two-bets-2026-06-06.md` §"What
  would actually test the central thesis": the **held-out composition gap**
  `GAP = comp_test(trained pairs) − heldout_pair_acc`, read off the same
  end-of-stream model. The regime discriminates iff GAP is large and CI-disjoint
  from 0 for the *simple* arms — i.e. replay + a soft weight anchor do **not**
  saturate it.
- **Required controls** per `CONTEXT-B.md` §"THE TRACER BULLET": scratch
  denominator; joint-train ceiling; the 2×2 factorial; the **abelian matched
  control**; frozen-feature probe.
- **Last verified result:** Report 139 + EWC addendum — soft weight-anchoring +
  replay clears the §8 interaction gate on the modular toy; fully recombinant.
- **Why now:** STATUS.md blocker (3) "single toy domain" and the retrospective's
  §4 task-selection confound. Every mechanism result in this program is
  uninterpretable until a regime exists where the simple method fails.

## THIS IS NOT A GRADUATION EXPERIMENT — it is regime validation

It deliberately tests no brain-distinctive mechanism. It tests the *precondition*
for such a test being meaningful. When the simple method already saturates a task,
no mechanism can show a **necessary** advantage, so "the brain mechanism added
nothing" is expected by construction. Reports 133-139 all ran in that regime.

## The task (validated — see Report 140)

`betb.tasks.PermutationCompositionFamily`. Operators are elements of `S_5` acting
coordinatewise on 3-tuples over 5 points (125 states = 125 classes). Uniform arity
5: `(op_a, op_b, x0, x1, x2)`, where a primitive is a composition with identity.
8 of the 30 ordered operator pairs are **never trained on, by any arm**.

Standalone validation, n=5 seeds, 20k steps: primitives generalize 0.977,
trained composition pairs 0.961, held-out pairs **0.319** → gap **+0.642**
[0.466, 0.808], and the gap *widens* to +0.669 at 60k steps (held-out flat at
0.320 while trained-pair climbs to 0.989). No seed trends toward closure.

**The abelian matched control (`--group cyclic`) is load-bearing, not optional.**
In `(Z_5)^3` composing two operators *is* pooling them, so the shortcut is
correct: held-out saturates at 0.908 and the gap collapses to +0.060 with a CI
including zero. That is what shows the gap is about non-commutative recombination
rather than task difficulty. It replaces the Report-134 scramble, which STATUS.md
records as INVALID.

**A prior design was killed here.** `CompositionalAffineFamily` (random affine
maps on Z_p) memorized perfectly and generalized at 0.000 after 20k steps — a
regime where nothing learns is as uninformative as one that saturates. Caught by
running the known-grokking modular task through the same harness. See Report 140.

Run: `PYTHONPATH=src python experiments/86_betb_compositional.py --tiny`
Real:  `... --seeds 8 --max-steps 20000` (~13 min/run; use OMP_NUM_THREADS=1 and
shard — 4 torch threads measured a 13x slowdown from contention)
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from energy_memory.betb import (  # noqa: E402
    ContinualNet,
    Provenance,
    apply_tiny,
    base_parser,
    boot_ci,
    build_family,
    evaluate,
    git_sha,
    log_ftsr,
    make_consolidator,
    merge_shards,
    resolve_device,
    retention_matrix_metrics,
    run_arm,
    to_tensors,
    write_result,
)

ARMS = ["scratch", "floor", "replay_only", "consol_only", "replay_plus_consol"]


def joint_train_ceiling(stream, *, embed, hidden, max_steps, lr, seed, device, head_mode):
    """Control (2): train one model on ALL tasks jointly — the ceiling.

    `CONTEXT-B.md` §8 requires the sequential stream's final accuracy to match
    this. Never implemented in exps 80-85, so the sequential numbers had no
    ceiling to be judged against.
    """
    m = ContinualNet(stream.vocab, stream.n_classes, len(stream), stream.n_inputs,
                     embed, hidden, seed, head_mode).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1.0)
    data = [(to_tensors(t.train, device), k) for k, t in enumerate(stream.tasks)]
    for _ in range(max_steps):
        loss = 0.0
        for ((x, y), k) in data:
            loss = loss + F.cross_entropy(m(x, k), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return {
        "final_acc": [evaluate(m, k, t.test, device) for k, t in enumerate(stream.tasks)],
        "heldout_acc": [evaluate(m, k, t.heldout, device) if t.heldout else float("nan")
                        for k, t in enumerate(stream.tasks)],
    }


def frozen_features_probe(stream, *, embed, hidden, max_steps, lr, seed, device, head_mode,
                          replay_frac, crit, eval_every):
    """Control (5): does the learned circuit ALREADY do the last task?

    `CONTEXT-B.md` §8 words this "frozen-model-in-context on T3 — guards the
    'frozen model already does it few-shot' escape." An MLP has no in-context
    mechanism, so the faithful analogue is a **frozen-feature probe**: run the
    stream to the final task, freeze embeddings + MLP, train only a fresh head.

    Naming this precisely matters. `reports/134/report.md:15` lists
    "frozen-in-context" among controls used, but grep over exps 80-85 finds it in
    a docstring only. Declaring the deviation is the difference between a control
    and a claim.
    """
    from energy_memory.betb.continual import train_task

    K = len(stream)
    m = ContinualNet(stream.vocab, stream.n_classes, K, stream.n_inputs,
                     embed, hidden, seed, head_mode).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1.0)
    g = torch.Generator().manual_seed(seed * 104729 + 7)
    buf = []
    for k, task in enumerate(stream.tasks[:-1]):
        train_task(m, opt, k, task, max_steps=max_steps, crit=crit, eval_every=eval_every,
                   buf=buf, replay_frac=replay_frac, gen=g, device=device)
        buf.append((task.train, k))

    for p in list(m.emb.parameters()) + list(m.mlp.parameters()):
        p.requires_grad_(False)
    head = m.head if head_mode == "shared" else m.heads[K - 1]
    for p in head.parameters():
        p.requires_grad_(True)
    opt2 = torch.optim.AdamW(list(head.parameters()), lr=lr, weight_decay=1.0)

    last = stream.tasks[-1]
    x, y = to_tensors(last.train, device)
    for _ in range(max_steps):
        opt2.zero_grad(set_to_none=True)
        F.cross_entropy(m(x, K - 1), y).backward()
        opt2.step()
    return {
        "probe_test_acc": evaluate(m, K - 1, last.test, device),
        "probe_heldout_acc": evaluate(m, K - 1, last.heldout, device) if last.heldout else float("nan"),
    }


def _family(args, group):
    return build_family("permutation", m=args.m, k=args.k, n_ops=args.n_ops,
                        heldout_frac=args.heldout_frac, group=group)


def collect(args, device):
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    seeds = [s for i, s in enumerate(seeds) if i % max(1, args.n_shards) == args.shard]
    out = {}

    for seed in seeds:
        stream = _family(args, args.group).build(
            args.K, args.frac, torch.Generator().manual_seed(seed * 7717 + 3))
        common = dict(embed=args.embed, hidden=args.hidden, max_steps=args.max_steps,
                      crit=args.crit, eval_every=args.eval_every,
                      replay_frac=args.replay_frac, lr=args.lr, seed=seed,
                      device=device, head_mode=args.head_mode)
        ewc = make_consolidator("ewc", lam=args.ewc_lambda)

        rec = {"arms": {}}
        for arm in ARMS:
            r = run_arm(arm, stream, consolidator_factory=ewc, **common)
            rec["arms"][arm] = {
                "steps": r.steps,
                "ret_after": r.ret_after,
                "test_final": r.test_final,
                "heldout_cells_final": r.heldout_cells_final,
            }

        # control (4): the ABELIAN matched control — the shortcut is legal there,
        # so the gap must collapse. Same arms, same everything else.
        rec["abelian"] = {}
        if args.run_abelian:
            ab = _family(args, "cyclic").build(
                args.K, args.frac, torch.Generator().manual_seed(seed * 7717 + 3))
            for arm in ("replay_only",):
                r = run_arm(arm, ab, consolidator_factory=ewc, **common)
                rec["abelian"][arm] = {
                    "test_final": r.test_final,
                    "heldout_cells_final": r.heldout_cells_final,
                }

        if args.run_controls:
            rec["joint_ceiling"] = joint_train_ceiling(
                stream, embed=args.embed, hidden=args.hidden,
                max_steps=max(200, args.max_steps // 8), lr=args.lr, seed=seed,
                device=device, head_mode=args.head_mode)
            rec["frozen_probe"] = frozen_features_probe(
                stream, embed=args.embed, hidden=args.hidden, max_steps=args.max_steps,
                lr=args.lr, seed=seed, device=device, head_mode=args.head_mode,
                replay_frac=args.replay_frac, crit=args.crit, eval_every=args.eval_every)

        rec["comp_tasks"] = [i for i, t in enumerate(stream.tasks) if t.kind == "composition"]
        out[str(seed)] = rec
    return out, seeds


def _gap_samples(per_seed, container, arm):
    """One (test_acc, cell_acc, gap) sample per (seed, held-out CELL).

    Bootstrapping over rows would treat 1000 correlated rows as independent when
    the effective n is 8 cells — per-cell accuracy in a single run ranges 0.000 to
    0.856, so row-level CIs would be ~11x too narrow.
    """
    tests, cells, gaps = [], [], []
    for rec in per_seed.values():
        src = rec.get(container, {}).get(arm)
        if not src or not src.get("heldout_cells_final"):
            continue
        for ti in rec["comp_tasks"]:
            if ti >= len(src["heldout_cells_final"]):
                continue
            tacc = src["test_final"][ti]
            for cacc in src["heldout_cells_final"][ti]:
                if cacc == cacc:
                    tests.append(tacc)
                    cells.append(cacc)
                    gaps.append(tacc - cacc)
    return tests, cells, gaps


def aggregate(per_seed, args):
    agg = {}
    for arm in ("floor", "replay_only", "consol_only", "replay_plus_consol"):
        t, c, g = _gap_samples(per_seed, "arms", arm)
        agg[arm] = {
            "comp_test_trained_pairs": boot_ci(t),
            "heldout_pair_acc": boot_ci(c),
            "HEADLINE_gap": boot_ci(g),
            "n_cell_samples": len(g),
        }
        lf = []
        for rec in per_seed.values():
            lf.extend(log_ftsr(rec["arms"]["scratch"]["steps"], rec["arms"][arm]["steps"]))
        agg[arm]["log_FTSR"] = boot_ci(lf)
        mats = [rec["arms"][arm]["ret_after"] for rec in per_seed.values()
                if rec["arms"][arm]["ret_after"]]
        agg[arm]["ACC"] = boot_ci([retention_matrix_metrics(m).get("ACC", float("nan")) for m in mats])
        agg[arm]["BWT"] = boot_ci([retention_matrix_metrics(m).get("BWT", float("nan")) for m in mats])

    t, c, g = _gap_samples(per_seed, "abelian", "replay_only")
    if g:
        agg["ABELIAN_CONTROL_replay_only"] = {
            "comp_test_trained_pairs": boot_ci(t),
            "heldout_pair_acc": boot_ci(c),
            "gap": boot_ci(g),
            "n_cell_samples": len(g),
        }

    jc = [a for rec in per_seed.values() if "joint_ceiling" in rec
          for a in rec["joint_ceiling"]["final_acc"]]
    if jc:
        agg["joint_ceiling_final_acc"] = boot_ci(jc)
    fp = [rec["frozen_probe"]["probe_heldout_acc"] for rec in per_seed.values()
          if "frozen_probe" in rec]
    if fp:
        agg["frozen_probe_heldout_acc"] = boot_ci(fp)

    gap = agg["replay_only"]["HEADLINE_gap"]
    ab = agg.get("ABELIAN_CONTROL_replay_only", {}).get("gap")
    agg["VERDICT"] = {
        "regime_is_discriminating": bool(gap[1] > 0.10),
        "criterion": "replay_only held-out GAP CI-lo > 0.10 (bootstrap over seed x cell)",
        "abelian_control_collapses": (bool(ab[1] <= 0.10) if ab else None),
        "abelian_criterion": "the matched abelian control's gap CI-lo must NOT exceed 0.10 — "
                             "if it does, the gap is task difficulty, not recombination",
        "note": ("Both clauses must hold. A large gap with a control gap that ALSO stays large "
                 "means the regime is merely hard, not discriminating."),
    }
    return agg


def main():
    ap = base_parser("exp86 — compositional (permutation) regime validation")
    ap.add_argument("--m", type=int, default=5, help="points the group acts on")
    ap.add_argument("--k", type=int, default=3, help="tuple width (state = m^k)")
    ap.add_argument("--n-ops", type=int, default=6)
    ap.add_argument("--heldout-frac", type=float, default=0.27)
    ap.add_argument("--group", default="symmetric", choices=["symmetric", "cyclic"])
    ap.add_argument("--ewc-lambda", type=float, default=0.01)
    ap.add_argument("--head-mode", default="shared", choices=["shared", "per_task"])
    ap.add_argument("--arms", default="")
    ap.add_argument("--no-abelian", dest="run_abelian", action="store_false")
    ap.add_argument("--no-controls", dest="run_controls", action="store_false")
    ap.set_defaults(run_abelian=True, run_controls=True)
    ap.set_defaults(embed=96, hidden=192, max_steps=20000, eval_every=500, K=2, crit=0.95)
    args = ap.parse_args()

    if args.merge is not None:
        merged = merge_shards(args.merge)
        merged["aggregate"] = aggregate(merged["per_seed_raw"], args)
        out = pathlib.Path(args.out or "reports/141_betb_compositional_regime/headline.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(merged, indent=2, sort_keys=True, default=str) + "\n")
        print(f"merged {len(args.merge)} shards -> {out}")
        return

    apply_tiny(args)
    if args.tiny:
        args.m, args.k, args.n_ops, args.K = 4, 2, 4, 2
        args.embed, args.hidden, args.max_steps, args.eval_every = 32, 64, 150, 50
    device = resolve_device(args.device)
    global ARMS
    if args.arms:
        ARMS = [a.strip() for a in args.arms.split(",") if a.strip()]

    per_seed, seeds = collect(args, device)
    executed = ["scratch_denominator", "factorial_2x2"] if len(ARMS) == 5 else []
    if args.run_abelian:
        executed.append("abelian_matched_control")
    if args.run_controls:
        executed += ["joint_train_ceiling", "frozen_features_probe"]

    prov = Provenance(
        experiment="exp86_betb_compositional_permutation",
        git_sha=git_sha(),
        argv=sys.argv,
        seeds=seeds,
        config={k: v for k, v in vars(args).items() if k not in ("merge", "out")},
        declared_controls=list(executed),
        executed_controls=executed,
        scenario="Class-IL (shared head, task inferred from input)"
        if args.head_mode == "shared" else "Task-IL (per-task head, task id at eval)",
        task_family=f"permutation_{args.group}_m{args.m}_k{args.k}",
    )
    payload = {"per_seed_raw": per_seed}
    if len(ARMS) == 5:
        payload["aggregate"] = aggregate(per_seed, args)

    write_result(args.out, payload, prov)
    if "aggregate" in payload:
        a = payload["aggregate"]
        g = a["replay_only"]["HEADLINE_gap"]
        print(f"\nHEADLINE replay_only GAP: {g[0]:+.3f} [{g[1]:+.3f}, {g[2]:+.3f}] "
              f"(n={a['replay_only']['n_cell_samples']} seed x cell)")
        print(f"  trained-pair {a['replay_only']['comp_test_trained_pairs'][0]:.3f} "
              f"vs held-out {a['replay_only']['heldout_pair_acc'][0]:.3f}")
        if "ABELIAN_CONTROL_replay_only" in a:
            ab = a["ABELIAN_CONTROL_replay_only"]["gap"]
            print(f"  ABELIAN control gap: {ab[0]:+.3f} [{ab[1]:+.3f}, {ab[2]:+.3f}] (must collapse)")
        print(f"  VERDICT: {a['VERDICT']}")


if __name__ == "__main__":
    main()
