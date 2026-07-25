"""experiments/86 — Bet B: is the compositional regime actually DISCRIMINATING?

## Preamble (per CLAUDE.md §Experiment preamble)

- **Active capability:** Bet B continual compounding-transfer (`CONTEXT-B.md`
  §"THE RE-GROUNDED TARGET").
- **Headline metric** per `notes/RETROSPECTIVE-two-bets-2026-06-06.md` §"What
  would actually test the central thesis": the **held-out composition gap** —
  `train_pair_acc − heldout_pair_acc` for the `replay_only` and
  `replay_plus_consol(ewc)` arms. The regime is discriminating iff that gap is
  large and CI-disjoint from 0, i.e. the simple method **does not saturate**.
- **Required controls** per `CONTEXT-B.md` §"THE TRACER BULLET" (Controls, all
  mandatory): scratch denominator; joint-train ceiling; the 2×2 factorial;
  scrambled control; frozen-model probe.
- **Last verified result:** Report 139 + EWC addendum — soft weight-anchoring +
  replay clears the §8 interaction gate on the modular toy; fully recombinant.
- **Why now:** STATUS.md blocker (3) "single toy domain" and the retrospective's
  §4 task-selection confound. Every mechanism result in this program is
  uninterpretable until a regime exists where the simple method fails.

## THIS IS NOT A GRADUATION EXPERIMENT — it is a regime-validation drill-down

It deliberately does **not** test a brain-distinctive mechanism. It tests the
*precondition* for such a test ever being meaningful. The retrospective §4 is
blunt about why: when the simple method already saturates the task, no mechanism
can show a *necessary* advantage, so "the brain mechanism added nothing" is
expected by construction. Reports 133-139 all ran in that regime.

So the question here is only: **does replay + a soft weight anchor fail to
generalize to held-out compositions?**

- If **yes** (large gap): the regime is discriminating. It becomes the venue for
  testing whether a genuinely *restructuring* consolidation
  (`betb.consolidators.SubspaceRestructure`) manufactures what protection cannot
  — the question `CONTEXT-B.md` §8 records as still unbuilt.
- If **no** (arms saturate held-out pairs): the regime is *not* discriminating
  and must be made harder before any mechanism claim is worth running. That is a
  real finding, not a failure — and cheaper to learn now than after another
  mechanism arc.

Both outcomes are informative, which is the property Reports 133-139 lacked.

## The task

`betb.tasks.CompositionalAffineFamily`. Affine operators on Z_p,
`o_i(x) = m_i x + c_i`. Uniform arity 3: a primitive is `(op_i, IDENTITY, x)`, a
composition is `(op_i, op_j, x) -> o_j(o_i(x))`. A fraction of `(i,j)` pairs is
**never trained on, in any arm**. Composition is genuinely recombinant —
`o_j(o_i(x))` has slope `m_j*m_i` and intercept `m_j*c_i + c_j`, neither of which
is either primitive's parameters — so pair memorization does not yield the rule.

Single shared head, task inferred from input tokens (`head_mode='shared'`).
Reports 134-139 used a per-task head indexed by task id at eval — Task-IL, the
easiest continual scenario, disclosed once and then dropped from the scope lines.

Run: `PYTHONPATH=src python experiments/86_betb_compositional.py --tiny`
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


# --------------------------------------------------------------------------
# controls that CONTEXT-B declares mandatory but no experiment implemented
# --------------------------------------------------------------------------
def joint_train_ceiling(stream, *, embed, hidden, max_steps, lr, seed, device, head_mode):
    """Control (2): train one model on ALL tasks jointly — the ceiling.

    `CONTEXT-B.md` §8 requires the sequential stream's final accuracy to match
    this. Never implemented in exps 80-85; declared nowhere, so the sequential
    numbers had no ceiling to be judged against.
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

    `CONTEXT-B.md` §8 words this as "frozen-model-in-context on T3 — guards the
    'frozen model already does it few-shot' escape." An MLP has no in-context
    mechanism, so the faithful analogue is a **frozen-feature probe**: run the
    stream up to the final task, freeze embeddings + MLP, and train only a fresh
    head on that task.

    Naming this precisely matters. `reports/134/report.md:15` lists
    "frozen-in-context" among the controls used, but `grep` over exps 80-85 finds
    it in a docstring only — it was never implemented. Declaring the deviation is
    the difference between a control and a claim.
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

    for p in m.emb.parameters():
        p.requires_grad_(False)
    for p in m.mlp.parameters():
        p.requires_grad_(False)
    head = m.head if head_mode == "shared" else m.heads[K - 1]
    for p in head.parameters():
        p.requires_grad_(True)
    opt2 = torch.optim.AdamW([p for p in head.parameters()], lr=lr, weight_decay=1.0)

    last = stream.tasks[-1]
    x, y = to_tensors(last.train, device)
    for _ in range(max_steps):
        opt2.zero_grad(set_to_none=True)
        F.cross_entropy(m(x, K - 1), y).backward()
        opt2.step()
    return {
        "probe_train_acc": evaluate(m, K - 1, last.test, device),
        "probe_heldout_acc": evaluate(m, K - 1, last.heldout, device) if last.heldout else float("nan"),
    }


# --------------------------------------------------------------------------
def collect(args, device):
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    seeds = [s for i, s in enumerate(seeds) if i % max(1, args.n_shards) == args.shard]
    out = {}

    for seed in seeds:
        g = torch.Generator().manual_seed(seed * 7717 + 3)
        fam = build_family("compositional", p=args.p, n_ops=args.n_ops,
                           heldout_frac=args.heldout_frac)
        stream = fam.build(args.K, args.frac, g)

        gs = torch.Generator().manual_seed(seed * 7717 + 3)
        fam_s = build_family("compositional", p=args.p, n_ops=args.n_ops,
                             heldout_frac=args.heldout_frac, scramble=True)
        stream_scram = fam_s.build(args.K, args.frac, gs)

        common = dict(embed=args.embed, hidden=args.hidden, max_steps=args.max_steps,
                      crit=args.crit, eval_every=args.eval_every,
                      replay_frac=args.replay_frac, lr=args.lr, seed=seed,
                      device=device, head_mode=args.head_mode)
        ewc = make_consolidator("ewc", lam=args.ewc_lambda)

        rec = {"arms": {}, "scramble": {}}
        for arm in ARMS:
            r = run_arm(arm, stream, consolidator_factory=ewc, **common)
            rec["arms"][arm] = {
                "steps": r.steps,
                "ret_after": r.ret_after,
                "heldout_after": r.heldout_after,
            }
        # control (4): scrambled — composition structure destroyed, transfer must die
        for arm in ("scratch", "replay_only"):
            r = run_arm(arm, stream_scram, consolidator_factory=ewc, **common)
            rec["scramble"][arm] = {"steps": r.steps}

        # control (2): joint ceiling
        rec["joint_ceiling"] = joint_train_ceiling(
            stream, embed=args.embed, hidden=args.hidden,
            max_steps=max(200, args.max_steps // 4), lr=args.lr, seed=seed,
            device=device, head_mode=args.head_mode)
        # control (5): frozen-feature probe
        rec["frozen_probe"] = frozen_features_probe(
            stream, embed=args.embed, hidden=args.hidden, max_steps=args.max_steps,
            lr=args.lr, seed=seed, device=device, head_mode=args.head_mode,
            replay_frac=args.replay_frac, crit=args.crit, eval_every=args.eval_every)

        rec["comp_tasks"] = [i for i, t in enumerate(stream.tasks) if t.kind == "composition"]
        out[str(seed)] = rec
    return out, seeds


def aggregate(per_seed, args):
    """Headline: the held-out composition gap for the simple arms."""
    agg = {}
    comp_idx = None
    for rec in per_seed.values():
        comp_idx = rec["comp_tasks"]
        break

    for arm in ("replay_only", "replay_plus_consol", "floor", "consol_only"):
        train_acc, held_acc, gaps = [], [], []
        for rec in per_seed.values():
            a = rec["arms"][arm]
            if not a["ret_after"]:
                continue
            final_ret, final_held = a["ret_after"][-1], a["heldout_after"][-1]
            for i in comp_idx:
                if i < len(final_ret):
                    tr, hd = final_ret[i], final_held[i]
                    if hd == hd:
                        train_acc.append(tr)
                        held_acc.append(hd)
                        gaps.append(tr - hd)
        agg[arm] = {
            "train_pair_acc": boot_ci(train_acc),
            "heldout_pair_acc": boot_ci(held_acc),
            "HEADLINE_heldout_gap": boot_ci(gaps),
        }

    # FTSR (log scale, Report 138) + ACC/BWT
    for arm in ("floor", "replay_only", "consol_only", "replay_plus_consol"):
        lf = []
        for rec in per_seed.values():
            lf.extend(log_ftsr(rec["arms"]["scratch"]["steps"], rec["arms"][arm]["steps"]))
        agg[arm]["log_FTSR"] = boot_ci(lf)
        mats = [rec["arms"][arm]["ret_after"] for rec in per_seed.values()
                if rec["arms"][arm]["ret_after"]]
        accs = [retention_matrix_metrics(m).get("ACC", float("nan")) for m in mats]
        bwts = [retention_matrix_metrics(m).get("BWT", float("nan")) for m in mats]
        agg[arm]["ACC"] = boot_ci(accs)
        agg[arm]["BWT"] = boot_ci(bwts)

    scram = []
    for rec in per_seed.values():
        scram.extend(log_ftsr(rec["scramble"]["scratch"]["steps"],
                              rec["scramble"]["replay_only"]["steps"]))
    agg["scramble_replay_only"] = {"log_FTSR": boot_ci(scram)}
    agg["joint_ceiling_final_acc"] = boot_ci(
        [a for rec in per_seed.values() for a in rec["joint_ceiling"]["final_acc"]])
    agg["frozen_probe_heldout_acc"] = boot_ci(
        [rec["frozen_probe"]["probe_heldout_acc"] for rec in per_seed.values()])

    gap = agg["replay_only"]["HEADLINE_heldout_gap"]
    agg["VERDICT"] = {
        "regime_is_discriminating": bool(gap[1] > 0.10),
        "criterion": "replay_only held-out gap CI-lo > 0.10",
        "note": ("If False, the simple method saturates held-out composition and this "
                 "regime does NOT discriminate — make it harder before running any "
                 "mechanism claim against it."),
    }
    return agg


def main():
    ap = base_parser("exp86 — compositional regime validation")
    ap.add_argument("--n-ops", type=int, default=6)
    ap.add_argument("--heldout-frac", type=float, default=0.3)
    ap.add_argument("--ewc-lambda", type=float, default=0.01)
    ap.add_argument("--head-mode", default="shared", choices=["shared", "per_task"])
    ap.add_argument("--arms", default="", help="comma-separated subset (smoke only)")
    args = ap.parse_args()

    if args.merge is not None:
        merged = merge_shards(args.merge)
        merged["aggregate"] = aggregate(merged["per_seed_raw"], args)
        out = pathlib.Path(args.out or "reports/146_betb_compositional/headline.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(merged, indent=2, sort_keys=True, default=str) + "\n")
        print(f"merged {len(args.merge)} shards -> {out}")
        return

    apply_tiny(args)
    if args.tiny:
        args.n_ops = 3
    device = resolve_device(args.device)
    global ARMS
    if args.arms:
        ARMS = [a.strip() for a in args.arms.split(",") if a.strip()]

    per_seed, seeds = collect(args, device)
    prov = Provenance(
        experiment="exp86_betb_compositional",
        git_sha=git_sha(),
        argv=sys.argv,
        seeds=seeds,
        config={k: v for k, v in vars(args).items() if k not in ("merge", "out")},
        declared_controls=["scratch_denominator", "factorial_2x2", "joint_train_ceiling",
                           "scrambled_control", "frozen_features_probe"],
        executed_controls=(["scratch_denominator", "factorial_2x2"]
                           + (["joint_train_ceiling", "scrambled_control",
                               "frozen_features_probe"] if len(ARMS) == 5 else [])),
        scenario="Class-IL (shared head, task inferred from input)"
        if args.head_mode == "shared" else "Task-IL (per-task head, task id at eval)",
        task_family="compositional_affine",
    )
    payload = {"per_seed_raw": per_seed}
    if len(ARMS) == 5:
        payload["aggregate"] = aggregate(per_seed, args)
    # a smoke run declares only what it ran
    if len(ARMS) != 5:
        prov.declared_controls = ["scratch_denominator", "factorial_2x2"]

    write_result(args.out, payload, prov)
    if "aggregate" in payload:
        v = payload["aggregate"]["VERDICT"]
        g = payload["aggregate"]["replay_only"]["HEADLINE_heldout_gap"]
        print(f"\nHEADLINE replay_only held-out gap: {g[0]:.3f} [{g[1]:.3f}, {g[2]:.3f}]")
        print(f"regime_is_discriminating = {v['regime_is_discriminating']}")


if __name__ == "__main__":
    main()
