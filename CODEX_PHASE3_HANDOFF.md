# Codex Full Handoff Prompt — Hypergraph Phase 3

Copy the text under **TASK PROMPT** into a new Codex task while the complete `hypergraph` repository is open.

The repository root should also contain `AGENTS.md`. Treat `AGENTS.md` as the permanent project contract and this prompt as the active execution task.

---

## TASK PROMPT

Take full ownership of the complete `hypergraph` research repository currently open in this workspace and finish Phase 3 as a scientifically valid, fully reproducible research artifact.

This repository was previously modified by an Antigravity agent. Do not trust its code changes, reports, metrics, PASS claims, or exported files without independent verification. Treat all Antigravity material as legacy evidence only.

Read `AGENTS.md` completely before making changes. Then inspect the entire repository, including source code, configurations, data and locked splits, notebooks, tests, results, checkpoints, predictions, reports, manifests, Git state, and any `handoff/antigravity_export` or `docs/legacy/antigravity` files.

## Operating rules

Work autonomously and continue through ordinary errors.

Do not stop after:
- producing a plan;
- inspecting files;
- finding one bug;
- editing code;
- passing only one gate;
- running only one seed;
- generating partial metrics;
- writing an unverified report;
- or creating an incomplete ZIP.

Do not repeatedly ask for permission for normal changes inside this repository. Ask only when there is a genuine external blocker or an action outside the repository that requires explicit approval.

If a command fails, diagnose the real cause, repair it, rerun the affected checks, and continue.

Do not claim completion until every mandatory acceptance criterion below has executable evidence.

## Persistent continuity

Create and continuously maintain:

`CODEX_PHASE3_STATUS.md`

It must contain:
- current branch and commit;
- authoritative project specification discovered;
- files inspected;
- changes made;
- commands executed and return codes;
- completed requirements;
- failed requirements;
- pending requirements;
- current blockers;
- exact next command to run.

Update this file after every major stage and before ending any session. If the task is resumed later, read this file first and continue from the exact pending step instead of restarting or guessing.

## Safety and preservation

Before modifying experiments or results:

1. Inspect `git status`, `git diff`, and relevant history.
2. Create or switch to a dedicated branch named `codex-phase3-final`. If it already exists, inspect it before continuing.
3. Archive all current Phase 3 outputs under:

   `results/phase3_archive/<UTC_TIMESTAMP>/`

4. Preserve source datasets, locked split definitions, notebooks, checkpoints, predictions, reports, logs, and manifests.
5. Never overwrite evidence without archiving it.
6. Use project-relative paths.
7. Do not copy Antigravity-exported Python files over the real source tree without a line-by-line diff and verification.

## Repository takeover audit

Inventory the repository and classify relevant files as:
- authoritative specification;
- current source;
- generated artifact;
- legacy Antigravity evidence;
- duplicate;
- obsolete cache;
- missing dependency;
- unsafe to delete.

Move chat-specific or agent-specific files out of the project root into a legacy folder, but preserve them until takeover is complete.

Compare every Antigravity-exported source file with the corresponding current source. Merge only changes that are mathematically correct and supported by tests.

Determine from the real specification and code:
1. the designated primary Phase 3 model;
2. the numerically best-performing model;
3. all required baselines;
4. all required ablations;
5. all official seeds;
6. all official splits;
7. exact definitions of Gates A, B, and C.

Do not infer the primary model merely from the best metric.

## Scientific integrity

Never:
- weaken a required Gate B threshold from `0.99` to `0.98` merely to pass;
- remove a failing structural or permutation audit;
- silently redefine an acceptance rule;
- attribute `LearnedCostSinkhorn` results to `MatchingNetwork`;
- attribute `MatchingNetwork` results to `LearnedCostSinkhorn`;
- manually edit reported metrics;
- tune using IID or OOD test sets;
- select checkpoints using test performance;
- claim five-seed evidence from one seed;
- present a training overfit test as final generalization performance;
- omit OOD-size results;
- issue a global PASS while a mandatory requirement is incomplete.

If an inherited requirement is mathematically incorrect, preserve its original result, explain the problem with evidence, and add a separately named corrected test. Do not silently replace it.

## Reproducibility repair

Make the repository runnable from a clean checkout.

Verify or create:
- complete source modules;
- `config_manager.py`;
- `oracles.py`;
- data loaders;
- model configurations;
- locked split definitions or deterministic split-generation scripts;
- dependency declarations and lock files;
- exact random seeds;
- tests;
- reproducibility commands;
- environment metadata.

Remove machine-specific absolute paths.

If raw data cannot be included, provide deterministic acquisition and preparation instructions/scripts.

## Dataset and split integrity

Audit:
- train;
- validation, if defined;
- IID test;
- OOD-template test;
- OOD-size test.

Verify:
- sample IDs;
- labels;
- expected sizes;
- class counts;
- incidence-matrix shapes;
- metadata consistency;
- masks;
- padding;
- no malformed or empty samples;
- no prohibited overlap;
- no duplicated underlying graph pairs across prohibited partitions;
- deterministic loading;
- deterministic split hashes.

Generate:

`results/phase3/split_overlap_audit.json`

Include explicit PASS/FAIL fields, counts, hashes, problematic sample IDs, and reasons.

## Gate A

Run the intended oracle verification.

Re-derive the mathematically correct Sinkhorn normalization from the implementation and specification.

Do not require a hard matched entry to equal `1.0` when a normalized coupling with `m` equally weighted valid rows should assign approximately `1/m` to each hard match.

Record:
- exact inputs;
- expected total mass;
- expected row and column marginals;
- observed matching cost;
- convergence iterations;
- residuals;
- tolerance;
- PASS/FAIL.

Save per-sample oracle predictions and evidence.

## Gate B

Restore and enforce the original non-weakened Gate B rule.

Use the official locked seeds. Unless the repository specification explicitly defines another set, use:

- 42
- 100
- 2026
- 777
- 999

For every failing seed:
1. reproduce it deterministically;
2. identify the exact failed sample;
3. inspect its label and structure;
4. inspect gradients;
5. inspect logits and probabilities;
6. inspect masks and padding;
7. verify temperature scheduling;
8. verify Sinkhorn convergence;
9. verify loss and optimizer behavior;
10. fix the underlying cause;
11. rerun every locked seed from scratch.

Generate a structured Gate B report with:
- model;
- seed;
- sample count;
- loss;
- accuracy;
- exact threshold;
- failed sample IDs;
- PASS/FAIL.

## Gate C and final experiments

Run Gate C according to the actual research specification.

Unless the specification says otherwise:
- run all five locked seeds;
- keep train, validation, IID, OOD-template, and OOD-size results separate;
- use validation only for checkpoint selection;
- save selection epoch and criterion;
- save final temperature;
- save training and evaluation Sinkhorn iteration counts;
- save runtime, device, dtype, and checkpoint hash.

Run the designated primary model, all required baselines, and all required ablations on identical locked splits and seeds.

Save:
- `results/phase3/per_seed_metrics.csv`
- `results/phase3/aggregate_metrics.csv`
- `results/phase3/checkpoints/<model>_seed<seed>.pt`
- `results/phase3/predictions/<model>_seed<seed>_<split>.csv`

Report at least:
- loss;
- accuracy;
- balanced accuracy;
- macro F1;
- ROC-AUC;
- PR-AUC where defined;
- Brier score where required;
- sample count;
- class counts.

Compute aggregate values programmatically from verified per-seed files.

## Complete float64 structural audit

Treat the previous numerical audit as unverified.

Use:
- float64;
- compact valid submatrices with padding removed;
- explicit masks and mathematically correct marginals;
- deterministic permutations;
- at least 100 Sinkhorn iterations;
- preferably 500 iterations for the final strict audit.

For every required seed and every applicable locked IID, OOD-template, and OOD-size sample, independently verify:

1. all cost-matrix values are finite;
2. all coupling values are finite;
3. coupling is non-negative within tolerance;
4. `C_yx ≈ C_xy.T`;
5. `P_yx ≈ P_xy.T`;
6. matching-cost root-swap invariance;
7. equality-score root-swap invariance;
8. final-logit root-swap invariance;
9. final-probability root-swap invariance;
10. correct total coupling mass;
11. correct row marginals;
12. correct column marginals;
13. direct assignment-transpose behavior at `P` level;
14. required node-row permutation invariance;
15. hyperedge/column permutation invariance if required by the mathematical formulation.

Do not remove a failed column-permutation test. Determine whether hyperedge order is semantically fixed or should be invariant. Support the conclusion using the research formulation, data generation, and architecture. Fix the model if it violates a mandatory invariance, otherwise document why the test is not a valid requirement.

Prefer `1e-8` float64 structural tolerances, with no acceptance ceiling looser than `1e-6` unless rigorously justified.

Generate:

`results/phase3/numerical_audit.json`

For each quantity record:
- number audited;
- failures;
- mean error;
- maximum error;
- p95;
- p99;
- tolerance;
- worst seed;
- worst split;
- worst sample ID;
- iteration count;
- convergence and marginal residuals;
- NaN and Inf counts.

A PASS requires zero unexplained mandatory failures.

## Metric and report consistency

Create an automated consistency checker.

It must compare every headline metric in the final report with `aggregate_metrics.csv` and fail on:
- wrong model;
- wrong split;
- wrong metric;
- wrong mean;
- wrong standard deviation;
- wrong seed count.

Clearly distinguish:
- designated primary model;
- numerically best-performing model;
- baselines;
- ablations.

Report exact IID, OOD-template, and OOD-size results with uncertainty.

## Notebooks

Update and cleanly execute from fresh kernels:
- `notebooks/Phase3_Research.ipynb`
- `notebooks/Phase3_Explained_FA.ipynb`

No hidden interactive state is allowed.

Synchronize:
- project formulation;
- primary model identity;
- locked split definitions;
- correct Sinkhorn mass interpretation;
- Gates A/B/C;
- per-seed metrics;
- aggregate metrics;
- IID results;
- OOD-template results;
- OOD-size results;
- ablations;
- complete numerical audit;
- limitations;
- exact reproduction commands.

The Persian notebook must explain the work naturally and technically accurately.

## Testing and logs

Run and log:
- dependency and import checks;
- repository tests;
- clean Phase 3 execution;
- split-integrity audit;
- Gate A;
- Gate B;
- Gate C;
- final model/baseline/ablation experiments;
- complete numerical audit;
- report consistency;
- clean notebook execution;
- checkpoint save/load round trip;
- manifest verification;
- final ZIP verification.

Store command, timestamp, stdout, stderr, runtime, and return code under:

`results/phase3/logs/`

A crashed, interrupted, timed-out, skipped, or unlogged command is not PASS.

## Reports and manifests

Create:

- `results/phase3/final_validation_report.md`
- `results/phase3/run_manifest.json`
- `results/phase3/final_artifact_manifest.json`

The final report must include:
- branch and commit;
- environment;
- archive path;
- authoritative specification;
- primary model;
- best-performing model;
- split audit;
- exact Gate definitions;
- Gate A evidence;
- Gate B evidence per seed;
- Gate C evidence per seed;
- model, baseline, and ablation comparisons;
- IID results;
- OOD-template results;
- OOD-size results;
- complete numerical audit;
- report consistency status;
- notebook execution status;
- known limitations;
- reproducibility commands;
- exact artifact list;
- explicit PASS/FAIL/PARTIAL/NOT RUN/BLOCKED status for every requirement.

Generate SHA-256 hashes for source, configs, dependency locks, split definitions, checkpoints, predictions, metrics, audits, notebooks, reports, and logs.

Recompute and verify all hashes after final files are frozen.

## Final reproducible export

Create a complete reproducible ZIP containing all required:
- source code;
- configs;
- dependency declarations and locks;
- split definitions or deterministic acquisition/generation scripts;
- tests;
- checkpoints;
- predictions;
- metrics;
- audits;
- reports;
- logs;
- executed notebooks;
- manifests.

Programmatically:
1. list ZIP contents;
2. verify required entries;
3. extract into a clean temporary directory;
4. verify hashes;
5. run documented import/smoke checks from the extracted copy.

Do not create another small ChatGPT review bundle that omits the project.

## Completion contract

Continue until every mandatory requirement is verified PASS.

The task is not complete if any mandatory item is:
- FAIL;
- PARTIAL;
- NOT RUN;
- BLOCKED;
- missing evidence;
- inconsistent with source metrics;
- or dependent on an untracked local file.

If execution time or session limits interrupt the task, update `CODEX_PHASE3_STATUS.md` with the exact state and next command. On the next run, resume from that file.

## Final response format

When the work is genuinely complete, report exact:

1. overall status;
2. branch and commit;
3. designated primary model;
4. numerically best-performing model;
5. IID metrics;
6. OOD-template metrics;
7. OOD-size metrics;
8. Gate A result;
9. Gate B result for each seed;
10. Gate C result for each seed;
11. maximum error for every numerical-audit quantity;
12. number of audited samples;
13. split-audit status;
14. report-consistency status;
15. notebook-execution status;
16. archive path;
17. final report path;
18. checkpoint directory;
19. prediction directory;
20. manifest paths;
21. final ZIP path;
22. Git diff summary;
23. remaining scientifically honest limitations.

Do not finish by asking whether to continue or whether notebooks should be updated. Those are already part of this task.
