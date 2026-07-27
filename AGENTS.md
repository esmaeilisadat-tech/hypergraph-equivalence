# Phase 3 Research Project — ZCode Instructions

## Mission

Take over this research project from the previous Antigravity agent and complete Phase 3 as a scientifically valid, fully reproducible experiment.

Treat all existing code, reports, metrics, checkpoints, and PASS claims as unverified until they are independently checked against the actual files and rerun results.

## Working behavior

Work autonomously.

Do not pause merely to report progress or ask whether you should continue.

Do not stop after:

- writing a plan;
- modifying code;
- finding one bug;
- passing only one gate;
- running only one seed;
- producing partial metrics;
- producing an incomplete ZIP;
- or writing a report without verifying it.

Continue inspecting, editing, running, debugging, rerunning, auditing, and validating until all mandatory criteria below pass.

Only stop early for a genuine external hard blocker that cannot be resolved from the repository. Before reporting a blocker, exhaust reasonable local solutions and record the exact command, error, attempted fixes, and missing dependency.

## Safety and preservation

Before modifying existing Phase 3 outputs:

1. Inspect Git status and Git diff.
2. Create a timestamped archive under:

   `results/phase3_archive/<UTC_TIMESTAMP>/`

3. Preserve all previous results, reports, checkpoints, predictions, logs, and manifests.
4. Never overwrite previous evidence without archiving it.
5. Do not delete source datasets.
6. Do not use machine-specific absolute paths.
7. Prefer project-relative paths.

## Scientific integrity rules

Never weaken, remove, reinterpret, or skip a required test merely to obtain PASS.

In particular:

- Do not change a required Gate B threshold from 0.99 to 0.98 merely because one sample fails.
- Do not remove a row, column, root-swap, assignment-transpose, or permutation audit merely because the implementation fails it.
- Do not attribute LearnedCostSinkhorn results to MatchingNetwork.
- Do not attribute MatchingNetwork results to LearnedCostSinkhorn.
- Do not manually edit metrics.
- Do not tune models on test sets.
- Do not select checkpoints using test performance.
- Do not silently omit OOD-size results.
- Do not claim a five-seed result if only Seed 42 was evaluated.
- Do not describe a Train-set gate as test-set performance.
- Do not report a global PASS while a mandatory requirement is partial or failed.

If a requirement is mathematically invalid, preserve its original result and explain the issue explicitly. Propose a corrected requirement separately; never silently replace it.

## Repository inspection

Before implementing new changes, inspect:

- all source files;
- all configurations;
- all notebooks;
- data-loading code;
- locked split files;
- oracle implementations;
- checkpoints;
- prediction files;
- metrics;
- audit files;
- reports;
- manifests;
- environment files;
- Git history and diff where useful.

Identify unambiguously:

1. the designated primary Phase 3 model;
2. the numerically best-performing model;
3. all required baselines;
4. all required ablations;
5. all required locked seeds;
6. all required evaluation splits;
7. any mismatch between reports and CSV files.

Do not infer the primary model merely from which model has the best result.

## Required seeds

Unless the project’s locked specification explicitly defines a different set, run:

- 42
- 100
- 2026
- 777
- 999

Every required final claim must clearly state how many seeds were completed.

## Dataset integrity

Audit all applicable splits:

- train;
- validation;
- IID test;
- OOD-template test;
- OOD-size test.

Verify:

- sample IDs;
- split sizes;
- class counts;
- valid labels;
- tensor dimensions;
- masks;
- padding;
- no prohibited sample overlap;
- no duplicated underlying graph pair across prohibited splits;
- no empty invalid sample;
- deterministic loading;
- deterministic split hashes.

Generate:

`results/phase3/split_overlap_audit.json`

The file must contain explicit PASS/FAIL fields, counts, hashes, problematic IDs, and reasons.

## Gate A

Run and verify the intended oracle test.

Report:

- exact inputs;
- coupling normalization;
- expected mass;
- observed cost;
- convergence residuals;
- tolerance;
- PASS/FAIL.

Do not require every hard Sinkhorn assignment entry to have probability 1.0 when total coupling mass is normalized. For `m` equally weighted valid rows, a hard matched entry normally has mass approximately `1/m`.

## Gate B

Restore the originally intended Gate B acceptance rule.

Run every required seed.

For any failing seed:

1. reproduce the failure;
2. identify the failed sample;
3. inspect the sample label and structure;
4. inspect gradients;
5. inspect logits and probabilities;
6. verify masks and padding;
7. verify temperature scheduling;
8. verify Sinkhorn convergence;
9. verify loss and optimizer behavior;
10. fix the underlying problem;
11. rerun all locked seeds from scratch.

Generate a structured Gate B report with:

- seed;
- sample count;
- accuracy;
- loss;
- failed sample IDs;
- exact threshold;
- PASS/FAIL.

## Gate C

Run Gate C according to the real project specification.

Unless the specification explicitly states otherwise:

- run all five locked seeds;
- keep Train, Validation, IID, OOD-template, and OOD-size separate;
- use Validation only for checkpoint selection;
- save checkpoint-selection epoch;
- save final temperature;
- save Sinkhorn iteration count;
- save one checkpoint per model and seed;
- save per-sample predictions.

Do not present Seed 42 Train performance as a five-seed final evaluation.

## Complete float64 numerical audit

Use:

- float64;
- compact valid submatrices with padding removed;
- at least 100 Sinkhorn iterations;
- preferably 500 iterations for final verification;
- explicit masks and marginals;
- deterministic permutations.

For every required seed and every applicable locked IID, OOD-template, and OOD-size sample, independently verify:

1. `C_yx ≈ C_xy.T`
2. `P_yx ≈ P_xy.T`
3. matching-cost invariance
4. equality-score invariance
5. final-logit invariance
6. final-probability invariance
7. no NaN
8. no Inf
9. non-negative coupling
10. correct total coupling mass
11. row-marginal residual
12. column-marginal residual
13. assignment-transpose behavior
14. valid node-row permutation invariance

Test hyperedge or column permutations when required by the mathematical specification. If they fail, determine whether:

- the model should mathematically be invariant;
- corresponding inputs or masks were permuted incorrectly;
- or the architecture genuinely violates the requirement.

Do not remove the test merely because it fails.

Recommended tolerances:

- structural float64 checks: `1e-8`;
- maximum final acceptance ceiling: `1e-6`;
- marginal residuals: `1e-8` where numerically feasible.

Generate:

`results/phase3/numerical_audit.json`

Record:

- audited sample count;
- failed sample count;
- error type;
- mean;
- maximum;
- p95;
- p99;
- tolerance;
- worst seed;
- worst split;
- worst sample ID;
- iteration count;
- convergence residuals.

A numerical PASS requires zero unexplained mandatory failures.

## Models and metrics

Run the designated primary model, every required baseline, and every required ablation on identical locked splits and seeds.

Report at least:

- loss;
- accuracy;
- balanced accuracy;
- macro F1;
- ROC-AUC;
- sample count;
- class counts.

Generate:

- `results/phase3/per_seed_metrics.csv`
- `results/phase3/aggregate_metrics.csv`
- `results/phase3/checkpoints/<model>_seed<seed>.pt`
- `results/phase3/predictions/<model>_seed<seed>_<split>.csv`

Compute aggregate metrics directly from verified per-seed files.

## Report consistency

Programmatically compare every headline metric in the report against `aggregate_metrics.csv`.

The consistency check must fail if any model name, split, mean, standard deviation, seed count, or metric differs.

Clearly distinguish:

- designated primary model;
- numerically best-performing model;
- baselines;
- ablations.

Include exact IID, OOD-template, and OOD-size results.

## Notebooks

Update and cleanly execute:

- `notebooks/Phase3_Research.ipynb`
- `notebooks/Phase3_Explained_FA.ipynb`

The notebooks must include:

- methodology;
- split definitions;
- model identity;
- correct Sinkhorn mass interpretation;
- Gate A;
- Gate B;
- Gate C;
- per-seed metrics;
- aggregate metrics;
- IID;
- OOD-template;
- OOD-size;
- ablations;
- numerical audit;
- limitations;
- reproducibility commands.

Execute every cell from a clean kernel.

No cell may rely on hidden interactive state.

The Persian notebook must explain the results naturally and accurately.

## Required artifacts

The final project must include all necessary code and evidence, including:

- source code;
- configurations;
- deterministic split definitions or generators;
- environment/dependency file;
- checkpoints;
- predictions;
- metrics;
- split audit;
- numerical audit;
- execution logs;
- notebooks;
- final report;
- manifests.

Generate:

- `results/phase3/final_validation_report.md`
- `results/phase3/run_manifest.json`
- `results/phase3/final_artifact_manifest.json`

Create SHA-256 hashes for all source, configuration, split, checkpoint, prediction, metric, audit, notebook, report, and log files used in the verified run.

Recompute every hash after finalization and confirm every manifest path exists.

## Final export

Create a complete reproducible ZIP containing everything required for independent verification.

The ZIP must not be a selected subset that omits:

- datasets or deterministic split generation;
- configurations;
- code dependencies;
- checkpoints;
- predictions;
- audit evidence;
- notebooks;
- logs;
- or manifests.

Verify ZIP contents programmatically before completion.

## Required final checks

Before declaring completion, run and save logs for:

- import checks;
- repository tests;
- clean Phase 3 execution;
- split-integrity audit;
- complete numerical audit;
- report-to-CSV consistency check;
- notebook clean-execution check;
- manifest hash verification;
- final ZIP content verification.

Store stdout, stderr, commands, timestamps, and return codes under:

`results/phase3/logs/`

A crashed, interrupted, timed-out, or unexecuted command is not PASS.

## Completion contract

The task is complete only when every mandatory requirement is verified and no mandatory item remains:

- FAIL;
- PARTIAL;
- NOT RUN;
- or BLOCKED.

Progress messages, code edits, generated plans, and confident summaries do not count as completion.

Continue automatically until the completion contract passes.

## Final response

At completion, report exact:

- overall status;
- primary model;
- best-performing model;
- IID metrics;
- OOD-template metrics;
- OOD-size metrics;
- Gate A result;
- Gate B per-seed result;
- Gate C per-seed result;
- numerical-audit maximum errors;
- audited sample count;
- archive path;
- report path;
- notebook paths;
- ZIP path;
- manifest paths;
- Git diff summary;
- remaining limitations.

Do not ask whether notebooks or reports should be updated. They are mandatory parts of the task.
