# ZCode Full Project Handoff — Hypergraph Phase 3

## How to use this file

1. Put this file in the root of the real `hypergraph` project folder.
2. Keep `AGENTS.md` in the project root as the permanent project rules.
3. Open the entire `hypergraph` folder as the ZCode workspace.
4. Start a new ZCode task.
5. Paste the prompt under **START PROMPT** into ZCode, including the `/goal` command.

---

# Recommended project folder structure

The workspace should contain the real project, not only the ChatGPT/Antigravity export:

```text
hypergraph/
├── .git/
├── .gitignore
├── AGENTS.md
├── ZCODE_HANDOFF_PROMPT.md
├── README.md
├── pyproject.toml
├── uv.lock                       # if the project uses uv
├── requirements.txt              # if used instead of/in addition to pyproject
│
├── src/
│   ├── __init__.py
│   ├── run_phase3_final.py
│   ├── differentiable_matching.py
│   ├── config_manager.py
│   ├── audit_splits.py
│   ├── audit_model_numerics.py
│   ├── oracles.py
│   └── all other required source modules
│
├── data/
│   ├── raw/                      # if applicable
│   ├── processed/                # if applicable
│   └── splits/
│       ├── train.json
│       ├── validation.json
│       ├── iid_test.json
│       ├── ood_template_test.json
│       └── ood_size_test.json
│
├── notebooks/
│   ├── Phase3_Research.ipynb
│   └── Phase3_Explained_FA.ipynb
│
├── tests/
│   └── project tests
│
├── docs/
│   ├── research_specification/
│   └── legacy/
│       └── antigravity/
│
├── handoff/
│   └── antigravity_export/
│       ├── audit_model_numerics.py
│       ├── audit_splits.py
│       ├── differentiable_matching.py
│       ├── run_phase3_final.py
│       ├── README_ChatGPT.md
│       └── original_export.zip
│
└── results/
    ├── phase3/
    └── phase3_archive/
```

## Files that should stay in the project root

Keep these in the root when available:

- `.git/`
- `.gitignore`
- `AGENTS.md`
- `README.md`
- `pyproject.toml`
- `uv.lock` or another dependency lock file
- project configuration files
- `src/`
- `data/`
- `notebooks/`
- `tests/`
- `docs/`
- `results/`

## Antigravity/ChatGPT files

Do not immediately delete uncertain files.

Move these into:

```text
docs/legacy/antigravity/
```

or:

```text
handoff/antigravity_export/
```

Examples:

- `README_ChatGPT.md`
- `implementation_plan.md`
- `task.md`
- copied chat transcripts
- Antigravity-generated summaries
- exported ZIP files
- old final reports that have not been independently verified

These files are historical evidence, not authoritative specifications.

The supplied `README_ChatGPT.md` is not reliable as a project README. It contains malformed control characters, claims files are present that are not in the ZIP, and defends previous implementation choices that must be independently audited. Preserve it only as legacy evidence.

Do not overwrite the real `src/` files with the four Python files from the Antigravity export. Put the exported versions under `handoff/antigravity_export/` and let ZCode compare them against the current repository.

## Files normally safe to remove after confirming they are generated

ZCode may clean these after preserving anything useful:

- `__pycache__/`
- `.pytest_cache/`
- `.mypy_cache/`
- `.ruff_cache/`
- `.ipynb_checkpoints/`
- temporary logs
- temporary extracted copies
- duplicate ZIP copies
- editor backup files
- obsolete generated caches

Do not delete:

- source code
- datasets
- split definitions
- notebooks
- checkpoints
- predictions
- metrics
- manifests
- research specifications
- Git history

A local `.venv/` may remain for running the project but should be excluded from Git and from the final reproducible export.

---

# START PROMPT

```text
/goal Take full ownership of the complete hypergraph research repository currently open as this workspace. This is a takeover from a previous Antigravity agent. Work autonomously until Phase 3 is scientifically valid, reproducible, internally consistent, fully executed, and packaged for independent verification.

Read and obey the root AGENTS.md as the permanent project contract. Treat this message as the session goal and AGENTS.md as the detailed acceptance specification.

IMPORTANT CONTEXT ABOUT THE HANDOFF

The folder handoff/antigravity_export contains candidate files exported from the previous agent. They are not automatically trusted and must not overwrite the current repository without comparison.

The received Antigravity export contains only:
- audit_model_numerics.py
- audit_splits.py
- differentiable_matching.py
- run_phase3_final.py
- README_ChatGPT.md

It is not a complete project. The exported run_phase3_final.py depends on modules such as src.config_manager and src.oracles that were not included in that export. The exported README_ChatGPT.md contains malformed characters, refers to files not present in the ZIP, and attempts to justify previous changes. Treat it only as historical evidence.

FIRST ACTIONS — DO NOT SKIP

1. Confirm that the current workspace is the real complete hypergraph repository.
2. Inspect the full directory tree, Git status, Git history where useful, dependency files, source modules, data, split files, notebooks, tests, results, checkpoints, predictions, reports, manifests, and legacy agent files.
3. Create a new Git branch named zcode-phase3-final, or a safe equivalent if that branch already exists.
4. Create a timestamped safety snapshot before editing:
   results/phase3_archive/<UTC_TIMESTAMP>/
5. Preserve all existing evidence before replacing anything.
6. Produce an inventory that labels every relevant file as:
   - authoritative source
   - current implementation
   - generated artifact
   - legacy Antigravity evidence
   - duplicate
   - missing dependency
   - obsolete cache
   - unsafe to delete
7. Compare every file under handoff/antigravity_export against the corresponding current repository file using actual diffs.
8. Merge only changes that are mathematically correct, compatible with the project specification, and supported by tests. Do not blindly copy the exported files.
9. Move legacy chat-specific files out of the project root into docs/legacy/antigravity/ or handoff/antigravity_export/. Do not delete them until the takeover is complete and their useful information has been preserved.
10. Keep the project root clean and suitable for a real research repository.

AUTONOMOUS WORK RULE

Do not stop after making a plan, inspecting files, finding a problem, editing code, passing one test, completing one seed, generating partial metrics, or writing a report.

Do not repeatedly ask me to approve ordinary repository-local changes.

Continue automatically through inspection, implementation, training, debugging, reruns, numerical auditing, notebook execution, consistency checks, manifest verification, and final packaging.

Progress updates do not count as completion.

Only stop for a genuine external hard blocker such as unavailable source data, unavailable credentials, denied system permissions, or hardware failure that cannot be solved locally. Before declaring a blocker, exhaust repository-local alternatives and record:
- exact command
- complete error
- files involved
- attempted fixes
- why each fix failed
- minimum user action required

SCIENTIFIC INTEGRITY — NON-NEGOTIABLE

Never:
- weaken Gate B from 0.99 to 0.98 merely to pass;
- remove a failing audit because the implementation fails it;
- claim a five-seed result from one seed;
- present a train-set overfit gate as test performance;
- attribute LearnedCostSinkhorn metrics to MatchingNetwork;
- attribute MatchingNetwork metrics to LearnedCostSinkhorn;
- manually edit metrics;
- tune on IID or OOD test sets;
- choose checkpoints using test performance;
- omit OOD-size results;
- report PASS without executable evidence;
- claim reproducibility while required files are missing.

If the original requirement is mathematically wrong, preserve the original test result, explain the mathematical issue with evidence, and introduce a corrected test as a separately named criterion. Do not silently redefine success.

RESOLVE THE PROJECT IDENTITY

Determine from the actual research specification, source code, configs, notebooks, and experiment design:
- the designated primary Phase 3 model;
- the numerically best-performing model;
- all required baselines;
- all required ablations;
- the official locked seeds;
- the official split definitions;
- Gate A, Gate B, and Gate C definitions.

Do not infer that the best-performing model is automatically the primary model.

REPRODUCIBILITY

Make the repository runnable from a clean checkout using project-relative paths.

Ensure the project contains or can deterministically generate:
- all source modules;
- config_manager.py;
- oracles.py;
- model configurations;
- data loaders;
- locked split definitions;
- dependency files and lock files;
- exact seeds;
- tests;
- notebooks;
- checkpoints;
- predictions;
- metrics;
- audits;
- reports;
- logs;
- manifests.

If raw data cannot legally or practically be included, provide a deterministic acquisition/preparation script and verify it.

DATASET AND SPLIT AUDIT

Audit train, validation, IID test, OOD-template test, and OOD-size test.

Verify:
- valid sample IDs;
- valid labels;
- expected sizes and class counts;
- valid B_x and B_y shapes;
- correct m and k metadata;
- padding and masks;
- no prohibited sample overlap;
- no duplicated underlying graph pairs across prohibited splits;
- deterministic loading;
- deterministic split hashes;
- no empty or malformed samples.

Generate:
results/phase3/split_overlap_audit.json

Do not continue to a final PASS while unexplained split-integrity failures remain.

GATE A

Execute the intended oracle verification and save per-sample oracle predictions.

Re-derive the expected Sinkhorn coupling normalization from the actual implementation and mathematical formulation.

Do not require a hard matched entry to equal 1.0 when a normalized coupling with m equal rows should assign approximately 1/m mass to each hard match.

Report expected total mass, row marginals, column marginals, convergence residuals, tolerance, and PASS/FAIL.

GATE B

Restore and enforce the intended non-weakened acceptance rule.

Run all official locked seeds. Unless the actual specification defines another set, use:
42, 100, 2026, 777, 999.

For every failure:
- reproduce it deterministically;
- identify the exact sample;
- inspect labels and graph structure;
- inspect gradients, logits, probabilities, masks, padding, temperature scheduling, Sinkhorn convergence, loss, optimizer, and checkpoint selection;
- fix the underlying issue;
- rerun all seeds from scratch.

Save a structured Gate B report with per-model and per-seed results, threshold, failed samples, and exact evidence.

GATE C AND FINAL TRAINING

Run the real Gate C definition across the required seeds and datasets.

Keep train, validation, IID, OOD-template, and OOD-size results separate.

Use validation only for checkpoint selection.

Record:
- selection epoch;
- validation criterion;
- final temperature;
- train and evaluation Sinkhorn iterations;
- seed;
- checkpoint hash;
- runtime;
- device and dtype.

Run the primary model, all required baselines, and all required ablations on identical locked splits and seeds.

Save:
- results/phase3/per_seed_metrics.csv
- results/phase3/aggregate_metrics.csv
- results/phase3/checkpoints/<model>_seed<seed>.pt
- results/phase3/predictions/<model>_seed<seed>_<split>.csv

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

FULL NUMERICAL AND STRUCTURAL AUDIT

The previous audit must be treated as unverified. Its comments appear uncertain about Sinkhorn row and column mass. Re-derive the correct marginals from the actual Sinkhorn code before writing acceptance checks.

Use:
- float64;
- compact valid submatrices with padding removed;
- explicit masks;
- deterministic permutations;
- at least 100 Sinkhorn iterations;
- preferably 500 for the final strict audit.

For every required model seed and every applicable locked IID, OOD-template, and OOD-size sample, independently check:
1. all C values are finite;
2. all P values are finite;
3. P is non-negative within tolerance;
4. C_yx ≈ C_xy.T;
5. P_yx ≈ P_xy.T;
6. matching cost is root-swap invariant;
7. equality score is root-swap invariant;
8. final logits are root-swap invariant;
9. final probabilities are root-swap invariant;
10. total coupling mass is correct;
11. row marginals match their mathematically correct target;
12. column marginals match their mathematically correct target;
13. assignment transpose holds directly at P level;
14. valid node-row permutations preserve the required outputs;
15. hyperedge/column permutations are tested if the mathematical problem requires them.

Do not simply remove column-permutation tests. Determine whether hyperedge order is semantically fixed by the dataset or should be permutation invariant. Support the conclusion using the research formulation, data generation, and model design. If the architecture violates a required invariance, fix it or report an explicit failure.

Use strict per-quantity tolerances, preferably 1e-8 in float64, with an absolute acceptance ceiling no looser than 1e-6 unless mathematically justified.

Generate:
results/phase3/numerical_audit.json

Record per check:
- count;
- failures;
- mean;
- maximum;
- p95;
- p99;
- tolerance;
- worst sample;
- worst split;
- worst seed;
- iterations;
- marginal residuals;
- NaN/Inf counts.

REPORT AND METRIC CONSISTENCY

Generate all aggregate results directly from verified per-seed files.

Create an automated report consistency checker that compares every headline statement against aggregate_metrics.csv and fails on:
- wrong model;
- wrong split;
- wrong metric;
- wrong mean;
- wrong standard deviation;
- wrong seed count.

Explicitly distinguish:
- designated primary model;
- best-performing model;
- baselines;
- ablations.

Include exact IID, OOD-template, and OOD-size results with uncertainty. Do not hide poor OOD performance behind qualitative language.

NOTEBOOKS

Update and cleanly execute from a fresh kernel:
- notebooks/Phase3_Research.ipynb
- notebooks/Phase3_Explained_FA.ipynb

No hidden state is allowed.

Synchronize:
- final model identity;
- methodology;
- split definitions;
- Gates A/B/C;
- per-seed results;
- aggregate results;
- IID;
- OOD-template;
- OOD-size;
- ablations;
- complete numerical audit;
- limitations;
- exact reproducibility commands.

The Persian notebook must be a natural, technically correct explanation, not a superficial translation.

TESTS, LOGS, AND MANIFESTS

Run and log:
- dependency/import checks;
- repository tests;
- clean Phase 3 execution;
- split audit;
- Gate A;
- Gate B;
- Gate C;
- final model and baseline runs;
- numerical audit;
- report consistency check;
- clean notebook execution;
- checkpoint round-trip tests;
- manifest verification;
- ZIP content verification.

Store commands, timestamps, stdout, stderr, runtime, and return code under:
results/phase3/logs/

A crash, interruption, timeout, skipped command, or missing log is not PASS.

Generate SHA-256 manifests:
- results/phase3/run_manifest.json
- results/phase3/final_artifact_manifest.json

Include hashes for source files, configs, dependency locks, split files, checkpoints, predictions, metrics, audits, notebooks, reports, and logs.

Recompute and verify all hashes only after final artifacts are frozen.

FINAL REPORT

Create:
results/phase3/final_validation_report.md

It must contain:
- repository commit and branch;
- environment;
- archive path;
- authoritative project specification;
- primary model;
- best-performing model;
- split integrity;
- exact Gate definitions;
- Gate A evidence;
- Gate B per-seed evidence;
- Gate C per-seed evidence;
- model/baseline/ablation results;
- IID results;
- OOD-template results;
- OOD-size results;
- full numerical audit;
- report consistency status;
- notebook execution status;
- known limitations;
- reproducibility commands;
- generated artifact list;
- explicit PASS/FAIL/PARTIAL/NOT RUN/BLOCKED status for every requirement.

Do not issue a global PASS if any mandatory requirement is not PASS.

FINAL EXPORT

Create a complete independently reproducible ZIP.

It must include all required:
- source code;
- configs;
- dependency definitions and locks;
- data split definitions or deterministic generation/acquisition scripts;
- tests;
- checkpoints;
- predictions;
- metrics;
- audits;
- reports;
- logs;
- executed notebooks;
- manifests.

Do not export only a small ChatGPT review bundle.

Programmatically list and verify ZIP contents, extract it into a temporary clean directory, verify hashes, install or validate dependencies, and run the documented smoke/reproduction checks from the extracted copy.

COMPLETION CONTRACT

Continue working until every mandatory item is verified PASS.

The goal is incomplete if any mandatory item is:
- FAIL
- PARTIAL
- NOT RUN
- BLOCKED
- missing evidence
- inconsistent with source metrics
- dependent on an untracked local file

Only then give the final response.

FINAL RESPONSE FORMAT

Provide exact:
1. overall status;
2. Git branch and commit;
3. primary model;
4. best-performing model;
5. IID metrics;
6. OOD-template metrics;
7. OOD-size metrics;
8. Gate A result;
9. Gate B result for every seed;
10. Gate C result for every seed;
11. maximum error for each numerical audit quantity;
12. number of samples audited;
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

Do not finish by asking me whether to continue, whether to update notebooks, or whether the current result is acceptable. Those are already mandatory parts of this goal.
```
