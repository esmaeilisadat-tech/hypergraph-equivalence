# Independent Review of `phase3_reproducible.zip`

## Overall verdict

**Technical packaging and internal consistency: PASS**

**Claim that the primary model learned the task from data: NOT SUPPORTED in the current design**

The archive is substantially better than the previous Antigravity export. It contains the active source, data splits, tests, notebooks, 35 checkpoints, 175 prediction files, metrics, gate reports, numerical audits, logs, and a complete SHA-256 artifact manifest.

## Checks independently reproduced

- ZIP extracted successfully: 337 files.
- Artifact-manifest entries checked: 336/336 hashes and sizes match.
- The only intentionally excluded file is `results/phase3/final_artifact_manifest.json`.
- `python -m pytest -q`: 5/5 tests pass.
- `python -m src.final_checks`: PASS.
- Counts verified:
  - checkpoints: 35
  - prediction CSVs: 175
  - per-seed metric rows: 175
  - aggregate rows: 35
- Saved prediction metrics recompute within `3.143554050399899e-08`.
- Both notebooks contain 7/7 executed code cells and zero error outputs.
- Locked split labels agree with exact multiset and Hungarian oracles.
- The reported numerical-audit artifact has zero failed mandatory quantities over 10,000 seed-sample cases.

## Critical scientific finding

The primary `MatchingNetwork` is already a perfect classifier before training.

Its architecture directly computes fixed L1 row-matching costs, while the label is defined by whether the row multisets match exactly. The classifier is manually initialized with:

- weight: `[-2, +2]`
- bias: `0`

Independent zero-training evaluation produced accuracy 1.0 on:

- train
- validation
- IID test
- OOD-template test
- OOD-size test

The same is true for the untrained `LearnedCostSinkhorn` and `StraightThroughHungarian` variants because they retain the exact fixed-L1 separator.

Negative Hungarian costs are:

- train: exactly 4
- validation: exactly 4
- IID: exactly 4
- OOD-size: exactly 4
- OOD-template: 4 to 12

Positive costs are always 0.

Therefore, training is not required to discover the class boundary. It only increases confidence/calibration.

## Seed issue

All five `MatchingNetwork` checkpoints have identical tensor state dictionaries.

All five prediction CSVs are byte-identical for each split.

Therefore the five seeds are repeated deterministic runs, not five independent optimization outcomes for the primary model. Reporting `mean ± std` is numerically correct but scientifically uninformative for this deterministic architecture.

## Provenance issue

`run_manifest.json` records only the takeover baseline commit:

`119343d0ad758f8c4a78d3831e8727d37255a3be`

It also records a dirty working tree with many modified and untracked final files. The final source and evidence are not tied to a final clean Git commit.

## Missing requested controls

The archive does not contain evidence for:

- a shuffled-label negative control;
- a genuinely fresh virtual-environment installation test;
- a final clean Git commit;
- a claim audit distinguishing an algorithmic oracle-like solver from a learned model.

## Correct scientific interpretation

The current results are valid as evidence that a fixed differentiable matching rule implements the synthetic exact-equivalence criterion robustly and with strong numerical invariance.

They are not valid as evidence that a neural model learned the equivalence rule from training data.

## Required correction

Choose one of two defensible framings:

### A. Algorithmic matching study

Keep `MatchingNetwork` as a deterministic differentiable solver.

- Call it an algorithmic or oracle-like method, not a learned discovery.
- Report zero-shot performance.
- Remove claims that five seeds demonstrate optimization robustness.
- Treat training as calibration only.

### B. Genuine learning study

- Move fixed-L1 MatchingNetwork and Hungarian to oracle/algorithmic baselines.
- Create a primary learned matcher with random initialization and without the exact fixed-L1 zero/nonzero separator.
- Add zero-shot, shuffled-label, and random-label controls.
- Rerun the learned model across independent seeds.
- Preserve the current dataset and results as a separate corrected deterministic benchmark version.

Before publication, also create a final clean Git commit and regenerate the manifests and ZIP from that commit.
