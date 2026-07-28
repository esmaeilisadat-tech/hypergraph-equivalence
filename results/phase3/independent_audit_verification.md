# Independent verification of `INDEPENDENT_PHASE3_AUDIT.md`

Verified against the live repository and the preserved
`results/phase3/phase3_reproducible.zip` before any scientific correction.
The pre-correction evidence is archived at
`results/phase3_archive/20260727T160053Z_independent_audit_corrections/`.

## Verdict

The audit's technical packaging claims are **CONFIRMED**. Its central
scientific finding is also **CONFIRMED**: the former designated primary is an
algorithmic fixed-L1 solver that is perfect before training, so its results do
not establish that the equivalence rule was learned from data.

## Claim-by-claim classification

| Audit claim | Classification | Independent evidence |
|---|---|---|
| Technical packaging and internal consistency pass | CONFIRMED | Extraction, manifest verification, `pytest`, and `src.final_checks` all returned 0 from both the live tree and extracted ZIP. |
| The claim that the former primary learned the task is unsupported | CONFIRMED | Zero-training evaluation is perfect on every split; fixed L1 and manual classifier evidence is at `src/differentiable_matching.py:102`, `src/differentiable_matching.py:109`, `src/differentiable_matching.py:373`, and `src/differentiable_matching.py:435`. |
| ZIP extracts successfully and has 337 entries | CONFIRMED | `results/phase3/phase3_reproducible.zip`, SHA-256 `03319a79b01d13576b151398e4b841540b70155adf74ecd21def0fa24d3536c3`; `ZipFile.testzip()` returned `None`; entry count 337. |
| Manifest has 336 correct entries | CONFIRMED | Extracted `results/phase3/final_artifact_manifest.json` has `entry_count=336`; independent size and SHA-256 loop found 0 failures. |
| The manifest itself is the only ZIP entry not listed | CONFIRMED | Set difference between ZIP names and manifest paths is exactly `results/phase3/final_artifact_manifest.json`. The ZIP is also intentionally not self-hashed because it cannot contain itself. |
| `python -m pytest -q` passes 5/5 | CONFIRMED | Live and extracted-ZIP runs both returned 0 with `5 passed`; cache provider was disabled to keep verification read-only. |
| `python -m src.final_checks` passes | CONFIRMED | Live and extracted-ZIP runs both returned 0 with status `PASS` and no failures. |
| Counts are 35 checkpoints, 175 prediction CSVs, 175 per-seed rows, and 35 aggregate rows | CONFIRMED | Independent filesystem and CSV counts match exactly. |
| Prediction metrics recompute within `3.143554050399899e-08` | CONFIRMED | Independent recomputation over all 175 prediction files obtained the same maximum error, in Brier score; all other listed discrepancies were no greater. |
| Both notebooks have 7/7 executed code cells and no error output | CONFIRMED | Direct `nbformat`/JSON inspection of both files found execution counts `[1,2,3,4,5,6,7]` and zero error outputs. |
| Locked labels agree with exact multiset and Hungarian oracles | CONFIRMED | Independent pass over all 3,600 samples found 0 oracle-label disagreements. |
| Reported numerical audit has zero mandatory failures over 10,000 seed-sample cases | CONFIRMED | `results/phase3/numerical_audit.json` reports 10,000 audited, 0 failed; every quantity has `failed_count=0`; extracted `src.final_checks` independently enforced those fields and returned 0. |
| Former `MatchingNetwork` is perfect before training on all five splits | CONFIRMED | Independent initialization/evaluation for seeds 42, 100, 2026, 777, and 999 produced accuracy 1.0 on train, validation, IID, OOD-template, and OOD-size. |
| Classifier is manually initialized to `[-2,+2]`, bias 0 | CONFIRMED | `src/differentiable_matching.py:109-116`. |
| The architecture directly computes fixed L1 row costs | CONFIRMED | `src/differentiable_matching.py:368-377`; downstream matching features explicitly use that base cost at lines 435-448. |
| Untrained `LearnedCostSinkhorn` and `StraightThroughHungarian` are also perfect | CONFIRMED | Independent five-seed zero-training evaluation produced accuracy 1.0 on all five splits for both. `LearnedCostSinkhorn` retains `cost_base` through `cost = cost_base + ...` at lines 400-403. |
| Negative Hungarian costs are 4 on train/validation/IID/OOD-size and 4-12 on OOD-template; positives are 0 | CONFIRMED | Independent all-sample audit found negative supports `{4}`, `{4}`, `{4}`, `{4,6,8,10,12}`, `{4}` respectively and positive support `{0}` on every split. |
| Training is unnecessary to discover the class boundary and only changes confidence/calibration | PARTIALLY CONFIRMED | Training is demonstrably unnecessary for classification. The old experiment did not include a dedicated calibration-only causal control, so “only” is stronger than the evidence even though the decision boundary is present at initialization. |
| All five former-primary checkpoint tensor states are identical | CONFIRMED | Tensor-by-tensor comparison after `torch.load(..., weights_only=False)` found exact equality for `temperature`, `classifier.weight`, and `classifier.bias` across all five seeds. Serialized file hashes differ because checkpoint metadata differs. |
| Former-primary prediction files are byte-identical across seeds for each split | CONFIRMED | Per-split SHA-256 grouping produced one unique hash across five seeds for each of the five splits. |
| The five former-primary seeds are repeated deterministic outcomes, not independent optimization outcomes | CONFIRMED | Exact tensor and prediction identity establish this for the former fixed-L1 primary. It does not apply to every old model: `LearnedCostSinkhorn` has five distinct tensor states and prediction hashes. |
| Manifest records only baseline commit `119343d...` and a dirty final tree | CONFIRMED | `results/phase3/run_manifest.json` has `baseline_commit=119343d0ad758f8c4a78d3831e8727d37255a3be` and embeds the long dirty `git_status`; live `git status --short` independently matches the provenance defect. |
| Final bundle lacks shuffled-label evidence | CONFIRMED | No active result, log, checkpoint, prediction, or control artifact exists. Hits occur only in preserved Antigravity legacy files and the audit text. |
| Final bundle lacks a genuinely fresh environment installation test | CONFIRMED | Status records reuse of a pre-existing parent `.venv`; no active install log or temporary-environment evidence exists. |
| Final evidence is not tied to a clean final commit | CONFIRMED | Live HEAD is the takeover baseline and all final evidence is modified or untracked. |
| Final bundle lacks a claim audit separating an algorithmic solver from learned discovery | CONFIRMED | The old report instead calls fixed-L1 `MatchingNetwork` primary and describes its manually initialized two-feature head as learned (`results/phase3/final_validation_report.md:70-74`). |
| Results are valid as evidence for an algorithmic differentiable matching rule, not learned discovery | CONFIRMED | This is the scientifically supported interpretation of the reproduced zero-shot, cost-separation, and seed-identity evidence. |

No material claim in the independent audit was classified **NOT CONFIRMED**.
The calibration-only wording was the sole **PARTIALLY CONFIRMED** claim.

## Mandatory pre-correction checks

### Zero-training performance

All seven reported model types were freshly initialized under all five locked
seeds and evaluated without optimization. Accuracy was:

| Model | Train | Validation | IID | OOD-template | OOD-size |
|---|---:|---:|---:|---:|---:|
| MLP_Baseline | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| NestedDeepSets | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| BipartiteEncoder | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| MatchingNetwork | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| LearnedCostSinkhorn | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| StraightThroughHungarian | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| NoSinkhorn | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |

### Prohibited metadata and label leakage

**No prohibited metadata path was found.** The matching forward signature is
`forward(self, bx, by, return_aux=False)`. `collate_samples` returns only
`B_x`, `B_y`, and labels, while `predict` passes only `B_x` and `B_y` into the
model (`src/phase3_experiments.py:97-111`). Sample IDs, split names, labels,
structure hashes, pair identifiers, oracle outputs, and precomputed exact costs
do not enter the forward path.

### Seed independence

The former deterministic primary fails the scientific independence criterion:
all five tensor states and per-split predictions are identical. Randomly
initialized learned baselines and old `LearnedCostSinkhorn` are seed-dependent,
showing that the experiment machinery can represent independent initial states;
the defect is the former primary's deterministic construction.

### Dataset difficulty

The old benchmark is balanced and nuisance-matched but algorithmically trivial
for exact row assignment: exact Hungarian cost is zero if and only if the pair
is equivalent, and almost all negatives have the same cost 4. This benchmark
must therefore remain a versioned deterministic/oracle benchmark rather than
serving as evidence that a neural model discovered equivalence from data.

## Commands used

All Python commands used
`C:/Users/esmae/Documents/Educx Kurs machine lerning/aufgabe/.venv/Scripts/python.exe`.

- `python -m pytest -q -p no:cacheprovider` — return code 0, live and extracted ZIP.
- `python -m src.final_checks` — return code 0, live and extracted ZIP.
- A read-only `zipfile` extraction, `testzip`, entry-set, size, and SHA-256 verification script — return code 0.
- A read-only all-sample exact-label/Hungarian-cost and all-prediction metric-recomputation script — return code 0.
- A read-only five-seed/all-model zero-training evaluation script using `load_splits`, `load_config`, `initialize_model`, `predict`, and `_metrics` — return code 0.
- A read-only tensor-state and prediction SHA-256 comparison script — return code 0.
- `git status --short`, `git log -5 --oneline --decorate`, and `git diff --stat` — return code 0.
- Source line inspection with `rg -n` and `Get-Content` — return code 0.
