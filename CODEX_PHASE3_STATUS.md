# Codex Phase 3 Status

Last reconciled: 2026-07-28 00:18:30 UTC

## Overall status

Phase 3 is **COMPLETE**. All mandatory scientific, structural, notebook,
consistency, repository, packaging, and fresh-environment checks pass. The
corrected learned study is committed on branch `codex-phase3-final`.

The immutable scientific evidence commit is:

`8425f6c780ad611f2502dd7f3e57119a07f07ee0`

The final handoff commit cannot be embedded inside itself; its exact SHA is
therefore recorded by `results/phase3/run_manifest.json` as `source_commit`.
That post-commit manifest records an empty Git status. The ignored artifact
manifest and ZIP were regenerated from the clean final HEAD and passed exact
verification. This self-reference-safe finalization does not alter scientific
source, splits, checkpoints, predictions, metrics, controls, audits, or
notebooks.

No process is currently running.

## Eight-stage plan

1. **Takeover inspection and preservation — COMPLETE**
   Git status/diff, full tree, source, configurations, splits, notebooks,
   metrics, predictions, checkpoints, manifests, logs, history, and legacy
   claims were inspected. Existing evidence was preserved before correction.

2. **Independent-audit verification — COMPLETE**
   Every finding in `INDEPENDENT_PHASE3_AUDIT.md` was independently classified
   as CONFIRMED, PARTIALLY CONFIRMED, or NOT CONFIRMED with executable/file
   evidence in
   `results/phase3/independent_audit_verification.md`.

3. **Scientific restructuring and benchmark locking — COMPLETE**
   Fixed-L1/Hungarian methods were reclassified as algorithmic/oracle
   controls. A randomly initialized learned primary and a versioned learned
   final benchmark were implemented and locked without test-based selection.

4. **Model selection and final five-seed experiment matrix — COMPLETE**
   Candidate selection used train/validation evidence only. All 10 required
   models × 5 locked seeds completed, with 50 checkpoints, 250 prediction
   files, 250 per-seed rows, and 50 aggregate rows.

5. **Gates, controls, split audit, and numerical audit — COMPLETE**
   Gates A/B/C, zero-shot, label access, seed independence, oracle separation,
   split integrity, invalid-label controls, and the full float64 audit pass.
   Failed attempts and their corrections remain preserved and disclosed.

6. **Report, notebooks, and consistency — COMPLETE**
   The technical report and both English/Persian notebooks were regenerated,
   clean-executed, and checked against the aggregate CSV.

7. **Repository verification and scientific commit — COMPLETE**
   Imports, all tests, final frozen-evidence checks, staged diff checks, and
   the complete scientific commit succeeded.

8. **Post-commit export and fresh-environment verification — COMPLETE**
   The post-commit manifests and ZIP passed exact verification, and that ZIP
   passed a genuinely fresh temporary-environment install and extracted
   verification. The final clean-HEAD manifest/ZIP regeneration and exact
   local verification also pass.

## Preservation and archives

- Main pre-correction archive:
  `results/phase3_archive/20260727T160053Z_independent_audit_corrections/`
  (342 files, 31,049,172 bytes).
- Original deterministic benchmark preserved under:
  `data/deterministic_benchmark_v2/` and
  `results/deterministic_benchmark_v2/`.
- Development-only v1 benchmark preserved under:
  `data/learned_benchmark_v1_development/` and
  `results/learned_benchmark_v1_development/`.
- Full-pipeline invalid-label failure archive:
  `results/phase3_archive/20260727T213558Z_label_control_protocol_correction/`
  (28 archived items, 715,073 bytes).
- First numerical-audit failure archive:
  `results/phase3_archive/20260727T233427Z_numerical_audit_failure/`.
- Legacy Antigravity active files were preserved under
  `docs/legacy/antigravity/`; they are not treated as verified current source.
- No source datasets were deleted. Archives and ZIPs are excluded from Git.

## Independent Audit Finding Verification

The audit's core scientific finding was CONFIRMED: the old fixed-L1
`MatchingNetwork` solved the deterministic benchmark perfectly before
training, its five primary checkpoints were tensor-identical, and per-split
predictions were byte-identical. The old result remains valid only as an
algorithmic equivalence solver, not learned discovery.

The old package/provenance and missing-control findings were also confirmed:
the old manifest referenced takeover baseline commit
`119343d0ad758f8c4a78d3831e8727d37255a3be`, recorded a dirty tree, and lacked
the requested shuffled/random-label and genuinely fresh-environment evidence.

Complete commands, files, line evidence, hashes, and classifications are in:

`results/phase3/independent_audit_verification.md`

## Deterministic Baselines

The current roles are explicit:

- `DeterministicDifferentiableMatching`: algorithmic baseline.
- `ExactHungarianOracle`: oracle.
- `FixedL1ResidualSinkhorn`: algorithmic hybrid.
- `FixedL1StraightThroughHungarian`: algorithmic baseline.

These methods are not described as learned evidence and are excluded from
learned-model ranking claims. Exact algorithmic/oracle test accuracy is 1.0.

## Learned Primary Model

The designated primary is `LearnedSinkhornMatcher` (`primary_learned`), not
the numerically convenient old deterministic model.

It:

- starts from seed-dependent random initialization;
- receives only `B_x`, `B_y`, and optional `return_aux`;
- has no fixed L1/Hamming separator or exact Hungarian result;
- does not consume labels, sample IDs, structure hashes, split names, or file
  names;
- learns symmetric column-pair representations, row-pair costs, Sinkhorn
  transport summaries, and a classifier;
- uses validation macro F1 followed by validation loss for checkpoint
  selection;
- never uses final-v2 tests for selection.

Five unique initial-state hashes, five unique final-state hashes, and five
unique first-epoch data-order hashes confirm independent runs.

## Benchmark and split integrity

Final benchmark: `learned_benchmark_v2_final`.

The development chronology is explicit: v2 was frozen before the first final
v2 test evaluation, but not before development optimization. Generator/model
development used the separate v1 benchmark; no final-v2 test result selected a
model or hyperparameter.

Split audit: `PASS`, zero problems and zero prohibited sample-ID,
matched-pair, base-hash, or invariant-structure overlap.

- Train: 800 samples, 400 class 0 / 400 class 1.
- Validation: 800, 400 / 400.
- IID test: 800, 400 / 400.
- OOD-template test: 800, 400 / 400.
- OOD-size test: 400, 200 / 200.

Locked split file SHA-256 values:

- Train:
  `6d16657a848ef31e820c597eb585199a0b402ccfac92c247ea33d78664c38edf`
- Validation:
  `69bfa5c61be13bf71199cd152f8f5dfe002451fd08e80f282204d00ce9f63ada`
- IID:
  `260a8f3963b77c36c5e0c6064838bdb2d3042ef85426728c23f6adafde358c4f`
- OOD-template:
  `e8de47dba98ee788f6658e276d3a85e50c448aa35d5666801b8c31f09b56c322`
- OOD-size:
  `58625a95222355f928a04e34e5456fa50535b85299b1ad7bc1827ee1d197b982`

Deterministic generation/loading, IDs, hashes, labels, shapes, masks, padding,
matched pairs, and margins all pass.

## Model selection

Candidates v1-v4 are preserved as failed/development evidence. Candidate v5
was selected using train/validation only.

Selected Seed 42 checkpoint:

`results/phase3/model_selection/candidate_v5_seed42.pt`

SHA-256:

`647a0750de3664b6f8f53b31d1f7a875e32c8658b326624c0581ece2f915e661`

The disclosed v1 test-exposure incident was not used for final-v2 claims.

## Zero-Shot Results

`results/phase3/zero_shot_control.json`: `PASS`.

For `LearnedSinkhornMatcher`, accuracy is exactly 0.5 for all five seeds on
Train, Validation, IID, OOD-template, and OOD-size (25 primary rows). Primary
mean zero-shot ROC-AUC is `0.5458902499999999`.

Mean primary train accuracy improves from 0.5 zero-shot to 1.0 trained, an
absolute improvement of 0.5.

## Final learned-primary metrics

All accuracy, balanced accuracy, macro F1, and ROC-AUC values below are
five-seed mean ± sample standard deviation:

- IID: accuracy `1.0 ± 0.0`; balanced accuracy `1.0 ± 0.0`; macro F1
  `1.0 ± 0.0`; ROC-AUC `1.0 ± 0.0`; loss
  `8.076873781895485e-06 ± 4.556830588707213e-06`.
- OOD-template: accuracy `1.0 ± 0.0`; balanced accuracy `1.0 ± 0.0`; macro F1
  `1.0 ± 0.0`; ROC-AUC `1.0 ± 0.0`; loss
  `8.280781531712364e-06 ± 4.206681827433285e-06`.
- OOD-size: accuracy `1.0 ± 0.0`; balanced accuracy `1.0 ± 0.0`; macro F1
  `1.0 ± 0.0`; ROC-AUC `1.0 ± 0.0`; loss
  `0.004025965099677455 ± 0.003397260359064635`.

Selection epochs/final temperatures:

- Seed 42: epoch 114; `0.0199782345443964`.
- Seed 100: epoch 96; `0.023092152550816536`.
- Seed 2026: epoch 135; `0.016871949657797813`.
- Seed 777: epoch 160; `0.013797296211123466`.
- Seed 999: epoch 135; `0.016871949657797813`.

## Baselines and ablations

All 10 models and five seeds are present in
`results/phase3/per_seed_metrics.csv` and
`results/phase3/aggregate_metrics.csv`.

Important ablation result:

- `LearnedNoSinkhorn` is near-perfect IID/OOD-template but OOD-size accuracy
  is `0.5685 ± 0.00993730345717586`, macro F1
  `0.4695740223473729 ± 0.017862316678784573`, and ROC-AUC `0.99872`.
  This exposes a transport/calibration failure that accuracy on easier splits
  would hide.
- `LearnedNoColumnContext` OOD-size accuracy is
  `0.9985 ± 0.003354101966249658`; macro F1
  `0.9984999156202538 ± 0.0033542906450984653`; ROC-AUC 1.0.

## Shuffled-Label Control

The first full-pipeline control was completed and preserved, not discarded.
It was scientifically confounded because label-free structural pretraining
itself supplies real task signal. The full run returned code 1 after
4,983.0988211 seconds: random-label mean accuracy was 0.665 on IID and 0.6915
on OOD-template, above the unchanged 0.65 maximum. Seed 777 selected epoch 0
(post-pretraining) and reached 0.9675 IID / 0.985 OOD-template accuracy.

The corrected `supervised_label_signal_only` protocol disables only that
label-free pretraining and otherwise retains the locked architecture,
optimizer, validation selection, seeds, and splits. It completed all 10 runs
in 5,128.729064899999 seconds with return code 0 and `PASS`.

Shuffled-label five-seed accuracy:

- IID: `0.49425 ± 0.020128`.
- OOD-template: `0.50325 ± 0.031962`.
- OOD-size: `0.5 ± 0.0`.

Shuffled-label macro F1 means are `0.489143`, `0.494272`, and `0.333333`.

Preserved confounded evidence:

- `results/phase3/controls/full_pipeline_pretraining_confound_controls.json`
- `results/phase3/controls/full_pipeline_pretraining_confound_metrics.csv`

Valid decomposed control:

- `results/phase3/controls/label_signal_controls.json`
- `results/phase3/controls/supervised_only_label_signal_control_metrics.csv`

## Random-Label Control

Random-label five-seed accuracy:

- IID: `0.48975 ± 0.090357`.
- OOD-template: `0.49425 ± 0.116641`.
- OOD-size: `0.5 ± 0.0`.

Random-label macro F1 means are `0.486514`, `0.490928`, and `0.333333`.

All aggregate accuracies satisfy the unchanged 0.65 maximum. Every corrected
control record states that self-supervised cost pretraining was not performed.

## Label access, seed independence, and claim audit

`results/phase3/scientific_claim_audit.json`: `PASS`.

All 10 checks pass:

- no fixed/metadata separator in the learned primary;
- chance-level zero-shot primary;
- meaningful training improvement;
- five independent primary seeds;
- algorithmic models not claimed as learned;
- oracle separation;
- locked final-test protocol;
- positive label-access control;
- valid supervised-only invalid-label controls;
- preserved/disclosed pretraining confound.

The positive label-access control reaches expected accuracy 1.0 on every split,
while source inspection finds no prohibited forward-input token.

## Gate A

`PASS` over 3,600 samples with exact oracle accuracy 1.0.

- Sinkhorn iterations: 500.
- Expected total coupling mass: 1.0.
- Hard matched entry interpretation: approximately `1/m`, not 1.0.
- Maximum total-mass error: `2.220446049250313e-16`.
- Maximum row residual: `7.353839759360881e-14`.
- Maximum column residual: `7.296940829348841e-14`.
- Tolerance: `1e-08`.

## Gate B

All five locked seeds `PASS` the unchanged intended rule with 64 samples,
accuracy 1.0, macro F1 1.0, both class F1 values 1.0, and no failed IDs.

Loss:

- Seed 42: `0.0005092812822928834`.
- Seed 100: `0.0004148976468337607`.
- Seed 2026: `0.0003978203914182482`.
- Seed 777: `0.0004778715588270414`.
- Seed 999: `0.0003934729328484188`.

## Gate C

All five locked seeds `PASS`. Every seed improves train accuracy from 0.5 to
1.0; checkpoint selection is validation-only; checkpoint/state/data-order
hashes are independent. Checkpoint round-trip is `PASS` for 50/50 checkpoints.

## Numerical audit

The first full 10,000-case audit exposed one real tolerance failure:
`joint_column_permutation_logit_error = 1.4486367660992983e-08` at tolerance
`1e-08`, Seed 999, IID sample `iid_test-pair-0006-class-1`. It was traced to
floating-point addition-order amplification: cost/coupling permutation errors
were only `4.54747350886464e-13` / `6.67112198815545e-13`.

The failure was archived. Float64 column-set reduction now uses a canonical
coordinate-wise order, preserving the mathematical sum while removing input
permutation dependence. The audit program also exits nonzero if its structured
status is FAIL.

The complete rerun took 1,717.530206800002 seconds, returned code 0, and is
`PASS`:

- dtype: float64;
- compact valid submatrices: true;
- deterministic permutations: true;
- Sinkhorn iterations: 500;
- audited seed-sample cases: 10,000;
- failed samples: 0;
- joint-column logit maximum: `1.25652022120448e-09`;
- node-row logit maximum: `9.74448965962438e-10`;
- matching-cost swap maximum: `2.44027020812609e-12`;
- coupling/assignment transpose maximum: `1.46854750582293e-13`;
- row marginal maximum: `1.20681242776755e-13`;
- column marginal maximum: `1.15296661107323e-13`;
- total mass maximum: `5.55111512312578e-16`;
- NaN/Inf/non-negativity/root-swap final logit/root-swap probability
  violations: maximum 0.

## Report and notebooks

- `results/phase3/final_validation_report.md`
- `notebooks/Phase3_Research.ipynb`
- `notebooks/Phase3_Explained_FA.ipynb`

Report-to-CSV consistency: `PASS`, exactly 210 checked headlines
(10 models × 3 test splits × 7 metrics), zero failures.

Each notebook has 9 code cells, 9 executed cells, and zero error outputs after
clean-kernel execution. The Persian notebook explains methodology, model
identity, normalized Sinkhorn mass, all gates, controls, five-seed results,
IID/OOD splits, ablations, numerical audit, limitations, and reproduction.

## Commands and return codes

Major final commands:

- `final_v2_full_experiments`: return 0; 8,833.618699200004 seconds.
- `final_v2_gates`: return 0; 342.8702173999991 seconds.
- `final_v2_zero_shot_control`: return 0; 659.4936151000002 seconds.
- Original full-pipeline label controls: return 1; preserved failure.
- Corrected supervised-only label controls: return 0.
- `final_v2_split_audit`: return 0.
- `final_v2_scientific_claim_audit`: return 0.
- First numerical audit: structured FAIL despite wrapper return 0; rejected
  and archived.
- `final_v2_numerical_audit_rerun`: return 0; structured PASS.
- `final_v2_report` and `final_v2_report_postcommit`: return 0.
- `final_v2_notebooks`: return 0.
- `final_v2_import_checks`: return 0; all 22 active modules imported.
- `final_v2_repository_tests`: return 0; `7 passed in 12.04s`.
- `final_v2_final_checks_precommit`: return 0; `PASS`.
- `final_v2_postcommit_artifacts`: return 0; manifest and ZIP verification
  `PASS`.
- `final_v2_fresh_environment`: return 0; 290.103666300005 seconds; `PASS`.

All commands have JSON metadata plus stdout/stderr logs under
`results/phase3/logs/`. Failed commands and retry evidence are retained.

## Repository checks

Final active-source checks before the scientific commit:

- 22 active modules import.
- `pytest -q`: 7 passed.
- `src.final_checks`: `PASS`.
- Checkpoints: 50.
- Predictions: 250.
- Run records: 50.
- Per-seed rows: 250.
- Aggregate rows: 50.
- Maximum saved-prediction metric recomputation error:
  `3.4009236205001514e-08`.
- Notebooks: 9/9 executed cells each, zero errors.
- Staged files: 2,000.
- `git diff --cached --check`: return 0.
- Scientific commit: 2,000 files changed, 3,770,206 insertions, 1,459
  deletions; 1,942 additions, 31 preservation renames, 18 modifications, and
  9 active legacy deletions/replacements.

## Post-commit manifests and ZIP

Final post-commit artifacts (all `PASS`):

- `results/phase3/run_manifest.json`
- `results/phase3/final_artifact_manifest.json`
- `results/phase3/phase3_reproducible.zip`

The run manifest is authoritative for the self-reference-safe final handoff
HEAD and records:

- branch `codex-phase3-final`;
- the exact final `source_commit`;
- empty Git status;
- 50 checkpoints;
- 250 predictions.

The final artifact manifest contains SHA-256 for every included source and
evidence file. The corresponding ZIP includes exactly those entries plus the
artifact manifest. CRC, exact entry set, required files, extracted hashes, and
extracted smoke inference all pass. The exact final ZIP size and SHA-256 are
reported by the final verification result and can be recomputed directly from
`results/phase3/phase3_reproducible.zip`.

## Fresh-environment verification

`results/phase3/fresh_environment_validation.json`: `PASS`.

A genuinely new temporary venv was created, used, and removed. Every step
returned code 0:

1. venv creation;
2. clean pinned dependency installation;
3. ZIP CRC/extraction/2,002 manifest-hash verification;
4. dependency and active-source imports;
5. repository tests (`7 passed in 10.60s`);
6. deterministic split audit;
7. primary zero-shot smoke (accuracy 0.5);
8. checkpoint load/inference matching saved probabilities on 800 IID samples;
9. all 210 report metrics;
10. frozen evidence checks.

Step stdout/stderr files are under
`results/phase3/logs/fresh_environment_steps/`.

## Final Git Commit

Scientific evidence commit:

`8425f6c780ad611f2502dd7f3e57119a07f07ee0`

This commit contains the complete corrected scientific source and evidence.
The handoff metadata commits are intentionally separate because a file inside
a Git commit cannot contain that same commit's own cryptographic hash. The
post-handoff run manifest is the authoritative exact record of the final HEAD;
the report and this status record the immutable scientific evidence commit.

## Required artifact paths

- Audit verification:
  `results/phase3/independent_audit_verification.md`
- Final report:
  `results/phase3/final_validation_report.md`
- Aggregate metrics:
  `results/phase3/aggregate_metrics.csv`
- Per-seed metrics:
  `results/phase3/per_seed_metrics.csv`
- Zero-shot metrics:
  `results/phase3/zero_shot_metrics.csv`
- Valid label-signal metrics:
  `results/phase3/controls/supervised_only_label_signal_control_metrics.csv`
- Preserved confounded label-control metrics:
  `results/phase3/controls/full_pipeline_pretraining_confound_metrics.csv`
- Scientific/seed-independence audit:
  `results/phase3/scientific_claim_audit.json`
- Numerical audit:
  `results/phase3/numerical_audit.json`
- Split audit:
  `results/phase3/split_overlap_audit.json`
- Run manifest:
  `results/phase3/run_manifest.json`
- Artifact manifest:
  `results/phase3/final_artifact_manifest.json`
- Fresh-environment attestation:
  `results/phase3/fresh_environment_validation.json`
- Reproducible ZIP:
  `results/phase3/phase3_reproducible.zip`

## Remaining Scientific Limitations

- The corpus is synthetic and addresses a locked local incidence-star
  formulation, not arbitrary empirical hypergraphs or every hierarchy level
  in the source paper.
- Five seeds measure optimization variability on fixed synthetic splits, not
  uncertainty over an external population.
- Exact row-multiset/Hungarian structure separates formal labels by
  definition. The learned primary is meaningful because that separator is
  prohibited from its implementation, not because the mathematical task lacks
  an exact algorithm.
- The original full-pipeline invalid-label control cannot isolate label signal
  because self-supervised pretraining carries structural task information.
  Both its FAIL and the valid decomposed control are reported.
- The v1 test-exposure incident remains part of the development history.
  New unseen final-v2 tests repair final evaluation but cannot erase that
  chronology.
- Final-v2 perfect primary metrics support only the locked generator families;
  they are not a universal theorem or causal claim.
- Git/manifest self-reference is handled with a scientific commit followed by
  a metadata commit and post-commit manifests. The manifest, not a file inside
  the same commit, records the final handoff HEAD.

## Exact next action

None. All Phase 3 contract work is complete. The only remaining handoff action
is to paste this complete file inline in Codex chat between the required
markers.
