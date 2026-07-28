# Codex Phase 3 Status

Last updated: 2026-07-28 00:07:08 UTC

## Current continuity checkpoint (authoritative)

- The original full-pipeline label-signal command completed naturally:
  `python -m src.logged_command final_v2_label_signal_controls {python} -m
  src.run_label_signal_controls`.
- It ran from `2026-07-27T20:10:39.279384+00:00` through
  `2026-07-27T21:33:42.327296+00:00` (4,983.0988211 seconds), completed all
  10 mode/seed runs, and returned code 1 because the locked acceptance ceiling
  was not met.
- Exact failure: random-label mean accuracy was `0.665` on IID and `0.6915`
  on OOD-template, above the unchanged `0.65` maximum. OOD-size mean accuracy
  did not fail. Shuffled labels did not fail the mean-accuracy ceiling.
- Root-cause evidence: the full pipeline includes label-free self-supervised
  structural pretraining. Random-label seed 777 selected epoch 0 (the
  post-pretraining state) and reached IID/OOD-template accuracy
  `0.9675/0.985`. Thus the completed full-pipeline run is not a valid
  label-only negative control: it contains genuine structural supervision
  independent of the corrupted labels.
- The complete failed evidence was preserved before any protocol correction
  under
  `results/phase3_archive/20260727T213558Z_label_control_protocol_correction/`
  (28 files/directories, 715,073 bytes). Active preserved copies are
  `results/phase3/controls/full_pipeline_pretraining_confound_controls.json`
  and
  `results/phase3/controls/full_pipeline_pretraining_confound_metrics.csv`.
- Files modified to implement the scientifically valid decomposed control:
  `src/phase3_experiments.py` adds a default-preserving `config_override`
  argument to `train_model`; `src/phase3_controls.py` disables only
  self-supervised pretraining for the supervised-label-signal control, uses
  new checkpoint/record names, and explicitly links the preserved confounded
  result. `python -m py_compile` for the three affected modules returned 0.
- No process is currently running at this checkpoint.
- Remaining stages: execute the corrected supervised-only shuffled/random
  controls; split and scientific-claim audits; full float64 numerical audit;
  report and clean notebook execution; repository/final consistency checks;
  final clean commit, manifests, ZIP, fresh-environment verification, final
  hash/ZIP reconciliation, and inline status handoff.
- Exact next action: launch the logged, resumable
  `final_v2_supervised_only_label_signal_controls` command and monitor it
  without interruption until all ten new records are durable.

### Label-signal correction completion

- The exact next action above completed naturally. Logged command
  `python -m src.logged_command
  final_v2_supervised_only_label_signal_controls {python} -m
  src.run_label_signal_controls` ran from
  `2026-07-27T21:37:30.211156+00:00` through
  `2026-07-27T23:02:58.919869+00:00`, took `5128.729064899999` seconds,
  returned code 0, and is `PASS`.
- All 10 new mode/seed records and 10 new checkpoints are durable under
  `results/phase3/controls/supervised_only_run_records/` and
  `results/phase3/controls/supervised_only_*_labels_seed*.pt`. Every record
  states that self-supervised cost pretraining was not performed.
- Exact five-seed test aggregates (mean ± sample standard deviation):
  shuffled-label IID accuracy `0.49425 ± 0.020128`, OOD-template
  `0.50325 ± 0.031962`, OOD-size `0.5 ± 0`; random-label IID
  `0.48975 ± 0.090357`, OOD-template `0.49425 ± 0.116641`, OOD-size
  `0.5 ± 0`.
- Random-label macro F1 means are `0.486514` (IID), `0.490928`
  (OOD-template), and `0.333333` (OOD-size). Shuffled-label macro F1 means
  are `0.489143`, `0.494272`, and `0.333333`, respectively.
- The unchanged `0.65` maximum is satisfied on every aggregate test split;
  `results/phase3/controls/label_signal_controls.json` reports 10 completed
  runs, zero failures, protocol `supervised_label_signal_only`, and `PASS`.
- `src/reporting.py`, `src/rebuild_notebooks.py`, and
  `src/scientific_claim_audit.py` now distinguish the valid supervised-only
  control from the preserved full-pipeline/pretraining-confounded FAIL. The
  latter remains a disclosed limitation and is never relabeled as PASS.
- No process is currently running.
- Remaining stages: split and scientific-claim audits; full float64 numerical
  audit; report and clean notebook execution; repository/final consistency
  checks; final clean commit, manifests, ZIP, fresh-environment verification,
  final hash/ZIP reconciliation, and inline status handoff.
- Exact next action: run logged split-integrity and scientific-claim audits
  against the completed corrected controls, then launch the numerical audit
  alone if both pass.

### Split and scientific-claim audit completion

- Logged command `python -m src.logged_command final_v2_split_audit
  {python} -m src.audit_splits` returned code 0 and `PASS`.
  `results/phase3/split_overlap_audit.json` reports zero problems across all
  five required splits and zero sample-ID, matched-pair, base-hash, or
  invariant-structure overlap in every prohibited pair.
- Split sizes and class counts are train `800 (400/400)`, validation
  `800 (400/400)`, IID `800 (400/400)`, OOD-template `800 (400/400)`, and
  OOD-size `400 (200/200)`. Deterministic generation/loading, shapes, labels,
  masks, padding, pair structure, margins, and hashes all pass.
- Logged command `python -m src.logged_command
  final_v2_scientific_claim_audit {python} -m
  src.scientific_claim_audit` returned code 0 and `PASS`.
- Exact scientific evidence: primary zero-shot accuracy min/max `0.5/0.5`
  over 25 rows; mean zero-shot ROC-AUC `0.5458902499999999`; mean train
  accuracy improved `0.5 -> 1.0`; unique initial/final/order hashes
  `5/5/5`; exact oracle mean test accuracy `1.0`; no prohibited model source
  or forward-input hits; final tests were not used for selection; valid
  supervised-only invalid-label controls passed; the original full-pipeline
  FAIL and its two exact failures remain preserved and disclosed.
- Files created/updated:
  `results/phase3/split_overlap_audit.json`,
  `results/phase3/splits_hashes.json`,
  `results/phase3/scientific_claim_audit.json`, and the corresponding
  `final_v2_split_audit*` and `final_v2_scientific_claim_audit*` logs.
- No process is currently running.
- Remaining stages: full float64 numerical audit; report and clean notebook
  execution; repository/final consistency checks; final clean commit,
  manifests, ZIP, fresh-environment verification, final hash/ZIP
  reconciliation, and inline status handoff.
- Exact next action: launch only
  `python -m src.logged_command final_v2_numerical_audit {python} -m
  src.audit_model_numerics`, monitor it without interruption, and inspect
  every mandatory quantity if it returns nonzero.

### Numerical-audit failure diagnosis and repair

- The first full final-v2 numerical command ran from
  `2026-07-27T23:05:06.952303+00:00` to
  `2026-07-27T23:33:05.195511+00:00` in `1678.2487435000003` seconds.
  The wrapper returned 0, but `results/phase3/numerical_audit.json` correctly
  reported `FAIL`; the wrapper status was therefore not accepted as evidence.
- The audit covered 10,000 seed-sample cases in float64 with compact
  submatrices and 500 Sinkhorn iterations. Nineteen mandatory quantities had
  zero failures. One case failed
  `joint_column_permutation_logit_error`: maximum
  `1.4486367660992983e-08` at tolerance `1e-08`, seed 999, IID sample
  `iid_test-pair-0006-class-1`. Its cost and coupling permutation errors were
  only `4.54747350886464e-13` and `6.67112198815545e-13`, identifying
  floating-point reduction-order amplification rather than structural
  non-invariance.
- The complete failing artifact and six command/background logs were
  preserved before repair at
  `results/phase3_archive/20260727T233427Z_numerical_audit_failure/`.
- `src/differentiable_matching.py` now uses a canonical coordinate-wise
  ordering for the float64 set reduction over columns. This preserves the
  mathematical sum and removes dependence on the input permutation's
  floating-point addition order; training/inference float32 results are not
  changed.
- `src/audit_model_numerics.py` now exits nonzero whenever its structured
  report is `FAIL`, preventing a future wrapper/semantic-status mismatch.
- An isolated exact reproduction of seed 999 sample
  `iid_test-pair-0006-class-1` at 500 iterations now gives zero for the
  column-permutation cost, coupling, logit, and probability errors; all 20
  measured quantities are within tolerance.
- No process is currently running.
- Remaining stages are unchanged.
- Exact next action: rerun the complete logged 10,000-case numerical audit
  from scratch, without changing its 500 iterations or `1e-08` tolerances.

### Numerical-audit completion

- The complete rerun
  `python -m src.logged_command final_v2_numerical_audit_rerun {python}
  -m src.audit_model_numerics` ran from
  `2026-07-27T23:35:17.694326+00:00` to
  `2026-07-28T00:03:55.219471+00:00`, took `1717.530206800002` seconds,
  returned code 0, and both the command log and structured artifact are
  `PASS`.
- `results/phase3/numerical_audit.json` covers 10,000 seed-sample cases,
  float64 compact valid submatrices, deterministic row/column permutations,
  explicit uniform marginals, and 500 Sinkhorn iterations. Failed sample
  count is zero across every mandatory quantity.
- Exact maxima: joint-column permutation logit
  `1.25652022120448e-09`; node-row permutation logit
  `9.74448965962438e-10`; matching-cost swap
  `2.44027020812609e-12`; coupling/assignment transpose
  `1.46854750582293e-13`; row/column marginal residuals
  `1.20681242776755e-13` and `1.15296661107323e-13`; total coupling mass
  error `5.55111512312578e-16`. NaN, Inf, negativity, final root-swap logit,
  and final root-swap probability violations all have maximum zero.
- Files updated: `results/phase3/numerical_audit.json` plus
  `results/phase3/logs/final_v2_numerical_audit_rerun.json`, stdout, stderr,
  and background logs.
- No process is currently running.
- Remaining stages: report and clean notebook execution; repository/final
  consistency checks; final clean commit, manifests, ZIP, fresh-environment
  verification, final hash/ZIP reconciliation, and inline status handoff.
- Exact next action: generate the technical report, run its exact
  report-to-CSV consistency check, rebuild both required notebooks, and
  execute every notebook cell from clean kernels.

### Report, notebook, and active-source verification completion

- Logged report command `final_v2_report` returned code 0. It generated
  `results/phase3/final_validation_report.md`; the embedded headline
  consistency check is `PASS` for exactly 210 values
  (10 models × 3 test splits × 7 metrics) with zero failures.
- Logged notebook command `final_v2_notebooks` returned code 0.
  `notebooks/Phase3_Research.ipynb` and
  `notebooks/Phase3_Explained_FA.ipynb` each have 9 code cells, 9 executed
  cells, and zero error outputs after clean-kernel execution. The Windows ZMQ
  selector-thread runtime warning was non-fatal and did not appear in a
  notebook error output.
- Logged final import command returned code 0 and imported all 22 active
  `src` modules.
- Logged final repository test command returned code 0:
  `7 passed in 12.04s`.
- Logged pre-commit final-check command returned code 0 and `PASS`: 50
  checkpoints, 250 predictions, 50 run records, 250 per-seed metric rows,
  50 aggregate metric rows, both notebooks clean, all five required seeds,
  and maximum saved-prediction metric recomputation error
  `3.4009236205001514e-08`.
- Files created/updated include the final report, both notebooks,
  `results/phase3/report_consistency.json`, and all
  `final_v2_report*`, `final_v2_notebooks*`, `final_v2_import_checks*`,
  `final_v2_repository_tests*`, and
  `final_v2_final_checks_precommit*` logs.
- Git reconciliation: branch is `codex-phase3-final`; Phase 3 currently has
  1,055 files totaling 42,985,927 bytes. The large tracked/untracked diff is
  the preserved takeover replacement of legacy active files plus the complete
  corrected evidence; archives and ZIPs remain ignored. Post-commit manifests
  are explicitly ignored so generating them from the final commit will not
  falsely dirty the committed tree.
- No process is currently running.
- Remaining stages: scientific evidence commit; post-commit manifests and
  reproducible ZIP; fresh temporary-environment installation and extracted-ZIP
  verification; final handoff metadata commit; final manifest/hash/ZIP
  regeneration and verification; inline status handoff.
- Exact next action: stage the complete reconciled scientific source,
  configurations, splits, checkpoints, predictions, metrics, controls,
  audits, logs, notebooks, report, and status; inspect the staged inventory;
  create the scientific evidence commit without including archives, caches,
  virtual environments, old ZIPs, or pre-commit manifests.

## Active correction-state override

This section supersedes historical deterministic-model statements retained
later in this continuity file. Those statements document completed legacy
stages; they are not current scientific claims.

- Current designated primary: `LearnedSinkhornMatcher`
  (`primary_learned`), selected from candidate v5 using train/validation
  evidence.
- Current benchmark: `learned_benchmark_v2_final`.
- The deterministic fixed-L1 model is now
  `DeterministicDifferentiableMatching`; it and exact Hungarian are explicitly
  algorithmic/oracle controls, not learned results.
- Final-v2 seed 42 was adopted from the validation-selected candidate without
  retraining and has a valid run record, checkpoint, and five split prediction
  files.
- The remaining final model-seed records are being produced by the one existing
  resumable experiment process; no completed run has been repeated.
- Final-v2 `LearnedSinkhornMatcher` seed 100 completed at
  2026-07-27 17:54:56 UTC. Its checkpoint and all five prediction hashes
  match the durable record; Gate C is `PASS`, zero-shot/train accuracy is
  `0.5/1.0`, train macro F1 is `1.0`, selection epoch is `96`, final
  temperature is `0.023092152550816536`, and IID/OOD-template/OOD-size
  accuracy, balanced accuracy, macro F1, and ROC-AUC are all `1.0`.
  OOD-size loss is `0.0011631047885656775`. Current matrix progress is
  at least 2/50 model-seed records.
- Final-v2 `LearnedSinkhornMatcher` seed 2026 completed at
  2026-07-27 18:17:54 UTC. Its checkpoint and five prediction hashes match;
  Gate C is `PASS`, zero-shot/train accuracy is `0.5/1.0`, train macro F1 is
  `1.0`, selection epoch is `135`, final temperature is
  `0.016871949657797813`, and IID/OOD-template/OOD-size accuracy, balanced
  accuracy, macro F1, and ROC-AUC are all `1.0`. OOD-size loss is
  `0.0002814287769662821`. Current matrix progress is 3/50; seed 777 is
  now complete.
- Final-v2 `LearnedSinkhornMatcher` seed 777 completed at
  2026-07-27 18:32:08 UTC. Its checkpoint and five prediction hashes match;
  Gate C is `PASS`, zero-shot/train accuracy is `0.5/1.0`, train macro F1 is
  `1.0`, selection epoch is `160`, final temperature is
  `0.013797296211123466`, and IID/OOD-template/OOD-size accuracy, balanced
  accuracy, macro F1, and ROC-AUC are all `1.0`. OOD-size loss is
  `0.003969094308256258`. Current matrix progress is 4/50; seed 999 is
  now complete.
- Final-v2 `LearnedSinkhornMatcher` seed 999 completed at
  2026-07-27 18:44:11 UTC. Its checkpoint and five prediction hashes match;
  Gate C is `PASS`, zero-shot/train accuracy is `0.5/1.0`, train macro F1 is
  `1.0`, selection epoch is `135`, final temperature is
  `0.016871949657797813`, and IID/OOD-template/OOD-size accuracy, balanced
  accuracy, macro F1, and ROC-AUC are all `1.0`. OOD-size loss is
  `0.006425756087883351`.
- All five seeds are complete for the primary, `MLP_Baseline`,
  `NestedDeepSets`, `BipartiteEncoder`,
  `DeterministicDifferentiableMatching`, `ExactHungarianOracle`,
  `FixedL1ResidualSinkhorn`, and
  `FixedL1StraightThroughHungarian`. Matrix progress is 40/50; the unchanged
  process subsequently completed all five `LearnedNoSinkhorn` seeds.
- Matrix progress is 45/50. The unchanged process is running
  `LearnedNoColumnContext` seed 42; only that ablation's five locked seeds
  remain before final aggregation, Gate C emission, and checkpoint round-trip
  verification.
- Corrected final-v2 experiment matrix completed naturally at
  2026-07-27 20:01:56 UTC. Logged command
  `python -m src.logged_command final_v2_full_experiments {python} -m
  src.run_final_experiments` returned code 0 after 8,833.619 seconds.
- Final inventory: 50/50 model-seed records, 50 checkpoints, 250 prediction
  files, 250 per-seed metric rows, and 50 aggregate model/split rows.
  Checkpoint round-trip is `PASS` for 50/50 checkpoints with zero failures.
- Gate C is `PASS` for all five primary seeds. Every seed improved from
  zero-shot train accuracy `0.5` to train accuracy, balanced accuracy, macro
  F1, and both class recalls `1.0`. The five checkpoint hashes and five
  first-epoch order hashes are distinct.
- Primary five-seed aggregate metrics: IID loss
  `8.076873781895485e-06 ± 4.556830588707213e-06`,
  OOD-template loss `8.280781531712364e-06 ±
  4.206681827433285e-06`, and OOD-size loss
  `0.004025965099677455 ± 0.003397260359064635`; accuracy, balanced
  accuracy, macro F1, and ROC-AUC are `1.0 ± 0.0` on all three splits.
- `LearnedNoSinkhorn` demonstrates the main ablation effect: OOD-size
  accuracy `0.5685 ± 0.00993730345717586` and macro F1
  `0.4695740223473729 ± 0.017862316678784573`, versus primary `1.0`.
  `LearnedNoColumnContext` retains OOD-size accuracy
  `0.9985 ± 0.003354101966249658`.
- Current running process: none. Exact next action: run final-v2 Gate A/B,
  preserving the completed matrix unchanged.
- Final-v2 Gate A/B logged command returned code 0 after 342.870 seconds.
  Gate A is `PASS` on 3,600 samples with oracle accuracy `1.0`, 500 float64
  iterations, total-mass error `2.220446049250313e-16`, maximum row residual
  `7.353839759360881e-14`, maximum column residual
  `7.296940829348841e-14`, and tolerance `1e-8`.
- Gate B is `PASS` for all five locked seeds under the unchanged accuracy and
  macro-F1 threshold `0.99` and per-class F1 threshold `0.98`. Every seed has
  accuracy, balanced accuracy, macro F1, and both class F1 values `1.0`, with
  no failed IDs. Losses are seed 42 `0.0005092812822928834`, seed 100
  `0.0004148976468337607`, seed 2026 `0.0003978203914182482`, seed 777
  `0.0004778715588270414`, and seed 999 `0.0003934729328484188`.
- Current running process: none. Exact next action: run the resumable
  shuffled-label and random-label controls for all five seeds.
- The v2 lock chronology was corrected at 17:43 UTC: candidate development
  used preserved v1 data, and v2 was frozen before its first final test
  evaluation, not before the first development optimization. The original
  policy wording remains preserved as written, while
  `results/phase3/learned_benchmark_lock.json` now records the exact applied
  chronology.
- Mandatory corrected-learning work still pending: remaining experiment
  records, label-signal controls, gates, scientific/numerical audits, corrected
  report and notebooks, clean commit, post-commit manifests/ZIP, and genuine
  fresh-environment verification. Therefore the current overall status is
  **IN PROGRESS**, not PASS.
- Corrected reporting and continuity tooling prepared while the experiment
  runs: `src/reporting.py`, `src/rebuild_notebooks.py`,
  `src/execute_notebooks.py`, `src/run_gates.py`,
  `src/run_label_signal_controls.py`, and
  `src/fresh_environment_validation.py`. The label-signal control runner now
  writes and hash-validates one durable model-seed record after each completed
  run so interrupted controls resume without repeating completed work.
- Compile-only checks for all newly prepared scripts returned code 0. No gate,
  control, notebook, numerical-audit, or fresh-environment command has been
  launched prematurely.

## Repository state

- Workspace: `hypergraph_equivalence`
- Branch: `codex-phase3-final`
- Baseline commit: `119343d0ad758f8c4a78d3831e8727d37255a3be`; the supplied workspace contained no `.git` directory or recoverable Git history, so Git was initialized in place.
- Legacy archive: `results/phase3_archive/20260727_102315`
- Archive inventory: 20 files, 1,789,035 bytes.
- Independent-audit correction archive:
  `results/phase3_archive/20260727T160053Z_independent_audit_corrections`
  (342 files, 31,049,172 bytes). The archived final ZIP is 4,143,893 bytes
  with SHA-256
  `03319a79b01d13576b151398e4b841540b70155adf74ecd21def0fa24d3536c3`.
- Current scientific status: **REOPENED FOR CORRECTION**. The previous
  deterministic packaging contract passed, but the former primary-model
  learning interpretation failed independent verification.

## Authoritative project specification discovered

- `AGENTS.md` is the permanent Phase 3 completion contract.
- `CODEX_PHASE3_HANDOFF.md` is the active takeover and execution contract.
- Existing Antigravity code, outputs, metrics, checkpoints, reports, notebooks, and PASS claims are unverified legacy evidence.
- Designated primary model: `MatchingNetwork`, the fixed-L1 Sinkhorn matching model. This is stated by `src/config_manager.py` and independently by the inherited gate script.
- Legacy numerically best model: `LearnedCostSinkhorn` in the archived CSVs, but the result is unverified and was misreported in the legacy narrative.
- Required baselines: `MLP_Baseline`, `NestedDeepSets`, `BipartiteEncoder`.
- Required ablations: `LearnedCostSinkhorn`, `StraightThroughHungarian`, `NoSinkhorn`.
- Gate A: exact multiset/Hungarian oracle verification plus mathematically correct normalized Sinkhorn evidence.
- Gate B: balanced tiny-batch capability test with the non-weakened `0.99` accuracy and macro-F1 rule across all five seeds; inherited source also specifies per-class F1 at least `0.98`.
- Gate C: final primary-model training requirements plus validation-only checkpoint selection and complete per-seed evaluation artifacts.
- Default locked seeds, unless superseded by an authoritative repository specification: `42`, `100`, `2026`, `777`, `999`.
- Required splits: train, validation, IID test, OOD-template test, and OOD-size test.

## Files inspected

- `AGENTS.md` (read completely)
- `CODEX_PHASE3_HANDOFF.md` (read completely)
- Complete non-`.git` path inventory via `rg --files -uu -g '!.git/**'`
- Workspace and parent directory listings
- Git metadata search in the workspace and all parents
- Initial `results/phase3` archive inventory
- All 27 files under `src/`
- All five repository test files
- All seven locked split JSON files and their shapes, label counts, and hashes
- All root notebooks, including cell sources, execution state, and cached errors
- Eidi et al. PDF: extracted text plus visual inspection of all 28 rendered pages
- Legacy archive `20260727_080146` report, metrics, audits, predictions/checkpoints manifest, and artifact manifest

## Changes made

- Initialized Git because the supplied workspace had no `.git` metadata.
- Created branch `codex-phase3-final`.
- Archived current `results/phase3` contents and root Phase 3 notebooks/logs under `results/phase3_archive/20260727_102315`.
- Created this persistent status file.
- Rendered and visually inspected the source paper under temporary `tmp/pdfs/eidi/`.
- Established the only usable Python environment at `C:/Users/esmae/Documents/Educx Kurs machine lerning/aufgabe/.venv/Scripts/python.exe` (Python 3.11.15, PyTorch 2.12.1 CPU).
- Added `.gitignore` for caches, temporary renders, ZIPs, and preserved archives.
- Committed the untouched takeover baseline on `codex-phase3-final`.
- Added the authoritative locked experiment specification (`phase3_spec.json`) and model configurations.
- Replaced nondeterministic/leaking split generation with deterministic, stable-ID generation that rejects invariant structure and base-graph overlap.
- Replaced the primary/ablation matching implementation with normalized-mass, explicitly symmetric Sinkhorn/Hungarian implementations and added required invariant baselines.
- Added epsilon-scaled log-domain Sinkhorn updates plus a transpose-equivariant Dykstra transport-polytope finish for the 500-iteration float64 verification path.
- Replaced the split audit with deterministic regeneration, oracle, tensor, hash, overlap, duplicate, and padding checks.
- Added a pinned project dependency declaration.
- Installed the declared `pytest==8.4.1` into the verified project environment and reconciled all dependency pins to installed versions.
- Regenerated and froze all five locked splits after rejecting ambiguous WL-digest collisions.
- Implemented the unified Gate A/B/C, seven-model/five-seed training, prediction, checkpoint, metric aggregation, and round-trip verification pipeline.
- Implemented the complete five-seed/all-test-sample compact float64 numerical audit, including root swap, assignment transpose, node-row, and joint-column permutations.
- Recorded the new final-handoff rule: preserve the eight-stage sequence, reconcile this file after each remaining major stage, and paste its complete contents inline at completion.
- Moved obsolete Antigravity runners, duplicate generators, cached notebooks, and chat-specific files into `docs/legacy/antigravity/`; they remain preserved and tracked but are no longer imported as active source.
- Added portable logging, frozen-evidence validation, report consistency, manifest hash verification, and extracted-ZIP smoke testing.
- Independently verified every claim in `INDEPENDENT_PHASE3_AUDIT.md`; the
  complete claim classification is in
  `results/phase3/independent_audit_verification.md`.
- Confirmed that the former primary is perfect before training because it
  directly uses fixed normalized L1 cost and a manually initialized
  `[-2,+2]`, zero-bias classifier.
- Confirmed exact tensor-state identity across all five former-primary
  checkpoints and byte-identical former-primary predictions per split.
- Confirmed no prohibited metadata or label enters the model forward path.
- Created the required pre-correction archive without overwriting or rerunning
  any completed deterministic experiment.
- Materialized the old corrected deterministic evidence separately at
  `results/deterministic_benchmark_v2/phase3` and its splits at
  `data/deterministic_benchmark_v2/splits`.
- Implemented the randomly initialized `LearnedSinkhornMatcher`; its forward
  path contains a learned symmetric column-pair encoder, learned pair-score/cost
  head, normalized Sinkhorn transport, learned transport summaries, and a
  learned classifier. It never computes fixed L1 or consumes Hungarian output.
- Reclassified the fixed solvers as
  `DeterministicDifferentiableMatching`, `ExactHungarianOracle`,
  `FixedL1ResidualSinkhorn`, and `FixedL1StraightThroughHungarian`.
- Added `LearnedNoSinkhorn` and `LearnedNoColumnContext` ablations while
  retaining the three original learned summary/set/bipartite baselines.
- Created `learned_benchmark_v1` with variable-switch hard negatives and
  preserved it as development-only after an early failed-candidate evaluator
  exposed its test metrics.
- Selected learned candidate v5 using only train/validation evidence, then
  generated `learned_benchmark_v2_final` with byte-identical train/validation
  sample data and entirely new IID/OOD test samples at unseen offsets.
- Locked the final v2 split, spec, model, config, and training hashes before
  final test evaluation.

## Commands executed and return codes

- `git status --short --branch; git diff --stat; git diff --cached --stat` — return code 1; confirmed no Git repository existed.
- `Get-Content AGENTS.md -Raw; Get-Content CODEX_PHASE3_HANDOFF.md -Raw` — return code 0.
- `rg --files -uu -g '!.git/**'` — return code 0.
- Parent/workspace directory and `.git` searches — return code 0, except one PowerShell formatting pipeline parse error (return code 1); the corrected directory searches confirmed no `.git`.
- `git init -b codex-phase3-final; git branch --show-current; git status ...` — return code 0.
- Timestamped archive creation and inventory — return code 0.
- Legacy source, test, notebook, report, manifest, metric, and audit inspection commands — return code 0.
- Source-paper PDF rendering/extraction — return code 0 after bypassing a broken wrapper and invoking Poppler directly.
- Baseline test scripts — three pass, two fail with missing `generate_binary_dataset`/`build_hypergraph_from_bipartite` imports.
- Split profiling — return code 0.
- Invariant leakage check — return code 0; train/validation/IID share 793 WL-invariant structures, and each contains seven within-split WL duplicates.
- Oracle/shape validation — return code 0; all labels and matrix shapes match the exact oracles.
- `python -m compileall -q src` — return code 0.
- Seven-model forward smoke test — one compatibility failure from unavailable `torch.flatnonzero`; repaired with `torch.nonzero(...).flatten()`, then return code 0.
- Float64 500-iteration Sinkhorn smoke audit on 100 generated samples — return code 0; maximum marginal residual `6.41e-14`, mass error `2.22e-16`, non-negativity violation `0`.
- Final deterministic split regeneration — return code 0; file SHA-256 values: train `68e7feabb56592cb5f3ccdf588f8d3eeba38b7815d94f9ab8287cad2071b7a7e`, validation `1a1d22b356ea82a1d9f7b983e2f9fc54813e9753a5da2dc63ad8db6b60418bdb`, IID `a91f93999714ebf409611fc09bcc5c582744b7afac7295e5bbba20dcd3dfb8d9`, OOD-template `c1fad4d214ce469c3fc66d47ec0b6d39ed44c25d9e24e90e87733dc043e537e6`, OOD-size `059f9b79259442640554907bae316af6b07a349cfa3cdd67638bd8f886616ace`.
- Complete split audit — return code 0; `PASS`, 0 problems, sizes 800/800/800/800/400, balanced classes, unique invariant structures 800/800/800/800/400, and zero sample/pair/base/invariant overlap for all prohibited split pairs.
- Gate A — return code 0; `PASS` on 3,600 samples, oracle accuracy `1.0`, maximum row residual `6.147859998861804e-15`, maximum column residual `5.2735593669694936e-15`, 500 iterations, tolerance `1e-8`.
- Gate B — return code 0; `PASS` for seeds 42, 100, 2026, 777, and 999. Every seed achieved accuracy `1.0` and macro F1 `1.0` on 64 balanced samples under the unchanged `0.99` thresholds.
- Full experiment command was launched at 12:53 local time. The shell tool returned code `124` after its two-hour observation window, but process inspection confirmed that the original experiment process remains active (`PID 4280`, compute child `PID 28804`). It has not been cancelled, restarted, or repeated.
- Live stage-5 inventory at 14:55 local time: 28/35 checkpoints and 140/175 prediction files complete. Completed runs cover all five seeds for MLP_Baseline, NestedDeepSets, BipartiteEncoder, MatchingNetwork, and LearnedCostSinkhorn, plus seeds 42, 100, and 2026 for StraightThroughHungarian.
- The original full-experiment process exited naturally at 15:05 local time and completed all 35 model/seed runs, 175 prediction files, 175 per-seed metric rows, and 35 aggregate model/split rows. No completed training was restarted.
- Gate C — `PASS` for all five seeds. MatchingNetwork training accuracy and macro F1 are `1.0` for seeds 42, 100, 2026, 777, and 999; every checkpoint selection epoch is 160 under validation macro F1 followed by validation loss.
- Checkpoint save/load round trip — return code 0; `PASS` for all 35 checkpoints with zero failures.
- Primary MatchingNetwork test metrics: IID accuracy/macro F1/ROC-AUC `1.0/1.0/1.0`; OOD-template `1.0/1.0/1.0`; OOD-size `1.0/1.0/1.0`; all standard deviations `0.0`, five seeds each.
- LearnedCostSinkhorn and StraightThroughHungarian also achieve accuracy/macro F1/ROC-AUC `1.0/1.0/1.0` on all three test splits. This tie does not change primary-model identity.
- NoSinkhorn accuracy is IID `0.6025 ± 0.140690`, OOD-template `0.6355 ± 0.185751`, and OOD-size `0.5175 ± 0.027386`. The three required baselines remain at accuracy `0.5`, as expected for margin-only summaries on matched nuisance statistics.
- First complete float64 audit run — return code 0 but scientific status `FAIL`, correctly preserved in the original command output. It audited all 10,000 seed–sample cases and found 1,653 samples with at least one failure. The gauge-fixed Newton finish left maximum row/column residuals `1.7478114698719904e-4`/`1.2801104745845115e-4`, coupling-transpose error `2.0661321823547418e-4`, and row-permutation logit error `1.0045240724384286e-2`.
- Root cause: the Newton balancing finish was ill-conditioned near hard assignments and its fixed gauge was not transpose-equivariant. Acceptance thresholds and tested quantities were not changed.
- Repair: replaced the Newton finish with Dykstra projection onto the nonnegative uniform-marginal transport polytope and averaged direct/transposed projection paths. Targeted reruns of all four worst samples now have coupling-transpose errors at most `5.55e-17`, row residual at most `8.18e-14`, column residual at most `8.67e-14`, and matching-cost swap error at most `3.12e-17`.
- Repaired full numerical audit — return code 0; `PASS` with 0 failed samples across 10,000 seed–sample cases at float64 and 500 iterations.
- Numerical maxima: cost/coupling finite violations `0`; non-negativity violation `0`; cost-transpose `0`; coupling/assignment-transpose `7.252086444224297e-15`; matching-cost swap `2.914335439641036e-16`; equality-score swap `1.1102230246251565e-15`; final logit/probability swap `0`; total mass `4.440892098500626e-16`; row marginal `1.0171030684347215e-13`; column marginal `1.0551282070281331e-13`; row-permutation coupling `9.096889908022376e-15`, logit `5.901945598907332e-12`, probability `1.0302869668521453e-13`; joint-column cost/coupling/logit/probability `0`.
- Initial replacement test collection failed because pytest discovered the legacy `scratch/test_sinkhorn.py`; resolved by locking `testpaths = tests` in `pytest.ini` while preserving the scratch file as legacy evidence.
- The first locked test run then exposed `StraightThroughHungarian` row-order dependence under optimal-assignment ties. Its equality feature was repaired to depend on invariant optimal total cost rather than an arbitrary tie-broken assignment. All five contract tests now pass.
- Before replacing affected evidence, archived the five old StraightThroughHungarian checkpoints, 25 predictions, and metric CSVs under `results/phase3_archive/20260727_161900_straight_through_tie_fix`.
- Reran only the five affected StraightThroughHungarian seeds; the other 30 model/seed runs were not repeated. The repaired ablation retains accuracy and macro F1 `1.0 ± 0.0` on IID, OOD-template, and OOD-size.
- Final checkpoint round trip — return code 0; `PASS` for all 35 repaired checkpoints.
- Final report generation and automated consistency check — return code 0; `PASS` across 126 model/split/metric headline records with exact model, role, mean, standard deviation, and five-seed count agreement.
- Clean notebook execution — return code 0 for both notebooks. `Phase3_Research.ipynb` and `Phase3_Explained_FA.ipynb` each have 7/7 code cells executed and zero error outputs.
- The technical report follows the technical-report workflow: answer-first summary, model comparison, definitions, methods, gates, robustness, limitations, and evidence-backed next questions.
- Final active-source import log — return code 0; all 13 active modules import.
- Final active-source tests — return code 0; 5/5 tests pass.
- Frozen-evidence metric validation — first return code 1 because CSV round-trip differences exceeded an unnecessarily strict `1e-10` metric tolerance; preserved that failed log. Corrected metric-file tolerance `1e-7` is below float32 decimal serialization precision and does not alter the `1e-8` structural threshold. Retry return code 0 with maximum recomputation error `3.143554050399899e-8`.
- Logged gate, numerical-audit, report-consistency, notebook-execution, split-integrity, manifest, and ZIP validations — all final return codes 0.
- First full manifest/ZIP packaging pass — `PASS`: 330 hashed evidence entries; 331 ZIP entries; all paths present; all hashes recomputed; clean extraction succeeded; extracted imports/data/model smoke test succeeded.
- Final artifact paths: report `results/phase3/final_validation_report.md`; notebooks `notebooks/Phase3_Research.ipynb` and `notebooks/Phase3_Explained_FA.ipynb`; checkpoints `results/phase3/checkpoints/`; predictions `results/phase3/predictions/`; manifests `results/phase3/run_manifest.json` and `results/phase3/final_artifact_manifest.json`; ZIP `results/phase3/phase3_reproducible.zip`.
- Independent ZIP verification — return code 0; SHA-256
  `03319a79b01d13576b151398e4b841540b70155adf74ecd21def0fa24d3536c3`,
  337 entries, `testzip=None`, 336/336 manifest hashes and sizes correct,
  with the manifest itself as the only unlisted ZIP entry.
- Independent live and extracted-ZIP tests — return code 0 twice; 5/5 tests
  pass in each location.
- Independent live and extracted-ZIP `python -m src.final_checks` — return
  code 0 twice; 35 checkpoints, 175 predictions, 175 per-seed rows, 35
  aggregate rows, and maximum prediction-metric recomputation error
  `3.143554050399899e-08`.
- Independent notebook inspection — return code 0; each notebook has 7/7
  code cells executed, execution counts 1 through 7, and zero error outputs.
- Independent exact-oracle and dataset-difficulty audit — return code 0;
  0 label disagreements over 3,600 samples. Positive Hungarian cost is always
  0. Negative cost is exactly 4 on train, validation, IID, and OOD-size, and
  belongs to `{4,6,8,10,12}` on OOD-template.
- Independent zero-training evaluation — return code 0; all five seeds of
  `MatchingNetwork`, `LearnedCostSinkhorn`, and
  `StraightThroughHungarian` have accuracy 1.0 on all five splits. The other
  four model types have accuracy 0.5 on all five splits.
- Independent checkpoint/prediction comparison — return code 0;
  former-primary tensors are exactly identical across all five seeds and each
  split has one unique prediction-file hash. Old `LearnedCostSinkhorn` retains
  five distinct learned tensor states and prediction hashes.
- Pre-correction preservation command — return code 0; archived 342 files and
  31,049,172 bytes at
  `results/phase3_archive/20260727T160053Z_independent_audit_corrections`.
- Deterministic-version copy — return code 0; copied 275 result files and seven
  split files to the explicit `deterministic_benchmark_v2` locations.
- Learned-model compile/shape/invariance smoke — return code 0 for all 10
  configured models; learned-primary maximum float64 error was
  `2.78e-17` for row permutation and `1.39e-17` for joint-column permutation.
- Final v1 split audit — return code 0; 0 problems, no prohibited overlaps,
  balanced 800/800/800/800/400 samples.
- V1 difficulty audit/lock — return code 0; nuisance scalar maximum AUC exactly
  `0.5`; negative exact-Hungarian costs span 4–20 across splits; 281 train,
  276 validation, 290 IID, 182 OOD-template, and 63 OOD-size cost-4 hard
  negatives.
- V1 all-model/five-seed zero-shot control — return code 0; 250 rows and
  status `PASS`; the new learned primary is at accuracy 0.5 on every split with
  five distinct initial-state hashes.
- Learned candidate v1 seed 42 — completed and preserved `FAIL`; train and
  validation accuracy 0.5. Root cause: nearly uniform transport and negligible
  class separation in learned cost.
- Learned candidate v2 seed 42 — completed train/validation-only and preserved
  `FAIL`; validation accuracy 0.5 and macro F1 `0.4990952407786564`.
- Learned candidate v3 seed 42 — completed train/validation-only and preserved
  `FAIL`; genuine improvement to validation accuracy `0.6325`, macro F1
  `0.6259327824112068`, and ROC-AUC `0.6618406250000001`, still below Gate C.
- Learned candidate v4 seed 42 — completed train/validation-only and preserved
  `FAIL`; validation accuracy `0.77125`, macro F1
  `0.7682513902542112`, and ROC-AUC `0.8688750000000001`, still below Gate C.
- Learned candidate v5 seed 42 — completed train/validation-only and selected:
  zero-shot validation accuracy `0.5`, macro F1 `0.3333333333333333`,
  ROC-AUC `0.48593125`; trained validation accuracy/balanced accuracy/macro F1
  and both recalls `1.0`; validation loss `0.0002557150620533875`.
- Candidate-v5 self-supervised learned-cost pretraining — validation accuracy
  rose from `0.5` to `1.0`; best epoch 29, validation loss
  `0.005644179182127118`. Construction uses train rows paired with themselves
  versus a random one-bit perturbation and does not use sample labels, IDs,
  exact costs, or test data.
- V2 generation — return code 0; train/validation sample payloads are
  byte-semantically identical to v1, while all three test payload hashes differ.
- Final v2 split audit — return code 0; `PASS`, 0 problems and 0 prohibited
  overlaps. File hashes: train
  `6d16657a848ef31e820c597eb585199a0b402ccfac92c247ea33d78664c38edf`,
  validation
  `69bfa5c61be13bf71199cd152f8f5dfe002451fd08e80f282204d00ce9f63ada`,
  IID
  `260a8f3963b77c36c5e0c6064838bdb2d3042ef85426728c23f6adafde358c4f`,
  OOD-template
  `e8de47dba98ee788f6658e276d3a85e50c448aa35d5666801b8c31f09b56c322`,
  OOD-size
  `58625a95222355f928a04e34e5456fa50535b85299b1ad7bc1827ee1d197b982`.
- Final v2 zero-shot control — return code 0; `PASS`, 10 models × 5 seeds ×
  5 splits = 250 metric rows.
- Adopted final learned-primary seed 42 without retraining — return code 0;
  Gate C `PASS`; IID/OOD-template/OOD-size accuracy, balanced accuracy, macro
  F1, and ROC-AUC all `1.0`; losses
  `1.3739592411442742e-05`, `1.2573125639298742e-05`, and
  `0.008290441536715706`, respectively.

## Completed requirements

- Read both control files completely.
- Inspected initial Git state and diff availability.
- Inspected the complete non-`.git` repository path inventory.
- Created the required dedicated branch.
- Created a timestamped archive before overwriting any Phase 3 result.
- Preserved all currently present Phase 3 output files plus root legacy notebooks and logs.
- Completed the takeover classification of the active source, tests, notebooks, data, and main legacy evidence.
- Identified the designated primary model, required baselines/ablations, locked seeds/splits, and inherited gate definitions.
- Implemented and smoke-tested the deterministic data/model/numerical foundations needed by the clean run.
- Completed stage 4 of 8: repaired deterministic data, model, gate, audit, and reproducibility foundations.
- Gate A is verified PASS.
- Gate B is verified PASS for all five locked seeds.
- Completed stage 5 of 8: all models, seeds, splits, checkpoints, predictions, metrics, and Gate C evidence.
- Completed stage 6 of 8: strict numerical audit, report consistency, repository tests, and clean-executed English/Persian notebooks.
- Completed stage 7 of 8: command logs, SHA-256 manifests, complete ZIP, extraction, hash verification, and extracted smoke tests.
- Completed stage 8 of 8: final frozen-evidence validation and exact artifact handoff.
- Previous deterministic Phase 3 packaging status: `PASS`, preserved as
  historical evidence.
- Current corrected-learning completion status: `IN PROGRESS`; the independent
  audit verification, preservation, learned-model selection, and final
  benchmark lock are complete.

## Historical failed evidence and resolved failures

- Legacy split integrity: FAIL because train, validation, and IID test reuse 793 permutation-equivalent underlying structures.
- Legacy report consistency: FAIL because the report weakens Gate B to `0.98+`, claims four validation seeds, omits exact OOD-size metrics, and conflicts with archived CSV values.
- Legacy numerical audit: FAIL as completion evidence because it omits required quantities, uses an incomplete schema, and inherited code contains contradictory mass assumptions.
- Baseline repository tests: FAIL (2/5 scripts fail import).
- First strict numerical audit: FAIL; preserved and resolved by the Dykstra transport projection, followed by a full 10,000-case PASS.
- First final metric-file reconciliation: FAIL only at `3.14e-8` float32 CSV round-trip error against `1e-10`; preserved and resolved with a documented `1e-7` metric serialization tolerance.
- Current mandatory failures: none.

## Pending requirements

- Reclassify fixed-L1 `MatchingNetwork` as
  `DeterministicDifferentiableMatching` and exact Hungarian as an
  algorithmic/oracle baseline.
- Implement a genuinely learned primary with random initialization, learned
  representations and matching costs, no fixed-L1 zero/nonzero separator, and
  validation-only selection.
- Freeze and audit a separate versioned learned benchmark, while preserving
  the current deterministic benchmark unchanged.
- Add and execute zero-shot, shuffled-label, random-label, label-access,
  oracle-separation, and seed-independence controls.
- Run five independent seeds for the learned primary and all required
  baselines/ablations on identical locked splits.
- Run corrected gates, split integrity, float64 structural audits, report/CSV
  consistency, notebooks, manifests, and final export verification.
- Create a final clean Git commit, generate manifests and ZIP from that
  committed state, and validate in a genuinely fresh temporary environment.

## Current blockers

- None.

## Current running process

- Resumable final v2 experiment matrix is running. Wrapper PID `5220`,
  logged-command PID `27032`, experiment wrapper PID `6904`, compute PID
  `33900`; launched 2026-07-27 17:34:37 UTC. Command:
  `python -m src.logged_command final_v2_full_experiments {python} -m src.run_final_experiments`.
  It validates and skips the completed adopted
  `LearnedSinkhornMatcher_seed42` record.

## Remaining stages

- Historical stages 1-4: completed and preserved.
- Independent-audit verification and pre-correction preservation: completed.
- Corrected learned-model/benchmark/control implementation: in progress.
- Five-seed corrected experiments: running; 1/50 model-seed records was
  complete at launch.
- Label-signal controls and scientific audits: pending.
- Corrected report and clean-executed notebooks: pending.
- Final commit, post-commit manifests/ZIP, fresh-environment verification, and
  exact inline handoff: pending.

## Final model results

- Designated primary: `MatchingNetwork`.
- Numerically best: exact three-way tie between `MatchingNetwork`, `LearnedCostSinkhorn`, and `StraightThroughHungarian` at mean macro F1 `1.0` across IID, OOD-template, and OOD-size.
- MatchingNetwork IID: loss is recorded per seed; accuracy `1.0 ± 0.0`, balanced accuracy `1.0 ± 0.0`, macro F1 `1.0 ± 0.0`, ROC-AUC `1.0 ± 0.0`, five seeds.
- MatchingNetwork OOD-template: accuracy `1.0 ± 0.0`, balanced accuracy `1.0 ± 0.0`, macro F1 `1.0 ± 0.0`, ROC-AUC `1.0 ± 0.0`, five seeds.
- MatchingNetwork OOD-size: accuracy `1.0 ± 0.0`, balanced accuracy `1.0 ± 0.0`, macro F1 `1.0 ± 0.0`, ROC-AUC `1.0 ± 0.0`, five seeds.
- Gate A: `PASS`, 3,600/3,600 correct, maximum row residual `6.147859998861804e-15`, maximum column residual `5.2735593669694936e-15`.
- Gate B seed 42: `PASS`, loss `0.004119258599548861`, accuracy/macro F1 `1.0/1.0`.
- Gate B seed 100: `PASS`, loss `0.004119258599548861`, accuracy/macro F1 `1.0/1.0`.
- Gate B seed 2026: `PASS`, loss `0.004119258599548861`, accuracy/macro F1 `1.0/1.0`.
- Gate B seed 777: `PASS`, loss `0.004119258599548861`, accuracy/macro F1 `1.0/1.0`.
- Gate B seed 999: `PASS`, loss `0.004119258599548861`, accuracy/macro F1 `1.0/1.0`.
- Gate C seeds 42, 100, 2026, 777, and 999: each `PASS` with training accuracy, balanced accuracy, macro F1, and both class recalls `1.0`; selected epoch 160; selected temperature `0.004999999888241291`.

## Git diff summary

- Branch `codex-phase3-final`, baseline commit `119343d0ad758f8c4a78d3831e8727d37255a3be`.
- Tracked diff summary before final generated/untracked artifacts: 57 files, 903,024 insertions, 11,313 deletions.
- Apparent deletions of legacy files are tracked moves into `docs/legacy/antigravity/`; no legacy evidence was discarded.

## Remaining scientifically honest limitations

- The corpus is synthetic and tests the locked local incidence-star formulation, not arbitrary empirical hypergraphs.
- Five seeds measure optimization variability on fixed data, not uncertainty over an external population.
- WL hashes are screening identifiers rather than graph-isomorphism proofs; exact labels use row-multiset equality and Hungarian cost.
- The three perfect matching variants are tied on these locked generators, so this experiment does not rank them beyond that tie.
- The system `python` launcher is unavailable on this machine; the verified environment is Python 3.11.15/PyTorch 2.12.1 CPU and all portable commands use the interpreter placeholder documented in the run manifest.

## Exact next command

Monitor the existing final v2 experiment process without interrupting it; while
it runs, update the corrected tests/audits/reporting code for the frozen model
taxonomy.
