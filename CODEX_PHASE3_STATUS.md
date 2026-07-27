# Codex Phase 3 Status

Last updated: 2026-07-27 10:52:00 UTC

## Repository state

- Workspace: `hypergraph_equivalence`
- Branch: `codex-phase3-final`
- Commit: none; the supplied workspace contained no `.git` directory or recoverable Git history, so Git was initialized in place.
- Legacy archive: `results/phase3_archive/20260727_102315`
- Archive inventory: 20 files, 1,789,035 bytes.

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

## Completed requirements

- Read both control files completely.
- Inspected initial Git state and diff availability.
- Inspected the complete non-`.git` repository path inventory.
- Created the required dedicated branch.
- Created a timestamped archive before overwriting any Phase 3 result.
- Preserved all currently present Phase 3 output files plus root legacy notebooks and logs.
- Completed the takeover classification of the active source, tests, notebooks, data, and main legacy evidence.
- Identified the designated primary model, required baselines/ablations, locked seeds/splits, and inherited gate definitions.

## Failed requirements

- Legacy split integrity: FAIL because train, validation, and IID test reuse 793 permutation-equivalent underlying structures.
- Legacy report consistency: FAIL because the report weakens Gate B to `0.98+`, claims four validation seeds, omits exact OOD-size metrics, and conflicts with archived CSV values.
- Legacy numerical audit: FAIL as completion evidence because it omits required quantities, uses an incomplete schema, and inherited code contains contradictory mass assumptions.
- Baseline repository tests: FAIL (2/5 scripts fail import).
- Reproducibility environment: FAIL because no dependency declaration or lock exists and the system `python` command is unusable.

## Pending requirements

- Repair the reproducibility pipeline.
- Replace leaking/non-deterministic split generation and regenerate locked split files with deterministic IDs and zero prohibited overlap.
- Repair model constructors, masks, Sinkhorn normalization, padding behavior, checkpoint schema, and all mandatory invariances.
- Replace stale tests and add gate/data/numerical/report/manifest/ZIP coverage.
- Run all required tests, gates, models, seeds, splits, and audits.
- Execute and validate both required notebooks from clean kernels.
- Generate final reports, metrics, predictions, checkpoints, logs, manifests, hashes, and complete ZIP.
- Verify the ZIP by extraction, hashes, and smoke tests.

## Current blockers

- None.

## Exact next command

`git add .gitignore AGENTS.md CODEX_PHASE3_HANDOFF.md CODEX_PHASE3_STATUS.md src tests data results/phase3 *.py *.ipynb *.md *.txt *.log *.pdf`
