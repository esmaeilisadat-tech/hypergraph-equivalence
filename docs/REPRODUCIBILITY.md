# Reproducibility and evidence map

## Environment

The saved experiment manifest records Python `3.11.15`, CPU execution and PyTorch `2.12.1+cpu`. The project's dependency pins are in [`requirements.txt`](../requirements.txt). No GPU is required by the recorded experiment.

```bash
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# POSIX shell: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.final_checks
```

Use Python 3.11 for the scientific environment. The standard-library presentation verifier can be run separately without installing the training dependencies:

```bash
python tools/build_github_presentation.py --check
# Rebuild the diagrams, results table and saved presentation check:
python tools/build_github_presentation.py
```

The generator uses locked sample matrices and saved CSV values. SVGs require no downloaded fonts, remote scripts or plotting packages.

## Locked experimental design

| Split | Samples | Classes 0 / 1 | Incident hyperedges m | Neighbor columns k |
|---|---:|---:|---|---|
| Train | 800 | 400 / 400 | 5–12 | 5–10 |
| Validation | 800 | 400 / 400 | 5–12 | 5–10 |
| IID | 800 | 400 / 400 | 5–12 | 5–10 |
| OOD-template | 800 | 400 / 400 | 5–12 | 5–10 |
| OOD-size | 400 | 200 / 200 | 15–30 | 12–20 |

The generator uses 1–4 switches for ordinary splits and 5–8 for OOD-template. See the specification and generation code for the precise sampling procedure. Each final experiment uses seeds `42`, `100`, `2026`, `777`, `999`. Checkpoint selection uses validation macro F1, then validation loss. Tests are evaluation-only.

## Preserve evidence before a scientific rerun

Training, gates, split audits, report generation and notebook execution can overwrite files under `results/phase3/` or `notebooks/`. First inspect `git status` and `git diff`, then archive existing outputs under a UTC timestamp in `results/phase3_archive/`. Also preserve the existing notebooks. The project contract in [`AGENTS.md`](../AGENTS.md) applies to scientific reruns.

`src.run_final_experiments` is resumable. To obtain a fresh experiment rather than reuse existing results, archive and move the current active run output out of `results/phase3/` before running it. Do not delete the datasets or change locked splits. Record the exact source revision and environment.

The following are the experiment entry points, not claims that they were freshly executed for this GitHub update:

```bash
python -m src.audit_splits
python -m src.run_gates
python -m src.run_final_experiments
python -m src.audit_model_numerics
python -m src.scientific_claim_audit
python -m src.reporting
python -m src.execute_notebooks
python -m src.final_checks
python -m src.artifacts
```

The full study also has negative-control runners in `src/phase3_controls.py` and `src/run_label_signal_controls.py`; inspect their entry points and the saved `run_manifest.json` command list when reproducing all controls. `src.artifacts` generates/verifies scientific manifests and a complete reproducibility ZIP. This takes longer than checking saved CSV files.

## Evidence locations

| Evidence | Location |
|---|---|
| Final specification and model roles | `phase3_spec.json`, `configs/phase3_models.json` |
| Final split lock and chronology | `results/phase3/learned_benchmark_lock.json` |
| Samples | `data/splits/{train,validation,iid_test,ood_template_test,ood_size_test}.json` |
| Seed metrics and aggregates | `results/phase3/per_seed_metrics.csv`, `aggregate_metrics.csv` |
| Predictions and weights | `results/phase3/predictions/`, `results/phase3/checkpoints/` |
| Gate A, B and C | `results/phase3/gate_{a,b,c}_report.json` |
| Dataset audit | `results/phase3/split_overlap_audit.json` |
| Float64 numerical audit | `results/phase3/numerical_audit.json` |
| Scientific claims and controls | `results/phase3/scientific_claim_audit.json`, `controls/` |
| Final report | `results/phase3/final_validation_report.md` |
| Historical commands and return codes | `results/phase3/logs/` |
| Presentation verification | `docs/github/presentation_validation.json` |

The historical `run_manifest.json`, `final_artifact_manifest.json` and reproducibility ZIP are ignored in ordinary Git by the original `.gitignore`; the publication commit explicitly includes the two manifest files so their historical provenance remains inspectable. The large archive is not required to read the repository because the source/evidence tree itself is included. Archives and local temporary folders are not silently substituted for the active dataset.

The historical manifest predates the GitHub documentation and the paper's author-name correction. It is not a manifest of the later presentation files. One pre-existing status-file difference is explicitly reported by the presentation check. A new publication file manifest under `docs/github/` records the added documentation separately.
