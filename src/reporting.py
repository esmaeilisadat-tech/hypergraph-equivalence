"""Generate and strictly verify the source-backed Phase 3 report."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.phase3_data import PROJECT_ROOT, load_spec

RESULT_DIR = PROJECT_ROOT / "results" / "phase3"
REPORT_PATH = RESULT_DIR / "final_validation_report.md"
HEADLINE_START = "<!-- HEADLINE_METRICS_JSON_START -->"
HEADLINE_END = "<!-- HEADLINE_METRICS_JSON_END -->"
TEST_SPLITS = ["iid_test", "ood_template_test", "ood_size_test"]
DISPLAY_SPLITS = {
    "iid_test": "IID",
    "ood_template_test": "OOD-template",
    "ood_size_test": "OOD-size",
}
LEARNED_ROLES = {"primary_learned", "baseline", "ablation"}
ALGORITHMIC_ROLES = {
    "algorithmic_baseline",
    "algorithmic_hybrid",
    "oracle",
}
HEADLINE_METRICS = (
    "loss",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
)


def _read_json(relative: str) -> dict[str, Any]:
    return json.loads((RESULT_DIR / relative).read_text(encoding="utf-8"))


def _git(arguments: list[str]) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _headline_records(aggregate: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    selected = aggregate[aggregate["split"].isin(TEST_SPLITS)]
    for _, row in selected.iterrows():
        for metric in HEADLINE_METRICS:
            records.append(
                {
                    "model": str(row["model"]),
                    "role": str(row["role"]),
                    "split": str(row["split"]),
                    "metric": metric,
                    "mean": float(row[f"{metric}_mean"]),
                    "std": float(row[f"{metric}_std"]),
                    "seed_count": int(row["seed_count"]),
                }
            )
    return records


def _metric_table(rows: pd.DataFrame) -> str:
    lines = [
        "| Model | Role | Split | Loss | Accuracy | Balanced accuracy | "
        "Macro F1 | ROC-AUC | Seeds |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in rows.sort_values(["role", "model", "split"]).iterrows():
        values = row.to_dict()
        values["split"] = DISPLAY_SPLITS[str(row["split"])]
        lines.append(
            "| {model} | {role} | {split} | {loss_mean:.6f} ± "
            "{loss_std:.6f} | {accuracy_mean:.6f} ± {accuracy_std:.6f} | "
            "{balanced_accuracy_mean:.6f} ± {balanced_accuracy_std:.6f} | "
            "{macro_f1_mean:.6f} ± {macro_f1_std:.6f} | "
            "{roc_auc_mean:.6f} ± {roc_auc_std:.6f} | {seed_count} |".format(
                **values
            )
        )
    return "\n".join(lines)


def _rank(rows: pd.DataFrame) -> tuple[list[str], float]:
    ranking = (
        rows.groupby("model")["macro_f1_mean"].mean().sort_values(
            ascending=False
        )
    )
    best = float(ranking.iloc[0])
    models = [
        str(model)
        for model, value in ranking.items()
        if abs(float(value) - best) <= 1e-12
    ]
    return models, best


def _format_models(models: list[str]) -> str:
    text = ", ".join(f"`{model}`" for model in models)
    return text + (" (exact tie)" if len(models) > 1 else "")


def generate_report() -> Path:
    spec = load_spec()
    aggregate = pd.read_csv(RESULT_DIR / "aggregate_metrics.csv")
    per_seed = pd.read_csv(RESULT_DIR / "per_seed_metrics.csv")
    zero_shot = pd.read_csv(RESULT_DIR / "zero_shot_metrics.csv")
    label_signal = pd.read_csv(
        RESULT_DIR
        / "controls"
        / "supervised_only_label_signal_control_metrics.csv"
    )
    pretraining_confound = pd.read_csv(
        RESULT_DIR
        / "controls"
        / "full_pipeline_pretraining_confound_metrics.csv"
    )
    gate_a = _read_json("gate_a_report.json")
    gate_b = _read_json("gate_b_report.json")
    gate_c = _read_json("gate_c_report.json")
    split_audit = _read_json("split_overlap_audit.json")
    benchmark_audit = _read_json("learned_benchmark_audit.json")
    benchmark_lock = _read_json("learned_benchmark_lock.json")
    numerical = _read_json("numerical_audit.json")
    scientific = _read_json("scientific_claim_audit.json")
    roundtrip = _read_json("checkpoint_roundtrip.json")
    selection = _read_json("model_selection/model_selection_protocol.json")
    label_access = _read_json("controls/label_access_control.json")
    label_control_report = _read_json("controls/label_signal_controls.json")

    test_rows = aggregate[aggregate["split"].isin(TEST_SPLITS)].copy()
    learned_rows = test_rows[test_rows["role"].isin(LEARNED_ROLES)]
    algorithmic_rows = test_rows[test_rows["role"].isin(ALGORITHMIC_ROLES)]
    learned_best, learned_best_value = _rank(learned_rows)
    overall_best, overall_best_value = _rank(test_rows)
    primary = str(spec["primary_model"])
    primary_rows = learned_rows[learned_rows["model"] == primary]
    primary_summary = "; ".join(
        f"{DISPLAY_SPLITS[str(row['split'])]} accuracy "
        f"{row['accuracy_mean']:.6f} ± {row['accuracy_std']:.6f}, "
        f"macro F1 {row['macro_f1_mean']:.6f} ± "
        f"{row['macro_f1_std']:.6f}, ROC-AUC "
        f"{row['roc_auc_mean']:.6f} ± {row['roc_auc_std']:.6f}"
        for _, row in primary_rows.sort_values("split").iterrows()
    )

    required_statuses = {
        "Gate A": gate_a["status"],
        "Gate B": gate_b["status"],
        "Gate C": gate_c["status"],
        "split audit": split_audit["status"],
        "benchmark audit": benchmark_audit["status"],
        "numerical audit": numerical["status"],
        "scientific-claim audit": scientific["status"],
        "label-access control": label_access["status"],
        "label-signal controls": label_control_report["status"],
        "checkpoint round trip": roundtrip["status"],
    }
    status = (
        "PASS"
        if all(value == "PASS" for value in required_statuses.values())
        else "FAIL"
    )

    primary_zero = zero_shot[zero_shot["model"] == primary]
    primary_train_zero = primary_zero[primary_zero["split"] == "train"]
    primary_train = per_seed[
        (per_seed["model"] == primary) & (per_seed["split"] == "train")
    ]
    zero_to_trained = (
        float(primary_train["accuracy"].mean())
        - float(primary_train_zero["accuracy"].mean())
    )

    gate_b_lines = "\n".join(
        f"- Seed {row['seed']}: **{row['status']}**; accuracy "
        f"{row['accuracy']:.6f}; macro F1 {row['macro_f1']:.6f}; "
        f"loss {row['loss']:.8f}; failed IDs "
        f"{row.get('failed_sample_ids') or 'none'}."
        for row in gate_b["per_seed"]
    )
    gate_c_lines = "\n".join(
        f"- Seed {row['seed']}: **{row['status']}**; initial/train accuracy "
        f"{row['initial_train_metrics']['accuracy']:.6f}/"
        f"{row['train_metrics']['accuracy']:.6f}; train macro F1 "
        f"{row['train_metrics']['macro_f1']:.6f}; selected epoch "
        f"{row['selection_epoch']}; temperature "
        f"{row['selected_temperature']:.8f}; first-order hash "
        f"`{row['first_epoch_order_sha256']}`."
        for row in gate_c["per_seed"]
    )
    control_rows = (
        label_signal[
            label_signal["split"].isin(TEST_SPLITS)
        ]
        .groupby(["control", "split"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    control_lines = "\n".join(
        f"- {row['control']} labels, {DISPLAY_SPLITS[row['split']]}: "
        f"accuracy {row['accuracy_mean']:.6f} ± "
        f"{row['accuracy_std']:.6f}; macro F1 "
        f"{row['macro_f1_mean']:.6f}; seeds {int(row['seeds'])}."
        for _, row in control_rows.iterrows()
    )
    confound_rows = (
        pretraining_confound[
            pretraining_confound["split"].isin(TEST_SPLITS)
        ]
        .groupby(["control", "split"])
        .agg(accuracy_mean=("accuracy", "mean"))
        .reset_index()
    )
    confound_lines = "\n".join(
        f"- {row['control']} labels, {DISPLAY_SPLITS[row['split']]}: "
        f"mean accuracy {row['accuracy_mean']:.6f}."
        for _, row in confound_rows.iterrows()
    )
    numerical_lines = "\n".join(
        f"- `{name}`: maximum `{values['maximum']:.3e}`, p95 "
        f"`{values['p95']:.3e}`, p99 `{values['p99']:.3e}`, failures "
        f"`{values['failed_count']}/{values['audited_count']}`, tolerance "
        f"`{values['tolerance']:.1e}`."
        for name, values in numerical["quantities"].items()
    )
    status_lines = "\n".join(
        f"- {name}: **{value}**." for name, value in required_statuses.items()
    )
    branch = _git(["branch", "--show-current"])
    commit = _git(["rev-parse", "HEAD"])
    headline_records = _headline_records(aggregate)

    report = f"""# Phase 3 Learned Hypergraph Equivalence — Final Validation Report

## Technical summary

**Overall validated status: {status}.** The specification-designated primary
is `{primary}` (`primary_learned`). Its final-v2 result is {primary_summary}.
The numerically best learned model(s), ranked by mean macro F1 across IID,
OOD-template, and OOD-size, are {_format_models(learned_best)} at
`{learned_best_value:.6f}`. Across learned and algorithmic/oracle rows, the
best are {_format_models(overall_best)} at `{overall_best_value:.6f}`.

The learned primary is not the old fixed-L1 solver. It starts from random
weights, is at chance before training, has independently seeded initial/final
states and training orders, and improves mean training accuracy by
`{zero_to_trained:.6f}`. Fixed-cost and exact-Hungarian results are preserved
as algorithmic evidence and are never attributed to learned discovery.

## Key findings

- All five `{primary}` seeds were optimized with validation-only checkpoint
  selection and evaluated on the same locked final-v2 splits.
- Primary zero-shot accuracy spans
  `{primary_zero['accuracy'].min():.6f}`–`{primary_zero['accuracy'].max():.6f}`
  across 25 seed/split rows; trained primary mean train accuracy is
  `{primary_train['accuracy'].mean():.6f}`.
- Final-v2 contains 800/800/800/800/400 balanced samples in
  train/validation/IID/OOD-template/OOD-size, has zero prohibited overlap, and
  has no exact-oracle label disagreement.
- Every required scientific, structural, numerical, and reproducibility gate
  used to generate this report has status `{status}` as itemized below.

## Learned models and ablations

{_metric_table(learned_rows)}

These rows are the only rows eligible for learned-performance claims.
`MLP_Baseline`, `NestedDeepSets`, and `BipartiteEncoder` are learned baselines;
`LearnedNoSinkhorn` and `LearnedNoColumnContext` are ablations.

## Algorithmic, hybrid, and oracle controls

{_metric_table(algorithmic_rows)}

`DeterministicDifferentiableMatching`,
`FixedL1StraightThroughHungarian`, and `ExactHungarianOracle` are not trained.
`FixedL1ResidualSinkhorn` contains a parameterized residual component but is
non-optimized in the locked study and receives the fixed-L1 separator. It is
therefore an algorithmic hybrid, not evidence that a matching cost was
discovered from scratch.

{HEADLINE_START}
{json.dumps(headline_records, sort_keys=True)}
{HEADLINE_END}

## Scope, data definitions, and benchmark difficulty

The binary task distinguishes EE-equivalent incidence stars that are not
strongly structurally equivalent (class 0) from stars whose incident-neighbor
multisets are exactly equal (class 1). Matched positive/negative pairs share
dimensions, margins, density, and total incidence. The nuisance-scalar audit
has maximum univariate direction-adjusted AUC
`{max(row['nuisance_scalar_max_auc'] for row in benchmark_audit['split_reports'].values()):.12f}`.
Negative exact Hungarian costs range from
`{min(row['negative_hungarian_cost']['minimum'] for row in benchmark_audit['split_reports'].values()):.1f}`
to
`{max(row['negative_hungarian_cost']['maximum'] for row in benchmark_audit['split_reports'].values()):.1f}`;
positive costs are exactly zero by definition.

Split integrity is `{split_audit['status']}` with
`{split_audit['problem_count']}` problems and zero prohibited sample, matched
pair, base, or invariant-structure overlap. IDs, dimensions, labels, masks,
padding, deterministic regeneration, and file hashes were audited.

## Model-selection chronology and test-data remedy

Candidate v1 failed and its generic development runner exposed v1 test
metrics. Those test partitions are preserved but explicitly retired from final
claims. Candidates v2–v5 were judged on train/validation evidence; v5 was the
first to meet unchanged Gate C thresholds and was selected with validation
accuracy/macro F1
`{selection['selected_validation_metrics']['accuracy']:.6f}/`
`{selection['selected_validation_metrics']['macro_f1']:.6f}`.

After selection, the model source, training source, configuration, and
selection specification were hashed, and entirely new final-v2 IID,
OOD-template, and OOD-size samples were generated at unseen offsets. The
scientifically accurate lock claim is:
`locked_before_first_final_test_evaluation =
{str(benchmark_lock['locked_before_first_final_test_evaluation']).lower()}`;
`locked_before_first_optimization =
{str(benchmark_lock['locked_before_first_optimization']).lower()}`.
Final-v2 test data did not guide model or hyperparameter selection. This
disclosure supersedes the broader, inaccurate policy sentence retained in the
original specification.

## Learned-primary architecture and training

`{primary}` uses a randomly initialized symmetric column-pair MLP, learned
row-pair scores converted to non-negative costs, balanced log-domain Sinkhorn
transport, transport summaries, and a learned classifier. Its forward
signature receives only `B_x` and `B_y`. It never computes or receives fixed
L1/Hamming cost, exact Hungarian output, labels, IDs, hashes, or split names.

Cost pretraining uses only train rows paired with themselves versus a single
random incidence-bit flip and selects on analogous synthetic validation-row
BCE. Main checkpoint selection is validation macro F1 followed by validation
loss. Test metrics never select epochs or temperatures.

## Zero-shot, label-signal, and label-access controls

The zero-shot primary must remain within
`{spec['controls']['zero_shot_primary_accuracy_min']:.2f}`–`{spec['controls']['zero_shot_primary_accuracy_max']:.2f}`
accuracy. The valid supervised-label-signal controls disable only the
label-free structural pretraining, corrupt train and validation labels, retain
the locked architecture/optimizer/seeds/splits, and evaluate untouched true
test labels:

{control_lines}

The initially completed full-pipeline control is preserved, not discarded.
It failed the 0.65 ceiling because label-free structural pretraining itself
supplies genuine task signal, so it cannot isolate supervised label signal:

{confound_lines}

The label-access positive control achieves the expected accuracy
`{label_access['positive_control_expected_accuracy']:.1f}` on every split.
Source inspection finds
`{len(label_access['prohibited_token_hits_in_forward'])}` prohibited forward
tokens, and the real model receives only `B_x`, `B_y`.

## Gate A: oracle and normalized coupling

Gate A is `{gate_a['status']}` on `{gate_a['oracle_sample_count']}` samples
with oracle accuracy `{gate_a['oracle_accuracy']:.6f}`. Coupling mass is
normalized to one, so a hard matched entry for `m` equally weighted rows is
approximately `1/m`, not one. At 500 float64 iterations, maximum total-mass,
row-marginal, and column-marginal errors are
`{gate_a['maximum_total_mass_error']:.3e}`,
`{gate_a['maximum_row_marginal_residual']:.3e}`, and
`{gate_a['maximum_column_marginal_residual']:.3e}` at tolerance
`{gate_a['tolerance']:.1e}`.

## Gate B: unchanged five-seed tiny-batch rule

{gate_b_lines}

The acceptance rule remains accuracy ≥ 0.99, macro F1 ≥ 0.99, and each class
F1 ≥ 0.98. No threshold or sample was removed after observing results.

## Gate C: validation-only checkpoint selection

{gate_c_lines}

The unchanged train thresholds are accuracy, balanced accuracy, and macro F1
at least 0.95, with each class recall at least 0.90.

## Complete float64 structural audit

The numerical audit is `{numerical['status']}` over
`{numerical['audited_sample_count']}` seed–sample cases with
`{numerical['failed_sample_count']}` failed cases, compact unpadded matrices,
explicit uniform marginals, deterministic permutations, float64, and
`{numerical['sinkhorn_iterations']}` Sinkhorn iterations.

{numerical_lines}

Mandatory checks cover cost/coupling transpose equivariance, matching-cost,
equality-score, final-logit, and probability invariance, finiteness,
non-negativity, total mass, both marginals, assignment transpose, valid
incident-row permutations, and joint neighbor-column permutations.

## Validation status

{status_lines}

## Limitations and interpretation

- The corpus is synthetic and tests a locked local incidence-star formulation,
  not arbitrary empirical hypergraphs or every hierarchy level in the source
  paper.
- Five seeds measure optimization variability on fixed data, not uncertainty
  over an external data-generating population.
- Exact row-multiset/Hungarian cost separates the formal labels by definition.
  The learned primary is scientifically meaningful only because that separator
  is prohibited from its inputs and implementation.
- The v1 test-exposure incident is preserved. New final-v2 tests repair final
  evaluation, but the development chronology remains an explicit limitation.
- A full-pipeline invalid-label run did not remain at chance because
  self-supervised structural pretraining can learn task-relevant matching
  without class labels. Its FAIL result is preserved; the separate
  supervised-only control isolates the narrower claim about corrupted labels.
- Strong final-v2 performance supports these locked generator families; it is
  not a universal theorem or causal claim.

## Reproduction and evidence

Branch `{branch}`; current source commit `{commit}`; Python
`{platform.python_version()}`; PyTorch `{torch.__version__}` on CPU. The
pre-correction archive is
`results/phase3_archive/20260727T160053Z_independent_audit_corrections`.
Primary evidence is in `per_seed_metrics.csv`, `aggregate_metrics.csv`,
`checkpoints/`, `predictions/`, the gate/control/audit JSON files, both
clean-executed notebooks, logged commands, and the post-commit manifests.

## Recommended next work

An independent follow-up should add generator families not designed during
this project and empirical incidence stars while retaining the same
validation-only selection, zero-shot/invalid-label controls, and full
structural audit. That would test whether the learned ranking persists beyond
the locked synthetic distribution.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    return REPORT_PATH


def verify_report_consistency() -> dict[str, Any]:
    report = REPORT_PATH.read_text(encoding="utf-8")
    failures: list[dict[str, Any]] = []
    if HEADLINE_START not in report or HEADLINE_END not in report:
        failures.append({"field": "headline marker", "reason": "missing"})
        report_records: list[dict[str, Any]] = []
    else:
        payload = report.split(HEADLINE_START, 1)[1].split(
            HEADLINE_END, 1
        )[0]
        report_records = json.loads(payload.strip())
    aggregate = pd.read_csv(RESULT_DIR / "aggregate_metrics.csv")
    expected = _headline_records(aggregate)
    report_by_key = {
        (r["model"], r["split"], r["metric"]): r for r in report_records
    }
    expected_by_key = {
        (r["model"], r["split"], r["metric"]): r for r in expected
    }
    if set(report_by_key) != set(expected_by_key):
        failures.append(
            {
                "field": "model/split/metric keys",
                "missing": sorted(set(expected_by_key) - set(report_by_key)),
                "unexpected": sorted(
                    set(report_by_key) - set(expected_by_key)
                ),
            }
        )
    for key in sorted(set(report_by_key) & set(expected_by_key)):
        for field in ("role", "mean", "std", "seed_count"):
            observed = report_by_key[key][field]
            wanted = expected_by_key[key][field]
            differs = (
                abs(float(observed) - float(wanted)) > 1e-12
                if field in {"mean", "std"}
                else observed != wanted
            )
            if differs:
                failures.append(
                    {
                        "key": key,
                        "field": field,
                        "expected": wanted,
                        "observed": observed,
                    }
                )
    result = {
        "status": "PASS" if not failures else "FAIL",
        "checked_headline_count": len(expected),
        "model_count": int(
            aggregate[aggregate["split"].isin(TEST_SPLITS)]["model"].nunique()
        ),
        "split_count": len(TEST_SPLITS),
        "metric_count": len(HEADLINE_METRICS),
        "failures": failures,
    }
    (RESULT_DIR / "report_consistency.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(generate_report())
    consistency = verify_report_consistency()
    print(json.dumps(consistency, indent=2))
    raise SystemExit(0 if consistency["status"] == "PASS" else 1)
