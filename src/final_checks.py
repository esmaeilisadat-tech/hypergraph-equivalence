"""Read-only final validation of every frozen Phase 3 evidence layer."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from src.config_manager import model_names
from src.phase3_data import PROJECT_ROOT, file_sha256, load_spec
from src.phase3_experiments import _metrics

RESULT_DIR = PROJECT_ROOT / "results" / "phase3"


def validate() -> dict[str, object]:
    spec = load_spec()
    failures: list[str] = []
    expected_checkpoints = len(model_names()) * len(spec["seeds"])
    expected_predictions = expected_checkpoints * len(spec["splits"])
    expected_records = expected_checkpoints
    checkpoints = sorted((RESULT_DIR / "checkpoints").glob("*.pt"))
    predictions = sorted((RESULT_DIR / "predictions").glob("*.csv"))
    run_records = sorted((RESULT_DIR / "run_records").glob("*.json"))
    if len(checkpoints) != expected_checkpoints:
        failures.append(
            f"checkpoint count {len(checkpoints)} != {expected_checkpoints}"
        )
    if len(predictions) != expected_predictions:
        failures.append(
            f"prediction count {len(predictions)} != {expected_predictions}"
        )
    if len(run_records) != expected_records:
        failures.append(
            f"run record count {len(run_records)} != {expected_records}"
        )

    observed_rows = []
    for prediction_path in predictions:
        stem = prediction_path.stem
        model_seed, split = stem.split("_seed", 1)
        seed_text, split = split.split("_", 1)
        # Split names contain underscores; identify the seed at the first separator.
        seed = int(seed_text)
        frame = pd.read_csv(prediction_path)
        metrics = _metrics(
            frame["label"].to_numpy(), frame["probability"].to_numpy()
        )
        observed_rows.append(
            {
                "model": model_seed,
                "seed": seed,
                "split": split,
                **metrics,
                "prediction_sha256": file_sha256(prediction_path),
            }
        )
    observed = pd.DataFrame(observed_rows)
    saved = pd.read_csv(RESULT_DIR / "per_seed_metrics.csv")
    metric_columns = [
        "loss",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "sample_count",
        "class_0_count",
        "class_1_count",
        "class_0_precision",
        "class_1_precision",
        "class_0_recall",
        "class_1_recall",
        "class_0_f1",
        "class_1_f1",
    ]
    merged = saved.merge(
        observed,
        on=["model", "seed", "split"],
        suffixes=("_saved", "_observed"),
        validate="one_to_one",
    )
    if len(merged) != expected_predictions:
        failures.append(
            f"metric merge count {len(merged)} != {expected_predictions}"
        )
    maximum_metric_error = 0.0
    for column in metric_columns:
        error = np.max(
            np.abs(
                merged[f"{column}_saved"].astype(float)
                - merged[f"{column}_observed"].astype(float)
            )
        )
        maximum_metric_error = max(maximum_metric_error, float(error))
        # Prediction probabilities originate as float32 and are round-tripped
        # through decimal CSV. A 1e-7 reconciliation tolerance is below the
        # representational precision and is unrelated to the 1e-8 float64
        # structural-audit acceptance threshold.
        if error > 1e-7:
            failures.append(f"{column} maximum recomputation error {error}")
    hash_mismatch = merged[
        merged["prediction_sha256_saved"]
        != merged["prediction_sha256_observed"]
    ]
    if len(hash_mismatch):
        failures.append(f"{len(hash_mismatch)} prediction hash mismatches")

    aggregate = pd.read_csv(RESULT_DIR / "aggregate_metrics.csv")
    if len(aggregate) != len(model_names()) * len(spec["splits"]) or not (
        aggregate["seed_count"] == len(spec["seeds"])
    ).all():
        failures.append("aggregate row or seed count invalid")
    for name in (
        "split_overlap_audit.json",
        "learned_benchmark_audit.json",
        "zero_shot_control.json",
        "gate_a_report.json",
        "gate_b_report.json",
        "gate_c_report.json",
        "numerical_audit.json",
        "scientific_claim_audit.json",
        "controls/label_access_control.json",
        "controls/label_signal_controls.json",
        "checkpoint_roundtrip.json",
        "report_consistency.json",
    ):
        payload = json.loads((RESULT_DIR / name).read_text(encoding="utf-8"))
        if payload["status"] != "PASS":
            failures.append(f"{name} status {payload['status']}")
    numerical = json.loads(
        (RESULT_DIR / "numerical_audit.json").read_text(encoding="utf-8")
    )
    expected_audited = len(spec["seeds"]) * sum(
        int(spec["splits"][split]["samples"])
        for split in spec["numerical_audit"]["splits"]
    )
    if numerical["audited_sample_count"] != expected_audited:
        failures.append(
            "numerical audited sample count "
            f"{numerical['audited_sample_count']} != {expected_audited}"
        )
    if any(
        quantity["failed_count"] != 0
        for quantity in numerical["quantities"].values()
    ):
        failures.append("numerical audit has quantity failures")

    notebook_status = {}
    for name in ("Phase3_Research.ipynb", "Phase3_Explained_FA.ipynb"):
        notebook = nbformat.read(PROJECT_ROOT / "notebooks" / name, as_version=4)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        executed = sum(cell.execution_count is not None for cell in code_cells)
        notebook_status[name] = {
            "code_cells": len(code_cells),
            "executed_cells": executed,
            "error_outputs": len(errors),
        }
        if executed != len(code_cells) or errors:
            failures.append(f"{name} is not cleanly executed")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checkpoint_count": len(checkpoints),
        "prediction_count": len(predictions),
        "run_record_count": len(run_records),
        "per_seed_metric_count": len(saved),
        "aggregate_metric_count": len(aggregate),
        "maximum_metric_recomputation_error": maximum_metric_error,
        "notebooks": notebook_status,
        "required_seeds": spec["seeds"],
    }


if __name__ == "__main__":
    result = validate()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
