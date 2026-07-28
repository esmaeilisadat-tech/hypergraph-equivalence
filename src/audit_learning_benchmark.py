"""Freeze and audit the versioned learned Phase 3 benchmark before training."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from src.phase3_data import (
    PROJECT_ROOT,
    REQUIRED_SPLITS,
    SPEC_PATH,
    SPLIT_DIR,
    exact_hungarian_cost,
    exact_label,
    file_sha256,
    load_spec,
    load_splits,
)

RESULT_DIR = PROJECT_ROOT / "results" / "phase3"
AUDIT_PATH = RESULT_DIR / "learned_benchmark_audit.json"
LOCK_PATH = RESULT_DIR / "learned_benchmark_lock.json"


def _symmetric_auc(labels: list[int], values: list[float]) -> float:
    auc = float(roc_auc_score(labels, values))
    return max(auc, 1.0 - auc)


def _sample_scalars(sample: dict[str, Any]) -> dict[str, float]:
    bx = np.asarray(sample["B_x"], dtype=np.float64)
    by = np.asarray(sample["B_y"], dtype=np.float64)
    return {
        "m": float(sample["m"]),
        "k": float(sample["k"]),
        "total_incidence_difference": float(abs(bx.sum() - by.sum())),
        "sorted_row_sum_l1": float(
            np.abs(np.sort(bx.sum(1)) - np.sort(by.sum(1))).sum()
        ),
        "column_degree_l1": float(np.abs(bx.sum(0) - by.sum(0)).sum()),
        "density_sum": float(bx.mean() + by.mean()),
    }


def audit_and_lock() -> dict[str, Any]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    spec = load_spec()
    splits = load_splits()
    reports: dict[str, Any] = {}
    failures: list[str] = []
    all_ids: set[str] = set()
    all_bases: set[str] = set()
    all_structures: set[str] = set()

    for split_name in REQUIRED_SPLITS:
        samples = splits[split_name]
        labels: list[int] = []
        costs: dict[int, list[float]] = {0: [], 1: []}
        scalars: dict[str, list[float]] = {}
        local_ids: set[str] = set()
        local_bases: set[str] = set()
        local_structures: set[str] = set()
        oracle_failures: list[str] = []
        for sample in samples:
            sample_id = str(sample["sample_id"])
            bx = np.asarray(sample["B_x"], dtype=np.float64)
            by = np.asarray(sample["B_y"], dtype=np.float64)
            label = int(sample["label"])
            labels.append(label)
            costs[label].append(exact_hungarian_cost(bx, by))
            if exact_label(bx, by) != label:
                oracle_failures.append(sample_id)
            local_ids.add(sample_id)
            local_bases.add(str(sample["base_hash"]))
            local_structures.add(str(sample["structure_hash"]))
            for name, value in _sample_scalars(sample).items():
                scalars.setdefault(name, []).append(value)
        if len(local_ids) != len(samples):
            failures.append(f"{split_name}: duplicate sample IDs")
        if local_ids & all_ids:
            failures.append(f"{split_name}: cross-split sample ID overlap")
        if local_bases & all_bases:
            failures.append(f"{split_name}: cross-split base overlap")
        if local_structures & all_structures:
            failures.append(f"{split_name}: cross-split structure overlap")
        if oracle_failures:
            failures.append(
                f"{split_name}: {len(oracle_failures)} oracle-label failures"
            )
        all_ids.update(local_ids)
        all_bases.update(local_bases)
        all_structures.update(local_structures)
        nuisance_auc = {
            name: _symmetric_auc(labels, values)
            for name, values in scalars.items()
        }
        reports[split_name] = {
            "sample_count": len(samples),
            "class_counts": dict(sorted(Counter(labels).items())),
            "file_sha256": file_sha256(SPLIT_DIR / f"{split_name}.json"),
            "oracle_label_failure_count": len(oracle_failures),
            "negative_hungarian_cost": {
                "minimum": min(costs[0]),
                "maximum": max(costs[0]),
                "unique": sorted(set(costs[0])),
                "hard_negative_count_cost_4": int(
                    sum(value == 4 for value in costs[0])
                ),
            },
            "positive_hungarian_cost": {
                "minimum": min(costs[1]),
                "maximum": max(costs[1]),
                "unique": sorted(set(costs[1])),
            },
            "nuisance_scalar_best_direction_auc": nuisance_auc,
            "nuisance_scalar_max_auc": max(nuisance_auc.values()),
        }

    audit = {
        "schema_version": "phase3-learned-benchmark-audit-v1",
        "status": "PASS" if not failures else "FAIL",
        "benchmark_version": spec["benchmark_version"],
        "failures": failures,
        "split_reports": reports,
        "interpretation": {
            "hard_negatives": "Cost-4 negatives are minimum Ryser-switch near-isomorphs.",
            "nuisance_rule": "Recorded size, density, row-margin, column-margin, and total-incidence scalars must not separate labels.",
            "oracle_caveat": "Exact Hungarian zero-versus-positive cost necessarily separates exact row-multiset equivalence by definition. The learned primary is prohibited from receiving that cost or any fixed L1 separator.",
            "policy_as_written_in_spec": spec["benchmark_lock_policy"],
            "applied_test_policy": (
                "Final v2 tests were generated and frozen after candidate v5 "
                "selection but before first final-v2 test evaluation. Candidate "
                "development used the preserved v1 benchmark."
            ),
        },
    }
    AUDIT_PATH.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lock = {
        "schema_version": "phase3-learned-benchmark-lock-v1",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_version": spec["benchmark_version"],
        "spec_sha256": file_sha256(SPEC_PATH),
        "model_definition_sha256": file_sha256(
            PROJECT_ROOT / "src" / "differentiable_matching.py"
        ),
        "model_config_sha256": file_sha256(
            PROJECT_ROOT / "configs" / "phase3_models.json"
        ),
        "training_implementation_sha256": file_sha256(
            PROJECT_ROOT / "src" / "phase3_experiments.py"
        ),
        "split_sha256": {
            split: file_sha256(SPLIT_DIR / f"{split}.json")
            for split in REQUIRED_SPLITS
        },
        "audit_sha256": file_sha256(AUDIT_PATH),
        # Candidate development used the separately preserved v1 benchmark.
        # This v2 benchmark was generated and frozen only after candidate v5
        # was selected, but before any final-v2 test evaluation.  Recording
        # that chronology explicitly avoids overstating the lock.
        "locked_before_first_optimization": False,
        "locked_before_first_final_test_evaluation": True,
        "development_optimization_used_separate_v1_benchmark": True,
        "test_data_used_for_model_or_hyperparameter_selection": False,
        "policy_as_written_in_spec": spec["benchmark_lock_policy"],
        "applied_policy": (
            "Final v2 split files and hashes were frozen before first final-v2 "
            "test evaluation. Candidate development and the disclosed v1 "
            "protocol incident used only the preserved v1 benchmark; final-v2 "
            "test data did not guide model or hyperparameter selection."
        ),
    }
    LOCK_PATH.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"audit": audit, "lock": lock}


if __name__ == "__main__":
    result = audit_and_lock()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["audit"]["status"] == "PASS" else 1)
