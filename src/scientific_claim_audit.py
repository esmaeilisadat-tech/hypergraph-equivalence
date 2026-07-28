"""Audit that final claims distinguish learned and algorithmic evidence."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.config_manager import load_config
from src.differentiable_matching import LearnedSinkhornMatcher, initialize_model
from src.phase3_controls import tensor_state_sha256
from src.phase3_data import PROJECT_ROOT, load_spec

RESULT_DIR = PROJECT_ROOT / "results" / "phase3"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_claims() -> dict[str, Any]:
    spec = load_spec()
    primary = str(spec["primary_model"])
    zero_shot = pd.read_csv(RESULT_DIR / "zero_shot_metrics.csv")
    trained = pd.read_csv(RESULT_DIR / "per_seed_metrics.csv")
    gate_c = _read_json(RESULT_DIR / "gate_c_report.json")
    lock = _read_json(RESULT_DIR / "learned_benchmark_lock.json")
    label_access = _read_json(
        RESULT_DIR / "controls" / "label_access_control.json"
    )
    label_signal = _read_json(
        RESULT_DIR / "controls" / "label_signal_controls.json"
    )
    confounded_path = (
        RESULT_DIR
        / "controls"
        / "full_pipeline_pretraining_confound_controls.json"
    )
    pretraining_confound = _read_json(confounded_path)
    checks: dict[str, dict[str, Any]] = {}

    source = inspect.getsource(LearnedSinkhornMatcher)
    forbidden_source_tokens = [
        token
        for token in (
            "C_base",
            "hungarian_coupling",
            "linear_sum_assignment",
            "exact_hungarian_cost",
            "sample_id",
            "structure_hash",
            "matched_pair_id",
        )
        if token in source
    ]
    signature = str(inspect.signature(LearnedSinkhornMatcher.forward))
    checks["learned_primary_has_no_fixed_or_metadata_separator"] = {
        "status": "PASS" if not forbidden_source_tokens else "FAIL",
        "forward_signature": signature,
        "forbidden_source_token_hits": forbidden_source_tokens,
        "model_inputs": ["B_x", "B_y"],
    }

    primary_zero = zero_shot[zero_shot["model"] == primary]
    zero_min = float(spec["controls"]["zero_shot_primary_accuracy_min"])
    zero_max = float(spec["controls"]["zero_shot_primary_accuracy_max"])
    checks["primary_zero_shot_is_chance_level"] = {
        "status": "PASS"
        if primary_zero["accuracy"].between(zero_min, zero_max).all()
        else "FAIL",
        "minimum_accuracy": float(primary_zero["accuracy"].min()),
        "maximum_accuracy": float(primary_zero["accuracy"].max()),
        "mean_roc_auc": float(primary_zero["roc_auc"].mean()),
        "row_count": len(primary_zero),
    }

    primary_train = trained[
        (trained["model"] == primary) & (trained["split"] == "train")
    ]
    primary_zero_train = primary_zero[primary_zero["split"] == "train"]
    improvement = float(
        primary_train["accuracy"].mean()
        - primary_zero_train["accuracy"].mean()
    )
    checks["training_improves_primary"] = {
        "status": "PASS"
        if improvement
        >= float(spec["controls"]["minimum_primary_train_accuracy_improvement"])
        else "FAIL",
        "zero_shot_train_accuracy_mean": float(
            primary_zero_train["accuracy"].mean()
        ),
        "trained_train_accuracy_mean": float(
            primary_train["accuracy"].mean()
        ),
        "improvement": improvement,
    }

    initial_hashes = set(primary_zero["initial_state_sha256"])
    final_hashes: set[str] = set()
    for seed in spec["seeds"]:
        path = (
            RESULT_DIR / "checkpoints" / f"{primary}_seed{seed}.pt"
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = initialize_model(payload["model_config"])
        model.load_state_dict(payload["state_dict"])
        final_hashes.add(tensor_state_sha256(model))
    data_orders = {
        str(row["first_epoch_order_sha256"]) for row in gate_c["per_seed"]
    }
    checks["five_independent_primary_seeds"] = {
        "status": "PASS"
        if len(initial_hashes) == len(spec["seeds"])
        and len(final_hashes) == len(spec["seeds"])
        and len(data_orders) == len(spec["seeds"])
        else "FAIL",
        "unique_initial_state_count": len(initial_hashes),
        "unique_final_state_count": len(final_hashes),
        "unique_first_epoch_order_count": len(data_orders),
    }

    algorithmic = set(spec["non_optimized_models"])
    algorithmic_rows = trained[trained["model"].isin(algorithmic)]
    role_failures = [
        model
        for model in algorithmic
        if spec["model_roles"][model]
        not in {"algorithmic_baseline", "algorithmic_hybrid", "oracle"}
    ]
    checks["algorithmic_models_are_not_claimed_as_learned"] = {
        "status": "PASS"
        if not role_failures
        and not algorithmic_rows["optimization_performed"].astype(bool).any()
        else "FAIL",
        "models": sorted(algorithmic),
        "role_failures": role_failures,
        "optimized_row_count": int(
            algorithmic_rows["optimization_performed"].astype(bool).sum()
        ),
    }

    exact = trained[
        (trained["model"] == "ExactHungarianOracle")
        & trained["split"].isin(
            ["iid_test", "ood_template_test", "ood_size_test"]
        )
    ]
    checks["oracle_separation_control"] = {
        "status": "PASS"
        if (exact["accuracy"] == 1.0).all()
        and primary_zero[
            primary_zero["split"].isin(
                ["iid_test", "ood_template_test", "ood_size_test"]
            )
        ]["accuracy"].mean()
        <= zero_max
        else "FAIL",
        "oracle_test_accuracy_mean": float(exact["accuracy"].mean()),
        "learned_primary_zero_shot_test_accuracy_mean": float(
            primary_zero[
                primary_zero["split"].isin(
                    ["iid_test", "ood_template_test", "ood_size_test"]
                )
            ]["accuracy"].mean()
        ),
    }

    checks["final_test_protocol_is_locked"] = {
        "status": "PASS"
        if lock["benchmark_version"] == "learned_benchmark_v2_final"
        and lock["test_data_used_for_model_or_hyperparameter_selection"] is False
        else "FAIL",
        "benchmark_version": lock["benchmark_version"],
        "locked_utc": lock["locked_utc"],
        "test_used_for_selection": lock[
            "test_data_used_for_model_or_hyperparameter_selection"
        ],
    }
    checks["label_access_positive_control"] = {
        "status": label_access["status"],
        "prohibited_forward_hits": label_access[
            "prohibited_token_hits_in_forward"
        ],
    }
    checks["invalid_label_negative_controls"] = {
        "status": (
            "PASS"
            if label_signal["status"] == "PASS"
            and label_signal.get("protocol")
            == "supervised_label_signal_only"
            and label_signal.get("self_supervised_structural_pretraining")
            is False
            else "FAIL"
        ),
        "protocol": label_signal.get("protocol"),
        "self_supervised_structural_pretraining": label_signal.get(
            "self_supervised_structural_pretraining"
        ),
        "failures": label_signal["failures"],
    }
    checks["pretraining_confound_is_preserved_and_disclosed"] = {
        "status": (
            "PASS"
            if confounded_path.exists()
            and pretraining_confound["status"] == "FAIL"
            and len(pretraining_confound.get("failures", [])) > 0
            else "FAIL"
        ),
        "path": str(confounded_path.relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        ),
        "original_status": pretraining_confound["status"],
        "original_failures": pretraining_confound.get("failures", []),
        "interpretation": (
            "The full-pipeline invalid-label run includes label-free "
            "structural pretraining and therefore cannot isolate supervised "
            "label signal."
        ),
    }

    report = {
        "schema_version": "phase3-scientific-claim-audit-v1",
        "status": "PASS"
        if all(check["status"] == "PASS" for check in checks.values())
        else "FAIL",
        "primary_model": primary,
        "checks": checks,
        "interpretation": (
            "Perfect fixed-L1/Hungarian results are algorithmic baselines. "
            "Learned claims are restricted to the randomly initialized "
            "LearnedSinkhornMatcher and its measured zero-shot-to-trained "
            "change. The preserved full-pipeline invalid-label failure shows "
            "that self-supervised pretraining supplies structural task signal; "
            "the supervised-only control isolates corrupted label signal."
        ),
    }
    output = RESULT_DIR / "scientific_claim_audit.json"
    output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = audit_claims()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
