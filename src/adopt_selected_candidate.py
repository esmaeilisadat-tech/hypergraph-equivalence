"""Adopt the validation-selected seed-42 candidate without retraining it."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import torch

from src.config_manager import load_config
from src.differentiable_matching import initialize_model
from src.phase3_data import PROJECT_ROOT, file_sha256, load_spec, load_splits
from src.phase3_experiments import (
    CHECKPOINT_DIR,
    PREDICTION_DIR,
    RESULT_DIR,
    RUN_RECORD_DIR,
    _metrics,
    predict,
)


def adopt() -> dict[str, object]:
    spec = load_spec()
    model_name = str(spec["primary_model"])
    seed = 42
    source = RESULT_DIR / "model_selection" / "candidate_v5_seed42.pt"
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload["model_name"] != model_name or int(payload["seed"]) != seed:
        raise RuntimeError("Selected candidate identity mismatch")
    if payload["model_config"] != load_config(model_name):
        raise RuntimeError("Selected candidate config differs from frozen config")
    model = initialize_model(payload["model_config"]).eval()
    model.load_state_dict(payload["state_dict"])
    metadata = payload["metadata"]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    RUN_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{model_name}_seed{seed}.pt"
    shutil.copyfile(source, checkpoint_path)
    checkpoint_sha256 = file_sha256(checkpoint_path)

    splits = load_splits()
    metric_rows: list[dict[str, object]] = []
    for split_name, samples in splits.items():
        labels, probabilities = predict(model, samples)
        predictions = (probabilities >= 0.5).astype(int)
        prediction_path = (
            PREDICTION_DIR / f"{model_name}_seed{seed}_{split_name}.csv"
        )
        pd.DataFrame(
            {
                "sample_id": [sample["sample_id"] for sample in samples],
                "matched_pair_id": [
                    sample["matched_pair_id"] for sample in samples
                ],
                "label": labels.astype(int),
                "probability": probabilities,
                "prediction": predictions,
                "correct": predictions == labels,
                "m": [sample["m"] for sample in samples],
                "k": [sample["k"] for sample in samples],
            }
        ).to_csv(prediction_path, index=False)
        metric_rows.append(
            {
                "model": model_name,
                "role": spec["model_roles"][model_name],
                "seed": seed,
                "split": split_name,
                **_metrics(labels, probabilities),
                "selection_epoch": metadata["selection_epoch"],
                "selected_temperature": metadata["selected_temperature"],
                "optimization_performed": True,
                "initial_accuracy": metadata["initial_train_metrics"][
                    "accuracy"
                ]
                if split_name == "train"
                else metadata["initial_validation_metrics"]["accuracy"]
                if split_name == "validation"
                else np.nan,
                "checkpoint_sha256": checkpoint_sha256,
                "prediction_sha256": file_sha256(prediction_path),
                "prediction_path": str(
                    prediction_path.relative_to(PROJECT_ROOT)
                ).replace("\\", "/"),
            }
        )
    train_row = next(row for row in metric_rows if row["split"] == "train")
    passed = (
        train_row["accuracy"]
        >= spec["gate_c"]["train_accuracy_threshold"]
        and train_row["balanced_accuracy"]
        >= spec["gate_c"]["train_balanced_accuracy_threshold"]
        and train_row["macro_f1"]
        >= spec["gate_c"]["train_macro_f1_threshold"]
        and min(train_row["class_0_recall"], train_row["class_1_recall"])
        >= spec["gate_c"]["train_min_class_recall_threshold"]
    )
    gate_c_row = {
        "seed": seed,
        "status": "PASS" if passed else "FAIL",
        "train_metrics": {
            key: train_row[key]
            for key in (
                "loss",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "class_0_recall",
                "class_1_recall",
            )
        },
        "initial_train_metrics": metadata["initial_train_metrics"],
        "initial_validation_metrics": metadata["initial_validation_metrics"],
        "selection_epoch": metadata["selection_epoch"],
        "selection_criterion": metadata["selection_criterion"],
        "selected_temperature": metadata["selected_temperature"],
        "train_sinkhorn_iterations": metadata["train_sinkhorn_iterations"],
        "evaluation_sinkhorn_iterations": metadata[
            "evaluation_sinkhorn_iterations"
        ],
        "epochs_executed": metadata["epochs_executed"],
        "first_epoch_order_sha256": metadata["first_epoch_order_sha256"],
        "checkpoint": str(
            checkpoint_path.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_sha256,
        "adopted_without_retraining": True,
    }
    record = {
        "schema_version": "phase3-model-seed-record-v1",
        "model": model_name,
        "seed": seed,
        "checkpoint": gate_c_row["checkpoint"],
        "checkpoint_sha256": checkpoint_sha256,
        "metric_rows": metric_rows,
        "gate_c_row": gate_c_row,
        "training_metadata": metadata,
        "provenance": (
            "Validation-selected candidate_v5_seed42; copied and evaluated "
            "on final v2 tests without retraining."
        ),
    }
    record_path = RUN_RECORD_DIR / f"{model_name}_seed{seed}.json"
    record_path.write_text(
        json.dumps(record, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return record


if __name__ == "__main__":
    print(json.dumps(adopt(), indent=2, allow_nan=True))
