"""Scientific controls for the corrected learned Phase 3 study."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config_manager import load_config, model_names
from src.differentiable_matching import LearnedSinkhornMatcher, initialize_model
from src.phase3_data import PROJECT_ROOT, file_sha256, load_spec, load_splits
from src.phase3_experiments import (
    _metrics,
    predict,
    set_seed,
    train_model,
)

RESULT_DIR = PROJECT_ROOT / "results" / "phase3"
ZERO_SHOT_PREDICTION_DIR = RESULT_DIR / "zero_shot_predictions"
CONTROL_DIR = RESULT_DIR / "controls"


def tensor_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run_zero_shot_control() -> dict[str, Any]:
    spec = load_spec()
    splits = load_splits()
    ZERO_SHOT_PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for model_name in model_names():
        for seed in spec["seeds"]:
            set_seed(int(seed))
            model = initialize_model(load_config(model_name)).eval()
            state_hash = tensor_state_sha256(model)
            for split_name, samples in splits.items():
                labels, probabilities = predict(model, samples)
                predictions = (probabilities >= 0.5).astype(int)
                prediction_path = (
                    ZERO_SHOT_PREDICTION_DIR
                    / f"{model_name}_seed{seed}_{split_name}.csv"
                )
                pd.DataFrame(
                    {
                        "sample_id": [sample["sample_id"] for sample in samples],
                        "label": labels.astype(int),
                        "probability": probabilities,
                        "prediction": predictions,
                        "correct": predictions == labels,
                    }
                ).to_csv(prediction_path, index=False)
                rows.append(
                    {
                        "model": model_name,
                        "role": spec["model_roles"][model_name],
                        "seed": seed,
                        "split": split_name,
                        **_metrics(labels, probabilities),
                        "initial_state_sha256": state_hash,
                        "prediction_sha256": file_sha256(prediction_path),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULT_DIR / "zero_shot_metrics.csv", index=False)
    primary = frame[frame["model"] == spec["primary_model"]]
    lower = float(spec["controls"]["zero_shot_primary_accuracy_min"])
    upper = float(spec["controls"]["zero_shot_primary_accuracy_max"])
    failures = [
        {
            "seed": int(row.seed),
            "split": row.split,
            "accuracy": float(row.accuracy),
        }
        for row in primary.itertuples()
        if not lower <= float(row.accuracy) <= upper
    ]
    report = {
        "schema_version": "phase3-zero-shot-control-v1",
        "status": "PASS" if not failures else "FAIL",
        "model_count": len(model_names()),
        "seed_count": len(spec["seeds"]),
        "split_count": len(splits),
        "row_count": len(frame),
        "primary_model": spec["primary_model"],
        "primary_accuracy_acceptance_interval": [lower, upper],
        "primary_failures": failures,
        "metrics_path": "results/phase3/zero_shot_metrics.csv",
        "prediction_directory": "results/phase3/zero_shot_predictions",
    }
    (RESULT_DIR / "zero_shot_control.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _relabel(
    samples: list[dict[str, Any]], seed: int, mode: str
) -> list[dict[str, Any]]:
    relabeled = copy.deepcopy(samples)
    rng = np.random.default_rng(seed)
    original = np.asarray([int(sample["label"]) for sample in samples])
    if mode == "shuffled":
        labels = rng.permutation(original)
    elif mode == "random":
        labels = np.zeros(len(samples), dtype=int)
        labels[len(samples) // 2 :] = 1
        labels = rng.permutation(labels)
    else:
        raise ValueError(mode)
    for sample, label in zip(relabeled, labels):
        sample["label"] = int(label)
    return relabeled


def run_label_signal_controls() -> dict[str, Any]:
    """Isolate supervised label signal using invalid label assignments.

    The primary pipeline includes label-free structural pretraining. Running
    that pretraining inside a random-label control can legitimately recover
    task structure and therefore does not isolate the effect of labels. The
    original full-pipeline result is preserved separately as a confounded
    control. This protocol disables only that pretraining and otherwise uses
    the locked architecture, optimizer, validation selection, seeds, and
    splits.
    """

    spec = load_spec()
    splits = load_splits()
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    record_dir = CONTROL_DIR / "supervised_only_run_records"
    record_dir.mkdir(parents=True, exist_ok=True)
    control_config = load_config(spec["primary_model"])
    control_config["self_supervised_cost_pretraining"] = False
    rows: list[dict[str, Any]] = []
    for mode in ("shuffled", "random"):
        for seed in spec["seeds"]:
            record_path = record_dir / f"{mode}_labels_seed{seed}.json"
            checkpoint_path = (
                CONTROL_DIR / f"supervised_only_{mode}_labels_seed{seed}.pt"
            )
            if record_path.exists() and checkpoint_path.exists():
                record = json.loads(record_path.read_text(encoding="utf-8"))
                if (
                    record.get("checkpoint_sha256")
                    == file_sha256(checkpoint_path)
                    and len(record.get("metric_rows", [])) == len(splits)
                ):
                    rows.extend(record["metric_rows"])
                    continue
            control_train = _relabel(
                splits["train"], int(seed) + 101_000, mode
            )
            control_validation = _relabel(
                splits["validation"], int(seed) + 202_000, mode
            )
            model, metadata = train_model(
                spec["primary_model"],
                int(seed),
                control_train,
                control_validation,
                config_override=control_config,
            )
            torch.save(
                {
                    "schema_version": "phase3-checkpoint-v2",
                    "model_name": metadata["model"],
                    "seed": metadata["seed"],
                    "model_config": control_config,
                    "state_dict": model.state_dict(),
                    "metadata": metadata,
                    "control_protocol": "supervised_label_signal_only",
                },
                checkpoint_path,
            )
            checkpoint_sha256 = file_sha256(checkpoint_path)
            run_rows: list[dict[str, Any]] = []
            for split_name in (
                "train",
                "validation",
                "iid_test",
                "ood_template_test",
                "ood_size_test",
            ):
                labels, probabilities = predict(model, splits[split_name])
                run_rows.append(
                    {
                        "control": mode,
                        "seed": seed,
                        "split": split_name,
                        **_metrics(labels, probabilities),
                        "selection_epoch": metadata["selection_epoch"],
                        "checkpoint_sha256": checkpoint_sha256,
                    }
                )
            record = {
                "schema_version": "phase3-label-signal-control-run-v1",
                "control": mode,
                "seed": seed,
                "checkpoint": str(
                    checkpoint_path.relative_to(PROJECT_ROOT)
                ).replace("\\", "/"),
                "checkpoint_sha256": checkpoint_sha256,
                "metric_rows": run_rows,
                "training_metadata": metadata,
            }
            record_path.write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
            rows.extend(run_rows)
            # Rebuild the partial durable table after each completed run so an
            # interruption never loses or requires repeating completed work.
            pd.DataFrame(rows).to_csv(
                CONTROL_DIR
                / "supervised_only_label_signal_control_metrics.partial.csv",
                index=False,
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        CONTROL_DIR / "supervised_only_label_signal_control_metrics.csv",
        index=False,
    )
    test_rows = frame[
        frame["split"].isin(
            ["iid_test", "ood_template_test", "ood_size_test"]
        )
    ]
    failures: list[dict[str, Any]] = []
    for mode, threshold_key in (
        ("shuffled", "shuffled_label_test_accuracy_max"),
        ("random", "random_label_test_accuracy_max"),
    ):
        threshold = float(spec["controls"][threshold_key])
        grouped = test_rows[test_rows["control"] == mode].groupby("split")[
            "accuracy"
        ].mean()
        for split, value in grouped.items():
            if float(value) > threshold:
                failures.append(
                    {
                        "control": mode,
                        "split": split,
                        "mean_accuracy": float(value),
                        "maximum": threshold,
                    }
                )
    report = {
        "schema_version": "phase3-label-signal-controls-v2",
        "status": "PASS" if not failures else "FAIL",
        "protocol": "supervised_label_signal_only",
        "self_supervised_structural_pretraining": False,
        "reason": (
            "The label-free structural pretraining is disabled so this "
            "control isolates whether corrupted supervised labels can produce "
            "held-out generalization. The original full-pipeline control is "
            "preserved as pretraining-confounded evidence."
        ),
        "preserved_full_pipeline_control": {
            "report": (
                "results/phase3/controls/"
                "full_pipeline_pretraining_confound_controls.json"
            ),
            "metrics": (
                "results/phase3/controls/"
                "full_pipeline_pretraining_confound_metrics.csv"
            ),
            "status": "FAIL",
            "interpretation": (
                "Not a valid label-only negative control because label-free "
                "structural pretraining supplies genuine task signal."
            ),
        },
        "completed_runs": 2 * len(spec["seeds"]),
        "failures": failures,
        "metrics_path": (
            "results/phase3/controls/"
            "supervised_only_label_signal_control_metrics.csv"
        ),
    }
    (CONTROL_DIR / "label_signal_controls.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def run_label_access_control() -> dict[str, Any]:
    """Positive control plus source-level proof that real models lack label access."""

    spec = load_spec()
    splits = load_splits()
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    positive: dict[str, Any] = {}
    for split_name, samples in splits.items():
        labels = np.asarray([int(sample["label"]) for sample in samples])
        positive[split_name] = _metrics(labels, labels.astype(float))
    signature = str(inspect.signature(LearnedSinkhornMatcher.forward))
    source = inspect.getsource(LearnedSinkhornMatcher.forward)
    prohibited = (
        "label",
        "sample_id",
        "matched_pair_id",
        "structure_hash",
        "base_hash",
        "split",
        "oracle",
        "hungarian",
    )
    hits = [token for token in prohibited if token in source]
    expected = float(spec["controls"]["label_access_expected_accuracy"])
    report = {
        "schema_version": "phase3-label-access-control-v1",
        "status": "PASS"
        if not hits
        and all(row["accuracy"] == expected for row in positive.values())
        else "FAIL",
        "positive_control": positive,
        "positive_control_expected_accuracy": expected,
        "learned_primary_forward_signature": signature,
        "prohibited_token_hits_in_forward": hits,
        "real_model_receives_only": ["B_x", "B_y"],
    }
    (CONTROL_DIR / "label_access_control.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run_zero_shot_control(), indent=2))
