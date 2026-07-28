"""Locked Phase 3 gates, training, prediction, and metric aggregation."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.config_manager import generate_configs, load_config, model_names
from src.differentiable_matching import (
    DifferentiableIncidenceMatchingNetwork,
    LearnedSinkhornMatcher,
    initialize_model,
)
from src.phase3_data import (
    PROJECT_ROOT,
    collate_samples,
    exact_hungarian_cost,
    exact_label,
    file_sha256,
    load_spec,
    load_splits,
)

RESULT_DIR = PROJECT_ROOT / "results" / "phase3"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
PREDICTION_DIR = RESULT_DIR / "predictions"
RUN_RECORD_DIR = RESULT_DIR / "run_records"
DEVICE = torch.device("cpu")
DTYPE = torch.float32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _tensor_batch(
    samples: list[dict[str, Any]], dtype: torch.dtype = DTYPE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bx, by, labels = collate_samples(samples, dtype=dtype)
    return bx.to(DEVICE), by.to(DEVICE), labels.to(DEVICE)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(int)
    loss = float(
        F.binary_cross_entropy(
            torch.tensor(probabilities, dtype=torch.float64).clamp(1e-12, 1 - 1e-12),
            torch.tensor(labels, dtype=torch.float64),
        ).item()
    )
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    return {
        "loss": loss,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "sample_count": int(len(labels)),
        "class_0_count": int((labels == 0).sum()),
        "class_1_count": int((labels == 1).sum()),
        "class_0_precision": float(precision[0]),
        "class_1_precision": float(precision[1]),
        "class_0_recall": float(recall[0]),
        "class_1_recall": float(recall[1]),
        "class_0_f1": float(per_class_f1[0]),
        "class_1_f1": float(per_class_f1[1]),
        "class_0_support": int(support[0]),
        "class_1_support": int(support[1]),
    }


def predict(
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            bx, by, labels = _tensor_batch(samples[start : start + batch_size])
            probabilities = torch.sigmoid(model(bx, by))
            all_labels.append(labels.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_probabilities)


def _validation_score(
    model: torch.nn.Module, samples: list[dict[str, Any]]
) -> tuple[float, float]:
    labels, probabilities = predict(model, samples)
    metrics = _metrics(labels, probabilities)
    return metrics["macro_f1"], metrics["loss"]


def _rows_by_width(
    samples: list[dict[str, Any]],
) -> dict[int, np.ndarray]:
    grouped: dict[int, list[np.ndarray]] = {}
    for sample in samples:
        matrix = np.asarray(sample["B_x"], dtype=np.float32)
        grouped.setdefault(matrix.shape[1], []).extend(matrix)
    return {
        width: np.stack(rows).astype(np.float32)
        for width, rows in grouped.items()
    }


def _pretraining_batch(
    rows: np.ndarray,
    rng: np.random.Generator,
    maximum_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = min(len(rows), maximum_rows)
    selected = rows[rng.choice(len(rows), size=count, replace=False)].copy()
    perturbed = selected.copy()
    columns = rng.integers(0, selected.shape[1], size=count)
    perturbed[np.arange(count), columns] = (
        1.0 - perturbed[np.arange(count), columns]
    )
    first = np.concatenate((selected, selected), axis=0)
    second = np.concatenate((selected, perturbed), axis=0)
    targets = np.concatenate(
        (np.ones(count, dtype=np.float32), np.zeros(count, dtype=np.float32))
    )
    order = rng.permutation(len(targets))
    return (
        torch.tensor(first[order], dtype=DTYPE, device=DEVICE),
        torch.tensor(second[order], dtype=DTYPE, device=DEVICE),
        torch.tensor(targets[order], dtype=DTYPE, device=DEVICE),
    )


def _pretrain_learned_cost(
    model: LearnedSinkhornMatcher,
    train_samples: list[dict[str, Any]],
    validation_samples: list[dict[str, Any]],
    seed: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    train_rows = _rows_by_width(train_samples)
    validation_rows = _rows_by_width(validation_samples)
    epochs = int(settings["self_supervised_cost_pretraining_epochs"])
    maximum_rows = int(settings["self_supervised_cost_pretraining_rows_per_k"])
    optimizer = torch.optim.AdamW(
        list(model.base_column_pair_mlp.parameters())
        + list(model.pair_score_head.parameters()),
        lr=float(settings["self_supervised_cost_pretraining_learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    train_rng = np.random.default_rng(seed + 303_000)
    validation_rng = np.random.default_rng(seed + 404_000)
    fixed_validation = {
        width: _pretraining_batch(rows, validation_rng, maximum_rows)
        for width, rows in validation_rows.items()
    }
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for rows in train_rows.values():
            first, second, targets = _pretraining_batch(
                rows, train_rng, maximum_rows
            )
            optimizer.zero_grad(set_to_none=True)
            scores = model.score_row_pairs(first, second)
            loss = F.binary_cross_entropy_with_logits(scores, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(settings["gradient_clip"])
            )
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        validation_losses: list[float] = []
        correct = 0
        count = 0
        with torch.no_grad():
            for first, second, targets in fixed_validation.values():
                scores = model.score_row_pairs(first, second)
                validation_losses.append(
                    float(
                        F.binary_cross_entropy_with_logits(
                            scores, targets
                        ).item()
                    )
                )
                correct += int(
                    ((torch.sigmoid(scores) >= 0.5) == (targets >= 0.5))
                    .sum()
                    .item()
                )
                count += len(targets)
        validation_loss = float(np.mean(validation_losses))
        validation_accuracy = correct / count
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {
        "performed": True,
        "construction": (
            "train rows paired with themselves versus one randomly flipped "
            "incidence bit; no sample label, ID, exact cost, or test split"
        ),
        "selection": "synthetic validation-row BCE loss",
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
        "validation_accuracy": history[best_epoch - 1][
            "validation_accuracy"
        ],
        "history": history,
    }


def train_model(
    model_name: str,
    seed: int,
    train_samples: list[dict[str, Any]],
    validation_samples: list[dict[str, Any]],
    *,
    max_epochs: int | None = None,
    patience: int | None = None,
    config_override: dict[str, Any] | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    spec = load_spec()
    settings = spec["training"]
    max_epochs = int(max_epochs or settings["max_epochs"])
    patience = int(patience or settings["patience"])
    set_seed(seed)
    config = (
        dict(config_override)
        if config_override is not None
        else load_config(model_name)
    )
    if config.get("model_name") != model_name:
        raise ValueError(
            "config_override model_name must match the requested model"
        )
    model = initialize_model(config).to(device=DEVICE, dtype=DTYPE)
    initial_train_labels, initial_train_probabilities = predict(
        model, train_samples
    )
    initial_train_metrics = _metrics(
        initial_train_labels, initial_train_probabilities
    )
    initial_validation_labels, initial_validation_probabilities = predict(
        model, validation_samples
    )
    initial_validation_metrics = _metrics(
        initial_validation_labels, initial_validation_probabilities
    )
    if model_name in spec.get("non_optimized_models", []):
        metadata = {
            "model": model_name,
            "seed": seed,
            "selection_epoch": 0,
            "selection_criterion": "not_applicable_algorithmic_baseline",
            "validation_macro_f1": initial_validation_metrics["macro_f1"],
            "validation_loss": initial_validation_metrics["loss"],
            "selected_temperature": float(
                getattr(model, "temp", config["tau_start"])
            ),
            "epochs_executed": 0,
            "runtime_seconds": 0.0,
            "device": str(DEVICE),
            "dtype": str(DTYPE),
            "train_sinkhorn_iterations": int(config["train_iters"]),
            "evaluation_sinkhorn_iterations": int(config["eval_iters"]),
            "optimization_performed": False,
            "initial_train_metrics": initial_train_metrics,
            "initial_validation_metrics": initial_validation_metrics,
            "history": [],
        }
        return model.eval(), metadata

    pretraining = {"performed": False}
    if isinstance(model, LearnedSinkhornMatcher) and bool(
        config.get("self_supervised_cost_pretraining", False)
    ):
        pretraining = _pretrain_learned_cost(
            model,
            train_samples,
            validation_samples,
            seed,
            settings,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    best_macro_f1, best_loss = _validation_score(model, validation_samples)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_tau = float(getattr(model, "temp", config["tau_start"]))
    epochs_without_improvement = 0
    patience_macro_f1 = best_macro_f1
    patience_loss = best_loss
    history: list[dict[str, float | int]] = []
    start_time = time.perf_counter()
    batch_size = int(settings["batch_size"])
    order_rng = np.random.default_rng(seed)
    first_epoch_order_hash = ""

    for epoch in range(1, max_epochs + 1):
        if isinstance(
            model,
            (
                DifferentiableIncidenceMatchingNetwork,
                LearnedSinkhornMatcher,
            ),
        ):
            fraction = epoch / max_epochs
            tau = config["tau_start"] * (
                config["tau_final"] / config["tau_start"]
            ) ** fraction
            model.temp = float(tau)
        order = order_rng.permutation(len(train_samples))
        if epoch == 1:
            first_epoch_order_hash = hashlib.sha256(order.tobytes()).hexdigest()
        epoch_losses: list[float] = []
        epoch_gradient_norms: list[float] = []
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            batch = [train_samples[int(index)] for index in batch_indices]
            bx, by, labels = _tensor_batch(batch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx, by)
            classification_loss = F.binary_cross_entropy_with_logits(
                logits, labels
            )
            positives = logits[labels > 0.5]
            negatives = logits[labels <= 0.5]
            if len(positives) and len(negatives):
                ranking_loss = F.softplus(
                    float(settings["pairwise_ranking_margin"])
                    - (
                        positives.unsqueeze(1)
                        - negatives.unsqueeze(0)
                    )
                ).mean()
            else:
                ranking_loss = logits.new_zeros(())
            loss = classification_loss + float(
                settings["pairwise_ranking_loss_weight"]
            ) * ranking_loss
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(settings["gradient_clip"])
                ).item()
            )
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            epoch_gradient_norms.append(gradient_norm)
        validation_macro_f1, validation_loss = _validation_score(
            model, validation_samples
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "validation_macro_f1": validation_macro_f1,
                "validation_loss": validation_loss,
                "gradient_norm": float(np.mean(epoch_gradient_norms)),
                "temperature": float(getattr(model, "temp", 0.0)),
            }
        )
        improved = validation_macro_f1 > best_macro_f1 + 1e-12 or (
            abs(validation_macro_f1 - best_macro_f1) <= 1e-12
            and validation_loss < best_loss - 1e-12
        )
        if improved:
            best_macro_f1 = validation_macro_f1
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_tau = float(getattr(model, "temp", config["tau_start"]))
        significant_improvement = (
            validation_macro_f1 > patience_macro_f1 + 1e-12
            or (
                abs(validation_macro_f1 - patience_macro_f1) <= 1e-12
                and validation_loss < patience_loss - 1e-4
            )
        )
        if significant_improvement:
            patience_macro_f1 = validation_macro_f1
            patience_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    metadata = {
        "model": model_name,
        "seed": seed,
        "selection_epoch": best_epoch,
        "selection_criterion": spec["gate_c"]["checkpoint_selection"],
        "validation_macro_f1": best_macro_f1,
        "validation_loss": best_loss,
        "selected_temperature": best_tau,
        "epochs_executed": len(history),
        "runtime_seconds": time.perf_counter() - start_time,
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "train_sinkhorn_iterations": int(config["train_iters"]),
        "evaluation_sinkhorn_iterations": int(config["eval_iters"]),
        "optimization_performed": True,
        "first_epoch_order_sha256": first_epoch_order_hash,
        "initial_train_metrics": initial_train_metrics,
        "initial_validation_metrics": initial_validation_metrics,
        "self_supervised_cost_pretraining": pretraining,
        "history": history,
    }
    return model, metadata


def run_gate_a() -> dict[str, Any]:
    spec = load_spec()
    splits = load_splits()
    rows: list[dict[str, Any]] = []
    oracle_failures: list[str] = []
    for split, samples in splits.items():
        for sample in samples:
            bx = np.asarray(sample["B_x"], dtype=np.float64)
            by = np.asarray(sample["B_y"], dtype=np.float64)
            prediction = exact_label(bx, by)
            cost = exact_hungarian_cost(bx, by)
            if prediction != int(sample["label"]):
                oracle_failures.append(sample["sample_id"])
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "split": split,
                    "label": sample["label"],
                    "oracle_prediction": prediction,
                    "hard_assignment_cost": cost,
                    "m": sample["m"],
                    "k": sample["k"],
                }
            )
    predictions_path = RESULT_DIR / "gate_a_predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)

    reference_samples = [
        splits["train"][0],
        splits["train"][1],
        splits["ood_size_test"][0],
        splits["ood_size_test"][1],
    ]
    model = initialize_model(
        load_config("DeterministicDifferentiableMatching")
    ).double().eval()
    bx, by, _ = _tensor_batch(reference_samples, torch.float64)
    with torch.no_grad():
        _, auxiliary, _ = model(bx, by, return_aux=True)
        coupling = model.sinkhorn(
            auxiliary["C"],
            auxiliary["mask_x"],
            auxiliary["mask_y"],
            num_iters=int(spec["gate_a"]["sinkhorn_iterations"]),
        )
    expected_rows = auxiliary["mask_x"] / auxiliary["mask_x"].sum(
        dim=1, keepdim=True
    )
    expected_columns = auxiliary["mask_y"] / auxiliary["mask_y"].sum(
        dim=1, keepdim=True
    )
    mass_error = float((coupling.sum((1, 2)) - 1).abs().max().item())
    row_residual = float(
        (coupling.sum(2) - expected_rows).abs().max().item()
    )
    column_residual = float(
        (coupling.sum(1) - expected_columns).abs().max().item()
    )
    observed_cost = (
        coupling * auxiliary["C_base"]
    ).sum((1, 2)).detach().cpu().numpy()
    tolerance = float(spec["gate_a"]["structural_tolerance"])
    status = (
        "PASS"
        if not oracle_failures
        and mass_error <= tolerance
        and row_residual <= tolerance
        and column_residual <= tolerance
        else "FAIL"
    )
    report = {
        "schema_version": "phase3-gate-a-v2",
        "status": status,
        "oracle_sample_count": len(rows),
        "oracle_accuracy": 1.0 - len(oracle_failures) / len(rows),
        "oracle_failed_sample_ids": oracle_failures,
        "coupling_normalization": "total mass 1; every valid row and column has marginal 1/m",
        "expected_total_mass": 1.0,
        "expected_hard_matched_entry": "1/m, not 1.0",
        "sinkhorn_iterations": int(spec["gate_a"]["sinkhorn_iterations"]),
        "maximum_total_mass_error": mass_error,
        "maximum_row_marginal_residual": row_residual,
        "maximum_column_marginal_residual": column_residual,
        "tolerance": tolerance,
        "reference_inputs": [
            {
                "sample_id": sample["sample_id"],
                "B_x": sample["B_x"],
                "B_y": sample["B_y"],
                "label": sample["label"],
                "m": sample["m"],
                "expected_row_column_marginal": 1.0 / sample["m"],
                "observed_matching_cost": float(observed_cost[index]),
            }
            for index, sample in enumerate(reference_samples)
        ],
        "predictions_path": str(predictions_path.relative_to(PROJECT_ROOT)),
    }
    (RESULT_DIR / "gate_a_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def run_gate_b() -> dict[str, Any]:
    spec = load_spec()
    train_samples = load_splits(["train"])["train"]
    selected = [
        sample for sample in train_samples if sample["label"] == 0
    ][:32] + [
        sample for sample in train_samples if sample["label"] == 1
    ][:32]
    per_seed: list[dict[str, Any]] = []
    for seed in spec["seeds"]:
        model, training = train_model(
            spec["primary_model"],
            seed,
            selected,
            selected,
            max_epochs=int(spec["gate_b"]["epochs"]),
            patience=20,
        )
        labels, probabilities = predict(model, selected, batch_size=64)
        metrics = _metrics(labels, probabilities)
        predictions = (probabilities >= 0.5).astype(int)
        failed_ids = [
            sample["sample_id"]
            for sample, label, prediction in zip(selected, labels, predictions)
            if int(label) != int(prediction)
        ]
        passed = (
            metrics["accuracy"] >= spec["gate_b"]["accuracy_threshold"]
            and metrics["macro_f1"] >= spec["gate_b"]["macro_f1_threshold"]
            and min(metrics["class_0_f1"], metrics["class_1_f1"])
            >= spec["gate_b"]["per_class_f1_threshold"]
        )
        history = training["history"]
        per_seed.append(
            {
                "seed": seed,
                "sample_count": len(selected),
                **metrics,
                "failed_sample_ids": failed_ids,
                "accuracy_threshold": spec["gate_b"]["accuracy_threshold"],
                "macro_f1_threshold": spec["gate_b"]["macro_f1_threshold"],
                "per_class_f1_threshold": spec["gate_b"]["per_class_f1_threshold"],
                "selection_epoch": training["selection_epoch"],
                "epochs_executed": training["epochs_executed"],
                "final_temperature": training["selected_temperature"],
                "gradient_norm_first": history[0]["gradient_norm"],
                "gradient_norm_last": history[-1]["gradient_norm"],
                "loss_first": history[0]["train_loss"],
                "loss_last": history[-1]["train_loss"],
                "sinkhorn_iterations": load_config(spec["primary_model"])[
                    "eval_iters"
                ],
                "status": "PASS" if passed else "FAIL",
            }
        )
    report = {
        "schema_version": "phase3-gate-b-v2",
        "model": spec["primary_model"],
        "status": "PASS"
        if all(row["status"] == "PASS" for row in per_seed)
        else "FAIL",
        "completed_seed_count": len(per_seed),
        "required_seed_count": len(spec["seeds"]),
        "per_seed": per_seed,
    }
    (RESULT_DIR / "gate_b_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _checkpoint_payload(
    model: torch.nn.Module, metadata: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "phase3-checkpoint-v2",
        "model_name": metadata["model"],
        "seed": metadata["seed"],
        "model_config": load_config(metadata["model"]),
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }


def run_model_seed(
    model_name: str,
    seed: int,
    splits: dict[str, list[dict[str, Any]]],
    *,
    resume: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Run and durably record one model/seed without repeating completed work."""

    RUN_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    record_path = RUN_RECORD_DIR / f"{model_name}_seed{seed}.json"
    if resume and record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        checkpoint_path = PROJECT_ROOT / record["checkpoint"]
        prediction_paths = [
            PROJECT_ROOT / row["prediction_path"]
            for row in record["metric_rows"]
        ]
        if (
            checkpoint_path.exists()
            and file_sha256(checkpoint_path) == record["checkpoint_sha256"]
            and all(
                path.exists()
                and file_sha256(path)
                == row["prediction_sha256"]
                for path, row in zip(prediction_paths, record["metric_rows"])
            )
        ):
            return record["metric_rows"], record.get("gate_c_row")

    spec = load_spec()
    model, metadata = train_model(
        model_name, seed, splits["train"], splits["validation"]
    )
    checkpoint_path = CHECKPOINT_DIR / f"{model_name}_seed{seed}.pt"
    torch.save(_checkpoint_payload(model, metadata), checkpoint_path)
    checkpoint_sha256 = file_sha256(checkpoint_path)
    metric_rows: list[dict[str, Any]] = []
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
                "optimization_performed": metadata["optimization_performed"],
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

    gate_c_row: dict[str, Any] | None = None
    if model_name == spec["primary_model"]:
        train_row = next(
            row for row in metric_rows if row["split"] == "train"
        )
        passed = (
            train_row["accuracy"]
            >= spec["gate_c"]["train_accuracy_threshold"]
            and train_row["balanced_accuracy"]
            >= spec["gate_c"]["train_balanced_accuracy_threshold"]
            and train_row["macro_f1"]
            >= spec["gate_c"]["train_macro_f1_threshold"]
            and min(
                train_row["class_0_recall"],
                train_row["class_1_recall"],
            )
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
            "initial_validation_metrics": metadata[
                "initial_validation_metrics"
            ],
            "selection_epoch": metadata["selection_epoch"],
            "selection_criterion": metadata["selection_criterion"],
            "selected_temperature": metadata["selected_temperature"],
            "train_sinkhorn_iterations": metadata[
                "train_sinkhorn_iterations"
            ],
            "evaluation_sinkhorn_iterations": metadata[
                "evaluation_sinkhorn_iterations"
            ],
            "epochs_executed": metadata["epochs_executed"],
            "first_epoch_order_sha256": metadata[
                "first_epoch_order_sha256"
            ],
            "checkpoint": str(
                checkpoint_path.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
            "checkpoint_sha256": checkpoint_sha256,
        }
    record = {
        "schema_version": "phase3-model-seed-record-v1",
        "model": model_name,
        "seed": seed,
        "checkpoint": str(
            checkpoint_path.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_sha256,
        "metric_rows": metric_rows,
        "gate_c_row": gate_c_row,
        "training_metadata": metadata,
    }
    record_path.write_text(
        json.dumps(record, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return metric_rows, gate_c_row


def run_experiments() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    spec = load_spec()
    splits = load_splits()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    RUN_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    generate_configs(RESULT_DIR / "model_configs")
    metric_rows: list[dict[str, Any]] = []
    gate_c_rows: list[dict[str, Any]] = []
    for model_name in model_names():
        for seed in spec["seeds"]:
            rows, gate_c_row = run_model_seed(
                model_name, int(seed), splits, resume=True
            )
            metric_rows.extend(rows)
            if gate_c_row is not None:
                gate_c_rows.append(gate_c_row)
    per_seed = pd.DataFrame(metric_rows)
    per_seed_path = RESULT_DIR / "per_seed_metrics.csv"
    per_seed.to_csv(per_seed_path, index=False)
    numeric_metrics = [
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
    aggregate = (
        per_seed.groupby(["model", "role", "split"], sort=False)[numeric_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in aggregate.columns
    ]
    counts = (
        per_seed.groupby(["model", "role", "split"], sort=False)["seed"]
        .nunique()
        .rename("seed_count")
        .reset_index()
    )
    aggregate = aggregate.merge(counts, on=["model", "role", "split"])
    aggregate_path = RESULT_DIR / "aggregate_metrics.csv"
    aggregate.to_csv(aggregate_path, index=False)
    gate_c = {
        "schema_version": "phase3-gate-c-v2",
        "model": spec["primary_model"],
        "status": "PASS"
        if len(gate_c_rows) == len(spec["seeds"])
        and all(row["status"] == "PASS" for row in gate_c_rows)
        else "FAIL",
        "completed_seed_count": len(gate_c_rows),
        "required_seed_count": len(spec["seeds"]),
        "thresholds": spec["gate_c"],
        "per_seed": gate_c_rows,
    }
    (RESULT_DIR / "gate_c_report.json").write_text(
        json.dumps(gate_c, indent=2) + "\n", encoding="utf-8"
    )
    return per_seed, aggregate, gate_c


def rerun_affected_model(model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rerun only one repaired model while preserving every unaffected run."""

    spec = load_spec()
    if model_name not in model_names():
        raise KeyError(model_name)
    splits = load_splits()
    existing = pd.read_csv(RESULT_DIR / "per_seed_metrics.csv")
    retained = existing[existing["model"] != model_name].copy()
    replacement_rows: list[dict[str, Any]] = []
    for seed in spec["seeds"]:
        model, metadata = train_model(
            model_name, seed, splits["train"], splits["validation"]
        )
        checkpoint_path = CHECKPOINT_DIR / f"{model_name}_seed{seed}.pt"
        torch.save(_checkpoint_payload(model, metadata), checkpoint_path)
        metadata["checkpoint_sha256"] = file_sha256(checkpoint_path)
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
            replacement_rows.append(
                {
                    "model": model_name,
                    "role": spec["model_roles"][model_name],
                    "seed": seed,
                    "split": split_name,
                    **_metrics(labels, probabilities),
                    "selection_epoch": metadata["selection_epoch"],
                    "selected_temperature": metadata["selected_temperature"],
                    "checkpoint_sha256": metadata["checkpoint_sha256"],
                    "prediction_sha256": file_sha256(prediction_path),
                }
            )
    per_seed = pd.concat(
        [retained, pd.DataFrame(replacement_rows)], ignore_index=True
    ).sort_values(["model", "seed", "split"])
    per_seed.to_csv(RESULT_DIR / "per_seed_metrics.csv", index=False)
    numeric_metrics = [
        column
        for column in replacement_rows[0]
        if column
        not in {
            "model",
            "role",
            "seed",
            "split",
            "selection_epoch",
            "selected_temperature",
            "checkpoint_sha256",
            "prediction_sha256",
        }
    ]
    aggregate = (
        per_seed.groupby(["model", "role", "split"], sort=False)[numeric_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in aggregate.columns
    ]
    counts = (
        per_seed.groupby(["model", "role", "split"], sort=False)["seed"]
        .nunique()
        .rename("seed_count")
        .reset_index()
    )
    aggregate = aggregate.merge(counts, on=["model", "role", "split"])
    aggregate.to_csv(RESULT_DIR / "aggregate_metrics.csv", index=False)
    return per_seed, aggregate


def verify_checkpoint_roundtrip() -> dict[str, Any]:
    failures: list[str] = []
    checked = 0
    sample = load_splits(["iid_test"])["iid_test"][:8]
    bx, by, _ = _tensor_batch(sample)
    for checkpoint_path in sorted(CHECKPOINT_DIR.glob("*.pt")):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        first = initialize_model(payload["model_config"]).to(dtype=DTYPE).eval()
        second = initialize_model(payload["model_config"]).to(dtype=DTYPE).eval()
        first.load_state_dict(payload["state_dict"])
        second.load_state_dict(payload["state_dict"])
        with torch.no_grad():
            error = float((first(bx, by) - second(bx, by)).abs().max().item())
        checked += 1
        if error != 0.0:
            failures.append(f"{checkpoint_path.name}: {error}")
    expected = len(model_names()) * len(load_spec()["seeds"])
    report = {
        "status": "PASS" if checked == expected and not failures else "FAIL",
        "checked_checkpoint_count": checked,
        "expected_checkpoint_count": expected,
        "failures": failures,
    }
    (RESULT_DIR / "checkpoint_roundtrip.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run_gate_a(), indent=2))
    print(json.dumps(run_gate_b(), indent=2))
    _, aggregate_frame, gate_c_report = run_experiments()
    print(aggregate_frame.to_string(index=False))
    print(json.dumps(gate_c_report, indent=2))
    print(json.dumps(verify_checkpoint_roundtrip(), indent=2))
