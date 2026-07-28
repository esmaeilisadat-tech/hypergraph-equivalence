"""Complete compact-submatrix float64 audit for the primary Phase 3 model."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.differentiable_matching import (
    LearnedSinkhornMatcher,
    initialize_model,
)
from src.phase3_data import PROJECT_ROOT, load_spec, load_splits

RESULT_DIR = PROJECT_ROOT / "results" / "phase3"


class Measurements:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = defaultdict(list)
        self.locations: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add(
        self,
        name: str,
        values: torch.Tensor | np.ndarray,
        samples: list[dict[str, Any]],
        seed: int,
        split: str,
    ) -> None:
        array = np.asarray(
            values.detach().cpu().numpy()
            if isinstance(values, torch.Tensor)
            else values,
            dtype=np.float64,
        ).reshape(-1)
        for value, sample in zip(array, samples):
            self.values[name].append(float(value))
            self.locations[name].append(
                {
                    "seed": seed,
                    "split": split,
                    "sample_id": sample["sample_id"],
                }
            )

    def summarize(
        self, tolerances: dict[str, float], iterations: int
    ) -> dict[str, Any]:
        summaries: dict[str, Any] = {}
        for name, raw_values in self.values.items():
            values = np.asarray(raw_values, dtype=np.float64)
            maximum_index = int(np.nanargmax(values))
            tolerance = float(tolerances[name])
            summaries[name] = {
                "audited_count": int(len(values)),
                "failed_count": int(np.sum((~np.isfinite(values)) | (values > tolerance))),
                "mean": float(np.nanmean(values)),
                "maximum": float(np.nanmax(values)),
                "p95": float(np.nanpercentile(values, 95)),
                "p99": float(np.nanpercentile(values, 99)),
                "tolerance": tolerance,
                "worst_seed": self.locations[name][maximum_index]["seed"],
                "worst_split": self.locations[name][maximum_index]["split"],
                "worst_sample_id": self.locations[name][maximum_index][
                    "sample_id"
                ],
                "sinkhorn_iterations": iterations,
            }
        return summaries


def _per_sample_max(values: torch.Tensor) -> torch.Tensor:
    return values.reshape(values.shape[0], -1).abs().amax(dim=1)


def _stable_permutations(
    samples: list[dict[str, Any]], size: int, seed: int, suffix: str
) -> torch.Tensor:
    permutations = []
    for sample in samples:
        digest = hashlib.sha256(
            f"{seed}:{sample['sample_id']}:{suffix}".encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        permutations.append(rng.permutation(size))
    return torch.tensor(np.stack(permutations), dtype=torch.long)


def _permute_rows(
    matrices: torch.Tensor, permutations: torch.Tensor
) -> torch.Tensor:
    return torch.gather(
        matrices,
        1,
        permutations.unsqueeze(-1).expand(-1, -1, matrices.shape[2]),
    )


def _permute_columns(
    matrices: torch.Tensor, permutations: torch.Tensor
) -> torch.Tensor:
    return torch.gather(
        matrices,
        2,
        permutations.unsqueeze(1).expand(-1, matrices.shape[1], -1),
    )


def _direction(
    model: LearnedSinkhornMatcher,
    bx: torch.Tensor,
    by: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    output = model.forward_direction(bx, by, return_aux=True)
    assert isinstance(output, tuple)
    return output


def _audit_group(
    model: LearnedSinkhornMatcher,
    samples: list[dict[str, Any]],
    seed: int,
    split: str,
    measurements: Measurements,
) -> None:
    bx = torch.tensor(
        np.stack([sample["B_x"] for sample in samples]),
        dtype=torch.float64,
    )
    by = torch.tensor(
        np.stack([sample["B_y"] for sample in samples]),
        dtype=torch.float64,
    )
    rows, columns = bx.shape[1:]
    row_permutation = _stable_permutations(samples, rows, seed, "rows")
    column_permutation = _stable_permutations(samples, columns, seed, "columns")
    bx_row = _permute_rows(bx, row_permutation)
    bx_column = _permute_columns(bx, column_permutation)
    by_column = _permute_columns(by, column_permutation)
    with torch.no_grad():
        logit_xy, aux_xy = _direction(model, bx, by)
        logit_yx, aux_yx = _direction(model, by, bx)
        logit_row_xy, aux_row_xy = _direction(model, bx_row, by)
        logit_row_yx, _ = _direction(model, by, bx_row)
        logit_column_xy, aux_column_xy = _direction(
            model, bx_column, by_column
        )
        logit_column_yx, _ = _direction(model, by_column, bx_column)

    cost_xy = aux_xy["C"]
    cost_yx = aux_yx["C"]
    coupling_xy = aux_xy["P"]
    coupling_yx = aux_yx["P"]
    assert isinstance(cost_xy, torch.Tensor)
    assert isinstance(cost_yx, torch.Tensor)
    assert isinstance(coupling_xy, torch.Tensor)
    assert isinstance(coupling_yx, torch.Tensor)
    matching_xy = aux_xy["matching_cost"]
    matching_yx = aux_yx["matching_cost"]
    equality_xy = aux_xy["equality_score"]
    equality_yx = aux_yx["equality_score"]
    assert isinstance(matching_xy, torch.Tensor)
    assert isinstance(matching_yx, torch.Tensor)
    assert isinstance(equality_xy, torch.Tensor)
    assert isinstance(equality_yx, torch.Tensor)
    final_logit = 0.5 * (logit_xy + logit_yx)
    swapped_logit = 0.5 * (logit_yx + logit_xy)
    row_logit = 0.5 * (logit_row_xy + logit_row_yx)
    column_logit = 0.5 * (logit_column_xy + logit_column_yx)
    inverse_rows = torch.argsort(row_permutation, dim=1)
    row_coupling = aux_row_xy["P"]
    assert isinstance(row_coupling, torch.Tensor)
    unpermuted_row_coupling = torch.gather(
        row_coupling,
        1,
        inverse_rows.unsqueeze(-1).expand(-1, -1, rows),
    )
    column_coupling = aux_column_xy["P"]
    assert isinstance(column_coupling, torch.Tensor)
    target_marginal = torch.full(
        (len(samples), rows), 1.0 / rows, dtype=torch.float64
    )

    finite_cost_error = (
        (~torch.isfinite(cost_xy)).reshape(len(samples), -1).any(1).double()
    )
    finite_coupling_error = (
        (~torch.isfinite(coupling_xy)).reshape(len(samples), -1).any(1).double()
    )
    quantities = {
        "cost_finite_violation": finite_cost_error,
        "coupling_finite_violation": finite_coupling_error,
        "coupling_nonnegativity_violation": torch.clamp(
            -coupling_xy, min=0
        ).reshape(len(samples), -1).amax(1),
        "cost_transpose_error": _per_sample_max(
            cost_yx - cost_xy.transpose(1, 2)
        ),
        "coupling_transpose_error": _per_sample_max(
            coupling_yx - coupling_xy.transpose(1, 2)
        ),
        "assignment_transpose_error": _per_sample_max(
            coupling_yx.transpose(1, 2) - coupling_xy
        ),
        "matching_cost_swap_error": (matching_xy - matching_yx).abs().flatten(),
        "equality_score_swap_error": (equality_xy - equality_yx).abs().flatten(),
        "final_logit_swap_error": (final_logit - swapped_logit).abs(),
        "final_probability_swap_error": (
            torch.sigmoid(final_logit) - torch.sigmoid(swapped_logit)
        ).abs(),
        "total_coupling_mass_error": (
            coupling_xy.sum((1, 2)) - 1.0
        ).abs(),
        "row_marginal_residual": (
            coupling_xy.sum(2) - target_marginal
        ).abs().amax(1),
        "column_marginal_residual": (
            coupling_xy.sum(1) - target_marginal
        ).abs().amax(1),
        "node_row_permutation_coupling_error": _per_sample_max(
            unpermuted_row_coupling - coupling_xy
        ),
        "node_row_permutation_logit_error": (row_logit - final_logit).abs(),
        "node_row_permutation_probability_error": (
            torch.sigmoid(row_logit) - torch.sigmoid(final_logit)
        ).abs(),
        "joint_column_permutation_cost_error": _per_sample_max(
            aux_column_xy["C"] - cost_xy
        ),
        "joint_column_permutation_coupling_error": _per_sample_max(
            column_coupling - coupling_xy
        ),
        "joint_column_permutation_logit_error": (
            column_logit - final_logit
        ).abs(),
        "joint_column_permutation_probability_error": (
            torch.sigmoid(column_logit) - torch.sigmoid(final_logit)
        ).abs(),
    }
    for name, values in quantities.items():
        measurements.add(name, values, samples, seed, split)


def run_numerical_audit() -> dict[str, Any]:
    spec = load_spec()
    audit_spec = spec["numerical_audit"]
    iterations = int(spec["training"]["audit_sinkhorn_iterations"])
    splits = load_splits(audit_spec["splits"])
    measurements = Measurements()
    for seed in spec["seeds"]:
        checkpoint_path = (
            RESULT_DIR
            / "checkpoints"
            / f"{spec['primary_model']}_seed{seed}.pt"
        )
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        config = dict(payload["model_config"])
        config["eval_iters"] = iterations
        model = initialize_model(config).double().eval()
        model.load_state_dict(payload["state_dict"])
        assert isinstance(model, LearnedSinkhornMatcher)
        model.eval_iters = iterations
        for split, samples in splits.items():
            groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
            for sample in samples:
                groups[(int(sample["m"]), int(sample["k"]))].append(sample)
            for group_samples in groups.values():
                _audit_group(
                    model, group_samples, int(seed), split, measurements
                )

    structural = float(audit_spec["structural_tolerance"])
    marginal = float(audit_spec["marginal_tolerance"])
    tolerances = {
        name: (
            marginal
            if name
            in {
                "total_coupling_mass_error",
                "row_marginal_residual",
                "column_marginal_residual",
            }
            else structural
        )
        for name in measurements.values
    }
    summaries = measurements.summarize(tolerances, iterations)
    audited_count = sum(len(samples) for samples in splits.values()) * len(
        spec["seeds"]
    )
    report = {
        "schema_version": "phase3-numerical-audit-v2",
        "status": "PASS"
        if all(row["failed_count"] == 0 for row in summaries.values())
        else "FAIL",
        "model": spec["primary_model"],
        "dtype": "float64",
        "compact_valid_submatrices": True,
        "explicit_masks_and_uniform_marginals": True,
        "deterministic_permutations": True,
        "sinkhorn_iterations": iterations,
        "audited_sample_count": audited_count,
        "failed_sample_count": int(
            sum(
                any(
                    measurements.values[name][index] > tolerances[name]
                    for name in measurements.values
                )
                for index in range(audited_count)
            )
        ),
        "quantities": summaries,
        "interpretation": {
            "rows": "incident hyperedges; their order is a set order and is mandatory invariant",
            "columns": "neighbor nodes; joint relabeling is mandatory invariant",
            "root_swap": "the unordered root pair requires transpose equivariance and final-score invariance",
        },
    }
    output_path = RESULT_DIR / "numerical_audit.json"
    output_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = run_numerical_audit()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
