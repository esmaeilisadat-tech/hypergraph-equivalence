from __future__ import annotations

import copy
import inspect

import numpy as np
import torch

from src.config_manager import load_config, model_names
from src.differentiable_matching import (
    LearnedSinkhornMatcher,
    initialize_model,
)
from src.phase3_data import (
    REQUIRED_SPLITS,
    collate_samples,
    exact_hungarian_cost,
    exact_label,
    generate_all_splits,
    load_spec,
    load_splits,
)


def test_locked_spec_identity() -> None:
    spec = load_spec()
    assert spec["primary_model"] == "LearnedSinkhornMatcher"
    assert spec["benchmark_version"] == "learned_benchmark_v2_final"
    assert (
        spec["model_roles"]["DeterministicDifferentiableMatching"]
        == "algorithmic_baseline"
    )
    assert spec["model_roles"]["ExactHungarianOracle"] == "oracle"
    assert spec["seeds"] == [42, 100, 2026, 777, 999]
    assert set(model_names()) == set(spec["model_roles"])
    assert spec["gate_b"]["accuracy_threshold"] == 0.99
    assert spec["gate_b"]["macro_f1_threshold"] == 0.99


def test_split_generation_is_byte_semantically_deterministic() -> None:
    spec = copy.deepcopy(load_spec())
    for split_settings in spec["splits"].values():
        split_settings["pairs"] = 3
    assert generate_all_splits(spec) == generate_all_splits(spec)


def test_locked_splits_have_valid_exact_labels_and_margins() -> None:
    splits = load_splits()
    assert tuple(splits) == REQUIRED_SPLITS
    occupied_ids: set[str] = set()
    occupied_bases: set[str] = set()
    occupied_structures: set[str] = set()
    spec = load_spec()
    for split, samples in splits.items():
        assert len(samples) == spec["splits"][split]["samples"]
        assert sum(sample["label"] == 0 for sample in samples) == len(samples) // 2
        assert sum(sample["label"] == 1 for sample in samples) == len(samples) // 2
        split_bases = {sample["base_hash"] for sample in samples}
        split_structures = {sample["structure_hash"] for sample in samples}
        assert not (split_bases & occupied_bases)
        assert not (split_structures & occupied_structures)
        assert len(split_structures) == len(samples)
        for sample in samples:
            bx = np.asarray(sample["B_x"])
            by = np.asarray(sample["B_y"])
            assert sample["sample_id"] not in occupied_ids
            assert bx.shape == by.shape == (sample["m"], sample["k"])
            assert np.all(np.isin(bx, [0, 1]))
            assert np.all(np.isin(by, [0, 1]))
            assert np.array_equal(np.sort(bx.sum(1)), np.sort(by.sum(1)))
            assert np.array_equal(bx.sum(0), by.sum(0))
            assert exact_label(bx, by) == sample["label"]
            assert (exact_hungarian_cost(bx, by) == 0) == bool(sample["label"])
            occupied_ids.add(sample["sample_id"])
        occupied_bases.update(split_bases)
        occupied_structures.update(split_structures)


def test_all_models_are_root_row_and_joint_column_invariant() -> None:
    samples = load_splits(["train"])["train"][:8]
    bx, by, _ = collate_samples(samples, dtype=torch.float64)
    row_permutation = torch.arange(bx.shape[1] - 1, -1, -1)
    column_permutation = torch.arange(bx.shape[2] - 1, -1, -1)
    for model_name in model_names():
        torch.manual_seed(42)
        model = initialize_model(load_config(model_name)).double().eval()
        if hasattr(model, "eval_iters"):
            model.eval_iters = 500
        with torch.no_grad():
            original = model(bx, by)
            swapped = model(by, bx)
            row_permuted = model(bx[:, row_permutation], by)
            column_permuted = model(
                bx[:, :, column_permutation], by[:, :, column_permutation]
            )
        assert torch.allclose(original, swapped, atol=1e-8, rtol=0)
        assert torch.allclose(original, row_permuted, atol=1e-8, rtol=0)
        assert torch.allclose(original, column_permuted, atol=1e-8, rtol=0)


def test_learned_primary_has_no_fixed_l1_or_metadata_input() -> None:
    model = initialize_model(load_config("LearnedSinkhornMatcher"))
    assert isinstance(model, LearnedSinkhornMatcher)
    signature = inspect.signature(model.forward)
    assert list(signature.parameters) == ["bx", "by", "return_aux"]
    source = inspect.getsource(LearnedSinkhornMatcher)
    assert "C_base" not in source
    assert "hungarian_coupling" not in source
    assert "linear_sum_assignment" not in source
    assert "sample_id" not in source
    assert "structure_hash" not in source


def test_learned_primary_initial_states_are_seed_dependent() -> None:
    states = []
    for seed in [42, 100, 2026, 777, 999]:
        torch.manual_seed(seed)
        model = initialize_model(load_config("LearnedSinkhornMatcher"))
        states.append(
            torch.cat(
                [
                    parameter.detach().flatten()
                    for parameter in model.parameters()
                ]
            )
        )
    assert all(
        not torch.equal(states[0], candidate)
        for candidate in states[1:]
    )


def test_float64_sinkhorn_mass_marginals_and_transpose() -> None:
    samples = load_splits(["ood_size_test"])["ood_size_test"][:8]
    bx, by, _ = collate_samples(samples, dtype=torch.float64)
    model = initialize_model(
        load_config("LearnedSinkhornMatcher")
    ).double().eval()
    with torch.no_grad():
        _, xy, yx = model(bx, by, return_aux=True)
        pxy = model.sinkhorn(
            xy["C"], xy["mask_x"], xy["mask_y"], num_iters=500
        )
        pyx = model.sinkhorn(
            yx["C"], yx["mask_x"], yx["mask_y"], num_iters=500
        )
    expected_x = xy["mask_x"] / xy["mask_x"].sum(1, keepdim=True)
    expected_y = xy["mask_y"] / xy["mask_y"].sum(1, keepdim=True)
    assert torch.all(torch.isfinite(pxy))
    assert torch.all(pxy >= 0)
    assert torch.allclose(pxy.sum((1, 2)), torch.ones(8, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(pxy.sum(2), expected_x, atol=1e-10, rtol=0)
    assert torch.allclose(pxy.sum(1), expected_y, atol=1e-10, rtol=0)
    assert torch.allclose(xy["C"], yx["C"].transpose(1, 2), atol=1e-12, rtol=0)
    assert torch.allclose(pxy, pyx.transpose(1, 2), atol=1e-10, rtol=0)
