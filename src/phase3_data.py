"""Deterministic generation, loading, and integrity checks for Phase 3 splits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "phase3_spec.json"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
REQUIRED_SPLITS = (
    "train",
    "validation",
    "iid_test",
    "ood_template_test",
    "ood_size_test",
)
SPLIT_OFFSETS = {
    "train": 0,
    "validation": 10_000,
    "iid_test": 20_000,
    "ood_template_test": 30_000,
    "ood_size_test": 40_000,
}


def load_spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_multiset(matrix: np.ndarray) -> Counter[tuple[int, ...]]:
    return Counter(tuple(int(v) for v in row) for row in matrix)


def exact_label(bx: np.ndarray, by: np.ndarray) -> int:
    return int(row_multiset(bx) == row_multiset(by))


def exact_hungarian_cost(bx: np.ndarray, by: np.ndarray) -> float:
    if bx.shape != by.shape:
        return float("inf")
    cost = np.abs(bx[:, None, :] - by[None, :, :]).sum(axis=2)
    row_index, column_index = linear_sum_assignment(cost)
    return float(cost[row_index, column_index].sum())


def _colored_pair_graph(bx: np.ndarray, by: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    for column in range(bx.shape[1]):
        graph.add_node(("n", column), color="neighbor")
    for side, matrix in (("x", bx), ("y", by)):
        for row_index, row in enumerate(matrix):
            row_node = (side, row_index)
            graph.add_node(row_node, color=side)
            for column in np.flatnonzero(row):
                graph.add_edge(row_node, ("n", int(column)))
    return graph


def invariant_structure_hash(bx: np.ndarray, by: np.ndarray) -> str:
    """WL digest invariant to row order, joint columns, and root swap."""

    digests: list[str] = []
    for first, second in ((bx, by), (by, bx)):
        graph = _colored_pair_graph(first, second)
        digests.append(
            nx.weisfeiler_lehman_graph_hash(
                graph,
                node_attr="color",
                iterations=8,
                digest_size=32,
            )
        )
    return min(digests)


def _base_hash(matrix: np.ndarray) -> str:
    graph = nx.Graph()
    for row in range(matrix.shape[0]):
        graph.add_node(("r", row), color="row")
    for column in range(matrix.shape[1]):
        graph.add_node(("c", column), color="column")
    for row, column in np.argwhere(matrix > 0):
        graph.add_edge(("r", int(row)), ("c", int(column)))
    return nx.weisfeiler_lehman_graph_hash(
        graph, node_attr="color", iterations=8, digest_size=32
    )


def _generate_base_matrix(
    rng: np.random.Generator,
    k: int,
    m: int,
    min_size: int,
    max_size: int,
) -> np.ndarray:
    matrix = np.zeros((m, k), dtype=np.int8)
    for row in range(m):
        size = int(rng.integers(min_size, min(max_size, k) + 1))
        columns = rng.choice(k, size=size, replace=False)
        matrix[row, columns] = 1
    for column in range(k):
        if matrix[:, column].sum() == 0:
            matrix[int(rng.integers(0, m)), column] = 1
    return matrix


def _ryser_switch(
    matrix: np.ndarray, rng: np.random.Generator, switches: int
) -> np.ndarray | None:
    result = matrix.copy()
    successful = 0
    for _ in range(max(100, switches * 100)):
        row_sums = result.sum(axis=1)
        groups = [
            np.flatnonzero(row_sums == value)
            for value in np.unique(row_sums)
            if np.sum(row_sums == value) >= 2
        ]
        if not groups:
            return None
        group = groups[int(rng.integers(0, len(groups)))]
        row_a, row_b = rng.choice(group, size=2, replace=False)
        a_only = np.flatnonzero((result[row_a] == 1) & (result[row_b] == 0))
        b_only = np.flatnonzero((result[row_a] == 0) & (result[row_b] == 1))
        if len(a_only) == 0 or len(b_only) == 0:
            continue
        column_a = int(rng.choice(a_only))
        column_b = int(rng.choice(b_only))
        result[row_a, column_a] = 0
        result[row_b, column_a] = 1
        result[row_a, column_b] = 1
        result[row_b, column_b] = 0
        successful += 1
        if successful >= switches:
            return result
    return None


def _permute_representation(
    matrix: np.ndarray, rng: np.random.Generator, column_order: np.ndarray
) -> np.ndarray:
    row_order = rng.permutation(matrix.shape[0])
    return matrix[row_order][:, column_order]


def _generate_pair(
    rng: np.random.Generator,
    split_name: str,
    pair_index: int,
    k: int,
    m: int,
    switches: int,
    min_size: int,
    max_size: int,
) -> list[dict[str, Any]]:
    for _ in range(10_000):
        bx_base = _generate_base_matrix(rng, k, m, min_size, max_size)
        by_negative = _ryser_switch(bx_base, rng, switches)
        if by_negative is None or exact_label(bx_base, by_negative) == 1:
            continue
        if exact_hungarian_cost(bx_base, by_negative) <= 0:
            continue
        break
    else:
        raise RuntimeError(f"Could not generate valid pair for {split_name}:{pair_index}")

    pair_id = f"{split_name}-pair-{pair_index:04d}"
    base_hash = _base_hash(bx_base)
    samples: list[dict[str, Any]] = []
    for label, by_base in ((0, by_negative), (1, bx_base.copy())):
        column_order = rng.permutation(k)
        bx = _permute_representation(bx_base, rng, column_order)
        by = _permute_representation(by_base, rng, column_order)
        structure_hash = invariant_structure_hash(bx, by)
        sample_id = f"{split_name}-pair-{pair_index:04d}-class-{label}"
        samples.append(
            {
                "sample_id": sample_id,
                "matched_pair_id": pair_id,
                "base_hash": base_hash,
                "structure_hash": structure_hash,
                "template_family": (
                    "ood_size"
                    if split_name == "ood_size_test"
                    else "ood_unseen_multi_switch"
                    if split_name == "ood_template_test"
                    else "iid_variable_switch"
                ),
                "B_x": bx.astype(float).tolist(),
                "B_y": by.astype(float).tolist(),
                "label": label,
                "k": k,
                "m": m,
                "switches": switches,
            }
        )
    return samples


def generate_all_splits(spec: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    spec = spec or load_spec()
    master_seed = int(spec["seeds"][0])
    generated: dict[str, dict[str, Any]] = {}
    occupied_structures: set[str] = set()
    occupied_bases: set[str] = set()

    for split_name in REQUIRED_SPLITS:
        split_spec = spec["splits"][split_name]
        configured_offsets = spec.get("split_offsets", SPLIT_OFFSETS)
        generation_offset = int(configured_offsets[split_name])
        rng = np.random.default_rng(master_seed + generation_offset)
        samples: list[dict[str, Any]] = []
        pair_index = 0
        attempts = 0
        while pair_index < int(split_spec["pairs"]):
            attempts += 1
            if attempts > int(split_spec["pairs"]) * 100:
                raise RuntimeError(f"Exhausted generation attempts for {split_name}")
            k = int(
                rng.integers(split_spec["k_range"][0], split_spec["k_range"][1] + 1)
            )
            m = int(
                rng.integers(split_spec["m_range"][0], split_spec["m_range"][1] + 1)
            )
            min_size = 2
            max_size = 5
            switch_range = split_spec.get("switch_range")
            switches = (
                int(
                    rng.integers(
                        int(switch_range[0]),
                        int(switch_range[1]) + 1,
                    )
                )
                if switch_range is not None
                else int(split_spec["switches"])
            )
            pair = _generate_pair(
                rng,
                split_name,
                pair_index,
                k,
                m,
                switches,
                min_size,
                max_size,
            )
            pair_structures = {sample["structure_hash"] for sample in pair}
            pair_base = pair[0]["base_hash"]
            if (
                len(pair_structures) != len(pair)
                or pair_structures & occupied_structures
                or pair_base in occupied_bases
            ):
                continue
            samples.extend(pair)
            occupied_structures.update(pair_structures)
            occupied_bases.add(pair_base)
            pair_index += 1
        generated[split_name] = {
            "schema_version": "phase3-learned-splits-v1",
            "benchmark_version": spec.get(
                "benchmark_version", "learned_benchmark_v1"
            ),
            "seed": master_seed,
            "split": split_name,
            "generation_offset": generation_offset,
            "data": samples,
        }
    return generated


def write_all_splits(output_dir: Path = SPLIT_DIR) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = generate_all_splits()
    hashes: dict[str, str] = {}
    for split_name, payload in generated.items():
        path = output_dir / f"{split_name}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        hashes[split_name] = file_sha256(path)
    return hashes


def load_splits(
    split_names: Iterable[str] = REQUIRED_SPLITS,
) -> dict[str, list[dict[str, Any]]]:
    loaded: dict[str, list[dict[str, Any]]] = {}
    for split_name in split_names:
        path = SPLIT_DIR / f"{split_name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded[split_name] = payload["data"]
    return loaded


def collate_samples(
    samples: list[dict[str, Any]], dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_m = max(int(sample["m"]) for sample in samples)
    max_k = max(int(sample["k"]) for sample in samples)
    bx_batch = np.zeros((len(samples), max_m, max_k), dtype=np.float64)
    by_batch = np.zeros((len(samples), max_m, max_k), dtype=np.float64)
    labels = np.zeros(len(samples), dtype=np.float64)
    for index, sample in enumerate(samples):
        bx = np.asarray(sample["B_x"], dtype=np.float64)
        by = np.asarray(sample["B_y"], dtype=np.float64)
        bx_batch[index, : bx.shape[0], : bx.shape[1]] = bx
        by_batch[index, : by.shape[0], : by.shape[1]] = by
        labels[index] = float(sample["label"])
    return (
        torch.tensor(bx_batch, dtype=dtype),
        torch.tensor(by_batch, dtype=dtype),
        torch.tensor(labels, dtype=dtype),
    )
