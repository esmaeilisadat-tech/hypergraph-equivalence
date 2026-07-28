"""Complete deterministic integrity and prohibited-overlap audit."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.phase3_data import (
    PROJECT_ROOT,
    REQUIRED_SPLITS,
    SPLIT_DIR,
    collate_samples,
    exact_hungarian_cost,
    exact_label,
    file_sha256,
    generate_all_splits,
    invariant_structure_hash,
    load_spec,
    sha256_value,
)

OUTPUT_PATH = PROJECT_ROOT / "results" / "phase3" / "split_overlap_audit.json"


def get_hashes(sample: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    bx = np.asarray(sample["B_x"], dtype=np.int8)
    by = np.asarray(sample["B_y"], dtype=np.int8)
    structure_hash = invariant_structure_hash(bx, by)
    labeled_hash = sha256_value([structure_hash, int(sample["label"])])
    metadata_hash = sha256_value(sample)
    return (
        structure_hash,
        labeled_hash,
        metadata_hash,
        structure_hash,
        str(sample["sample_id"]),
        str(sample["matched_pair_id"]),
    )


def _problem(
    problems: list[dict[str, Any]],
    split: str,
    sample_id: str | None,
    reason: str,
) -> None:
    problems.append({"split": split, "sample_id": sample_id, "reason": reason})


def audit_splits() -> dict[str, Any]:
    spec = load_spec()
    fresh = generate_all_splits(spec)
    problems: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}
    structures: dict[str, set[str]] = {}
    bases: dict[str, set[str]] = {}
    matched_ids: dict[str, set[str]] = {}
    sample_ids: dict[str, set[str]] = {}

    for split_name in REQUIRED_SPLITS:
        path = SPLIT_DIR / f"{split_name}.json"
        first_text = path.read_text(encoding="utf-8")
        second_text = path.read_text(encoding="utf-8")
        payload = json.loads(first_text)
        data = payload.get("data", [])
        expected_count = int(spec["splits"][split_name]["samples"])
        labels = Counter(int(sample.get("label", -1)) for sample in data)
        local_sample_ids: list[str] = []
        local_matched_ids: list[str] = []
        local_structures: list[str] = []
        local_bases: list[str] = []
        paired: dict[str, list[dict[str, Any]]] = defaultdict(list)
        shape_failures = 0
        label_failures = 0
        mask_failures = 0
        margin_failures = 0
        hash_failures = 0

        if len(data) != expected_count:
            _problem(
                problems,
                split_name,
                None,
                f"expected {expected_count} samples, found {len(data)}",
            )
        if labels != Counter({0: expected_count // 2, 1: expected_count // 2}):
            _problem(
                problems,
                split_name,
                None,
                f"unexpected label distribution {dict(labels)}",
            )

        for sample in data:
            sample_id = str(sample.get("sample_id", ""))
            matched_id = str(sample.get("matched_pair_id", ""))
            local_sample_ids.append(sample_id)
            local_matched_ids.append(matched_id)
            paired[matched_id].append(sample)
            bx = np.asarray(sample.get("B_x", []), dtype=np.float64)
            by = np.asarray(sample.get("B_y", []), dtype=np.float64)
            expected_shape = (int(sample.get("m", -1)), int(sample.get("k", -1)))
            if bx.shape != expected_shape or by.shape != expected_shape:
                shape_failures += 1
                _problem(problems, split_name, sample_id, "matrix shape mismatch")
                continue
            if bx.size == 0 or by.size == 0:
                mask_failures += 1
                _problem(problems, split_name, sample_id, "empty matrix")
                continue
            if (
                not np.isin(bx, [0.0, 1.0]).all()
                or not np.isin(by, [0.0, 1.0]).all()
                or np.any(bx.sum(axis=1) <= 0)
                or np.any(by.sum(axis=1) <= 0)
                or np.any((bx.sum(axis=0) + by.sum(axis=0)) <= 0)
            ):
                mask_failures += 1
                _problem(
                    problems,
                    split_name,
                    sample_id,
                    "invalid binary values, empty row, or empty joint column",
                )
            if (
                sorted(bx.sum(axis=1).tolist())
                != sorted(by.sum(axis=1).tolist())
                or not np.array_equal(bx.sum(axis=0), by.sum(axis=0))
            ):
                margin_failures += 1
                _problem(
                    problems,
                    split_name,
                    sample_id,
                    "Ryser row/column margins are not preserved",
                )
            observed_label = exact_label(bx.astype(np.int8), by.astype(np.int8))
            hungarian_label = int(exact_hungarian_cost(bx, by) == 0)
            if observed_label != int(sample["label"]) or hungarian_label != int(
                sample["label"]
            ):
                label_failures += 1
                _problem(
                    problems,
                    split_name,
                    sample_id,
                    "label disagrees with exact multiset/Hungarian oracle",
                )
            structure_hash = invariant_structure_hash(
                bx.astype(np.int8), by.astype(np.int8)
            )
            local_structures.append(structure_hash)
            local_bases.append(str(sample.get("base_hash", "")))
            if structure_hash != sample.get("structure_hash"):
                hash_failures += 1
                _problem(
                    problems,
                    split_name,
                    sample_id,
                    "stored invariant structure hash is stale",
                )

        duplicate_sample_ids = [
            value for value, count in Counter(local_sample_ids).items() if count > 1
        ]
        duplicate_labeled_structures = []
        labeled_structures = [
            (sample["structure_hash"], int(sample["label"])) for sample in data
        ]
        duplicate_labeled_structures = [
            f"{key[0]}:label={key[1]}"
            for key, count in Counter(labeled_structures).items()
            if count > 1
        ]
        malformed_pairs = [
            pair_id
            for pair_id, samples in paired.items()
            if len(samples) != 2
            or {int(sample["label"]) for sample in samples} != {0, 1}
            or len({sample["base_hash"] for sample in samples}) != 1
        ]
        for duplicate in duplicate_sample_ids:
            _problem(problems, split_name, duplicate, "duplicate sample_id")
        for duplicate in duplicate_labeled_structures:
            _problem(
                problems,
                split_name,
                None,
                f"duplicate invariant labeled structure {duplicate}",
            )
        for pair_id in malformed_pairs:
            _problem(
                problems, split_name, None, f"malformed matched pair {pair_id}"
            )

        bx_batch, by_batch, _ = collate_samples(data)
        padding_valid = bool(
            np.all((bx_batch.numpy().sum(axis=2) > 0).sum(axis=1) > 0)
            and np.all((by_batch.numpy().sum(axis=2) > 0).sum(axis=1) > 0)
        )
        if not padding_valid:
            _problem(problems, split_name, None, "padded batch contains empty sample")

        deterministic_generation = sha256_value(payload) == sha256_value(
            fresh[split_name]
        )
        if not deterministic_generation:
            _problem(
                problems,
                split_name,
                None,
                "locked file differs from a fresh deterministic generation",
            )
        deterministic_loading = first_text == second_text
        structures[split_name] = set(local_structures)
        bases[split_name] = set(local_bases)
        matched_ids[split_name] = set(local_matched_ids)
        sample_ids[split_name] = set(local_sample_ids)
        split_reports[split_name] = {
            "status": "PASS"
            if not any(problem["split"] == split_name for problem in problems)
            else "FAIL",
            "sample_count": len(data),
            "expected_sample_count": expected_count,
            "class_counts": {str(key): value for key, value in sorted(labels.items())},
            "unique_sample_ids": len(set(local_sample_ids)),
            "unique_matched_pair_ids": len(set(local_matched_ids)),
            "unique_invariant_structures": len(set(local_structures)),
            "unique_base_hashes": len(set(local_bases)),
            "duplicate_sample_ids": duplicate_sample_ids,
            "duplicate_labeled_structures": duplicate_labeled_structures,
            "malformed_matched_pairs": malformed_pairs,
            "shape_failure_count": shape_failures,
            "label_failure_count": label_failures,
            "mask_failure_count": mask_failures,
            "margin_failure_count": margin_failures,
            "hash_failure_count": hash_failures,
            "padding_valid": padding_valid,
            "deterministic_loading": deterministic_loading,
            "deterministic_generation": deterministic_generation,
            "file_sha256": file_sha256(path),
            "payload_sha256": sha256_value(payload),
        }

    overlaps: dict[str, Any] = {}
    for index, left in enumerate(REQUIRED_SPLITS):
        for right in REQUIRED_SPLITS[index + 1 :]:
            structure_overlap = sorted(structures[left] & structures[right])
            base_overlap = sorted(bases[left] & bases[right])
            matched_overlap = sorted(matched_ids[left] & matched_ids[right])
            sample_overlap = sorted(sample_ids[left] & sample_ids[right])
            pair_name = f"{left}_vs_{right}"
            pair_status = (
                "PASS"
                if not (
                    structure_overlap
                    or base_overlap
                    or matched_overlap
                    or sample_overlap
                )
                else "FAIL"
            )
            overlaps[pair_name] = {
                "status": pair_status,
                "sample_id_overlap_count": len(sample_overlap),
                "matched_pair_id_overlap_count": len(matched_overlap),
                "base_hash_overlap_count": len(base_overlap),
                "invariant_structure_overlap_count": len(structure_overlap),
                "problematic_sample_ids": sample_overlap,
                "problematic_matched_pair_ids": matched_overlap,
                "problematic_base_hashes": base_overlap,
                "problematic_structure_hashes": structure_overlap,
                "reason": ""
                if pair_status == "PASS"
                else "prohibited cross-split identity or invariant structure overlap",
            }
            if pair_status == "FAIL":
                _problem(
                    problems,
                    pair_name,
                    None,
                    overlaps[pair_name]["reason"],
                )

    report = {
        "schema_version": "phase3-split-audit-v2",
        "status": "PASS" if not problems else "FAIL",
        "required_splits": list(REQUIRED_SPLITS),
        "split_reports": split_reports,
        "overlap_checks": overlaps,
        "problem_count": len(problems),
        "problems": problems,
        "reasons": [] if not problems else sorted({p["reason"] for p in problems}),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    report = audit_splits()
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
