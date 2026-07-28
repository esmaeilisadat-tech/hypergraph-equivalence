"""Final manifests, hash verification, and reproducible ZIP export checks."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.phase3_data import PROJECT_ROOT, file_sha256, load_spec

RESULT_DIR = PROJECT_ROOT / "results" / "phase3"
RUN_MANIFEST = RESULT_DIR / "run_manifest.json"
ARTIFACT_MANIFEST = RESULT_DIR / "final_artifact_manifest.json"
ZIP_PATH = RESULT_DIR / "phase3_reproducible.zip"


def _git(arguments: list[str]) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _artifact_paths() -> list[Path]:
    paths: set[Path] = set()
    explicit = [
        PROJECT_ROOT / "AGENTS.md",
        PROJECT_ROOT / "CODEX_PHASE3_HANDOFF.md",
        PROJECT_ROOT / "CODEX_PHASE3_STATUS.md",
        PROJECT_ROOT / "phase3_spec.json",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "pytest.ini",
        PROJECT_ROOT / ".gitignore",
        PROJECT_ROOT
        / "Eidi+et+al_2026_J._Phys._Complex._10.1088_2632-072X_ae6f5e.pdf",
        PROJECT_ROOT / "eidi_paper.txt",
    ]
    paths.update(path for path in explicit if path.exists())
    patterns = [
        "src/**/*.py",
        "configs/**/*.json",
        "tests/**/*.py",
        "data/**/*.json",
        "notebooks/*.ipynb",
        "results/phase3/**/*",
        "results/deterministic_benchmark_v2/**/*",
        "results/learned_benchmark_v1_development/**/*",
        "docs/legacy/antigravity/**/*",
    ]
    for pattern in patterns:
        paths.update(path for path in PROJECT_ROOT.glob(pattern) if path.is_file())
    paths.discard(ARTIFACT_MANIFEST)
    paths.discard(ZIP_PATH)
    return sorted(paths, key=lambda path: path.as_posix())


def generate_run_manifest() -> dict[str, Any]:
    spec = load_spec()
    log_reports = []
    for path in sorted((RESULT_DIR / "logs").glob("*.json")):
        log_reports.append(json.loads(path.read_text(encoding="utf-8")))
    manifest = {
        "schema_version": "phase3-run-manifest-v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "branch": _git(["branch", "--show-current"]),
        "source_commit": _git(["rev-parse", "HEAD"]),
        "git_status": _git(["status", "--short"]),
        "archive_paths": [
            "results/phase3_archive/20260727_102315",
            "results/phase3_archive/20260727T160053Z_independent_audit_corrections",
        ],
        "benchmark_version": spec["benchmark_version"],
        "primary_model": spec["primary_model"],
        "model_roles": spec["model_roles"],
        "seeds": spec["seeds"],
        "splits": spec["splits"],
        "python": {
            "executable": Path(sys.executable).name,
            "version": platform.python_version(),
            "platform": platform.platform(),
        },
        "torch": {
            "version": torch.__version__,
            "device": "cpu",
            "cuda_available": torch.cuda.is_available(),
        },
        "commands": log_reports,
        "completed_checkpoint_count": len(
            list((RESULT_DIR / "checkpoints").glob("*.pt"))
        ),
        "completed_prediction_count": len(
            list((RESULT_DIR / "predictions").glob("*.csv"))
        ),
    }
    RUN_MANIFEST.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def generate_artifact_manifest() -> dict[str, Any]:
    entries = []
    for path in _artifact_paths():
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    manifest = {
        "schema_version": "phase3-artifact-manifest-v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hash_algorithm": "SHA-256",
        "self_exclusion": (
            "This manifest and the ZIP are excluded to avoid recursive hashes; "
            "every source/evidence file inside the ZIP is listed."
        ),
        "entry_count": len(entries),
        "entries": entries,
    }
    ARTIFACT_MANIFEST.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify_manifest() -> dict[str, Any]:
    manifest = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    failures: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        path = PROJECT_ROOT / entry["path"]
        if not path.is_file():
            failures.append({"path": entry["path"], "reason": "missing"})
            continue
        observed_size = path.stat().st_size
        observed_hash = file_sha256(path)
        if observed_size != entry["size_bytes"] or observed_hash != entry["sha256"]:
            failures.append(
                {
                    "path": entry["path"],
                    "reason": "size or hash mismatch",
                    "expected_size": entry["size_bytes"],
                    "observed_size": observed_size,
                    "expected_sha256": entry["sha256"],
                    "observed_sha256": observed_hash,
                }
            )
    return {
        "status": "PASS" if not failures else "FAIL",
        "checked_entry_count": len(manifest["entries"]),
        "failures": failures,
    }


def create_zip() -> dict[str, Any]:
    manifest = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    paths = [PROJECT_ROOT / entry["path"] for entry in manifest["entries"]]
    paths.append(ARTIFACT_MANIFEST)
    with zipfile.ZipFile(
        ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in paths:
            archive.write(path, path.relative_to(PROJECT_ROOT).as_posix())
    return {
        "path": ZIP_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": ZIP_PATH.stat().st_size,
        "sha256": file_sha256(ZIP_PATH),
        "entry_count": len(paths),
    }


def verify_zip() -> dict[str, Any]:
    required = {
        "phase3_spec.json",
        "requirements.txt",
        "src/phase3_data.py",
        "src/differentiable_matching.py",
        "src/phase3_experiments.py",
        "data/splits/train.json",
        "data/splits/validation.json",
        "data/splits/iid_test.json",
        "data/splits/ood_template_test.json",
        "data/splits/ood_size_test.json",
        "results/phase3/per_seed_metrics.csv",
        "results/phase3/aggregate_metrics.csv",
        "results/phase3/split_overlap_audit.json",
        "results/phase3/learned_benchmark_audit.json",
        "results/phase3/learned_benchmark_lock.json",
        "results/phase3/zero_shot_control.json",
        "results/phase3/controls/label_access_control.json",
        "results/phase3/controls/label_signal_controls.json",
        "results/phase3/scientific_claim_audit.json",
        "results/phase3/gate_a_report.json",
        "results/phase3/gate_b_report.json",
        "results/phase3/gate_c_report.json",
        "results/phase3/numerical_audit.json",
        "results/phase3/checkpoint_roundtrip.json",
        "results/phase3/report_consistency.json",
        "results/phase3/final_validation_report.md",
        "results/phase3/run_manifest.json",
        "results/phase3/final_artifact_manifest.json",
        "notebooks/Phase3_Research.ipynb",
        "notebooks/Phase3_Explained_FA.ipynb",
    }
    failures: list[Any] = []
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = set(archive.namelist())
        corrupt = archive.testzip()
        if corrupt is not None:
            failures.append({"first_corrupt_entry": corrupt})
        missing = sorted(required - names)
        if missing:
            failures.append({"missing_required_entries": missing})
        with tempfile.TemporaryDirectory(prefix="phase3_zip_verify_") as temp:
            extract_root = Path(temp)
            archive.extractall(extract_root)
            extracted_manifest = json.loads(
                (extract_root / ARTIFACT_MANIFEST.relative_to(PROJECT_ROOT)).read_text(
                    encoding="utf-8"
                )
            )
            expected_names = {
                entry["path"] for entry in extracted_manifest["entries"]
            } | {
                ARTIFACT_MANIFEST.relative_to(PROJECT_ROOT).as_posix()
            }
            if names != expected_names:
                failures.append(
                    {
                        "unlisted_zip_entries": sorted(names - expected_names),
                        "manifest_entries_missing_from_zip": sorted(
                            expected_names - names
                        ),
                    }
                )
            for entry in extracted_manifest["entries"]:
                path = extract_root / entry["path"]
                if not path.is_file() or file_sha256(path) != entry["sha256"]:
                    failures.append(
                        {"path": entry["path"], "reason": "extracted hash mismatch"}
                    )
            smoke = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from src.phase3_data import load_spec, load_splits;"
                        "from src.config_manager import load_config;"
                        "from src.differentiable_matching import initialize_model;"
                        "assert load_spec()['primary_model']=='LearnedSinkhornMatcher';"
                        "assert len(load_splits(['iid_test'])['iid_test'])==800;"
                        "initialize_model(load_config('LearnedSinkhornMatcher'));"
                        "print('extracted smoke PASS')"
                    ),
                ],
                cwd=extract_root,
                capture_output=True,
                text=True,
            )
            if smoke.returncode != 0:
                failures.append(
                    {
                        "extracted_smoke_return_code": smoke.returncode,
                        "stdout": smoke.stdout,
                        "stderr": smoke.stderr,
                    }
                )
    return {
        "status": "PASS" if not failures else "FAIL",
        "zip_path": ZIP_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "zip_sha256": file_sha256(ZIP_PATH),
        "zip_size_bytes": ZIP_PATH.stat().st_size,
        "failures": failures,
    }


if __name__ == "__main__":
    print(json.dumps(generate_run_manifest(), indent=2))
    print(json.dumps(generate_artifact_manifest(), indent=2))
    print(json.dumps(verify_manifest(), indent=2))
    print(json.dumps(create_zip(), indent=2))
    print(json.dumps(verify_zip(), indent=2))
