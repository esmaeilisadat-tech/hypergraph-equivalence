"""Validate an exported Phase 3 package in a genuinely new virtual environment."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.phase3_data import PROJECT_ROOT

RESULT_DIR = PROJECT_ROOT / "results" / "phase3"
LOG_DIR = RESULT_DIR / "logs" / "fresh_environment_steps"
ZIP_PATH = RESULT_DIR / "phase3_reproducible.zip"


def _run(
    name: str,
    command: list[str],
    cwd: Path,
    reports: list[dict[str, Any]],
) -> None:
    started = datetime.now(timezone.utc)
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    ended = datetime.now(timezone.utc)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = LOG_DIR / f"{name}.stdout.log"
    stderr_path = LOG_DIR / f"{name}.stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    reports.append(
        {
            "name": name,
            "command": command,
            "working_directory": str(cwd),
            "started_utc": started.isoformat(),
            "ended_utc": ended.isoformat(),
            "return_code": completed.returncode,
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "stdout_path": str(stdout_path.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
            "stderr_path": str(stderr_path.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
        }
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed with return code {completed.returncode}"
        )


def main() -> None:
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(ZIP_PATH)
    reports: list[dict[str, Any]] = []
    overall_status = "FAIL"
    error: str | None = None
    with tempfile.TemporaryDirectory(prefix="phase3_fresh_env_") as temp:
        temp_root = Path(temp)
        environment = temp_root / "venv"
        extracted = temp_root / "extracted"
        fresh_python = (
            environment / "Scripts" / "python.exe"
            if sys.platform == "win32"
            else environment / "bin" / "python"
        )
        try:
            _run(
                "create_venv",
                [sys.executable, "-m", "venv", str(environment)],
                PROJECT_ROOT,
                reports,
            )
            _run(
                "clean_install",
                [
                    str(fresh_python),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--requirement",
                    str(PROJECT_ROOT / "requirements.txt"),
                ],
                PROJECT_ROOT,
                reports,
            )
            extraction_code = (
                "import hashlib,json,pathlib,sys,zipfile;"
                "z=pathlib.Path(sys.argv[1]);d=pathlib.Path(sys.argv[2]);"
                "a=zipfile.ZipFile(z);assert a.testzip() is None;a.extractall(d);"
                "m=json.loads((d/'results/phase3/final_artifact_manifest.json')"
                ".read_text(encoding='utf-8'));"
                "bad=[e['path'] for e in m['entries'] if "
                "hashlib.sha256((d/e['path']).read_bytes()).hexdigest()!=e['sha256']];"
                "assert not bad,bad;print('ZIP CRC, extraction, and manifest hashes PASS',"
                "len(m['entries']))"
            )
            _run(
                "extract_and_verify_zip",
                [
                    str(fresh_python),
                    "-c",
                    extraction_code,
                    str(ZIP_PATH),
                    str(extracted),
                ],
                PROJECT_ROOT,
                reports,
            )
            _run(
                "import_checks",
                [
                    str(fresh_python),
                    "-c",
                    (
                        "import torch,numpy,pandas,scipy,sklearn,networkx,"
                        "nbformat,nbclient;import src.phase3_data,"
                        "src.differentiable_matching,src.phase3_experiments,"
                        "src.audit_model_numerics,src.reporting;"
                        "print('fresh imports PASS')"
                    ),
                ],
                extracted,
                reports,
            )
            _run(
                "repository_tests",
                [str(fresh_python), "-m", "pytest", "-q"],
                extracted,
                reports,
            )
            _run(
                "deterministic_split_audit",
                [str(fresh_python), "-m", "src.audit_splits"],
                extracted,
                reports,
            )
            zero_shot_code = (
                "from src.phase3_data import load_splits;"
                "from src.config_manager import load_config;"
                "from src.differentiable_matching import initialize_model;"
                "from src.phase3_experiments import predict,set_seed;"
                "set_seed(42);m=initialize_model(load_config("
                "'LearnedSinkhornMatcher'));s=load_splits(['iid_test'])['iid_test'];"
                "y,p=predict(m,s);a=float(((p>=.5)==y).mean());"
                "assert .35<=a<=.65,a;print('zero-shot accuracy',a)"
            )
            _run(
                "zero_shot_primary_smoke",
                [str(fresh_python), "-c", zero_shot_code],
                extracted,
                reports,
            )
            inference_code = (
                "import numpy as np,pandas as pd,torch;"
                "from pathlib import Path;"
                "from src.phase3_data import load_splits;"
                "from src.differentiable_matching import initialize_model;"
                "from src.phase3_experiments import predict;"
                "r=Path('results/phase3');q=torch.load("
                "r/'checkpoints/LearnedSinkhornMatcher_seed42.pt',"
                "map_location='cpu',weights_only=False);"
                "m=initialize_model(q['model_config']);"
                "m.load_state_dict(q['state_dict']);"
                "s=load_splits(['iid_test'])['iid_test'];y,p=predict(m,s);"
                "saved=pd.read_csv(r/'predictions/"
                "LearnedSinkhornMatcher_seed42_iid_test.csv');"
                "assert np.max(np.abs(p-saved.probability.to_numpy()))<1e-7;"
                "print('checkpoint inference PASS',len(p))"
            )
            _run(
                "checkpoint_inference",
                [str(fresh_python), "-c", inference_code],
                extracted,
                reports,
            )
            _run(
                "report_metric_consistency",
                [
                    str(fresh_python),
                    "-c",
                    (
                        "from src.reporting import verify_report_consistency;"
                        "r=verify_report_consistency();assert r['status']=='PASS',r;"
                        "print(r)"
                    ),
                ],
                extracted,
                reports,
            )
            _run(
                "frozen_evidence_checks",
                [str(fresh_python), "-m", "src.final_checks"],
                extracted,
                reports,
            )
            overall_status = "PASS"
        except Exception as exc:  # result is persisted before propagating
            error = f"{type(exc).__name__}: {exc}"
        result = {
            "schema_version": "phase3-fresh-environment-validation-v1",
            "status": overall_status,
            "genuinely_new_temporary_virtual_environment": True,
            "environment_removed_after_validation": True,
            "zip_path": str(ZIP_PATH.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
            "steps": reports,
            "error": error,
        }
        output = RESULT_DIR / "fresh_environment_validation.json"
        output.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        if overall_status != "PASS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
