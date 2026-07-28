"""Run a command and save its complete reproducibility log."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.phase3_data import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if not arguments.command:
        parser.error("a command is required")
    log_dir = PROJECT_ROOT / "results" / "phase3" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    monotonic_start = time.perf_counter()
    executable_command = [
        sys.executable if part == "{python}" else part
        for part in arguments.command
    ]
    completed = subprocess.run(
        executable_command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    ended = datetime.now(timezone.utc)
    stdout_path = log_dir / f"{arguments.name}.stdout.log"
    stderr_path = log_dir / f"{arguments.name}.stderr.log"
    metadata_path = log_dir / f"{arguments.name}.json"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    metadata = {
        "name": arguments.name,
        "command": arguments.command,
        "working_directory": ".",
        "started_utc": started.isoformat(),
        "ended_utc": ended.isoformat(),
        "runtime_seconds": time.perf_counter() - monotonic_start,
        "return_code": completed.returncode,
        "stdout_path": str(stdout_path.relative_to(PROJECT_ROOT)),
        "stderr_path": str(stderr_path.relative_to(PROJECT_ROOT)),
        "status": "PASS" if completed.returncode == 0 else "FAIL",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
