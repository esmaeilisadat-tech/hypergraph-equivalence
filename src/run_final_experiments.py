"""Resumable final v2 model/seed experiment entry point."""

from __future__ import annotations

import json

from src.phase3_experiments import (
    run_experiments,
    verify_checkpoint_roundtrip,
)


def main() -> None:
    per_seed, aggregate, gate_c = run_experiments()
    roundtrip = verify_checkpoint_roundtrip()
    result = {
        "status": (
            "PASS"
            if gate_c["status"] == "PASS"
            and roundtrip["status"] == "PASS"
            else "FAIL"
        ),
        "per_seed_metric_rows": len(per_seed),
        "aggregate_metric_rows": len(aggregate),
        "gate_c": gate_c,
        "checkpoint_roundtrip": roundtrip,
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
