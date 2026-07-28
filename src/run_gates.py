"""Run the final Gate A and Gate B checks without rerunning Gate C training."""

from __future__ import annotations

import json

from src.phase3_experiments import run_gate_a, run_gate_b


def main() -> None:
    gate_a = run_gate_a()
    gate_b = run_gate_b()
    result = {
        "status": (
            "PASS"
            if gate_a["status"] == gate_b["status"] == "PASS"
            else "FAIL"
        ),
        "gate_a": gate_a,
        "gate_b": gate_b,
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
