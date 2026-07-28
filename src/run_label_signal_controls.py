"""Run the shuffled-label and random-label negative controls."""

from __future__ import annotations

import json

from src.phase3_controls import run_label_signal_controls


if __name__ == "__main__":
    result = run_label_signal_controls()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
