"""Authoritative model configuration loading and artifact export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "phase3_models.json"
RESULT_CONFIG_DIR = PROJECT_ROOT / "results" / "phase3" / "model_configs"


def _configuration_document() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_config(model_name: str) -> dict[str, Any]:
    document = _configuration_document()
    if model_name not in document["models"]:
        raise KeyError(f"Unknown Phase 3 model: {model_name}")
    config = dict(document["common"])
    config.update(document["models"][model_name])
    config["model_name"] = model_name
    return config


def model_names() -> list[str]:
    return list(_configuration_document()["models"])


def generate_configs(output_dir: Path = RESULT_CONFIG_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name in model_names():
        path = output_dir / f"{model_name}_config.json"
        path.write_text(
            json.dumps(load_config(model_name), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (output_dir / "final_model_config.json").write_text(
        json.dumps(
            load_config("LearnedSinkhornMatcher"),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    generate_configs()
