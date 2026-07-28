"""Rebuild and cleanly execute both required Phase 3 notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from src.phase3_data import PROJECT_ROOT
from src.rebuild_notebooks import main as rebuild


def execute(path: Path) -> dict[str, object]:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, path)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    return {
        "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "code_cell_count": len(code_cells),
        "executed_cell_count": sum(
            cell.execution_count is not None for cell in code_cells
        ),
        "error_output_count": len(errors),
        "status": "PASS"
        if not errors
        and all(cell.execution_count is not None for cell in code_cells)
        else "FAIL",
    }


def main() -> None:
    rebuild()
    reports = [
        execute(PROJECT_ROOT / "notebooks" / name)
        for name in ("Phase3_Research.ipynb", "Phase3_Explained_FA.ipynb")
    ]
    result = {
        "status": (
            "PASS"
            if all(report["status"] == "PASS" for report in reports)
            else "FAIL"
        ),
        "notebooks": reports,
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
