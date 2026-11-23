"""The example notebook stays stripped, portable and runnable (its code cells are executed here, no Jupyter needed)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "audioset_tools_demo.ipynb"
EXECUTED = ROOT / "docs" / "audioset_tools_demo_executed.ipynb"


def code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_example_notebook_is_stripped_and_portable():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert not re.search(r"/(?:Users|home|private|tmp)/\w+", text), "absolute path in the notebook"


def test_executed_copy_has_the_same_code_and_recorded_outputs():
    assert code_cells(EXECUTED) == code_cells(NOTEBOOK)
    executed = json.loads(EXECUTED.read_text(encoding="utf-8"))
    assert any(cell.get("outputs") for cell in executed["cells"] if cell["cell_type"] == "code")


def test_example_notebook_code_cells_run(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import warnings

    import matplotlib.pyplot as plt

    monkeypatch.chdir(ROOT / "examples")
    monkeypatch.setenv("TQDM_DISABLE", "1")
    namespace: dict = {"__name__": "__main__"}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
        for index, source in enumerate(code_cells(NOTEBOOK), start=1):
            exec(compile(source, f"<notebook cell {index}>", "exec"), namespace)  # noqa: S102
    plt.close("all")
    assert len(namespace["audioset"]) == 30 and len(namespace["vehicle_subset"]) == 5
    assert os.path.isdir(namespace["OUTPUT_DIR"])
