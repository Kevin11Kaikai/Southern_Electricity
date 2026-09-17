"""Execute only the new, read-only evidence notebook in a project-local kernel."""

import os
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from .common import ROOT, PROTOCOL_PATH, Ledger, read_json, sha256, write_json
from .run import run_directory


def main():
    protocol = read_json(PROTOCOL_PATH)
    run_dir = run_directory(protocol)
    ledger = Ledger(run_dir, protocol)
    if ledger.remaining_seconds() < 60:
        raise RuntimeError("not enough remaining development budget for notebook execution")
    runtime = run_dir / "notebook_runtime"
    runtime.mkdir(exist_ok=True)
    for folder in ["matplotlib", "jupyter_runtime", "ipython"]:
        (runtime / folder).mkdir(exist_ok=True)
    os.environ["JUPYTER_PATH"] = str(runtime)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime / "jupyter_runtime")
    os.environ["IPYTHONDIR"] = str(runtime / "ipython")
    os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    notebook_path = ROOT / "notebooks/12_codex_notebook.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            for forbidden in [".fit(", "train_all(", "run_job(", "subprocess.", "evaluate("]:
                if forbidden in cell.source:
                    raise ValueError(f"notebook includes disallowed executable entry {forbidden}")
    before_training = ledger.launches()
    started = time.perf_counter()
    ledger.add("notebook_execution_start", path=str(notebook_path.relative_to(ROOT)))
    client = NotebookClient(notebook, timeout=60, kernel_name="south-grid-codex", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbformat.validate(notebook)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    errors = [o for c in code_cells for o in c.get("outputs", []) if o.output_type == "error"]
    if errors or any(c.execution_count is None for c in code_cells) or ledger.launches() != before_training:
        raise AssertionError("notebook execution failed or unexpectedly started training")
    nbformat.write(notebook, notebook_path)
    result = {
        "code_cells_executed": len(code_cells), "errors": len(errors),
        "images_embedded": sum("image/png" in o.get("data", {}) for c in code_cells for o in c.get("outputs", [])),
        "notebook_sha256": sha256(notebook_path), "training_launched": False,
        "seconds": time.perf_counter()-started,
    }
    write_json(run_dir / "notebook_execution.json", result)
    ledger.add("notebook_execution_end", **result)
    print(result)


if __name__ == "__main__":
    main()
