from __future__ import annotations

import argparse
import json
import os
import pprint
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
REINVOCATION_FLAG = "MOLTBOOK_PIPELINE_VENV_REINVOCATION"


def _preferred_python_executable() -> str:
    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


@dataclass(frozen=True)
class PipelineStage:
    key: str
    title: str
    kind: str  # script | notebook
    target: str
    rationale: str


PIPELINE_STAGES: List[PipelineStage] = [
    PipelineStage(
        key="collect",
        title="Collect MoltBook raw data",
        kind="script",
        target="scripts/run_moltbook_collection.py",
        rationale="Build reproducible raw snapshots from public MoltBook pages.",
    ),
    PipelineStage(
        key="raw_to_staged",
        title="Consolidate raw to staged comments",
        kind="script",
        target="scripts/process_raw_to_staged.py",
        rationale="Create a deduplicated staged corpus for downstream analysis.",
    ),
    PipelineStage(
        key="eda_summary",
        title="Generate EDA summary artifacts",
        kind="script",
        target="scripts/run_moltbook_sentiment.py",
        rationale="Quantify corpus quality, missingness, and distribution baselines.",
    ),
    PipelineStage(
        key="eda_notebook",
        title="Visual EDA notebook",
        kind="notebook",
        target="notebooks/moltbook_eda_visualizations.ipynb",
        rationale="Inspect visual diagnostics and potential scraping artifacts.",
    ),
    PipelineStage(
        key="preprocess_notebook",
        title="Preprocessing methodology notebook",
        kind="notebook",
        target="notebooks/moltbook_preprocessing_steps.ipynb",
        rationale="Audit each NLP cleaning/filtering step and row-retention logic.",
    ),
    PipelineStage(
        key="polarity",
        title="Run strict preprocessing + polarity scoring",
        kind="script",
        target="scripts/run_moltbook_polarity.py",
        rationale="Create research-grade polarity outputs and modeling-ready dataset.",
    ),
    PipelineStage(
        key="polarity_notebook",
        title="Polarity assessment notebook",
        kind="notebook",
        target="notebooks/moltbook_polarity_assessment.ipynb",
        rationale="Validate score distributions, subgroup contrasts, and examples.",
    ),
    PipelineStage(
        key="interaction_network",
        title="Build author reply interaction network",
        kind="script",
        target="scripts/run_moltbook_interaction_network.py",
        rationale=(
            "Construct a directed author graph and report in-degree/out-degree, "
            "reciprocity, clustering, and thread-level interaction structure."
        ),
    ),
    PipelineStage(
        key="modeling",
        title="Train lightweight benchmark models",
        kind="script",
        target="scripts/run_moltbook_modeling.py",
        rationale="Benchmark interpretable, efficient models for Phase 1 reporting.",
    ),
]


def _stage_map() -> Dict[str, PipelineStage]:
    return {stage.key: stage for stage in PIPELINE_STAGES}


def _repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")


def _parse_stage_keys(raw_keys: Sequence[str]) -> List[str]:
    stage_lookup = _stage_map()
    keys: List[str] = []
    seen: Set[str] = set()

    for item in raw_keys:
        for token in item.split(","):
            key = token.strip().lower()
            if not key:
                continue
            if key not in stage_lookup:
                valid = ", ".join(stage_lookup.keys())
                raise ValueError(f"Unknown stage '{key}'. Valid stages: {valid}")
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def _resolve_stage_sequence(
    run_all: bool,
    stages: Optional[List[str]],
    from_stage: str,
    to_stage: str,
    include_notebooks: bool,
) -> List[PipelineStage]:
    if not (run_all or stages or from_stage or to_stage):
        raise ValueError("Specify one of --all, --stages, or --from-stage/--to-stage.")

    ordered = PIPELINE_STAGES
    lookup = _stage_map()

    if stages:
        selected = [lookup[key] for key in stages]
    else:
        if run_all:
            selected = ordered
        else:
            start = from_stage.strip().lower() if from_stage else ordered[0].key
            end = to_stage.strip().lower() if to_stage else ordered[-1].key
            if start not in lookup or end not in lookup:
                valid = ", ".join(lookup.keys())
                raise ValueError(f"Invalid range. Valid stages: {valid}")

            start_idx = [s.key for s in ordered].index(start)
            end_idx = [s.key for s in ordered].index(end)
            if start_idx > end_idx:
                raise ValueError("--from-stage must come before --to-stage in pipeline order.")
            selected = ordered[start_idx : end_idx + 1]

    if include_notebooks:
        return selected
    return [stage for stage in selected if stage.kind == "script"]


def _parse_cell_spec(cell_spec: str, max_cell_count: int) -> List[int]:
    if not cell_spec:
        return list(range(1, max_cell_count + 1))

    selected: Set[int] = set()
    parts = [p.strip() for p in cell_spec.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            if start <= 0 or end <= 0 or start > end:
                raise ValueError(f"Invalid range '{part}'. Use positive ascending ranges.")
            for i in range(start, end + 1):
                selected.add(i)
        else:
            idx = int(part)
            if idx <= 0:
                raise ValueError(f"Invalid cell number '{part}'. Use 1-based positive integers.")
            selected.add(idx)

    out = sorted(selected)
    if out and out[-1] > max_cell_count:
        raise ValueError(
            f"Cell selection exceeds notebook length ({max_cell_count} code/markdown cells)."
        )
    return out


def _load_notebook(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_notebook_cells(notebook_path: Path, cell_spec: str, fail_fast: bool = True) -> None:
    nb = _load_notebook(notebook_path)
    cells = nb.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError(f"Notebook file is malformed: {_repo_rel(notebook_path)}")

    selected_numbers = set(_parse_cell_spec(cell_spec, len(cells)))
    def _display_fallback(*args: object, **kwargs: object) -> None:
        for value in args:
            pprint.pprint(value)
        for key, value in kwargs.items():
            print(f"{key}=")
            pprint.pprint(value)

    namespace = {
        "__name__": "__main__",
        "__file__": str(notebook_path),
        "display": _display_fallback,
    }

    print(f"Notebook: {_repo_rel(notebook_path)}")
    print(f"Running cells: {sorted(selected_numbers)}")

    executed = 0
    for idx, cell in enumerate(cells, start=1):
        if idx not in selected_numbers:
            continue

        cell_type = str(cell.get("cell_type", ""))
        if cell_type != "code":
            print(f"  Skipping Cell {idx} (non-code)")
            continue

        source = cell.get("source", [])
        if isinstance(source, list):
            code = "".join(source)
        else:
            code = str(source)

        if not code.strip():
            print(f"  Skipping Cell {idx} (empty code)")
            continue

        print(f"  Executing Cell {idx}")
        try:
            exec(compile(code, f"{notebook_path.name}:Cell{idx}", "exec"), namespace)
            executed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  Cell {idx} failed: {type(exc).__name__}: {exc}")
            if fail_fast:
                raise

    print(f"Notebook run complete. Executed code cells: {executed}")


def _resolve_notebook_path(notebook_name_or_path: str) -> Path:
    candidate = Path(notebook_name_or_path)
    if candidate.suffix.lower() != ".ipynb":
        candidate = candidate.with_suffix(".ipynb")

    if candidate.is_absolute() and candidate.exists():
        return candidate

    rel_from_repo = REPO_ROOT / candidate
    if rel_from_repo.exists():
        return rel_from_repo

    in_notebooks = NOTEBOOK_DIR / candidate.name
    if in_notebooks.exists():
        return in_notebooks

    raise FileNotFoundError(f"Notebook not found: {notebook_name_or_path}")


def _run_script(script_rel_path: str, extra_args: Optional[Sequence[str]] = None) -> None:
    script_path = REPO_ROOT / script_rel_path
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_rel_path}")

    cmd = [_preferred_python_executable(), str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"Running script: {script_rel_path}")
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Script failed ({completed.returncode}): {script_rel_path}")


def _print_pipeline() -> None:
    print("Research-aligned NLP pipeline order:")
    for i, stage in enumerate(PIPELINE_STAGES, start=1):
        print(f"{i:02d}. {stage.key:<18} [{stage.kind}] {stage.title}")
        print(f"    rationale: {stage.rationale}")


def _run_stages(stages: Iterable[PipelineStage], notebook_cells: Dict[str, str]) -> None:
    selected = list(stages)
    if not selected:
        print("No stages selected. Nothing to run.")
        return

    print("Executing pipeline stages:")
    for i, stage in enumerate(selected, start=1):
        print(f"  {i}. {stage.key} ({stage.kind})")

    for stage in selected:
        if stage.kind == "script":
            _run_script(stage.target)
            continue

        notebook_path = _resolve_notebook_path(stage.target)
        cell_spec = notebook_cells.get(stage.key, "")
        _run_notebook_cells(notebook_path, cell_spec)


def _parse_notebook_cells_arg(values: Sequence[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for raw in values:
        text = raw.strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(
                "Invalid --notebook-cells format. Use '<stage_key>:<cells>' e.g. preprocess_notebook:1-8,10"
            )
        key, spec = text.split(":", 1)
        stage_key = key.strip().lower()
        if stage_key not in _stage_map():
            valid = ", ".join(_stage_map().keys())
            raise ValueError(f"Unknown notebook stage '{stage_key}'. Valid stages: {valid}")
        if _stage_map()[stage_key].kind != "notebook":
            raise ValueError(f"Stage '{stage_key}' is not a notebook stage.")
        parsed[stage_key] = spec.strip()
    return parsed


def main() -> None:
    preferred_python = _preferred_python_executable()
    current_python = str(Path(sys.executable).resolve())
    preferred_python_resolved = str(Path(preferred_python).resolve())
    if (
        preferred_python_resolved != current_python
        and os.environ.get(REINVOCATION_FLAG) != "1"
    ):
        env = os.environ.copy()
        env[REINVOCATION_FLAG] = "1"
        cmd = [preferred_python_resolved, str(Path(__file__).resolve())] + sys.argv[1:]
        completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, env=env)
        raise SystemExit(completed.returncode)

    parser = argparse.ArgumentParser(
        description="Unified MoltBook NLP system: ordered pipeline execution + notebook cell control."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_list = subparsers.add_parser("list", help="List full pipeline order and rationale.")
    p_list.set_defaults(action="list")

    p_run = subparsers.add_parser("run", help="Run script/notebook stages in pipeline order.")
    p_run.add_argument("--all", action="store_true", help="Run the full research pipeline range.")
    p_run.add_argument(
        "--stages",
        nargs="+",
        default=[],
        help="Specific stage keys (comma-separated or space-separated).",
    )
    p_run.add_argument("--from-stage", default="", help="Start stage key for a contiguous range.")
    p_run.add_argument("--to-stage", default="", help="End stage key for a contiguous range.")
    p_run.add_argument(
        "--include-notebooks",
        action="store_true",
        help="Include notebook stages (disabled by default for fast script-only runs).",
    )
    p_run.add_argument(
        "--notebook-cells",
        nargs="*",
        default=[],
        help=(
            "Optional cell selectors for notebook stages. Format: "
            "<stage_key>:<cells>. Example: preprocess_notebook:1-8 polarity_notebook:1-6"
        ),
    )
    p_run.set_defaults(action="run")

    p_notebook = subparsers.add_parser(
        "run-notebook",
        help="Run a specific notebook with optional cell-number selection.",
    )
    p_notebook.add_argument("--notebook", required=True, help="Notebook name or path.")
    p_notebook.add_argument(
        "--cells",
        default="",
        help="Cell numbers/ranges, e.g. 1-5,8. Default runs all notebook cells.",
    )
    p_notebook.set_defaults(action="run-notebook")

    args, passthrough = parser.parse_known_args()

    if args.action == "list":
        _print_pipeline()
        return

    if args.action == "run-notebook":
        notebook_path = _resolve_notebook_path(args.notebook)
        _run_notebook_cells(notebook_path, args.cells)
        return

    stage_keys = _parse_stage_keys(args.stages)
    notebook_cells = _parse_notebook_cells_arg(args.notebook_cells)
    selected = _resolve_stage_sequence(
        run_all=args.all,
        stages=stage_keys if stage_keys else None,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        include_notebooks=args.include_notebooks,
    )

    if passthrough:
        print("Passing extra args to script stages:", passthrough)

    # Script passthrough is not yet stage-specific; run one-by-one without extras.
    if passthrough:
        print("Note: extra args are currently ignored. Use stage scripts directly for custom flags.")

    _run_stages(selected, notebook_cells)


if __name__ == "__main__":
    main()
