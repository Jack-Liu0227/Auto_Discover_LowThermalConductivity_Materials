"""Safe reset helpers for isolated BO and LLM experiment roots."""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path


def archive_active_run(project_root: Path, run_mode: str) -> Path | None:
    """Archive an active run root before starting a fresh experiment.

    ``--reset`` is an explicit user request to start a new experiment.  Move
    the complete mode directory instead of deleting only progress markers, so
    old results cannot contaminate the new data/model/result chain and remain
    available for audit.
    """
    project_root = Path(project_root).resolve()
    if not run_mode or Path(run_mode).name != run_mode or Path(run_mode).parent != Path("."):
        raise ValueError(f"Invalid run mode for reset: {run_mode!r}")

    active_root = project_root / run_mode
    if not active_root.exists():
        return None
    if not active_root.is_dir():
        raise NotADirectoryError(f"Active run root is not a directory: {active_root}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = project_root / f"{run_mode}_old_{stamp}"
    counter = 1
    while archive_root.exists():
        archive_root = project_root / f"{run_mode}_old_{stamp}_{counter}"
        counter += 1

    shutil.move(str(active_root), str(archive_root))
    (project_root / run_mode).mkdir(parents=True, exist_ok=True)
    return archive_root


def _iteration_numbers(root: Path) -> list[int]:
    numbers: list[int] = []
    if not root.exists():
        return numbers
    for path in root.glob("iteration_*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"iteration_(\d+)", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return sorted(set(numbers))


def _version_numbers(root: Path) -> list[int]:
    numbers: list[int] = []
    if not root.exists():
        return numbers
    for path in root.glob("v0.0.*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"v0\.0\.(\d+)", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return sorted(set(numbers))


def _copy_then_remove(source: Path, archive: Path) -> int:
    """Copy an active artifact to audit storage, then remove only its active copy."""
    if not source.exists():
        return 0
    archive.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, archive)
        file_count = sum(1 for path in archive.rglob("*") if path.is_file())
        shutil.rmtree(source)
    else:
        shutil.copy2(source, archive)
        file_count = 1
        source.unlink()
    return file_count


def _archive_and_filter_summary(path: Path, archive: Path, from_iteration: int) -> int:
    """Preserve an aggregate CSV and keep only its pre-rebuild rows active."""
    if not path.exists() or not path.is_file():
        return 0
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, archive)
    try:
        import pandas as pd

        frame = pd.read_csv(path, encoding="utf-8-sig")
        if "iteration" not in frame.columns:
            return 1
        iterations = pd.to_numeric(frame["iteration"], errors="coerce")
        frame = frame[iterations < int(from_iteration)].copy()
        temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            frame.to_csv(temp, index=False, encoding="utf-8-sig")
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()
    except Exception as exc:
        raise RuntimeError(f"Failed to filter summary CSV {path}: {exc}") from exc
    return 1


def rebuild_active_chain(
    path_config,
    tracker,
    *,
    from_iteration: int,
    run_mode: str,
) -> Path:
    """Safely invalidate and archive an active iteration dependency suffix.

    This is intentionally separate from ``--reset``.  Iterations before
    ``from_iteration`` remain active, while every dependent result/data/model/
    document artifact at or after the rebuild boundary is copied to an audit
    directory before its active copy is removed.  The tracker is then reset
    only for the invalidated suffix, allowing the normal contiguous resume
    cursor to start at exactly ``from_iteration``.
    """
    from_iteration = int(from_iteration)
    if from_iteration <= 0:
        raise ValueError("from_iteration must be greater than zero")
    if not run_mode or Path(run_mode).name != run_mode or Path(run_mode).parent != Path("."):
        raise ValueError(f"Invalid run mode for rebuild: {run_mode!r}")
    incomplete_predecessors = [
        iteration
        for iteration in range(1, from_iteration)
        if not tracker.is_round_completed(iteration)
    ]
    if incomplete_predecessors:
        first_incomplete = incomplete_predecessors[0]
        raise ValueError(
            f"Cannot rebuild from iteration {from_iteration}: "
            f"continuous predecessor prefix is incomplete at iteration {first_incomplete}"
        )

    project_root = Path(path_config.project_root).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = project_root / f"{run_mode}_rebuild_{stamp}"
    counter = 1
    while archive_root.exists():
        archive_root = project_root / f"{run_mode}_rebuild_{stamp}_{counter}"
        counter += 1

    archived: list[dict[str, object]] = []

    def archive_suffix(root: Path, archive_base: Path, minimum: int) -> None:
        for number in _iteration_numbers(root):
            if number < minimum:
                continue
            source = root / f"iteration_{number}"
            destination = archive_base / f"iteration_{number}"
            count = _copy_then_remove(source, destination)
            archived.append({"source": str(source), "archive": str(destination), "files": count})

    archive_suffix(Path(path_config.results_root), archive_root / "results", from_iteration)
    archive_suffix(Path(path_config.data_root), archive_root / "data", from_iteration)
    archive_suffix(Path(path_config.models_root), archive_root / "models" / "GPR", max(from_iteration - 1, 0))

    if path_config.doc_root is not None:
        doc_root = Path(path_config.doc_root)
        for number in _version_numbers(doc_root):
            if number < from_iteration:
                continue
            source = doc_root / f"v0.0.{number}"
            destination = archive_root / "doc" / f"v0.0.{number}"
            count = _copy_then_remove(source, destination)
            archived.append({"source": str(source), "archive": str(destination), "files": count})

    results_root = Path(path_config.results_root)
    progress_path = results_root / "progress.json"
    if progress_path.exists():
        progress_archive = archive_root / "results" / "progress.json"
        progress_archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(progress_path, progress_archive)
        archived.append({"source": str(progress_path), "archive": str(progress_archive), "files": 1})

    # Aggregate summaries are derived from per-iteration extraction files. Keep
    # their pre-rebuild rows active, but preserve the original complete files.
    summary_paths = list(results_root.glob("*.csv")) + list((results_root / "summary").glob("*.csv"))
    for summary_path in sorted(set(summary_paths)):
        archive_path = archive_root / "results" / summary_path.relative_to(results_root)
        count = _archive_and_filter_summary(summary_path, archive_path, from_iteration)
        archived.append({"source": str(summary_path), "archive": str(archive_path), "files": count})

    for number in sorted(
        int(match.group(1))
        for key in tracker.progress
        if (match := re.fullmatch(r"iteration_(\d+)", str(key)))
        and int(match.group(1)) >= from_iteration
    ):
        tracker.reset_round(number)

    manifest = {
        "run_mode": run_mode,
        "from_iteration": from_iteration,
        "created_at": datetime.now().isoformat(),
        "predecessor_iteration": from_iteration - 1,
        "archived": archived,
        "note": "Active suffix was explicitly rebuilt; archived copies are audit-only and never resume inputs.",
    }
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "rebuild_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return archive_root
