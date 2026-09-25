"""Compatibility entry point for theory updates based on successful materials."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    from .deduplicate_success import deduplicate_success_materials
    from .update_document import update_theory_from_success
except ImportError:
    from deduplicate_success import deduplicate_success_materials
    from update_document import update_theory_from_success

logger = logging.getLogger(__name__)
THEORY_DOC_NAME = "Theoretical_principle_document.md"


def _resolve_previous_theory_doc(doc_root: str | Path, iteration_num: int) -> Path:
    """Return the previous versioned theory document, with baseline fallback."""
    root = Path(doc_root)
    if iteration_num <= 1:
        return root / "v0.0.0" / THEORY_DOC_NAME

    previous = root / f"v0.0.{iteration_num - 1}" / THEORY_DOC_NAME
    if previous.exists():
        return previous

    baseline = root / "v0.0.0" / THEORY_DOC_NAME
    logger.warning("Previous theory document missing; using baseline: %s", baseline)
    return baseline


def analyze_success_and_update_theory(
    success_csv_path: str,
    iteration_num: int = 1,
    version: int | None = None,
    results_root: str = "results",
    doc_root: str = "llm/doc",
) -> dict[str, Any] | None:
    """Deduplicate successful materials and update the versioned theory document."""
    del version  # Retained for compatibility with older workflow callers.
    success_csv = Path(success_csv_path)
    if not success_csv.exists():
        logger.warning("Success CSV not found: %s", success_csv)
        return None

    if success_csv.stem.endswith("_deduped"):
        target_csv = success_csv
    else:
        target_csv = success_csv.with_name(f"{success_csv.stem}_deduped{success_csv.suffix}")
        processed_csv = deduplicate_success_materials(str(success_csv), str(target_csv))
        if processed_csv is None:
            logger.warning("Deduplication failed; using original CSV: %s", success_csv)
            target_csv = success_csv

    output_dir = Path(doc_root) / f"v0.0.{iteration_num}"
    original_doc = _resolve_previous_theory_doc(doc_root, iteration_num)
    updated_doc = update_theory_from_success(
        original_doc_path=str(original_doc),
        success_csv=str(target_csv),
        output_dir=str(output_dir),
        iteration_num=iteration_num,
        results_root=results_root,
    )
    if not updated_doc:
        return None

    return {
        "success": True,
        "analysis_report": str(Path(results_root) / f"iteration_{iteration_num}" / "reports" / "llm_theory_update_output.md"),
        "updated_doc": updated_doc,
        "iteration_num": iteration_num,
    }
