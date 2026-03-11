"""
Utilities for learning from successful materials and updating the theory doc.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


try:
    from .deduplicate_success import deduplicate_success_materials
    from .update_document import update_theory_from_success
except ImportError:
    try:
        from deduplicate_success import deduplicate_success_materials
        from update_document import update_theory_from_success
    except ImportError:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from deduplicate_success import deduplicate_success_materials
        from update_document import update_theory_from_success


logger = logging.getLogger(__name__)
THEORY_DOC_NAME = "Theoretical_principle_document.md"


def learn_from_success_materials(
    success_csv: str,
    output_dir: str = "llm/doc/v0.0.2",
    iteration_num: int = 1,
    doc_path: str = "llm/doc/v0.0.1/Theoretical_principle_document.md",
    results_root: str = "results",
) -> Optional[str]:
    """
    Deduplicate success cases and update the theory document.
    """
    logger.info("=" * 80)
    logger.info("Starting success-material learning for iteration %s", iteration_num)
    logger.info("=" * 80)

    try:
        csv_dir = os.path.dirname(success_csv)
        csv_name = os.path.basename(success_csv)
        base_name, ext = os.path.splitext(csv_name)
        dedup_csv_path = os.path.join(csv_dir, f"{base_name}_deduped{ext}")

        processed_csv = deduplicate_success_materials(success_csv, dedup_csv_path)
        target_csv = processed_csv or success_csv

        if processed_csv:
            logger.info("Using deduplicated success CSV: %s", processed_csv)
        else:
            logger.warning("Deduplication failed, using original success CSV: %s", success_csv)

        updated_doc_path = update_theory_from_success(
            original_doc_path=doc_path,
            success_csv=target_csv,
            output_dir=output_dir,
            iteration_num=iteration_num,
            results_root=results_root,
        )

        if updated_doc_path:
            logger.info("Updated theory document: %s", updated_doc_path)
            return updated_doc_path

        logger.warning("Theory document update produced no file")
        return None
    except Exception as exc:
        logger.exception("Learning from success materials failed: %s", exc)
        return None


def _resolve_previous_theory_doc(doc_root: str, iteration_num: int) -> str:
    if iteration_num <= 1:
        return f"{doc_root}/v0.0.0/{THEORY_DOC_NAME}"

    prev_doc = f"{doc_root}/v0.0.{iteration_num - 1}/{THEORY_DOC_NAME}"
    if os.path.exists(prev_doc):
        return prev_doc

    baseline_doc = f"{doc_root}/v0.0.0/{THEORY_DOC_NAME}"
    logger.warning("Previous theory document missing, fallback to baseline: %s", baseline_doc)
    return baseline_doc


def analyze_success_and_update_theory(
    success_csv_path: str,
    iteration_num: int = 1,
    version: int = 1,
    results_root: str = "results",
    doc_root: str = "llm/doc",
) -> Optional[dict]:
    """
    Analyze successful materials and update the versioned theory doc.
    """
    del version

    logger.info("=" * 80)
    logger.info("Starting theory update for iteration %s", iteration_num)
    logger.info("=" * 80)

    try:
        original_doc_path = _resolve_previous_theory_doc(doc_root, iteration_num)
        output_dir = f"{doc_root}/v0.0.{iteration_num}"

        logger.info("Original theory document: %s", original_doc_path)
        logger.info("Success CSV: %s", success_csv_path)

        csv_name = os.path.basename(success_csv_path)
        if "_deduped" in csv_name:
            target_csv = success_csv_path
            logger.info("Using pre-deduplicated CSV: %s", target_csv)
        else:
            csv_dir = os.path.dirname(success_csv_path)
            base_name, ext = os.path.splitext(csv_name)
            dedup_csv_path = os.path.join(csv_dir, f"{base_name}_deduped{ext}")
            processed_csv = deduplicate_success_materials(success_csv_path, dedup_csv_path)
            target_csv = processed_csv or success_csv_path
            if processed_csv:
                logger.info("Using deduplicated CSV: %s", processed_csv)
            else:
                logger.warning("Deduplication failed, using original CSV: %s", success_csv_path)

        updated_doc_path = update_theory_from_success(
            original_doc_path=original_doc_path,
            success_csv=target_csv,
            output_dir=output_dir,
            iteration_num=iteration_num,
            results_root=results_root,
        )

        if not updated_doc_path:
            logger.warning("Theory update returned no output")
            return None

        logger.info("Theory document updated: %s", updated_doc_path)
        return {
            "success": True,
            "analysis_report": f"{results_root}/iteration_{iteration_num}/reports/llm_theory_update_output.md",
            "updated_doc": updated_doc_path,
            "iteration_num": iteration_num,
        }
    except Exception as exc:
        logger.exception("Theory update failed: %s", exc)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_csv = "results/iteration_1/success_examples/success_materials.csv"
    if os.path.exists(test_csv):
        print(analyze_success_and_update_theory(success_csv_path=test_csv, iteration_num=1, version=1))
    else:
        print(f"Test CSV not found: {test_csv}")
