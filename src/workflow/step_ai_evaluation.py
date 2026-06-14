# -*- coding: utf-8 -*-
"""
Workflow Step 3: AI Material Evaluation
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from agents.material_evaluator import (
    MaterialEvaluator,
    _extract_candidate_scores,
    _extract_selected_materials,
    _limit_selected_materials,
    save_evaluation_report,
)
from utils.path_config import PathConfig


CANONICAL_DOC_NAME = "Theoretical_principle_document.md"
LEGACY_DOC_NAME = "\u7406\u8bba\u539f\u7406\u6587\u6863.md"


def _parse_doc_version(version_dir: Path) -> tuple[int, ...]:
    name = version_dir.name
    if not name.startswith("v"):
        return (0,)
    try:
        return tuple(int(part) for part in name[1:].split("."))
    except Exception:
        return (0,)


def _resolve_theory_doc_path(
    iteration_num: int,
    path_config: Optional[PathConfig],
    results_root: str,
    doc_root: str,
    init_doc_path: str | None,
) -> Optional[Path]:
    expected_doc_path: Optional[Path] = None

    if path_config:
        try:
            expected_doc_path = path_config.get_theory_doc_path(iteration_num)
        except ValueError:
            return None
    else:
        if iteration_num == 1:
            if init_doc_path:
                expected_doc_path = Path(init_doc_path)
                if not expected_doc_path.is_absolute():
                    expected_doc_path = project_root / expected_doc_path
            else:
                expected_doc_path = project_root / doc_root / "v0.0.0" / CANONICAL_DOC_NAME
        else:
            expected_doc_path = project_root / doc_root / f"v0.0.{iteration_num - 1}" / CANONICAL_DOC_NAME

    if expected_doc_path and expected_doc_path.exists():
        return expected_doc_path

    if expected_doc_path:
        print(f"WARNING theory document missing: {expected_doc_path}")

    llm_doc_root = path_config.doc_root if path_config and path_config.doc_root else (project_root / doc_root)
    candidate_paths: list[Path] = []
    candidate_paths.append(project_root / "doc" / CANONICAL_DOC_NAME)

    for i in range(iteration_num - 1, -1, -1):
        candidate_paths.append(llm_doc_root / f"v0.0.{i}" / CANONICAL_DOC_NAME)
        candidate_paths.append(llm_doc_root / f"v0.0.{i}" / LEGACY_DOC_NAME)

    candidate_paths.append(llm_doc_root / "v0.0.0" / CANONICAL_DOC_NAME)
    candidate_paths.append(llm_doc_root / "v0.0.0" / LEGACY_DOC_NAME)

    if llm_doc_root.exists():
        version_dirs = sorted(
            [p for p in llm_doc_root.iterdir() if p.is_dir() and p.name.startswith("v")],
            key=_parse_doc_version,
            reverse=True,
        )
        for vdir in version_dirs:
            candidate_paths.append(vdir / CANONICAL_DOC_NAME)
            candidate_paths.append(vdir / LEGACY_DOC_NAME)

    legacy_root = project_root / "llm_first_version" / "doc"
    if legacy_root.exists():
        version_dirs = sorted(
            [p for p in legacy_root.iterdir() if p.is_dir() and p.name.startswith("v")],
            key=_parse_doc_version,
            reverse=True,
        )
        for vdir in version_dirs:
            candidate_paths.append(vdir / CANONICAL_DOC_NAME)
            candidate_paths.append(vdir / LEGACY_DOC_NAME)

    candidate_paths.append(project_root / "assets" / "theory.md")

    resolved = next((p for p in candidate_paths if p.exists()), None)
    if not resolved:
        return None

    canonical_target = llm_doc_root / "v0.0.0" / CANONICAL_DOC_NAME
    try:
        canonical_target.parent.mkdir(parents=True, exist_ok=True)
        if resolved != canonical_target and not canonical_target.exists():
            shutil.copy2(resolved, canonical_target)
            print(f"INFO backfilled theory document to: {canonical_target}")
    except Exception as exc:
        print(f"WARNING failed to backfill theory document: {exc}")

    return resolved


def normalize_formula(formula: str) -> str:
    subscript_map = {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
    }
    normalized = formula
    for subscript, normal in subscript_map.items():
        normalized = normalized.replace(subscript, normal)
    return normalized


def step_ai_evaluation(
    iteration_num: int,
    candidate_materials: list,
    n_select: int = 5,
    evaluation_mode: str = "selected_materials",
    path_config: Optional[PathConfig] = None,
    results_root: str = "llm/results",
    doc_root: str = "llm/doc",
    init_doc_path: str | None = None,
    extra_instructions: str | None = None,
):
    """
    Step 3: AI material evaluation.

    Returns either final selected materials or full candidate scores depending on
    evaluation_mode.
    """
    print("=" * 80)
    print(f"Step 3: AI Material Evaluation (Iteration {iteration_num})")
    print("=" * 80)

    if path_config:
        try:
            resolved_results_root = str(path_config.results_root.relative_to(path_config.project_root))
        except Exception:
            resolved_results_root = results_root
    else:
        resolved_results_root = results_root

    doc_path = _resolve_theory_doc_path(
        iteration_num=iteration_num,
        path_config=path_config,
        results_root=results_root,
        doc_root=doc_root,
        init_doc_path=init_doc_path,
    )

    if not doc_path or not doc_path.exists():
        print("ERROR theory document not found")
        return {
            "success": False,
            "error": "Theory document not found",
        }

    print(f"theory document: {doc_path}")
    print(f"candidate count: {len(candidate_materials)}")
    print(f"target select count: {n_select}")
    print(f"evaluation mode: {evaluation_mode}")

    try:
        evaluator = MaterialEvaluator(doc_path=str(doc_path))
        normalized_mode = "candidate_scores" if evaluation_mode == "candidate_scores" else "selected_materials"
        evaluation_result = evaluator.evaluate_materials(
            candidate_materials,
            n_select=n_select,
            iteration_num=iteration_num,
            results_root=resolved_results_root,
            extra_instructions=extra_instructions,
            evaluation_mode=normalized_mode,
        )

        artifact_path = save_evaluation_report(evaluation_result)
        print(f"saved evaluation artifact: {artifact_path}")

        if normalized_mode == "candidate_scores":
            candidate_scores = _extract_candidate_scores(evaluation_result["evaluation"])
            if not candidate_scores:
                print("ERROR no candidate scores extracted")
                return {
                    "success": False,
                    "error": "No candidate scores extracted",
                }
            return {
                "success": True,
                "n_candidates": len(candidate_scores),
                "candidate_scores": candidate_scores,
                "csv_path": str(artifact_path) if artifact_path else None,
                "report_path": str(
                    project_root / resolved_results_root / f"iteration_{iteration_num}" / "reports" / "llm_candidate_scoring_output.md"
                ),
            }

        selected_materials = _limit_selected_materials(
            _extract_selected_materials(evaluation_result["evaluation"]),
            n_select,
        )
        if not selected_materials:
            print("ERROR no materials selected")
            return {
                "success": False,
                "error": "No materials selected",
            }

        selected_csv_dir = project_root / results_root / f"iteration_{iteration_num}" / "selected_results"
        selected_csv_dir.mkdir(parents=True, exist_ok=True)
        selected_csv_path = selected_csv_dir / "ai_selected_materials.csv"

        df_selected = pd.DataFrame(selected_materials)
        if "formula" in df_selected.columns:
            df_selected["formula"] = df_selected["formula"].apply(normalize_formula)

        df_selected.to_csv(selected_csv_path, index=False, encoding="utf-8-sig")
        print(f"saved selected materials to: {selected_csv_path}")

        return {
            "success": True,
            "n_selected": len(selected_materials),
            "selected_materials": selected_materials,
            "csv_path": str(selected_csv_path),
            "report_path": str(
                project_root / resolved_results_root / f"iteration_{iteration_num}" / "reports" / "llm_evaluation_output.md"
            ),
        }

    except Exception as exc:
        print(f"ERROR AI evaluation failed: {exc}")
        import traceback

        traceback.print_exc()
        return {
            "success": False,
            "error": str(exc),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--n-select", type=int, default=5)
    args = parser.parse_args()

    mock_materials = [
        {"formula": "AgBiS2", "predicted_k": 0.5, "ei": 0.1},
        {"formula": "CuSbSe2", "predicted_k": 0.6, "ei": 0.09},
    ]

    result = step_ai_evaluation(args.iteration, mock_materials, args.n_select)
    print(f"\nresult: {result}")
