from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
from pydantic import ValidationError

from agents.screening_agent import enrich_topn_with_websearch, rank_by_ei
from schemas import WorkflowInput
from utils.bo_runtime import extract_initial_samples_from_result
from utils.config_loader import get_effective_thresholds
from utils.param_sheet import persist_param_values
from utils.theory_doc_context import build_websearch_theory_context
from utils.workflow_resume import (
    load_saved_ai_evaluation_result,
    load_saved_bayesian_result,
    load_saved_document_update_result,
    load_saved_extract_result,
    reset_steps_from,
)
from workflow.agno_state import AGNO_SESSION_STATE_DEFAULT, AGNO_STATE_KEYS
from workflow.step_ai_evaluation import step_ai_evaluation
from workflow.step_bayesian_optimization import step_bayesian_optimization
from workflow.step_extract_materials import step_extract_materials
from workflow.step_merge_results import step_merge_results
from workflow.step_structure_calculation import step_structure_calculation
from workflow.step_train_model import step_train_model
from workflow.step_update_data_doc import step_update_data_and_doc


def run_train_step(iteration_num: int, config: dict[str, Any], tracker=None) -> dict[str, Any]:
    step_key = "train_model"
    if tracker and tracker.is_step_completed(iteration_num, step_key):
        return {"success": True, "skipped": True}

    result = step_train_model(
        iteration_num=iteration_num,
        data_root=config["data_root"],
        models_root=config["models_root"],
        path_config=config.get("path_config"),
    )
    if result.get("success") and tracker:
        tracker.mark_step_completed(iteration_num, step_key)
    return result


def run_bayesian_step(
    iteration_num: int,
    config: dict[str, Any],
    tracker=None,
    initial_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    step_key = "bayesian_optimization"
    if tracker and tracker.is_step_completed(iteration_num, step_key):
        cached = load_saved_bayesian_result(
            results_root=config["results_root"],
            iteration_num=iteration_num,
            n_top=int(config.get("top_k_bayes", 10)),
        )
        if cached is not None:
            print(f"[resume] loaded saved BO artifacts for iteration {iteration_num}: {cached.get('artifact_path')}")
            return cached
        print(f"[resume] BO artifacts missing for iteration {iteration_num}, resetting progress from {step_key}")
        reset_steps_from(tracker, iteration_num, step_key)

    result = step_bayesian_optimization(
        iteration_num=iteration_num,
        xi=config["xi"],
        n_samples=config["samples"],
        n_top=config["top_k_bayes"],
        initial_samples=initial_samples,
        seed=config.get("seed"),
        seed_stride=int(config.get("seed_stride", 1000)),
        path_config=config.get("path_config"),
        models_root=config["models_root"],
        results_root=config["results_root"],
    )
    if result.get("success") and tracker:
        tracker.mark_step_completed(iteration_num, step_key)
    return result


def run_extract_step(iteration_num: int, config: dict[str, Any], tracker=None) -> dict[str, Any]:
    step_key = "success_extraction"
    if tracker and tracker.is_step_completed(iteration_num, step_key):
        cached = load_saved_extract_result(config["results_root"], iteration_num)
        if cached is not None:
            print(f"[resume] loaded saved extraction artifacts for iteration {iteration_num}")
            return cached
        print(f"[resume] extraction artifacts missing for iteration {iteration_num}, resetting progress from {step_key}")
        reset_steps_from(tracker, iteration_num, step_key)

    thresholds = get_effective_thresholds()
    raw_k_threshold = config.get("k_threshold")
    raw_imag_tol = config.get("phonon_imag_tol")
    k_threshold = float(
        thresholds["thermal_conductivity"] if raw_k_threshold is None else raw_k_threshold
    )
    imag_tol = float(
        thresholds["dynamic_min_frequency"] if raw_imag_tol is None else raw_imag_tol
    )

    result = step_extract_materials(
        iteration_num=iteration_num,
        k_threshold=k_threshold,
        imag_tol=imag_tol,
        results_root=config["results_root"],
    )
    if (result.get("success") or result.get("no_materials")) and tracker:
        tracker.mark_step_completed(iteration_num, step_key)
    return result


def run_ai_evaluation_step(
    iteration_num: int,
    config: dict[str, Any],
    candidate_materials: list[dict[str, Any]],
    tracker=None,
) -> dict[str, Any]:
    step_key = "ai_evaluation"
    if tracker and tracker.is_step_completed(iteration_num, step_key):
        cached = load_saved_ai_evaluation_result(config["results_root"], iteration_num)
        if cached is not None:
            print(f"[resume] loaded saved AI screening artifacts for iteration {iteration_num}")
            return cached
        print(f"[resume] AI screening artifacts missing for iteration {iteration_num}, resetting progress from {step_key}")
        reset_steps_from(tracker, iteration_num, step_key)

    screening_mode = str(config.get("screening_mode") or "llm_bo_fusion")
    if screening_mode == "bo_direct":
        return {
            "success": False,
            "error": (
                "screening_mode=bo_direct has moved to main_bo_only.py. "
                "Use the standalone BO-only entrypoint for this mode."
            ),
            "screening_mode": screening_mode,
        }
    elif screening_mode == "llm_full_rerank":
        ai_result = step_ai_evaluation(
            iteration_num=iteration_num,
            candidate_materials=candidate_materials,
            n_select=config["top_k_screen"],
            evaluation_mode="selected_materials",
            path_config=config.get("path_config"),
            results_root=config["results_root"],
            doc_root=config.get("doc_root", "llm/doc"),
            init_doc_path=config.get("init_doc_path"),
        )
        if not ai_result.get("success"):
            return ai_result
        result = _build_full_rerank_selection(
            iteration_num=iteration_num,
            results_root=config["results_root"],
            candidate_materials=candidate_materials,
            ai_result=ai_result,
            screening_mode=screening_mode,
        )
    else:
        ai_result = step_ai_evaluation(
            iteration_num=iteration_num,
            candidate_materials=candidate_materials,
            n_select=config["top_k_screen"],
            evaluation_mode="candidate_scores",
            path_config=config.get("path_config"),
            results_root=config["results_root"],
            doc_root=config.get("doc_root", "llm/doc"),
            init_doc_path=config.get("init_doc_path"),
        )
        if not ai_result.get("success"):
            return ai_result
        result = _build_fusion_selection(
            iteration_num=iteration_num,
            results_root=config["results_root"],
            candidate_materials=candidate_materials,
            candidate_scores=ai_result.get("candidate_scores", []),
            n_select=int(config["top_k_screen"]),
            screening_mode=screening_mode,
            report_path=ai_result.get("report_path"),
        )

    if result.get("success") and tracker:
        tracker.mark_step_completed(
            iteration_num,
            step_key,
            metadata={
                "selected_materials": len(result.get("selected_materials", [])),
                "report_file": result.get("report_path"),
                "selection_trace": result.get("trace_path"),
                "screening_mode": result.get("screening_mode"),
            },
        )
    return result


def run_structure_step(
    iteration_num: int,
    config: dict[str, Any],
    materials: list[dict[str, Any]],
    tracker=None,
) -> dict[str, Any]:
    step_key = "structure_calculation"
    if tracker and tracker.is_step_completed(iteration_num, step_key):
        results_root = Path(config["results_root"]) / f"iteration_{iteration_num}"
        processed_dir = results_root / "processed_structures"
        relax_dir = results_root / "MyRelaxStructure"
        if processed_dir.exists() or relax_dir.exists():
            print(f"[resume] loaded saved structure artifacts for iteration {iteration_num}")
            return {
                "success": True,
                "skipped": True,
                "completed": True,
                "gen_output_dir": str(processed_dir),
                "relax_output_dir": str(relax_dir),
            }
        print(f"[resume] structure artifacts missing for iteration {iteration_num}, resetting progress from {step_key}")
        reset_steps_from(tracker, iteration_num, step_key)

    return step_structure_calculation(
        iteration_num=iteration_num,
        materials=materials,
        n_structures=config["n_structures"],
        max_workers=config["max_workers"],
        relax_workers=config["relax_workers"],
        phonon_workers=config["phonon_workers"],
        pressure=config["pressure"],
        device=config["device"],
        gpus=config["gpus"],
        results_root=config["results_root"],
        seed=config.get("seed"),
        tracker=tracker,
        allow_partial_completion=config.get("allow_partial_structure", False),
        path_config=config.get("path_config"),
        relax_timeout_sec=config.get("relax_timeout_sec", 900),
        prefer_isolated_relax_process=config.get("prefer_isolated_relax_process", True),
        allow_in_process_relax_fallback=config.get("allow_in_process_relax_fallback", True),
    )


def run_merge_step(iteration_num: int, config: dict[str, Any], tracker=None) -> dict[str, Any]:
    step_key = "merge_results"
    if tracker and tracker.is_step_completed(iteration_num, step_key):
        relax_dir = Path(config["results_root"]) / f"iteration_{iteration_num}" / "MyRelaxStructure"
        if relax_dir.exists():
            print(f"[resume] loaded saved merge artifacts for iteration {iteration_num}")
            return {"success": True, "skipped": True}
        print(f"[resume] merge artifacts missing for iteration {iteration_num}, resetting progress from {step_key}")
        reset_steps_from(tracker, iteration_num, step_key)

    return step_merge_results(iteration_num=iteration_num, results_root=config["results_root"], tracker=tracker)


def run_document_update_step(
    iteration_num: int,
    config: dict[str, Any],
    extraction_result: dict[str, Any],
    tracker=None,
) -> dict[str, Any]:
    step_key = "document_update"
    if tracker and tracker.is_step_completed(iteration_num, step_key):
        cached = load_saved_document_update_result(
            results_root=config["results_root"],
            data_root=config.get("data_root", "llm/data"),
            doc_root=config.get("doc_root", "llm/doc"),
            iteration_num=iteration_num,
        )
        if cached is not None:
            print(f"[resume] loaded saved document update artifacts for iteration {iteration_num}")
            return cached
        print(f"[resume] document update artifacts missing for iteration {iteration_num}, resetting progress from {step_key}")
        reset_steps_from(tracker, iteration_num, step_key)

    result = step_update_data_and_doc(
        iteration_num=iteration_num,
        extraction_result=extraction_result,
        version=config.get("version", 1),
        path_config=config.get("path_config"),
        data_root=config.get("data_root", "llm/data"),
        results_root=config.get("results_root", "llm/results"),
        doc_root=config.get("doc_root", "llm/doc"),
        skip_doc_update=config.get("skip_doc_update", False),
    )
    if result.get("success") and tracker:
        tracker.mark_step_completed(
            iteration_num,
            step_key,
            metadata={
                "updated_data_path": result.get("updated_data_path"),
                "updated_doc_path": result.get("updated_doc_path"),
            },
        )
    return result


def _selected_results_dir(results_root: str | Path, iteration_num: int) -> Path:
    path = Path(results_root) / f"iteration_{iteration_num}" / "selected_results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _formula_key(material: dict[str, Any]) -> str:
    value = str(material.get("formula") or material.get("composition") or material.get("name") or "").strip()
    return (
        value.replace("₀", "0")
        .replace("₁", "1")
        .replace("₂", "2")
        .replace("₃", "3")
        .replace("₄", "4")
        .replace("₅", "5")
        .replace("₆", "6")
        .replace("₇", "7")
        .replace("₈", "8")
        .replace("₉", "9")
        .replace(" ", "")
    )


def _candidate_rows(candidate_materials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, material in enumerate(candidate_materials, start=1):
        row = dict(material)
        row["formula"] = _formula_key(material)
        row["original_bo_rank"] = int(material.get("rank") or idx)
        row["k_pred"] = _safe_float(material.get("k_pred"))
        row["ei"] = _safe_float(material.get("ei", material.get("score")))
        row["sigma_log"] = _safe_float(material.get("sigma_log"))
        rows.append(row)
    return rows


def _lookup_candidates(candidate_materials: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        row["formula"]: row
        for row in _candidate_rows(candidate_materials)
        if row.get("formula")
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _persist_selection_outputs(
    *,
    iteration_num: int,
    results_root: str,
    screening_mode: str,
    selected_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    report_path: str | None = None,
) -> dict[str, str]:
    base_dir = _selected_results_dir(results_root, iteration_num)
    selected_csv_path = base_dir / "ai_selected_materials.csv"
    trace_csv_path = base_dir / "selection_trace.csv"
    trace_json_path = base_dir / "selection_trace.json"

    _write_csv(selected_csv_path, selected_rows)
    _write_csv(trace_csv_path, trace_rows)
    _write_json(
        trace_json_path,
        {
            "iteration": iteration_num,
            "screening_mode": screening_mode,
            "selected_count": len(selected_rows),
            "selected_formulas": [row.get("formula") for row in selected_rows],
            "report_path": report_path,
            "rows": trace_rows,
        },
    )
    return {
        "selected_csv": str(selected_csv_path),
        "trace_csv": str(trace_csv_path),
        "trace_json": str(trace_json_path),
    }


def _build_rank_cutoff_selection(
    *,
    iteration_num: int,
    results_root: str,
    candidate_materials: list[dict[str, Any]],
    n_select: int,
    screening_mode: str,
) -> dict[str, Any]:
    candidate_rows = _candidate_rows(candidate_materials)
    selected_formulas: set[str] = set()
    selected_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(candidate_rows, start=1):
        selected = idx <= n_select
        trace_row = {
            "formula": row["formula"],
            "original_bo_rank": row["original_bo_rank"],
            "k_pred": row["k_pred"],
            "ei": row["ei"],
            "sigma_log": row["sigma_log"],
            "screening_mode": screening_mode,
            "selected_for_calc": selected,
            "final_rank": idx if selected else "",
            "selection_reason": "bo_rank_cutoff" if selected else "",
        }
        trace_rows.append(trace_row)
        if selected:
            selected_formulas.add(row["formula"])
            selected_rows.append({**row, **trace_row})

    artifact_paths = _persist_selection_outputs(
        iteration_num=iteration_num,
        results_root=results_root,
        screening_mode=screening_mode,
        selected_rows=selected_rows,
        trace_rows=trace_rows,
    )
    return {
        "success": True,
        "n_selected": len(selected_rows),
        "selected_materials": selected_rows,
        "csv_path": artifact_paths["selected_csv"],
        "trace_path": artifact_paths["trace_json"],
        "trace_csv_path": artifact_paths["trace_csv"],
        "selection_trace": trace_rows,
        "screening_mode": screening_mode,
    }


def _build_full_rerank_selection(
    *,
    iteration_num: int,
    results_root: str,
    candidate_materials: list[dict[str, Any]],
    ai_result: dict[str, Any],
    screening_mode: str,
) -> dict[str, Any]:
    candidate_lookup = _lookup_candidates(candidate_materials)
    selected_rows: list[dict[str, Any]] = []
    selected_lookup: dict[str, dict[str, Any]] = {}

    for item in ai_result.get("selected_materials", []):
        formula = _formula_key(item)
        candidate = candidate_lookup.get(formula)
        if not formula or candidate is None or formula in selected_lookup:
            continue
        row = {
            **candidate,
            "formula": formula,
            "original_bo_rank": int(candidate.get("original_bo_rank", item.get("original_rank") or 0)),
            "screening_mode": screening_mode,
            "selected_for_calc": True,
            "final_rank": int(item.get("final_rank") or len(selected_rows) + 1),
            "ranking_reason": str(item.get("ranking_reason", "")).strip(),
            "main_risk": str(item.get("main_risk", "")).strip(),
        }
        selected_rows.append(row)
        selected_lookup[formula] = row

    selected_rows = sorted(selected_rows, key=lambda row: (int(row.get("final_rank", 10**9)), row.get("formula", "")))
    for idx, row in enumerate(selected_rows, start=1):
        row["final_rank"] = idx

    trace_rows: list[dict[str, Any]] = []
    for row in _candidate_rows(candidate_materials):
        selected = selected_lookup.get(row["formula"])
        trace_rows.append(
            {
                "formula": row["formula"],
                "original_bo_rank": row["original_bo_rank"],
                "k_pred": row["k_pred"],
                "ei": row["ei"],
                "sigma_log": row["sigma_log"],
                "screening_mode": screening_mode,
                "selected_for_calc": bool(selected),
                "final_rank": selected.get("final_rank") if selected else "",
                "ranking_reason": selected.get("ranking_reason", "") if selected else "",
                "main_risk": selected.get("main_risk", "") if selected else "",
            }
        )

    artifact_paths = _persist_selection_outputs(
        iteration_num=iteration_num,
        results_root=results_root,
        screening_mode=screening_mode,
        selected_rows=selected_rows,
        trace_rows=trace_rows,
        report_path=ai_result.get("report_path"),
    )
    return {
        "success": True,
        "n_selected": len(selected_rows),
        "selected_materials": selected_rows,
        "csv_path": artifact_paths["selected_csv"],
        "trace_path": artifact_paths["trace_json"],
        "trace_csv_path": artifact_paths["trace_csv"],
        "selection_trace": trace_rows,
        "report_path": ai_result.get("report_path"),
        "raw_ai_result": ai_result,
        "screening_mode": screening_mode,
    }


def _minmax_normalize(values: list[float], *, inverse: bool = False) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if abs(maximum - minimum) < 1e-12:
        normalized = [1.0 for _ in values]
    else:
        normalized = [(value - minimum) / (maximum - minimum) for value in values]
    if inverse:
        normalized = [1.0 - value for value in normalized]
    return normalized


def _build_fusion_selection(
    *,
    iteration_num: int,
    results_root: str,
    candidate_materials: list[dict[str, Any]],
    candidate_scores: list[dict[str, Any]],
    n_select: int,
    screening_mode: str,
    report_path: str | None = None,
) -> dict[str, Any]:
    candidate_rows = _candidate_rows(candidate_materials)
    if len(candidate_scores) < len(candidate_rows):
        return {
            "success": False,
            "error": f"Candidate score coverage mismatch: expected {len(candidate_rows)}, got {len(candidate_scores)}",
        }

    scores_by_formula = {
        _formula_key(item): item
        for item in candidate_scores
        if _formula_key(item)
    }

    if len(scores_by_formula) < len(candidate_rows):
        return {
            "success": False,
            "error": f"Candidate score uniqueness mismatch: expected {len(candidate_rows)}, got {len(scores_by_formula)}",
        }

    ei_norm = _minmax_normalize([row["ei"] for row in candidate_rows])
    k_pred_norm = _minmax_normalize([row["k_pred"] for row in candidate_rows], inverse=True)
    sigma_norm = _minmax_normalize([row["sigma_log"] for row in candidate_rows])
    rank_norm = _minmax_normalize([row["original_bo_rank"] for row in candidate_rows], inverse=True)

    trace_rows: list[dict[str, Any]] = []
    forced_keep_formulas: set[str] = set()
    eligible_pool: list[dict[str, Any]] = []

    for idx, row in enumerate(candidate_rows):
        formula = row["formula"]
        score_row = scores_by_formula.get(formula)
        if score_row is None:
            return {
                "success": False,
                "error": f"Missing candidate score for formula: {formula}",
            }

        bo_base_score = (
            0.40 * ei_norm[idx]
            + 0.30 * k_pred_norm[idx]
            + 0.15 * sigma_norm[idx]
            + 0.15 * rank_norm[idx]
        )
        llm_raw = (
            0.50 * (_safe_float(score_row.get("mechanism_fit_score")) / 10.0)
            - 0.60 * (_safe_float(score_row.get("stability_risk_score")) / 10.0)
            + 0.20 * (_safe_float(score_row.get("novelty_bonus_score")) / 10.0)
        )
        confidence = int(score_row.get("bo_override_confidence", 0) or 0)
        if confidence >= 8:
            llm_adjustment = 1.0 * llm_raw
        elif confidence >= 5:
            llm_adjustment = 0.5 * llm_raw
        else:
            llm_adjustment = 0.2 * llm_raw

        final_score = 0.85 * bo_base_score + 0.15 * llm_adjustment
        original_bo_rank = int(row["original_bo_rank"])

        if original_bo_rank <= 3:
            selection_gate_status = "protected_top3"
        elif original_bo_rank >= 13:
            if (
                int(score_row.get("mechanism_fit_score", 0) or 0) >= 8
                and int(score_row.get("stability_risk_score", 0) or 0) <= 4
                and confidence >= 8
            ):
                selection_gate_status = "tail_promotion_allowed"
            else:
                selection_gate_status = "tail_blocked"
        else:
            selection_gate_status = "normal_pool"

        trace_row = {
            **row,
            "formula": formula,
            "screening_mode": screening_mode,
            "selected_for_calc": False,
            "mechanism_fit_score": int(score_row.get("mechanism_fit_score", 0) or 0),
            "stability_risk_score": int(score_row.get("stability_risk_score", 0) or 0),
            "novelty_bonus_score": int(score_row.get("novelty_bonus_score", 0) or 0),
            "bo_override_confidence": confidence,
            "short_reason": str(score_row.get("short_reason", "")).strip(),
            "main_risk": str(score_row.get("main_risk", "")).strip(),
            "bo_base_score": bo_base_score,
            "llm_adjustment": llm_adjustment,
            "final_score": final_score,
            "selection_gate_status": selection_gate_status,
        }
        trace_rows.append(trace_row)

        protected = original_bo_rank <= 3 and not (
            trace_row["stability_risk_score"] >= 9 and confidence >= 8
        )
        if protected:
            forced_keep_formulas.add(formula)

        if selection_gate_status == "tail_blocked":
            continue
        eligible_pool.append(trace_row)

    selected_lookup: dict[str, dict[str, Any]] = {}
    for row in trace_rows:
        if row["formula"] in forced_keep_formulas:
            selected_lookup[row["formula"]] = dict(row)

    remaining = [row for row in eligible_pool if row["formula"] not in selected_lookup]
    remaining = sorted(remaining, key=lambda row: (-row["final_score"], row["original_bo_rank"], row["formula"]))

    for row in remaining:
        if len(selected_lookup) >= n_select:
            break
        selected_lookup[row["formula"]] = dict(row)

    selected_rows = sorted(
        selected_lookup.values(),
        key=lambda row: (-_safe_float(row.get("final_score")), int(row.get("original_bo_rank", 10**9)), row.get("formula", "")),
    )[:n_select]
    for idx, row in enumerate(selected_rows, start=1):
        row["selected_for_calc"] = True
        row["final_rank"] = idx

    selected_formula_set = {row["formula"] for row in selected_rows}
    for row in trace_rows:
        row["selected_for_calc"] = row["formula"] in selected_formula_set
        if row["formula"] in selected_formula_set:
            row["final_rank"] = next(item["final_rank"] for item in selected_rows if item["formula"] == row["formula"])
        else:
            row["final_rank"] = ""

    artifact_paths = _persist_selection_outputs(
        iteration_num=iteration_num,
        results_root=results_root,
        screening_mode=screening_mode,
        selected_rows=selected_rows,
        trace_rows=trace_rows,
        report_path=report_path,
    )
    return {
        "success": True,
        "n_selected": len(selected_rows),
        "selected_materials": selected_rows,
        "candidate_scores": candidate_scores,
        "csv_path": artifact_paths["selected_csv"],
        "trace_path": artifact_paths["trace_json"],
        "trace_csv_path": artifact_paths["trace_csv"],
        "selection_trace": trace_rows,
        "report_path": report_path,
        "screening_mode": screening_mode,
    }


def _load_formula_set(csv_path: str | None) -> set[str]:
    if not csv_path:
        return set()
    path = Path(csv_path)
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return set()
    formulas: set[str] = set()
    for column in ("formula", "Formula", "composition"):
        if column not in df.columns:
            continue
        formulas.update(str(value).strip() for value in df[column].tolist() if str(value).strip())
    return formulas


def _best_kappa_for_selected_success(extraction_result: dict[str, Any], selected_formulas: set[str]) -> float | None:
    success_path = extraction_result.get("success_deduped_file") or extraction_result.get("success_file")
    if not success_path or not Path(success_path).exists():
        return None
    try:
        df = pd.read_csv(success_path, encoding="utf-8-sig")
    except Exception:
        return None
    if "formula" not in df.columns:
        return None
    filtered = df[df["formula"].astype(str).isin(selected_formulas)]
    if filtered.empty:
        return None
    for column in ("thermal_conductivity_w_mk", "thermal_conductivity", "kappa", "k_pred"):
        if column in filtered.columns:
            series = pd.to_numeric(filtered[column], errors="coerce").dropna()
            if not series.empty:
                return float(series.min())
    return None


def _update_screening_summary(
    *,
    iteration_num: int,
    results_root: str,
    screen_result: dict[str, Any],
    extraction_result: dict[str, Any],
) -> str:
    summary_path = Path(results_root) / "screening_summary.csv"
    selected_rows = list(screen_result.get("selected_materials", []))
    selected_formulas = {str(row.get("formula") or "").strip() for row in selected_rows if str(row.get("formula") or "").strip()}
    success_formulas = _load_formula_set(extraction_result.get("success_deduped_file") or extraction_result.get("success_file"))
    stable_formulas = _load_formula_set(extraction_result.get("stable_deduped_file") or extraction_result.get("stable_file"))

    selected_count = len(selected_rows)
    success_count = sum(1 for row in selected_rows if str(row.get("formula") or "").strip() in success_formulas)
    stable_count = sum(1 for row in selected_rows if str(row.get("formula") or "").strip() in stable_formulas)
    selected_from_bo_top3_count = sum(1 for row in selected_rows if int(row.get("original_bo_rank", 10**9) or 10**9) <= 3)
    selected_from_bo_13_20_count = sum(
        1 for row in selected_rows if 13 <= int(row.get("original_bo_rank", 10**9) or 10**9) <= 20
    )
    best_kappa = _best_kappa_for_selected_success(extraction_result, selected_formulas)
    tail_promotions_count = sum(
        1
        for row in selected_rows
        if row.get("selection_gate_status") == "tail_promotion_allowed"
    )
    tail_promotions_success_count = sum(
        1
        for row in selected_rows
        if row.get("selection_gate_status") == "tail_promotion_allowed"
        and str(row.get("formula") or "").strip() in success_formulas
    )
    protected_top3_dropped_count = sum(
        1
        for row in screen_result.get("selection_trace", [])
        if row.get("selection_gate_status") == "protected_top3" and not bool(row.get("selected_for_calc"))
    )

    summary_row = {
        "iteration": iteration_num,
        "screening_mode": screen_result.get("screening_mode"),
        "selected_count": selected_count,
        "success_count": success_count,
        "stable_count": stable_count,
        "success_rate_at_k": (success_count / selected_count) if selected_count else 0.0,
        "stable_rate_at_k": (stable_count / selected_count) if selected_count else 0.0,
        "best_kappa_in_selected_success": best_kappa if best_kappa is not None else "",
        "selected_from_bo_top3_count": selected_from_bo_top3_count,
        "selected_from_bo_13_20_count": selected_from_bo_13_20_count,
        "tail_promotions_count": tail_promotions_count,
        "tail_promotions_success_count": tail_promotions_success_count,
        "protected_top3_dropped_count": protected_top3_dropped_count,
    }

    if summary_path.exists():
        existing = pd.read_csv(summary_path, encoding="utf-8-sig")
        if "iteration" in existing.columns:
            existing = existing[existing["iteration"] != iteration_num]
        updated = pd.concat([existing, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        updated = pd.DataFrame([summary_row])
    updated = updated.sort_values(by=["iteration"], kind="mergesort")
    updated.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return str(summary_path)


def _save_screening_artifacts(
    iteration_num: int,
    results_root: str,
    novel_pool: list[dict[str, Any]],
    websearch_enriched_candidates: list[dict[str, Any]],
) -> dict[str, str]:
    base_dir = Path(results_root) / f"iteration_{iteration_num}" / "selected_results"
    base_dir.mkdir(parents=True, exist_ok=True)

    def _to_csv(rows: list[dict[str, Any]], filename: str) -> str:
        path = base_dir / filename
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return str(path)

    return {
        "novel_pool_csv": _to_csv(novel_pool, "novel_candidates.csv"),
        "websearch_enriched_csv": _to_csv(websearch_enriched_candidates, "websearch_enriched_candidates.csv"),
    }


def _resolve_websearch_theory_template(iteration_num: int, config: dict[str, Any]) -> str | None:
    explicit_template = str(config.get("websearch_theory_template") or "").strip()
    if explicit_template:
        return explicit_template

    doc_path: Path | None = None
    path_config = config.get("path_config")
    if path_config:
        try:
            doc_path = path_config.get_theory_doc_path(iteration_num)
        except Exception:
            doc_path = None
    else:
        init_doc_path = str(config.get("init_doc_path") or "").strip()
        doc_root = Path(str(config.get("doc_root") or "llm/doc"))
        if iteration_num == 1 and init_doc_path:
            doc_path = Path(init_doc_path)
        elif iteration_num == 1:
            doc_path = doc_root / "v0.0.0" / "Theoretical_principle_document.md"
        else:
            doc_path = doc_root / f"v0.0.{iteration_num - 1}" / "Theoretical_principle_document.md"
        if doc_path and not doc_path.is_absolute():
            doc_path = Path.cwd() / doc_path

    if not doc_path or not doc_path.exists():
        return None

    try:
        doc_text = doc_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"[websearch] failed to read theory doc for query context: {doc_path}, error={exc}")
        return None

    theory_template = build_websearch_theory_context(doc_text, max_chars=700)
    if theory_template:
        print(f"[websearch] loaded round-aware theory context from: {doc_path}")
    return theory_template or None


def _step_input_to_text(step_input: Any) -> str:
    if step_input is None:
        return ""

    chunks: list[str] = []
    for key in ("input", "content", "message", "query", "text"):
        value = getattr(step_input, key, None)
        if value:
            chunks.append(str(value))

    if isinstance(step_input, dict):
        for key in ("input", "content", "message", "query", "text"):
            value = step_input.get(key)
            if value:
                chunks.append(str(value))

    chunks.append(str(step_input))
    return " ".join(chunks)


def _extract_requested_iterations(text: str) -> int | None:
    if not text:
        return None

    patterns = [
        r"第\s*(\d+)\s*轮",
        r"(\d+)\s*轮",
        r"(\d+)\s*iterations?",
        r"iterations?\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                value = int(match.group(1))
                if value > 0:
                    return value
            except Exception:
                continue
    return None

def _extract_step_payload(step_input: Any) -> dict[str, Any]:
    if step_input is None:
        return {}

    # Preferred path in Agno StepInput
    raw_input = getattr(step_input, "input", None)
    if isinstance(raw_input, dict):
        return raw_input
    if raw_input is None and isinstance(step_input, dict):
        raw_input = step_input.get("input", step_input)
    elif raw_input is None:
        raw_input = step_input

    if isinstance(raw_input, dict):
        return raw_input
    if isinstance(raw_input, WorkflowInput):
        return raw_input.model_dump()
    if hasattr(raw_input, "model_dump"):
        try:
            dumped = raw_input.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    return {}


def _build_theory_template_from_payload(payload: dict[str, Any]) -> str | None:
    material_type = str(payload.get("material_type") or "").strip()
    goal = str(payload.get("goal") or "").strip()
    composition = payload.get("composition") if isinstance(payload.get("composition"), dict) else {}
    processing = payload.get("processing") if isinstance(payload.get("processing"), dict) else {}
    features = payload.get("features") if isinstance(payload.get("features"), dict) else {}
    if not any([material_type, goal, composition, processing, features]):
        return None

    return (
        f"material type: {material_type or 'N/A'} | "
        f"goal: {goal or 'N/A'} | "
        f"composition constraints: {json.dumps(composition, ensure_ascii=False)} | "
        f"processing constraints: {json.dumps(processing, ensure_ascii=False)} | "
        f"features: {json.dumps(features, ensure_ascii=False)} | "
        "low lattice thermal conductivity mechanisms, phonon scattering, anharmonicity, mass disorder, lone pair, rattling"
    )


def _extract_runtime_overrides(step_input: Any, base_config: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
    payload = _extract_step_payload(step_input)
    if not payload:
        return {}, None

    try:
        wf_input = WorkflowInput.model_validate(payload)
        # Only use explicitly provided fields; avoid clobbering with schema defaults.
        normalized = wf_input.model_dump(exclude_unset=True, exclude_none=True)
    except ValidationError as exc:
        print(f"[agentos] workflow input validation failed, fallback to defaults: {exc}")
        normalized = payload

    overrides: dict[str, Any] = {}
    allowed_keys = {
        "samples",
        "n_structures",
        "top_k_bayes",
        "top_k_screen",
        "screening_mode",
        "websearch_enabled",
        "websearch_top_n",
        "phonon_imag_tol",
        "seed",
    }
    for key in allowed_keys:
        value = normalized.get(key)
        if value is not None and key in base_config:
            overrides[key] = value

    theory_template = _build_theory_template_from_payload(normalized)
    if theory_template:
        overrides["websearch_theory_template"] = theory_template

    requested_iterations = normalized.get("max_iterations")
    if requested_iterations is not None:
        try:
            requested_iterations = int(requested_iterations)
        except Exception:
            requested_iterations = None

    return overrides, requested_iterations


def _extract_formula(item: dict[str, Any]) -> str:
    return str(item.get("formula") or item.get("composition") or item.get("name") or "").strip()


def _extract_kappa(item: dict[str, Any]) -> float | None:
    for key in ("kappa", "k", "k_pred", "thermal_conductivity", "kappa_slack"):
        value = item.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _extract_result_path(item: dict[str, Any]) -> str:
    for key in ("Relative_CIF_Path", "CSV路径", "csv_path", "path", "result_path", "cif_path"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def _persist_runtime_memory(run_config: dict[str, Any], requested_iterations: int | None = None) -> None:
    csv_path = str(run_config.get("params_csv_path") or "").strip()
    if not csv_path:
        return
    payload = {
        "samples": run_config.get("samples"),
        "n_structures": run_config.get("n_structures"),
        "top_k_bayes": run_config.get("top_k_bayes"),
        "top_k_screen": run_config.get("top_k_screen"),
        "screening_mode": run_config.get("screening_mode"),
        "websearch_enabled": run_config.get("websearch_enabled"),
        "websearch_top_n": run_config.get("websearch_top_n"),
        "phonon_imag_tol": run_config.get("phonon_imag_tol"),
        "seed": run_config.get("seed"),
        "relax_timeout_sec": run_config.get("relax_timeout_sec"),
        "skip_doc_update": run_config.get("skip_doc_update"),
        "agentos_default_iterations": requested_iterations if requested_iterations is not None else run_config.get("agentos_default_iterations"),
        "agentos_ws_ping_interval": run_config.get("agentos_ws_ping_interval"),
        "agentos_ws_ping_timeout": run_config.get("agentos_ws_ping_timeout"),
    }
    updated, warnings = persist_param_values(csv_path, payload, keys=list(payload.keys()), enable_for_new_keys=False)
    if warnings:
        print(f"[agentos] param memory warnings: {warnings}")
    elif updated:
        print(f"[agentos] remembered runtime params to csv: {updated}")

def _compact_materials(materials: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for idx, item in enumerate(materials[:limit], start=1):
        compact.append(
            {
                "rank": idx,
                "name": _extract_formula(item),
                "path": _extract_result_path(item),
                "kappa_w_mk": _extract_kappa(item),
            }
        )
    return compact


def _first_existing_value(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _aggregate_materials_from_results(results_root: str) -> dict[str, Any]:
    base = Path(results_root)
    summary_dir = base / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "success": ("success_materials.csv", "success_materials_deduped.csv"),
        "stable": ("stable_materials.csv", "stable_materials_deduped.csv"),
    }

    formula_keys = ["formula", "Formula", "composition", "name", "组分", "组成"]
    kappa_keys = ["thermal_conductivity_w_mk", "thermal_conductivity", "kappa", "k", "k_pred", "热导率(W/m·K)"]
    path_keys = ["relative_cif_path", "Relative_CIF_Path", "cif_file", "CIF文件", "csv_path", "CSV路径", "path"]
    sid_keys = ["structure_id", "结构ID", "Structure_ID", "id"]

    aggregate: dict[str, pd.DataFrame] = {}
    written_paths: dict[str, str] = {}

    for tag, filename_candidates in spec.items():
        frames: list[pd.DataFrame] = []
        for success_examples_dir in sorted(base.glob("iteration_*/success_examples")):
            csv_path = None
            for filename in filename_candidates:
                candidate = success_examples_dir / filename
                if candidate.exists():
                    csv_path = candidate
                    break
            if csv_path is None:
                continue

            iter_match = re.search(r"iteration_(\d+)", str(success_examples_dir))
            iteration = int(iter_match.group(1)) if iter_match else None
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
            except Exception:
                continue
            if df.empty:
                continue

            df = df.copy()
            if "iteration" not in df.columns:
                df.insert(0, "iteration", iteration)
            else:
                df["iteration"] = df["iteration"].where(df["iteration"].notna(), iteration)
            df["source_type"] = tag
            df["source_file"] = str(csv_path)
            frames.append(df)

        if frames:
            out_df = pd.concat(frames, ignore_index=True)
            records = out_df.to_dict(orient="records")
            out_df["_summary_formula"] = [str(_first_existing_value(r, formula_keys) or "").strip() for r in records]
            out_df["_summary_rel_path"] = [str(_first_existing_value(r, path_keys) or "").strip() for r in records]
            out_df["_summary_sid"] = [str(_first_existing_value(r, sid_keys) or "").strip() for r in records]

            def _extract_kappa_value(record: dict[str, Any]) -> float | None:
                kappa = _first_existing_value(record, kappa_keys)
                if kappa is None:
                    for col_name, col_val in record.items():
                        key_text = str(col_name)
                        if "W/m" in key_text and ("K" in key_text or "k" in key_text):
                            kappa = col_val
                            break
                try:
                    return float(kappa) if kappa is not None else None
                except Exception:
                    return None

            out_df["_summary_kappa"] = [_extract_kappa_value(r) for r in records]
            out_df = out_df[out_df["_summary_formula"].astype(bool)]
            out_df = out_df[out_df["_summary_kappa"].notna()]
            out_df["dedup_key"] = out_df.apply(
                lambda r: f"{r['_summary_formula']}||{r['_summary_rel_path'] or r['_summary_sid'] or ''}", axis=1
            )
            out_df = out_df.sort_values(["_summary_kappa", "_summary_formula", "dedup_key"], ascending=True, kind="mergesort")
            out_df = out_df.drop_duplicates(subset=["dedup_key"], keep="first")
            out_df = out_df.drop(columns=["_summary_formula", "_summary_rel_path", "_summary_sid", "_summary_kappa", "dedup_key"])
        else:
            out_df = pd.DataFrame(columns=["iteration", "source_type", "source_file"])

        aggregate[tag] = out_df
        output_path = summary_dir / f"{tag}_materials_summary.csv"
        out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        written_paths[f"{tag}_summary_csv"] = str(output_path)

    all_df = pd.concat([aggregate["success"], aggregate["stable"]], ignore_index=True)
    if not all_df.empty:
        records = all_df.to_dict(orient="records")
        all_df["_summary_formula"] = [str(_first_existing_value(r, formula_keys) or "").strip() for r in records]
        all_df["_summary_rel_path"] = [str(_first_existing_value(r, path_keys) or "").strip() for r in records]
        all_df["_summary_sid"] = [str(_first_existing_value(r, sid_keys) or "").strip() for r in records]

        def _extract_kappa_value_all(record: dict[str, Any]) -> float | None:
            kappa = _first_existing_value(record, kappa_keys)
            if kappa is None:
                for col_name, col_val in record.items():
                    key_text = str(col_name)
                    if "W/m" in key_text and ("K" in key_text or "k" in key_text):
                        kappa = col_val
                        break
            try:
                return float(kappa) if kappa is not None else None
            except Exception:
                return None

        all_df["_summary_kappa"] = [_extract_kappa_value_all(r) for r in records]
        all_df = all_df[all_df["_summary_formula"].astype(bool)]
        all_df = all_df[all_df["_summary_kappa"].notna()]
        all_df["dedup_key"] = all_df.apply(
            lambda r: f"{r['_summary_formula']}||{r['_summary_rel_path'] or r['_summary_sid'] or ''}", axis=1
        )
        all_df = all_df.sort_values(["_summary_kappa", "_summary_formula", "dedup_key"], ascending=True, kind="mergesort")
        all_df = all_df.drop_duplicates(subset=["dedup_key"], keep="first")
        all_df = all_df.drop(columns=["_summary_formula", "_summary_rel_path", "_summary_sid", "_summary_kappa", "dedup_key"])
    output_path = summary_dir / "all_materials_summary.csv"
    all_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    written_paths["all_summary_csv"] = str(output_path)
    return {
        "success_summary_csv": written_paths.get("success_summary_csv"),
        "stable_summary_csv": written_paths.get("stable_summary_csv"),
        "all_summary_csv": written_paths.get("all_summary_csv"),
    }


def _compact_iteration_result(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    theory = result.get("theory", {}) if isinstance(result.get("theory"), dict) else {}
    top10 = result.get("top10", []) if isinstance(result.get("top10"), list) else []
    top20 = result.get("top20", []) if isinstance(result.get("top20"), list) else []
    chosen = top10 if top10 else top20
    return {
        "success": bool(result.get("success")),
        "iteration_num": result.get("iteration_num"),
        "failed_step": result.get("failed_step"),
        "materials": _compact_materials(chosen, limit=20),
        "updated_data_path": theory.get("updated_data_path"),
        "updated_doc_path": theory.get("updated_doc_path"),
    }


def _run_single_iteration(
    iteration_num: int,
    config: dict[str, Any],
    tracker=None,
    initial_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    train_result = run_train_step(iteration_num, config, tracker)
    if not train_result.get("success"):
        return {"success": False, "failed_step": "train_model", "train": train_result}

    bayes_result = run_bayesian_step(iteration_num, config, tracker, initial_samples)
    if not bayes_result.get("success"):
        return {"success": False, "failed_step": "bayesian_optimization", "bayes": bayes_result}

    all_candidates = bayes_result.get("all_materials") or bayes_result.get("top_materials", [])
    bayes_pool_size = int(config.get("samples", 100))
    bayes_pool = list(all_candidates)[:bayes_pool_size]

    novel_pool_ranked = rank_by_ei(bayes_pool)
    novel_top20 = novel_pool_ranked[: config["top_k_bayes"]]
    print(
        f"[screening] input_count={len(bayes_pool)}, "
        f"novel_count={len(novel_pool_ranked)}, "
        f"novel_top20_count={len(novel_top20)}"
    )

    websearch_theory_template = _resolve_websearch_theory_template(iteration_num, config)
    websearch_enriched_candidates = enrich_topn_with_websearch(
        candidates=novel_top20,
        top_n=int(config.get("websearch_top_n", 5)),
        enabled=bool(config.get("websearch_enabled", True)),
        strategy=str(config.get("websearch_strategy", "hybrid")),
        queries_per_candidate=int(config.get("websearch_queries_per_candidate", 2)),
        theory_template=websearch_theory_template,
    )
    websearch_attempted_candidates = min(len(novel_top20), int(config.get("websearch_top_n", 5)))
    total_queries = sum(len(item.get("websearch_queries", [])) for item in websearch_enriched_candidates[:websearch_attempted_candidates])
    success_queries = sum(int(item.get("websearch_success_count", 0)) for item in websearch_enriched_candidates[:websearch_attempted_candidates])
    failed_queries = max(total_queries - success_queries, 0)
    err_counter = Counter()
    for item in websearch_enriched_candidates[:websearch_attempted_candidates]:
        for err in item.get("websearch_errors", []):
            err_counter[str(err)] += 1
    top_errors = err_counter.most_common(3)
    print(
        f"[websearch] candidates={websearch_attempted_candidates}, queries={total_queries}, "
        f"success={success_queries}, failed={failed_queries}, top_errors={top_errors}"
    )

    artifact_paths = _save_screening_artifacts(
        iteration_num=iteration_num,
        results_root=config["results_root"],
        novel_pool=novel_pool_ranked,
        websearch_enriched_candidates=websearch_enriched_candidates,
    )

    screen_result = run_ai_evaluation_step(
        iteration_num=iteration_num,
        config=config,
        candidate_materials=websearch_enriched_candidates,
        tracker=tracker,
    )

    if screen_result.get("success"):
        screened_top10 = screen_result.get("selected_materials", [])
        screening_mode = str(screen_result.get("screening_mode") or config.get("screening_mode") or "llm_bo_fusion")
    else:
        fallback_result = _build_rank_cutoff_selection(
            iteration_num=iteration_num,
            results_root=config["results_root"],
            candidate_materials=websearch_enriched_candidates,
            n_select=int(config["top_k_screen"]),
            screening_mode="heuristic_fallback",
        )
        screened_top10 = fallback_result.get("selected_materials", [])
        screen_result = fallback_result
        screening_mode = "heuristic_fallback"

    calculate_result = run_structure_step(
        iteration_num=iteration_num,
        config=config,
        materials=screened_top10,
        tracker=tracker,
    )
    if not calculate_result.get("success"):
        return {
            "success": False,
            "failed_step": "calculation",
            "calculate": calculate_result,
            "top20": novel_top20,
            "top10": screened_top10,
        }
    if not calculate_result.get("completed"):
        return {
            "success": False,
            "failed_step": "calculation_incomplete",
            "calculate": calculate_result,
            "top20": novel_top20,
            "top10": screened_top10,
        }

    merge_result = run_merge_step(iteration_num=iteration_num, config=config, tracker=tracker)
    if not merge_result.get("success"):
        return {
            "success": False,
            "failed_step": "merge",
            "merge": merge_result,
            "top20": novel_top20,
            "top10": screened_top10,
        }

    extract_result = run_extract_step(iteration_num, config, tracker)
    if not extract_result.get("success") and not extract_result.get("no_materials"):
        return {
            "success": False,
            "failed_step": "extract",
            "extract": extract_result,
            "top20": novel_top20,
            "top10": screened_top10,
        }

    screening_summary_path = _update_screening_summary(
        iteration_num=iteration_num,
        results_root=config["results_root"],
        screen_result=screen_result,
        extraction_result=extract_result,
    )

    materials_summary = _aggregate_materials_from_results(config["results_root"])
    print(f"[summary] summary files: {materials_summary}")

    theory_result = run_document_update_step(
        iteration_num=iteration_num,
        config=config,
        extraction_result=extract_result,
        tracker=tracker,
    )
    if not theory_result.get("success"):
        return {
            "success": False,
            "failed_step": "theory",
            "theory": theory_result,
            "top20": novel_top20,
            "top10": screened_top10,
            "extract": extract_result,
            "materials_summary": materials_summary,
            "screening_summary_path": screening_summary_path,
        }

    return {
        "success": True,
        "iteration_num": iteration_num,
        "top20": novel_top20,
        "top10": screened_top10,
        "screening_mode": screening_mode,
        "bayes_pool_size": len(bayes_pool),
        "novel_pool": novel_pool_ranked,
        "novel_top20": novel_top20,
        "websearch_enriched_candidates": websearch_enriched_candidates,
        "train": train_result,
        "bayes": bayes_result,
        "screen": {
            "success": True,
            "selected_materials": screened_top10,
            "raw_ai_result": screen_result,
            "novel_pool": novel_pool_ranked,
            "novel_top20": novel_top20,
            "websearch_enriched_candidates": websearch_enriched_candidates,
            "artifact_paths": artifact_paths,
            "screening_mode": screening_mode,
        },
        "calculate": {"success": True, "structure": calculate_result, "merge": merge_result},
        "extract": extract_result,
        "theory": theory_result,
        "materials_summary": materials_summary,
        "screening_summary_path": screening_summary_path,
    }


def build_aslk_steps(
    config: dict[str, Any],
    tracker,
    start_iteration: int,
    max_iterations: int,
    initial_samples: list[dict[str, Any]] | None = None,
) -> list[Step]:
    from agno.workflow.step import Step, StepInput, StepOutput

    def orchestration_executor(step_input: StepInput, run_context=None) -> StepOutput:
        run_config = dict(config)
        verbose_output = bool(run_config.get("agentos_verbose_output", False))
        runtime_overrides, requested_iterations = _extract_runtime_overrides(step_input, run_config)
        print(f"[agentos] form payload keys={sorted(list(runtime_overrides.keys()))}, requested_iterations={requested_iterations}")
        if runtime_overrides:
            run_config.update(runtime_overrides)
            print(f"[agentos] runtime overrides from form: {runtime_overrides}")

        if bool(config.get("agentos_allow_text_iteration_override", True)):
            step_text = _step_input_to_text(step_input)
            text_requested_iterations = _extract_requested_iterations(step_text)
            if text_requested_iterations is not None:
                requested_iterations = text_requested_iterations
        loop_end = max_iterations
        if bool(run_config.get("max_iterations_locked", False)):
            requested_iterations = None
            loop_end = max_iterations
            print(
                f"[agentos] max_iterations locked by CLI/config, "
                f"start={start_iteration}, locked_end={loop_end}"
            )
        elif requested_iterations is not None:
            cap = int(run_config.get("agentos_max_iterations_cap", 20))
            loop_end = min(max(start_iteration, requested_iterations), cap)
            print(
                f"[agentos] requested_iterations={requested_iterations}, "
                f"start={start_iteration}, default_end={max_iterations}, resolved_end={loop_end}, cap={cap}"
            )
        _persist_runtime_memory(run_config, requested_iterations=requested_iterations)

        local_samples = initial_samples
        all_results: list[dict[str, Any]] = []
        compact_all_results: list[dict[str, Any]] = []

        session_state = getattr(run_context, "session_state", None)
        if isinstance(session_state, dict):
            for k, v in AGNO_SESSION_STATE_DEFAULT.items():
                session_state.setdefault(k, v if not isinstance(v, list) else list(v))

        for iteration in range(start_iteration, loop_end + 1):
            if iteration > int(max_iterations):
                print(
                    f"[agentos] stop guard reached: iteration={iteration}, "
                    f"max_iterations={max_iterations}"
                )
                break
            result = _run_single_iteration(iteration, run_config, tracker, local_samples)
            all_results.append(result)
            compact_result = _compact_iteration_result(result)
            compact_all_results.append(compact_result)
            try:
                next_samples, sample_source = extract_initial_samples_from_result(result.get("extract"))
                if next_samples:
                    local_samples = next_samples
                    print(
                        f"[warm-start] prepared {len(next_samples)} samples for next iteration "
                        f"(source: {sample_source})"
                    )
                elif local_samples and isinstance(result.get("extract"), dict):
                    local_samples = None
                    print("[warm-start] no reusable success/stable samples; next iteration falls back to config sampler")
            except Exception as exc:
                print(f"[warm-start] failed to prepare next-iteration samples: {exc}")

            if isinstance(session_state, dict):
                session_state[AGNO_STATE_KEYS["last_iteration"]] = iteration
                session_state[AGNO_STATE_KEYS["last_result"]] = result if verbose_output else compact_result
                session_state[AGNO_STATE_KEYS["all_results"]] = all_results if verbose_output else compact_all_results
                session_state[AGNO_STATE_KEYS["candidate_top20"]] = result.get("top20", []) if verbose_output else compact_result.get("materials", [])
                session_state[AGNO_STATE_KEYS["screened_top10"]] = result.get("top10", []) if verbose_output else compact_result.get("materials", [])
                session_state[AGNO_STATE_KEYS["calculation_results"]] = result.get("calculate", {}) if verbose_output else {}
                session_state[AGNO_STATE_KEYS["extraction_result"]] = result.get("extract", {}) if verbose_output else {}
                theory_result = result.get("theory", {})
                session_state[AGNO_STATE_KEYS["updated_data_path"]] = theory_result.get("updated_data_path")
                session_state[AGNO_STATE_KEYS["updated_doc_path"]] = theory_result.get("updated_doc_path")
                if not result.get("success"):
                    session_state[AGNO_STATE_KEYS["errors"]] = session_state.get(AGNO_STATE_KEYS["errors"], []) + [
                        result.get("failed_step", "unknown")
                    ]

            if not result.get("success"):
                break

        summary = {
            "runs": len(all_results),
            "all_success": all(r.get("success") for r in all_results),
            "last_result": (all_results[-1] if all_results else {}) if verbose_output else (compact_all_results[-1] if compact_all_results else {}),
            "all_results": all_results if verbose_output else compact_all_results,
            "materials_summary": (all_results[-1].get("materials_summary", {}) if all_results else {}),
            "requested_iterations": requested_iterations,
            "resolved_end_iteration": loop_end,
        }
        return StepOutput(content=summary)

    return [
        Step(
            name="aslk_orchestration",
            description="ASLK iterative orchestration with Agno Step executor",
            executor=orchestration_executor,
        )
    ]


__all__ = [
    "build_aslk_steps",
    "run_train_step",
    "run_bayesian_step",
    "run_extract_step",
]
