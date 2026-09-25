# -*- coding: utf-8 -*-
"""Guarded BO+AI screening utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils.workflow_resume import reset_steps_from
from workflow.step_ai_evaluation import step_ai_evaluation


GUARDED_LOCK_COUNT = 5
GUARDED_TARGET_COUNT = 10
GUARDED_SUPPLEMENT_COUNT = GUARDED_TARGET_COUNT - GUARDED_LOCK_COUNT
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def normalize_formula(value: Any) -> str:
    text = str(value or "").strip()
    return "".join(text.split()).translate(SUBSCRIPT_MAP)


def _selected_results_dir(results_root: str | Path, iteration_num: int) -> Path:
    return Path(results_root) / f"iteration_{iteration_num}" / "selected_results"


def _trace_path(results_root: str | Path, iteration_num: int) -> Path:
    return _selected_results_dir(results_root, iteration_num) / "selection_trace.json"


def _summary_path(results_root: str | Path) -> Path:
    return Path(results_root) / "guarded_selection_summary.csv"


def _to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "rank",
                "final_rank",
                "bo_rank",
                "original_rank",
                "formula",
                "ranking_reason",
                "main_risk",
                "k_pred",
                "mu_log",
                "sigma_log",
                "ei",
                "k_lower",
                "k_upper",
                "elements",
                "n_elements",
                "total_atoms",
                "selection_source",
            ]
        )
    return pd.DataFrame(rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    _to_dataframe(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def _copy_material(material: dict[str, Any], *, bo_rank: int, selection_source: str, rank: int | None = None) -> dict[str, Any]:
    row = dict(material)
    row["formula"] = normalize_formula(row.get("formula"))
    row["bo_rank"] = int(bo_rank)
    row["selection_source"] = selection_source
    if rank is not None:
        row["rank"] = int(rank)
    elif "rank" in row:
        try:
            row["rank"] = int(row["rank"])
        except Exception:
            row["rank"] = int(bo_rank)
    row["final_rank"] = int(row["rank"])
    return row


def _merge_rerank_fields(base_row: dict[str, Any], reranked_row: dict[str, Any] | None) -> dict[str, Any]:
    if not reranked_row:
        return base_row
    merged = dict(base_row)
    for key in ("ranking_reason", "main_risk", "original_rank"):
        value = reranked_row.get(key)
        if value is None:
            continue
        merged[key] = value
    return merged


def _build_selection_constraints() -> str:
    return (
        "- The first 5 BO-ranked materials are already locked by the workflow and must not be replaced.\n"
        "- You are only selecting 5 supplemental materials from the provided pool.\n"
        "- You must only choose formulas from the provided candidate list in this prompt.\n"
        "- Return exactly 5 materials when possible.\n"
        "- Do not invent formulas, do not repeat formulas, and do not refer to any locked BO material."
    )


def load_saved_guarded_selection_result(results_root: str | Path, iteration_num: int) -> dict[str, Any] | None:
    base_dir = _selected_results_dir(results_root, iteration_num)
    final_selected_path = base_dir / "final_selected_10.csv"
    locked_path = base_dir / "bo_guarded_top5.csv"
    ai_path = base_dir / "ai_selected_materials.csv"
    trace_path = base_dir / "selection_trace.json"

    if not final_selected_path.exists() or not locked_path.exists() or not ai_path.exists() or not trace_path.exists():
        return None

    try:
        final_selected = pd.read_csv(final_selected_path, encoding="utf-8-sig").to_dict(orient="records")
        locked_materials = pd.read_csv(locked_path, encoding="utf-8-sig").to_dict(orient="records")
        ai_selected = pd.read_csv(ai_path, encoding="utf-8-sig").to_dict(orient="records")
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    return {
        "success": True,
        "skipped": True,
        "selected_materials": final_selected,
        "locked_materials": locked_materials,
        "supplemented_materials": ai_selected,
        "selection_trace": trace,
        "csv_path": str(final_selected_path),
        "trace_path": str(trace_path),
    }


def run_guarded_ai_selection(
    iteration_num: int,
    candidate_materials: list[dict[str, Any]],
    config: dict[str, Any],
    tracker=None,
) -> dict[str, Any]:
    step_key = "ai_evaluation"
    if tracker and tracker.is_step_completed(iteration_num, step_key):
        cached = load_saved_guarded_selection_result(config["results_root"], iteration_num)
        if cached is not None:
            print(f"[resume] loaded saved guarded AI screening artifacts for iteration {iteration_num}")
            return cached
        print(f"[resume] guarded AI screening artifacts missing for iteration {iteration_num}, resetting progress from {step_key}")
        reset_steps_from(tracker, iteration_num, step_key)

    base_dir = _selected_results_dir(config["results_root"], iteration_num)
    base_dir.mkdir(parents=True, exist_ok=True)

    locked_pool = candidate_materials[:GUARDED_LOCK_COUNT]
    supplement_pool = candidate_materials[GUARDED_LOCK_COUNT:int(config.get("top_k_bayes", 20))]

    locked_rows = [
        _copy_material(material, bo_rank=index, selection_source="bo_locked", rank=index)
        for index, material in enumerate(locked_pool, start=1)
    ]
    locked_csv_path = base_dir / "bo_guarded_top5.csv"
    _write_csv(locked_csv_path, locked_rows)

    ai_result = step_ai_evaluation(
        iteration_num=iteration_num,
        candidate_materials=supplement_pool,
        n_select=GUARDED_SUPPLEMENT_COUNT,
        path_config=config.get("path_config"),
        results_root=config["results_root"],
        doc_root=config.get("doc_root", "llm/doc"),
        init_doc_path=config.get("init_doc_path"),
        extra_instructions=_build_selection_constraints(),
    )

    raw_ai_selected = ai_result.get("selected_materials", [])
    supplement_lookup = {
        normalize_formula(material.get("formula")): (idx, material)
        for idx, material in enumerate(supplement_pool, start=GUARDED_LOCK_COUNT + 1)
        if normalize_formula(material.get("formula"))
    }

    supplement_rows: list[dict[str, Any]] = []
    selected_formulas: set[str] = {normalize_formula(row.get("formula")) for row in locked_rows}

    reranked_lookup = {
        normalize_formula(item.get("formula")): item
        for item in raw_ai_selected
        if normalize_formula(item.get("formula"))
    }

    for raw_row in raw_ai_selected:
        normalized = normalize_formula(raw_row.get("formula"))
        if not normalized or normalized in selected_formulas or normalized not in supplement_lookup:
            continue
        bo_rank, material = supplement_lookup[normalized]
        merged_row = _merge_rerank_fields(
            _copy_material(
                material,
                bo_rank=bo_rank,
                selection_source="ai_supplement",
                rank=len(supplement_rows) + 1,
            ),
            reranked_lookup.get(normalized),
        )
        supplement_rows.append(merged_row)
        selected_formulas.add(normalized)
        if len(supplement_rows) >= GUARDED_SUPPLEMENT_COUNT:
            break

    for fallback_rank, fallback_material in enumerate(supplement_pool, start=GUARDED_LOCK_COUNT + 1):
        if len(supplement_rows) >= GUARDED_SUPPLEMENT_COUNT:
            break
        normalized = normalize_formula(fallback_material.get("formula"))
        if not normalized or normalized in selected_formulas:
            continue
        merged_row = _merge_rerank_fields(
            _copy_material(
                fallback_material,
                bo_rank=fallback_rank,
                selection_source="bo_backfill",
                rank=len(supplement_rows) + 1,
            ),
            reranked_lookup.get(normalized),
        )
        supplement_rows.append(merged_row)
        selected_formulas.add(normalized)

    sanitized_ai_csv = base_dir / "ai_selected_materials.csv"
    ai_supplement_csv = base_dir / "ai_supplement_5.csv"
    _write_csv(sanitized_ai_csv, supplement_rows)
    _write_csv(ai_supplement_csv, supplement_rows)

    final_selected_rows = []
    for combined_rank, row in enumerate(locked_rows + supplement_rows, start=1):
        combined_row = dict(row)
        combined_row["rank"] = combined_rank
        combined_row["final_rank"] = combined_rank
        final_selected_rows.append(combined_row)
    final_selected_csv = base_dir / "final_selected_10.csv"
    _write_csv(final_selected_csv, final_selected_rows)

    trace = {
        "iteration": iteration_num,
        "locked_count": len(locked_rows),
        "supplement_target": GUARDED_SUPPLEMENT_COUNT,
        "candidate_count": len(candidate_materials),
        "locked_formulas": [row.get("formula") for row in locked_rows],
        "supplement_pool_formulas": [normalize_formula(item.get("formula")) for item in supplement_pool],
        "raw_ai_selected_formulas": [normalize_formula(item.get("formula")) for item in raw_ai_selected],
        "final_selected": [
            {
                "formula": row.get("formula"),
                "bo_rank": int(row.get("bo_rank", 0) or 0),
                "selection_source": row.get("selection_source"),
            }
            for row in final_selected_rows
        ],
        "paths": {
            "bo_guarded_top5": str(locked_csv_path),
            "ai_selected": str(sanitized_ai_csv),
            "ai_supplement_5": str(ai_supplement_csv),
            "final_selected_10": str(final_selected_csv),
        },
    }
    trace_path = _trace_path(config["results_root"], iteration_num)
    trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")

    result = {
        "success": len(final_selected_rows) > 0,
        "selected_materials": final_selected_rows,
        "locked_materials": locked_rows,
        "supplemented_materials": supplement_rows,
        "selection_trace": trace,
        "csv_path": str(final_selected_csv),
        "trace_path": str(trace_path),
        "n_selected": len(final_selected_rows),
    }
    if tracker:
        tracker.mark_step_completed(
            iteration_num,
            step_key,
            metadata={
                "selected_materials": len(final_selected_rows),
                "locked_materials": len(locked_rows),
                "supplemented_materials": len(supplement_rows),
                "report_file": ai_result.get("report_path"),
                "selection_trace": str(trace_path),
            },
        )
    return result


def _load_formula_set(csv_path: str | Path | None) -> set[str]:
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
    for column in ("formula", "composition", "Formula"):
        if column not in df.columns:
            continue
        formulas.update(normalize_formula(value) for value in df[column].tolist() if normalize_formula(value))
    return formulas


def update_guarded_selection_summary(
    iteration_num: int,
    results_root: str | Path,
    extraction_result: dict[str, Any],
) -> str | None:
    loaded = load_saved_guarded_selection_result(results_root, iteration_num)
    if loaded is None:
        return None

    success_formulas = _load_formula_set(
        extraction_result.get("success_deduped_file") or extraction_result.get("success_file")
    )
    stable_formulas = _load_formula_set(
        extraction_result.get("stable_deduped_file") or extraction_result.get("stable_file")
    )
    selected_rows = loaded["selected_materials"]

    def count_by(source: str, formula_set: set[str]) -> int:
        return sum(
            1
            for row in selected_rows
            if row.get("selection_source") == source and normalize_formula(row.get("formula")) in formula_set
        )

    summary_row = {
        "iteration": iteration_num,
        "bo_locked_count": sum(1 for row in selected_rows if row.get("selection_source") == "bo_locked"),
        "ai_supplement_count": sum(1 for row in selected_rows if row.get("selection_source") == "ai_supplement"),
        "bo_backfill_count": sum(1 for row in selected_rows if row.get("selection_source") == "bo_backfill"),
        "bo_locked_success_count": count_by("bo_locked", success_formulas),
        "ai_supplement_success_count": count_by("ai_supplement", success_formulas),
        "bo_backfill_success_count": count_by("bo_backfill", success_formulas),
        "bo_locked_stable_count": count_by("bo_locked", stable_formulas),
        "ai_supplement_stable_count": count_by("ai_supplement", stable_formulas),
        "bo_backfill_stable_count": count_by("bo_backfill", stable_formulas),
        "final_success_count": len(success_formulas),
        "final_stable_count": len(stable_formulas),
    }

    summary_path = _summary_path(results_root)
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path, encoding="utf-8-sig")
        summary_df = summary_df[summary_df["iteration"] != iteration_num]
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
        summary_df = summary_df.sort_values("iteration")
    else:
        summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return str(summary_path)
