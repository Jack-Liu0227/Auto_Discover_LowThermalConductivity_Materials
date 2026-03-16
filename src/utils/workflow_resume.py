from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


THEORY_DOC_NAME = "Theoretical_principle_document.md"


def load_saved_bayesian_result(results_root: str | Path, iteration_num: int, n_top: int) -> dict[str, Any] | None:
    base_dir = Path(results_root) / f"iteration_{iteration_num}" / "selected_results"
    all_samples_path = base_dir / "all_samples.csv"
    if not all_samples_path.exists():
        return None

    try:
        df = pd.read_csv(all_samples_path, encoding="utf-8-sig")
    except Exception:
        return None

    rows = df.to_dict(orient="records")
    return {
        "success": True,
        "skipped": True,
        "n_materials": len(rows),
        "n_top": n_top,
        "top_materials": rows[:n_top],
        "top10_materials": rows[:10],
        "all_materials": rows,
        "artifact_path": str(all_samples_path),
    }


def load_saved_extract_result(results_root: str | Path, iteration_num: int) -> dict[str, Any] | None:
    success_dir = Path(results_root) / f"iteration_{iteration_num}" / "success_examples"
    if not success_dir.exists():
        return None

    success_file = success_dir / "success_materials.csv"
    stable_file = success_dir / "stable_materials.csv"
    success_deduped_file = success_dir / "success_materials_deduped.csv"
    stable_deduped_file = success_dir / "stable_materials_deduped.csv"
    novelty_csv = success_dir / "final_materials_db_novelty.csv"
    novelty_json = success_dir / "final_materials_db_novelty.json"
    novelty_summary = success_dir / "final_materials_db_novelty_summary.md"

    has_success = success_file.exists() or success_deduped_file.exists()
    has_stable = stable_file.exists() or stable_deduped_file.exists()
    if not has_success and not has_stable:
        return None

    return {
        "success": True,
        "skipped": True,
        "has_success": has_success,
        "has_stable": has_stable,
        "success_file": str(success_file) if success_file.exists() else None,
        "stable_file": str(stable_file) if stable_file.exists() else None,
        "success_deduped_file": str(success_deduped_file) if success_deduped_file.exists() else None,
        "stable_deduped_file": str(stable_deduped_file) if stable_deduped_file.exists() else None,
        "final_db_novelty_file": str(novelty_csv) if novelty_csv.exists() else None,
        "final_db_novelty_json": str(novelty_json) if novelty_json.exists() else None,
        "final_db_novelty_summary_file": str(novelty_summary) if novelty_summary.exists() else None,
        "novelty_summary": {},
    }


def load_saved_ai_evaluation_result(results_root: str | Path, iteration_num: int) -> dict[str, Any] | None:
    selected_path = Path(results_root) / f"iteration_{iteration_num}" / "selected_results" / "ai_selected_materials.csv"
    if not selected_path.exists():
        return None

    try:
        df = pd.read_csv(selected_path, encoding="utf-8-sig")
    except Exception:
        return None

    return {
        "success": True,
        "skipped": True,
        "n_selected": len(df),
        "selected_materials": df.to_dict(orient="records"),
        "csv_path": str(selected_path),
        "report_path": str(Path(results_root) / f"iteration_{iteration_num}" / "reports" / "llm_evaluation_output.md"),
    }


def load_saved_document_update_result(
    results_root: str | Path,
    data_root: str | Path,
    doc_root: str | Path,
    iteration_num: int,
) -> dict[str, Any] | None:
    data_path = Path(data_root) / f"iteration_{iteration_num}" / "data.csv"
    doc_path = Path(doc_root) / f"v0.0.{iteration_num}" / THEORY_DOC_NAME
    if not data_path.exists() or not doc_path.exists():
        return None

    return {
        "success": True,
        "skipped": True,
        "has_success_materials": True,
        "updated_data_path": str(data_path),
        "updated_doc_path": str(doc_path),
    }


def reset_steps_from(tracker, iteration_num: int, start_step: str) -> list[str]:
    reset_steps: list[str] = []
    should_reset = False
    for step in tracker.steps:
        if step == start_step:
            should_reset = True
        if not should_reset:
            continue
        if tracker.progress.get(f"iteration_{iteration_num}", {}).get(step) is None:
            continue
        tracker.reset_step(iteration_num, step)
        reset_steps.append(step)
    return reset_steps


def reconcile_progress_with_filesystem(tracker, path_config) -> list[str]:
    messages: list[str] = []
    results_root = path_config.results_root
    models_root = path_config.models_root
    doc_root = path_config.doc_root

    for key in sorted(tracker.progress.keys()):
        if not key.startswith("iteration_"):
            continue
        try:
            iteration_num = int(key.split("_", 1)[1])
        except Exception:
            continue

        # Step 1 trains the model for iteration_{iteration_num - 1} using the
        # previous round's dataset. Resume validation must follow the same
        # artifact convention or it will incorrectly reset completed rounds.
        model_iteration = max(iteration_num - 1, 0)
        model_candidates = [
            models_root / f"iteration_{model_iteration}" / "gpr_thermal_conductivity.joblib",
            # Backward compatibility for older runs/tests that stored the model
            # under the same iteration number instead of iteration-1.
            models_root / f"iteration_{iteration_num}" / "gpr_thermal_conductivity.joblib",
        ]
        all_samples_path = results_root / f"iteration_{iteration_num}" / "selected_results" / "all_samples.csv"
        ai_selected_path = results_root / f"iteration_{iteration_num}" / "selected_results" / "ai_selected_materials.csv"
        processed_dir = results_root / f"iteration_{iteration_num}" / "processed_structures"
        relax_dir = results_root / f"iteration_{iteration_num}" / "MyRelaxStructure"
        success_dir = results_root / f"iteration_{iteration_num}" / "success_examples"
        versioned_doc = doc_root / f"v0.0.{iteration_num}" / THEORY_DOC_NAME if doc_root else None

        round_data = tracker.progress.get(key, {})
        if round_data.get("train_model", {}).get("completed") and not any(path.exists() for path in model_candidates):
            reset_steps = reset_steps_from(tracker, iteration_num, "train_model")
            messages.append(f"iteration_{iteration_num}: reset {reset_steps} because model artifact is missing")
            continue

        if round_data.get("bayesian_optimization", {}).get("completed") and not all_samples_path.exists():
            reset_steps = reset_steps_from(tracker, iteration_num, "bayesian_optimization")
            messages.append(f"iteration_{iteration_num}: reset {reset_steps} because BO artifacts are missing")
            continue

        if round_data.get("ai_evaluation", {}).get("completed") and not ai_selected_path.exists():
            reset_steps = reset_steps_from(tracker, iteration_num, "ai_evaluation")
            messages.append(f"iteration_{iteration_num}: reset {reset_steps} because AI screening artifacts are missing")
            continue

        if round_data.get("structure_calculation") and not processed_dir.exists() and not relax_dir.exists():
            reset_steps = reset_steps_from(tracker, iteration_num, "structure_calculation")
            messages.append(f"iteration_{iteration_num}: reset {reset_steps} because structure artifacts are missing")
            continue

        if round_data.get("success_extraction", {}).get("completed") and not success_dir.exists():
            reset_steps = reset_steps_from(tracker, iteration_num, "success_extraction")
            messages.append(f"iteration_{iteration_num}: reset {reset_steps} because extraction artifacts are missing")
            continue

        if versioned_doc is not None and round_data.get("document_update", {}).get("completed") and not versioned_doc.exists():
            reset_steps = reset_steps_from(tracker, iteration_num, "document_update")
            messages.append(f"iteration_{iteration_num}: reset {reset_steps} because versioned theory doc is missing")

        missing_step: str | None = None
        seen_completed_after_gap = False
        for step in tracker.steps:
            if round_data.get(step) is None:
                if missing_step is None:
                    missing_step = step
                continue
            if missing_step is not None:
                seen_completed_after_gap = True
                break
        if missing_step is not None and seen_completed_after_gap:
            reset_steps = reset_steps_from(tracker, iteration_num, missing_step)
            messages.append(f"iteration_{iteration_num}: reset {reset_steps} because downstream steps existed after missing {missing_step}")

    return messages
