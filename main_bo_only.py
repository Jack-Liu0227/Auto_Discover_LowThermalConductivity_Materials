# -*- coding: utf-8 -*-
"""
Bayesian Optimization material discovery pipeline (BO-only).

This entrypoint owns the standalone BO baseline that used to be exposed as the
`bo_direct` screening mode in `main.py`:
- train the model
- run BO
- take the BO top-k directly for structure calculation
- extract success/stable materials
- update the dataset for the next iteration
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Force UTF-8 console IO on Windows to avoid mojibake.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
os.environ["PYTHONIOENCODING"] = "utf-8"
# PyTorch on Windows does not support the expandable_segments allocator
# option. Leaving it enabled makes CrystaLLM child processes fail before
# relaxation starts. Keep the allocator tuning only on non-Windows hosts.
if os.name == "nt":
    if "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").lower():
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
else:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

project_root = Path(__file__).resolve().parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from workflow.step_train_model import step_train_model
    from workflow.step_bayesian_optimization import step_bayesian_optimization
    from workflow.step_structure_calculation import step_structure_calculation
    from workflow.step_merge_results import step_merge_results
    from workflow.step_extract_materials import step_extract_materials
    from utils.update_dataset import update_dataset
    from utils.progress_tracker import ProgressTracker
    from utils.path_config import PathConfig
    from utils.bo_runtime import extract_initial_samples_from_result, load_bo_runtime_defaults
    from utils.experiment_reset import archive_active_run, rebuild_active_chain
    from utils.gpu_parallel import normalize_gpu_list, validate_gpu_devices
    from utils.reproducibility import setup_reproducibility
    from utils.workflow_resume import (
        WorkflowProgressInconsistencyError,
        file_sha256,
        get_contiguous_completed_rounds,
        load_saved_extract_result,
        reconcile_progress_with_filesystem,
        reset_steps_from,
        is_valid_structure_artifacts,
        _is_valid_merge_artifacts,
    )
except Exception as exc:
    print(f"[FATAL] Failed to import workflow steps: {exc}")
    sys.exit(1)


RUN_MODE = "bo"
RESULTS_ROOT = f"{RUN_MODE}/results"
MODELS_ROOT = f"{RUN_MODE}/models/GPR"
DATA_ROOT = f"{RUN_MODE}/data"

DEFAULT_CONFIG = {
    "samples": 100,
    "xi": 0.01,
    "top_k_bayes": 20,
    "n_structures": 5,
    "max_workers": 4,
    "relax_workers": 1,
    "phonon_workers": 1,
    "postprocess_workers": 2,
    "novelty_workers": 4,
    "pressure": 0.0,
    "device": "cuda",
    "gpus": ["cuda:0"],
    "k_threshold": 1.0,
    "top_k_screen": 10,
    "seed": 42,
    "seed_stride": 1000,
    "deterministic_torch": True,
    "allow_partial_structure": False,
    "relax_timeout_sec": 900,
    "prefer_isolated_relax_process": True,
    "allow_in_process_relax_fallback": True,
    "screening_mode": "bo_direct",
}

BO_STEPS = [
    "train_model",
    "bayesian_optimization",
    "structure_calculation",
    "merge_results",
    "success_extraction",
    "data_update",
]


def _atomic_write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp_file, path)
    finally:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass
    return str(path)


def _atomic_write_csv(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        pd.DataFrame(rows).to_csv(temp_file, index=False, encoding="utf-8-sig")
        os.replace(temp_file, path)
    finally:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass
    return str(path)


def _candidate_formula(material: dict) -> str:
    return str(
        material.get("formula")
        or material.get("composition")
        or material.get("name")
        or ""
    ).strip().replace(" ", "")


def _load_all_bo_candidates(results_base: Path, iteration_num: int, fallback: list[dict]) -> list[dict]:
    all_samples_path = results_base / f"iteration_{iteration_num}" / "selected_results" / "all_samples.csv"
    if not all_samples_path.exists():
        return fallback
    try:
        frame = pd.read_csv(all_samples_path, encoding="utf-8-sig")
        rows = frame.to_dict(orient="records")
        return rows if rows else fallback
    except (OSError, ValueError, pd.errors.ParserError):
        return fallback


def _write_bo_selection_outputs(
    *,
    results_base: Path,
    iteration_num: int,
    all_candidates: list[dict],
    bo_candidates: list[dict],
    n_select: int,
) -> dict[str, str]:
    """Persist BO-only selection artifacts using the same trace layout as LLM runs."""
    selected_formulas = {
        _candidate_formula(material)
        for material in bo_candidates[:n_select]
        if _candidate_formula(material)
    }
    bo_formulae = {
        _candidate_formula(material)
        for material in bo_candidates
        if _candidate_formula(material)
    }
    trace_rows: list[dict] = []
    selected_rows: list[dict] = []
    final_rank = 1

    for index, material in enumerate(all_candidates, start=1):
        row = dict(material)
        formula = _candidate_formula(row)
        rank_value = row.get("rank")
        try:
            original_rank = int(rank_value) if rank_value is not None and not pd.isna(rank_value) else index
        except (TypeError, ValueError):
            original_rank = index
        is_bo_candidate = formula in bo_formulae
        selected = formula in selected_formulas
        if selected:
            selection_reason = "bo_rank_cutoff"
            selected_rank = final_rank
            final_rank += 1
        elif is_bo_candidate:
            selection_reason = "outside_structure_top_k"
            selected_rank = ""
        else:
            selection_reason = "outside_bo_top_k"
            selected_rank = ""

        trace_row = {
            "formula": formula,
            "original_bo_rank": original_rank,
            "k_pred": row.get("k_pred"),
            "ei": row.get("ei"),
            "sigma_log": row.get("sigma_log"),
            "screening_mode": "bo_direct",
            "selected_for_calc": selected,
            "final_rank": selected_rank,
            "selection_reason": selection_reason,
        }
        trace_rows.append(trace_row)
        if selected:
            selected_rows.append({**row, **trace_row})

    base_dir = results_base / f"iteration_{iteration_num}" / "selected_results"
    selected_csv = base_dir / "bo_selected_materials.csv"
    trace_csv = base_dir / "selection_trace.csv"
    trace_json = base_dir / "selection_trace.json"
    _atomic_write_csv(selected_csv, selected_rows)
    _atomic_write_csv(trace_csv, trace_rows)
    _atomic_write_json(
        trace_json,
        {
            "iteration": iteration_num,
            "screening_mode": "bo_direct",
            "candidate_count": len(all_candidates),
            "bo_candidate_count": len(bo_candidates),
            "selected_count": len(selected_rows),
            "selected_formulas": [row["formula"] for row in selected_rows],
            "rows": trace_rows,
        },
    )
    return {
        "selected_csv": str(selected_csv),
        "trace_csv": str(trace_csv),
        "trace_json": str(trace_json),
    }


def _update_iteration_summary_csv(target_file: Path, source_path: str, iteration_num: int) -> int:
    df_new = pd.read_csv(source_path, encoding="utf-8-sig").copy()
    if "iteration" in df_new.columns:
        df_new["iteration"] = iteration_num
    else:
        df_new.insert(0, "iteration", iteration_num)

    if target_file.exists():
        df_existing = pd.read_csv(target_file, encoding="utf-8-sig")
        if "iteration" in df_existing.columns:
            df_existing = df_existing[df_existing["iteration"] != iteration_num]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    sort_cols = [
        col
        for col in [
            "iteration",
            "formula",
            "composition",
            "thermal_conductivity_w_mk",
            "structure_id",
            "cif_file",
        ]
        if col in df_combined.columns
    ]
    if sort_cols:
        df_combined = df_combined.sort_values(by=sort_cols, kind="mergesort", na_position="last")

    temp_file = target_file.with_name(f".{target_file.name}.{os.getpid()}.tmp")
    try:
        df_combined.to_csv(temp_file, index=False, encoding="utf-8-sig")
        os.replace(temp_file, target_file)
    finally:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass
    return len(df_combined)


def prepare_initial_data(init_data_path: str | None = None):
    """Prepare `iteration_0/data.csv` for the BO workflow."""
    bo_data_root = project_root / DATA_ROOT
    bo_iter0 = bo_data_root / "iteration_0" / "data.csv"

    if bo_iter0.exists():
        print(f"[INFO] Initial data already present: {bo_iter0}")
        return

    print(f"[INFO] Initial data not found: {bo_iter0}")
    sources = []

    if init_data_path:
        custom_data = Path(init_data_path)
        if not custom_data.is_absolute():
            custom_data = project_root / custom_data
        sources.append(custom_data)
        print(f"[INFO] Using custom initial data: {custom_data}")

    sources.extend(
        [
            project_root / "llm" / "data" / "iteration_0" / "data.csv",
            project_root / "data" / "processed_data.csv",
            project_root / "data" / "iteration_0" / "data.csv",
        ]
    )

    for src in sources:
        if not src.exists():
            continue
        print(f"[INFO] Found source data: {src}")
        bo_iter0.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, bo_iter0)
        print(f"[INFO] Copied to: {bo_iter0}")
        return

    print("[WARN] No valid initial data source found; the first iteration may fail.")


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Bayesian Optimization materials discovery (BO-only).")
    parser.add_argument("--max-iterations", type=int, default=20, help="maximum iterations (default: 20)")
    parser.add_argument("--samples", type=int, default=None, help="number of BO samples per iteration")
    parser.add_argument(
        "--top-k-bayes",
        type=int,
        default=None,
        help="number of top BO candidates kept for downstream calculation",
    )
    parser.add_argument("--n-top-candidates", dest="top_k_bayes", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--n-structures",
        type=int,
        default=None,
        help="number of generated structures per composition",
    )
    parser.add_argument(
        "--top-k-screen",
        type=int,
        default=None,
        help="number of materials selected for structure calculation",
    )
    parser.add_argument("--n-select", dest="top_k_screen", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="number of GPUs to use (for example 3 -> cuda:0,cuda:1,cuda:2)",
    )
    parser.add_argument(
        "--postprocess-workers",
        type=int,
        default=None,
        help="bounded CPU workers for structure deduplication, extraction, and merge",
    )
    parser.add_argument(
        "--novelty-workers",
        type=int,
        default=None,
        help="bounded workers for final database novelty queries",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="device for MatterSim structure calculations (default: cuda)",
    )
    parser.add_argument(
        "--relax-timeout-sec",
        type=int,
        default=None,
        help="timeout in seconds for one relax+phonon task",
    )
    parser.add_argument(
        "--phonon-imag-tol",
        type=float,
        default=None,
        help="minimum allowed phonon frequency (THz)",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument(
        "--non-deterministic-torch",
        action="store_true",
        help="disable deterministic torch kernels",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=None,
        help="explicit start iteration; default resumes at the first incomplete iteration",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="archive the active BO run and start a fresh isolated experiment",
    )
    parser.add_argument(
        "--rebuild-from",
        type=int,
        default=None,
        help="archive active artifacts from this iteration onward and resume the chain there",
    )
    parser.add_argument(
        "--allow-partial-structure",
        action="store_true",
        help="allow workflow to continue when some structure tasks fail",
    )
    parser.add_argument(
        "--init-data",
        type=str,
        default="data/processed_data.csv",
        help="path to initial dataset (default: data/processed_data.csv)",
    )
    args = parser.parse_args(argv)
    if args.max_iterations <= 0:
        parser.error("--max-iterations must be > 0")
    if args.start_iteration is not None and args.start_iteration <= 0:
        parser.error("--start-iteration must be > 0")
    if args.top_k_bayes is not None and args.top_k_bayes <= 0:
        parser.error("--top-k-bayes must be > 0")
    if args.n_structures is not None and args.n_structures <= 0:
        parser.error("--n-structures must be > 0")
    if args.top_k_screen is not None and args.top_k_screen <= 0:
        parser.error("--top-k-screen must be > 0")
    if args.num_gpus is not None and args.num_gpus <= 0:
        parser.error("--num-gpus must be > 0")
    if args.postprocess_workers is not None and args.postprocess_workers <= 0:
        parser.error("--postprocess-workers must be > 0")
    if args.novelty_workers is not None and args.novelty_workers <= 0:
        parser.error("--novelty-workers must be > 0")
    if args.relax_timeout_sec is not None and args.relax_timeout_sec <= 0:
        parser.error("--relax-timeout-sec must be > 0")
    if args.samples is not None and args.samples <= 0:
        parser.error("--samples must be > 0")
    if args.start_iteration is not None and args.start_iteration > args.max_iterations:
        parser.error("--start-iteration cannot be greater than --max-iterations")
    if args.rebuild_from is not None and args.rebuild_from <= 0:
        parser.error("--rebuild-from must be > 0")
    if args.reset and args.rebuild_from is not None:
        parser.error("Use either --reset or --rebuild-from, not both")
    return args


def run_single_iteration(iteration_num: int, config: dict, tracker: ProgressTracker, initial_samples=None):
    """Run one BO-only iteration."""
    print("\n" + "=" * 80)
    print(f">> Start Iteration {iteration_num} (Mode: BO-only / bo_direct)")
    print("=" * 80)
    sys.stdout.flush()

    results = {}
    path_config = config.get("path_config")
    results_base = path_config.results_root if path_config is not None else project_root / RESULTS_ROOT
    models_base = path_config.models_root if path_config is not None else project_root / MODELS_ROOT
    data_base = path_config.data_root if path_config is not None else project_root / DATA_ROOT

    step_key = "train_model"
    if tracker.is_step_completed(iteration_num, step_key):
        print(f"[SKIP] Step 1 ({step_key}) already completed")
        results["train"] = {"success": True}
    else:
        tracker.mark_step_started(iteration_num, step_key)
        train_result = step_train_model(
            iteration_num=iteration_num,
            data_root=str(data_base),
            models_root=str(models_base),
            path_config=config.get("path_config"),
        )
        results["train"] = train_result
        if train_result["success"]:
            tracker.mark_step_completed(
                iteration_num,
                step_key,
                metadata={
                    "training_data": train_result.get("training_data"),
                    "training_data_sha256": train_result.get("training_data_sha256"),
                    "model_file": train_result.get("model_file"),
                },
            )
        else:
            print(f"[ERROR] Step 1 failed: {train_result.get('error')}")
            return results

    step_key = "bayesian_optimization"
    candidate_materials = []
    all_candidate_materials = []
    save_dir = results_base / f"iteration_{iteration_num}" / "selected_results"
    save_file = save_dir / "bo_candidates.json"

    if tracker.is_step_completed(iteration_num, step_key):
        print(f"[SKIP] Step 2 ({step_key}) already completed")
        if save_file.exists():
            try:
                with open(save_file, "r", encoding="utf-8") as f:
                    candidate_materials = json.load(f)
                if not isinstance(candidate_materials, list):
                    raise ValueError("candidate file must contain a JSON list")
                print(f"[INFO] Loaded {len(candidate_materials)} BO candidates from file: {save_file}")
                all_candidate_materials = _load_all_bo_candidates(
                    results_base,
                    iteration_num,
                    candidate_materials,
                )
                results["bayes"] = {
                    "success": True,
                    "top_materials": candidate_materials,
                    "all_materials": all_candidate_materials,
                }
            except Exception as exc:
                print(f"[WARN] Failed to load BO candidates: {exc}; resetting BO step")
                reset_steps_from(tracker, iteration_num, step_key)
        else:
            print(f"[WARN] Candidate file not found: {save_file}; resetting BO step")
            reset_steps_from(tracker, iteration_num, step_key)

    if not tracker.is_step_completed(iteration_num, step_key):
        print("")
        print("#" * 80)
        print("Step 2/5: Bayesian Optimization")
        print("#" * 80)
        tracker.mark_step_started(iteration_num, step_key)
        bayes_result = step_bayesian_optimization(
            iteration_num=iteration_num,
            xi=config["xi"],
            n_samples=config["samples"],
            n_top=config["top_k_bayes"],
            initial_samples=initial_samples,
            seed=config.get("seed"),
            seed_stride=config.get("seed_stride", 1000),
            models_root=str(models_base),
            results_root=str(results_base),
            sampling_params=config.get("sampling_params"),
            path_config=config.get("path_config"),
        )
        results["bayes"] = bayes_result

        if not bayes_result["success"]:
            print(f"[ERROR] Step 2 failed: {bayes_result.get('error')}")
            return results

        candidate_materials = bayes_result.get("top_materials", [])
        all_candidate_materials = bayes_result.get("all_materials") or candidate_materials
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(candidate_materials, f, indent=2, ensure_ascii=False)
        tracker.mark_step_completed(iteration_num, step_key)
        print(f"[OK] Saved candidates: {save_file}")

    top_k_screen = config.get("top_k_screen", 10)
    all_candidate_materials = all_candidate_materials or candidate_materials
    if candidate_materials:
        trace_paths = _write_bo_selection_outputs(
            results_base=results_base,
            iteration_num=iteration_num,
            all_candidates=all_candidate_materials,
            bo_candidates=candidate_materials,
            n_select=top_k_screen,
        )
        results["bayes"] = {
            **(results.get("bayes") or {}),
            "selection_trace_csv": trace_paths["trace_csv"],
            "selection_trace_json": trace_paths["trace_json"],
            "selected_csv": trace_paths["selected_csv"],
        }
    selected_materials = candidate_materials[:top_k_screen] if candidate_materials else []
    if candidate_materials:
        print(
            f"[SELECT] bo_direct selected top {len(selected_materials)} "
            f"from {len(candidate_materials)} BO candidates"
        )

    step_key = "structure_calculation"
    if tracker.is_step_completed(iteration_num, step_key):
        expected_formulas = [
            str(material.get("formula") or "").strip()
            for material in selected_materials
            if str(material.get("formula") or "").strip()
        ]
        if expected_formulas and is_valid_structure_artifacts(
            results_base,
            iteration_num,
            expected_formulas=expected_formulas,
        ):
            print(f"[SKIP] Step 3 ({step_key}) already completed with valid artifacts")
            results["structure"] = {"success": True, "completed": True, "skipped": True}
        else:
            print(f"[WARN] Step 3 ({step_key}) marker has invalid artifacts; resetting from structure")
            reset_steps_from(tracker, iteration_num, step_key)

    if not tracker.is_step_completed(iteration_num, step_key):
        print(f"\n{'#' * 80}")
        print("Step 3/5: Structure generation and calculation")
        print(f"{'#' * 80}")

        if not selected_materials:
            print("[ERROR] No candidate materials available for structure generation.")
            return results

        tracker.mark_step_started(iteration_num, step_key)
        structure_result = step_structure_calculation(
            iteration_num=iteration_num,
            materials=selected_materials,
            n_structures=config["n_structures"],
            max_workers=config["max_workers"],
            relax_workers=config["relax_workers"],
            phonon_workers=config["phonon_workers"],
            postprocess_workers=max(1, int(config.get("postprocess_workers", 1))),
            pressure=config["pressure"],
            device=config["device"],
            gpus=config.get("gpus", ["cuda:0"]),
            allow_partial_completion=config.get("allow_partial_structure", False),
            results_root=str(results_base),
            seed=config.get("seed"),
            tracker=tracker,
            path_config=config.get("path_config"),
            relax_timeout_sec=config.get("relax_timeout_sec", 900),
            prefer_isolated_relax_process=config.get("prefer_isolated_relax_process", True),
            allow_in_process_relax_fallback=config.get("allow_in_process_relax_fallback", True),
        )
        results["structure"] = structure_result

        if structure_result.get("completed"):
            expected_formulas = [
                str(material.get("formula") or "").strip()
                for material in selected_materials
                if str(material.get("formula") or "").strip()
            ]
            if expected_formulas and is_valid_structure_artifacts(
                results_base,
                iteration_num,
                expected_formulas=expected_formulas,
            ):
                tracker.mark_step_completed(
                    iteration_num,
                    step_key,
                    metadata={
                        "gen_output_dir": structure_result.get("gen_output_dir"),
                        "relax_output_dir": structure_result.get("relax_output_dir"),
                    },
                )
            else:
                structure_result["success"] = False
                structure_result["completed"] = False
                structure_result["error"] = (
                    "Structure completion contract failed after calculation; "
                    "final artifacts are incomplete"
                )

        if not tracker.is_step_completed(iteration_num, step_key):
            print("[INFO] Structure calculation not completed; keep progress and continue later.")
            return results

    step_key = "merge_results"
    if tracker.is_step_completed(iteration_num, step_key):
        if _is_valid_merge_artifacts(results_base, iteration_num):
            print("[SKIP] Step merge_results completed with valid artifacts")
            results["merge"] = {"success": True, "skipped": True}
        else:
            print("[WARN] Merge marker has invalid artifacts; resetting from merge_results")
            reset_steps_from(tracker, iteration_num, step_key)

    if not tracker.is_step_completed(iteration_num, step_key):
        tracker.mark_step_started(iteration_num, step_key)
        merge_result = step_merge_results(
            iteration_num=iteration_num,
            results_root=str(results_base),
            tracker=tracker,
            path_config=config.get("path_config"),
            max_workers=max(1, int(config.get("postprocess_workers", 1))),
        )
        results["merge"] = merge_result
        if not merge_result.get("success"):
            print(f"[ERROR] merge_results failed: {merge_result.get('error')}")
            return results

    step_key = "success_extraction"
    if tracker.is_step_completed(iteration_num, step_key):
        print(f"[SKIP] Step 4 ({step_key}) already completed")
        extract_result = load_saved_extract_result(results_base, iteration_num)
        if extract_result is None:
            round_data = tracker.progress.get(f"iteration_{iteration_num}", {})
            step_data = round_data.get(step_key, {}) if isinstance(round_data, dict) else {}
            metadata = step_data.get("metadata", {}) if isinstance(step_data, dict) else {}
            if isinstance(metadata, dict) and metadata.get("no_materials") is True:
                print(f"[resume] loaded no-material extraction state for iteration {iteration_num}")
                results["extract"] = {
                    "success": True,
                    "no_materials": True,
                    "has_success": False,
                    "has_stable": False,
                }
            else:
                print("[WARN] Cached extraction artifacts are missing; resetting extraction and downstream steps")
                reset_steps_from(tracker, iteration_num, step_key)
        else:
            results["extract"] = extract_result
    if not tracker.is_step_completed(iteration_num, step_key):
        print(f"\n{'#' * 80}")
        print("Step 4/5: Extract success and stable materials")
        print(f"{'#' * 80}")

        tracker.mark_step_started(iteration_num, step_key)
        extract_result = step_extract_materials(
            iteration_num=iteration_num,
            k_threshold=config["k_threshold"],
            imag_tol=config.get("phonon_imag_tol"),
            results_root=str(results_base),
            path_config=config.get("path_config"),
            postprocess_workers=max(1, int(config.get("postprocess_workers", 1))),
            novelty_workers=max(1, int(config.get("novelty_workers", 1))),
        )
        results["extract"] = extract_result

        if not (extract_result.get("success") or extract_result.get("no_materials")):
            print(f"[ERROR] Step 4 failed: {extract_result.get('error')}")
            return results

        tracker.mark_step_completed(
            iteration_num,
            step_key,
            metadata={
                "no_materials": bool(extract_result.get("no_materials")),
                "has_success": bool(extract_result.get("has_success")),
                "has_stable": bool(extract_result.get("has_stable")),
            },
        )

        try:
            aggregated_results_dir = results_base
            aggregated_results_dir.mkdir(exist_ok=True, parents=True)
            summary_files = {
                "success": aggregated_results_dir / "success_materials.csv",
                "stable": aggregated_results_dir / "stable_materials.csv",
            }

            source_files = {}
            if extract_result.get("success_deduped_file"):
                source_files["success"] = extract_result["success_deduped_file"]
            elif extract_result.get("success_file"):
                source_files["success"] = extract_result["success_file"]

            if extract_result.get("stable_deduped_file"):
                source_files["stable"] = extract_result["stable_deduped_file"]
            elif extract_result.get("stable_file"):
                source_files["stable"] = extract_result["stable_file"]

            for key, source_path in source_files.items():
                if source_path and os.path.exists(source_path):
                    target_file = summary_files[key]
                    total_rows = _update_iteration_summary_csv(target_file, source_path, iteration_num)
                    print(f"  [LOG] Synced summary: {RESULTS_ROOT}/{target_file.name} (total: {total_rows})")
        except Exception as exc:
            print(f"[WARN] Failed to update summary files: {exc}")

        if extract_result.get("no_materials"):
            print("[INFO] No materials met success/stability criteria in this iteration.")
            extract_result["success"] = True
            extract_result["has_success"] = False
            extract_result["has_stable"] = False

    step_key = "data_update"
    if tracker.is_step_completed(iteration_num, step_key):
        print(f"[SKIP] Step 5 ({step_key}) already completed")
        results["update"] = {"success": True}
    else:
        print(f"\n{'#' * 80}")
        print("Step 5/5: Update dataset (CSV only)")
        print(f"{'#' * 80}")

        has_success = results["extract"].get("has_success", False)
        has_stable = results["extract"].get("has_stable", False)
        success_csv = results["extract"].get("success_deduped_file") or results["extract"].get("success_file")
        stable_csv = results["extract"].get("stable_deduped_file") or results["extract"].get("stable_file")

        target_csv = None
        if has_success and success_csv:
            target_csv = success_csv
            print(f"Prepare to merge success materials: {target_csv}")
        elif has_stable and stable_csv:
            target_csv = stable_csv
            print(f"Prepare to merge stable materials: {target_csv}")

        prev_iteration = iteration_num - 1
        required_origin = data_base / f"iteration_{prev_iteration}" / "data.csv"
        origin_csv = required_origin if required_origin.exists() else None
        if origin_csv is None:
            print(f"[ERROR] Immediate previous dataset not found: {required_origin}")

        output_dir = data_base / f"iteration_{iteration_num}"
        updated_path = None

        tracker.mark_step_started(iteration_num, step_key)
        if target_csv and origin_csv:
            print(f"Merge source: {target_csv}")
            print(f"Base dataset: {origin_csv}")
            try:
                updated_path = update_dataset(
                    success_csv=str(target_csv),
                    origin_csv=str(origin_csv),
                    output_dir=str(output_dir),
                )
            except Exception as exc:
                print(f"[ERROR] Failed to update dataset: {exc}")
        elif not origin_csv:
            print(f"[ERROR] Previous dataset not found (searched iteration 0..{prev_iteration}).")

        if not updated_path:
            print("No new materials or update failed; copy previous dataset for continuity.")
            if origin_csv:
                output_dir.mkdir(parents=True, exist_ok=True)
                dest = output_dir / "data.csv"
                shutil.copy2(origin_csv, dest)
                updated_path = str(dest)
                print(f"Copied dataset: {dest}")
            else:
                print("[ERROR] No historical dataset available to copy.")

        if updated_path:
            tracker.mark_step_completed(
                iteration_num,
                step_key,
                metadata={
                    "data_path": updated_path,
                    "data_sha256": file_sha256(updated_path),
                    "source_data": str(origin_csv) if origin_csv else None,
                },
            )
            results["update"] = {"success": True, "path": updated_path}
            print(f"[OK] Dataset update completed: {updated_path}")
        else:
            results["update"] = {"success": False}

    print("\n" + "=" * 80)
    print(f"[OK] Iteration {iteration_num} completed.")
    print("=" * 80)
    return results


def resolve_bo_start_iteration(tracker, requested_start: int | None) -> tuple[list[int], int, int]:
    """Resolve a non-regressing BO resume cursor from the contiguous prefix."""
    completed_rounds, last_completed = get_contiguous_completed_rounds(tracker)
    resume_start = max(1, last_completed + 1)
    if requested_start is None or requested_start <= resume_start:
        return completed_rounds, last_completed, resume_start
    if requested_start > 1 and not tracker.is_round_completed(requested_start - 1):
        raise ValueError(
            f"Explicit start iteration {requested_start} is invalid: "
            f"predecessor iteration {requested_start - 1} is incomplete."
        )
    return completed_rounds, last_completed, requested_start


def _iteration_failure_step(results: dict) -> str | None:
    """Return the first incomplete step instead of allowing a broken round to continue."""
    requirements = (
        ("train", "success"),
        ("bayes", "success"),
        ("structure", "completed"),
        ("merge", "success"),
        ("extract", "success"),
        ("update", "success"),
    )
    for result_key, field in requirements:
        result = results.get(result_key)
        if not isinstance(result, dict) or not result.get(field, False):
            return result_key
    return None


def main():
    try:
        import multiprocessing

        multiprocessing.freeze_support()
        args = parse_args()

        if args.reset:
            archived_root = archive_active_run(project_root, RUN_MODE)
            if archived_root is not None:
                print(f"[RESET] Archived previous active run: {archived_root}")
            else:
                print(f"[RESET] No existing active run found; creating fresh {RUN_MODE}/")

        results_dir = project_root / RESULTS_ROOT
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = results_dir / f"run_{timestamp}.log"
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        handlers = [
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ]

        logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers, force=True)
        logger = logging.getLogger(__name__)
        logger.info("=" * 80)
        logger.info("Program started. Log file: %s", log_file)
        logger.info("=" * 80)

        for handler in logging.root.handlers:
            handler.flush()

        print(f"\n{'=' * 80}")
        print(f"[LOG] Log file: {log_file}")
        print(f"{'=' * 80}\n")

        config = DEFAULT_CONFIG.copy()
        config.update(load_bo_runtime_defaults())

        if args.samples is not None:
            config["samples"] = args.samples
        if args.top_k_bayes is not None:
            config["top_k_bayes"] = args.top_k_bayes
        if args.n_structures is not None:
            config["n_structures"] = args.n_structures
        if args.postprocess_workers is not None:
            config["postprocess_workers"] = args.postprocess_workers
        if args.novelty_workers is not None:
            config["novelty_workers"] = args.novelty_workers
        if args.top_k_screen is not None:
            config["top_k_screen"] = args.top_k_screen
        if args.num_gpus is not None:
            config["gpus"] = [f"cuda:{i}" for i in range(args.num_gpus)]
            print(f"  GPU config: {config['gpus']}")
        if args.device is not None:
            config["device"] = args.device
            config["gpus"] = ["cpu"] if args.device == "cpu" else config.get("gpus", ["cuda:0"])
            print(f"  MatterSim device: {config['device']}")
        if args.relax_timeout_sec is not None:
            config["relax_timeout_sec"] = int(args.relax_timeout_sec)
        if args.phonon_imag_tol is not None:
            config["phonon_imag_tol"] = float(args.phonon_imag_tol)

        config["seed"] = int(args.seed)
        config["deterministic_torch"] = not bool(args.non_deterministic_torch)
        config["allow_partial_structure"] = args.allow_partial_structure

        try:
            config["gpus"] = normalize_gpu_list(
                config.get("gpus"),
                device=config.get("device", "cuda"),
            )
            validate_gpu_devices(config["gpus"])
        except (RuntimeError, ValueError) as exc:
            print(f"[FATAL] Invalid GPU configuration: {exc}")
            raise SystemExit(2) from exc

        repro_info = setup_reproducibility(
            seed=config["seed"],
            deterministic_torch=config["deterministic_torch"],
        )
        print(f"  Reproducibility seed: {repro_info['seed']}")
        print(f"  Deterministic torch: {repro_info['deterministic_torch']}")

        print("=" * 80)
        print("Bayesian Optimization Materials Discovery")
        print("Entrypoint: main_bo_only.py (former main.py --screening-mode bo_direct)")
        print(f"Run mode: {RUN_MODE}")
        print("Screening mode: bo_direct")
        print(f"Data root: {DATA_ROOT}")
        print(f"Models root: {MODELS_ROOT}")
        print(f"Results root: {RESULTS_ROOT}")
        print("=" * 80)

        prepare_initial_data(init_data_path=args.init_data)

        path_config = PathConfig.from_run_mode(
            project_root=project_root,
            run_mode=RUN_MODE,
            init_data_path=args.init_data,
            init_doc_path=None,
        )
        config["path_config"] = path_config
        # Keep downstream artifacts anchored to the project, not the caller's
        # current working directory.
        config["results_root"] = str(path_config.results_root)
        config["models_root"] = str(path_config.models_root)
        config["data_root"] = str(path_config.data_root)

        tracker = ProgressTracker(base_dir=path_config.results_root, steps=BO_STEPS)
        if args.rebuild_from is not None:
            try:
                rebuild_archive = rebuild_active_chain(
                    path_config,
                    tracker,
                    from_iteration=args.rebuild_from,
                    run_mode=RUN_MODE,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                print(f"[FATAL] Rebuild failed: {exc}")
                sys.exit(1)
            print(f"[REBUILD] Archived invalidated active chain to: {rebuild_archive}")
        try:
            reconcile_messages = reconcile_progress_with_filesystem(
                tracker,
                path_config,
                allow_inconsistent_history=False,
            )
        except WorkflowProgressInconsistencyError as exc:
            print(f"[FATAL] {exc}")
            print("[INFO] No iteration was started and progress/history was preserved.")
            print("[INFO] Repair the missing predecessor or explicitly use --reset to rebuild the chain.")
            sys.exit(1)
        for message in reconcile_messages:
            print(f"[INFO] Progress reconciled: {message}")

        try:
            completed_rounds, last_completed, start_iteration = resolve_bo_start_iteration(
                tracker,
                args.start_iteration,
            )
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)
        print(
            f"[RESUME] completed_rounds={completed_rounds}, "
            f"contiguous_last={last_completed}, start_iteration={start_iteration}"
        )
        if start_iteration > 1 and not tracker.is_round_completed(start_iteration - 1):
            print(
                f"[ERROR] Explicit start iteration {start_iteration} is invalid: "
                f"predecessor iteration {start_iteration - 1} is incomplete."
            )
            sys.exit(1)

        initial_samples = None
        if start_iteration > 1:
            cached_extract = load_saved_extract_result(results_dir, start_iteration - 1)
            initial_samples, sample_source = extract_initial_samples_from_result(cached_extract)
            if initial_samples:
                print(
                    f"[INFO] Loaded {len(initial_samples)} warm-start samples from "
                    f"iteration_{start_iteration - 1} ({sample_source})"
                )

        for iteration_num in range(start_iteration, args.max_iterations + 1):
            if iteration_num > 1 and not tracker.is_round_completed(iteration_num - 1):
                print(
                    f"[ERROR] Cannot start iteration {iteration_num}: "
                    f"predecessor iteration {iteration_num - 1} is incomplete."
                )
                print(
                    "[INFO] Resume from the first incomplete predecessor with the same "
                    "command and without --reset."
                )
                sys.exit(1)

            if tracker.is_round_completed(iteration_num):
                print(f"\n[SKIP] Iteration {iteration_num} already completed in tracker")
                cached_extract = load_saved_extract_result(results_dir, iteration_num)
                initial_samples, sample_source = extract_initial_samples_from_result(cached_extract)
                if initial_samples:
                    print(
                        f"[resume] loaded {len(initial_samples)} warm-start samples from "
                        f"iteration_{iteration_num} ({sample_source})"
                    )
                continue

            results = run_single_iteration(iteration_num, config, tracker, initial_samples)

            failure_step = _iteration_failure_step(results)
            if failure_step is None and not tracker.is_round_completed(iteration_num):
                failure_step = tracker.get_next_incomplete_step(iteration_num) or "unknown"
                results.setdefault(failure_step, {})
            if failure_step is not None:
                failed_result = results.get(failure_step) or {}
                print(
                    f"[ERROR] Iteration {iteration_num} stopped at incomplete step "
                    f"'{failure_step}': {failed_result.get('error', 'no completed result')}"
                )
                print("[INFO] Progress was preserved; rerun the same command without --reset to resume.")
                sys.exit(1)

            new_initial_samples = None
            extract_res = results.get("extract")
            if extract_res:
                try:
                    new_initial_samples, source_text = extract_initial_samples_from_result(extract_res)
                    if new_initial_samples:
                        print(
                            f"[INFO] Extracted {len(new_initial_samples)} initial samples "
                            f"for next iteration (source: {source_text})"
                        )
                    elif extract_res.get("has_success") or extract_res.get("has_stable"):
                        print("[WARN] Extracted initial samples are empty (possibly filtered by K threshold).")
                except Exception as exc:
                    print(f"[ERROR] Failed to read initial samples: {exc}")

            if new_initial_samples:
                initial_samples = new_initial_samples
            elif initial_samples:
                initial_samples = None
                print("[INFO] No new materials; force random sampling next iteration")

    except Exception as exc:
        print(f"[FATAL ERROR] {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
