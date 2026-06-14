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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

project_root = Path(__file__).parent
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
    from utils.reproducibility import setup_reproducibility
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

    df_combined.to_csv(target_file, index=False, encoding="utf-8-sig")
    return len(df_combined)


def load_fallback_bo_candidates(limit: int = 5, fallback_iteration: int = 15) -> list[dict]:
    fallback_path = (
        project_root / RESULTS_ROOT / f"iteration_{fallback_iteration}" / "selected_results" / "bo_candidates.json"
    )
    if not fallback_path.exists():
        return []

    try:
        with open(fallback_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to load fallback BO candidates: {exc}")
        return []

    if not isinstance(data, list):
        return []

    samples = []
    for item in data[:limit]:
        formula = item.get("formula")
        if not formula:
            continue
        entry = {"formula": formula}
        if "composition" in item:
            entry["composition"] = item["composition"]
        samples.append(entry)
    return samples


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


def parse_args():
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
        "--relax-timeout-sec",
        type=int,
        default=None,
        help="timeout in seconds for one relax+phonon task",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument(
        "--non-deterministic-torch",
        action="store_true",
        help="disable deterministic torch kernels",
    )
    parser.add_argument("--start-iteration", type=int, default=1, help="start iteration (default: 1)")
    parser.add_argument("--reset", action="store_true", help="reset all progress and start from scratch")
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
    args = parser.parse_args()
    if args.max_iterations <= 0:
        parser.error("--max-iterations must be > 0")
    if args.start_iteration <= 0:
        parser.error("--start-iteration must be > 0")
    if args.top_k_bayes is not None and args.top_k_bayes <= 0:
        parser.error("--top-k-bayes must be > 0")
    if args.n_structures is not None and args.n_structures <= 0:
        parser.error("--n-structures must be > 0")
    if args.top_k_screen is not None and args.top_k_screen <= 0:
        parser.error("--top-k-screen must be > 0")
    if args.num_gpus is not None and args.num_gpus <= 0:
        parser.error("--num-gpus must be > 0")
    if args.relax_timeout_sec is not None and args.relax_timeout_sec <= 0:
        parser.error("--relax-timeout-sec must be > 0")
    if args.samples is not None and args.samples <= 0:
        parser.error("--samples must be > 0")
    if args.start_iteration > args.max_iterations:
        parser.error("--start-iteration cannot be greater than --max-iterations")
    return args


def run_single_iteration(iteration_num: int, config: dict, tracker: ProgressTracker, initial_samples=None):
    """Run one BO-only iteration."""
    print("\n" + "=" * 80)
    print(f">> Start Iteration {iteration_num} (Mode: BO-only / bo_direct)")
    print("=" * 80)
    sys.stdout.flush()

    results = {}

    step_key = "train_model"
    if tracker.is_step_completed(iteration_num, step_key):
        print(f"[SKIP] Step 1 ({step_key}) already completed")
        results["train"] = {"success": True}
    else:
        train_result = step_train_model(
            iteration_num=iteration_num,
            data_root=DATA_ROOT,
            models_root=MODELS_ROOT,
            path_config=config.get("path_config"),
        )
        results["train"] = train_result
        if train_result["success"]:
            tracker.mark_step_completed(iteration_num, step_key)
        else:
            print(f"[ERROR] Step 1 failed: {train_result.get('error')}")
            return results

    step_key = "bayesian_optimization"
    candidate_materials = []
    save_dir = project_root / RESULTS_ROOT / f"iteration_{iteration_num}" / "selected_results"
    save_file = save_dir / "bo_candidates.json"

    if tracker.is_step_completed(iteration_num, step_key):
        print(f"[SKIP] Step 2 ({step_key}) already completed")
        if not save_file.exists():
            print(f"[ERROR] Candidate file not found: {save_file}")
            return results
        try:
            with open(save_file, "r", encoding="utf-8") as f:
                candidate_materials = json.load(f)
            print(f"[INFO] Loaded {len(candidate_materials)} BO candidates from file: {save_file}")
            results["bayes"] = {"success": True, "top_materials": candidate_materials}
        except Exception as exc:
            print(f"[ERROR] Failed to load BO candidates: {exc}")
            return results
    else:
        print("")
        print("#" * 80)
        print("Step 2/5: Bayesian Optimization")
        print("#" * 80)
        bayes_result = step_bayesian_optimization(
            iteration_num=iteration_num,
            xi=config["xi"],
            n_samples=config["samples"],
            n_top=config["top_k_bayes"],
            initial_samples=initial_samples,
            seed=config.get("seed"),
            seed_stride=config.get("seed_stride", 1000),
            models_root=MODELS_ROOT,
            results_root=RESULTS_ROOT,
            path_config=config.get("path_config"),
        )
        results["bayes"] = bayes_result

        if not bayes_result["success"]:
            print(f"[ERROR] Step 2 failed: {bayes_result.get('error')}")
            return results

        candidate_materials = bayes_result.get("top_materials", [])
        tracker.mark_step_completed(iteration_num, step_key)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(candidate_materials, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved candidates: {save_file}")

    top_k_screen = config.get("top_k_screen", 10)
    selected_materials = candidate_materials[:top_k_screen] if candidate_materials else []
    if candidate_materials:
        print(
            f"[SELECT] bo_direct selected top {len(selected_materials)} "
            f"from {len(candidate_materials)} BO candidates"
        )

    step_key = "structure_calculation"
    if tracker.is_step_completed(iteration_num, step_key):
        print(f"[SKIP] Step 3 ({step_key}) already completed")
        results["structure"] = {"success": True}
    else:
        print(f"\n{'#' * 80}")
        print("Step 3/5: Structure generation and calculation")
        print(f"{'#' * 80}")

        if not selected_materials:
            print("[ERROR] No candidate materials available for structure generation.")
            return results

        structure_result = step_structure_calculation(
            iteration_num=iteration_num,
            materials=selected_materials,
            n_structures=config["n_structures"],
            max_workers=config["max_workers"],
            relax_workers=config["relax_workers"],
            phonon_workers=config["phonon_workers"],
            pressure=config["pressure"],
            device=config["device"],
            gpus=config.get("gpus", ["cuda:0"]),
            allow_partial_completion=config.get("allow_partial_structure", False),
            results_root=RESULTS_ROOT,
            seed=config.get("seed"),
            tracker=tracker,
            path_config=config.get("path_config"),
            relax_timeout_sec=config.get("relax_timeout_sec", 900),
            prefer_isolated_relax_process=config.get("prefer_isolated_relax_process", True),
            allow_in_process_relax_fallback=config.get("allow_in_process_relax_fallback", True),
        )
        results["structure"] = structure_result

        if structure_result.get("completed"):
            tracker.mark_step_completed(iteration_num, step_key)
        else:
            print("[INFO] Structure calculation not completed; keep progress and continue later.")
            return results

    step_key = "merge_results"
    if tracker.is_step_completed(iteration_num, step_key):
        print("[SKIP] Step merge_results completed")
        results["merge"] = {"success": True}
    else:
        merge_result = step_merge_results(
            iteration_num=iteration_num,
            results_root=RESULTS_ROOT,
            tracker=tracker,
        )
        results["merge"] = merge_result
        if not merge_result.get("success"):
            print(f"[ERROR] merge_results failed: {merge_result.get('error')}")
            return results

    step_key = "success_extraction"
    if tracker.is_step_completed(iteration_num, step_key):
        print(f"[SKIP] Step 4 ({step_key}) already completed")
        extract_result = {"success": True, "has_success": True}
        results["extract"] = extract_result
    else:
        print(f"\n{'#' * 80}")
        print("Step 4/5: Extract success and stable materials")
        print(f"{'#' * 80}")

        extract_result = step_extract_materials(
            iteration_num=iteration_num,
            k_threshold=config["k_threshold"],
            results_root=RESULTS_ROOT,
        )
        results["extract"] = extract_result

        if not (extract_result.get("success") or extract_result.get("no_materials")):
            print(f"[ERROR] Step 4 failed: {extract_result.get('error')}")
            return results

        tracker.mark_step_completed(iteration_num, step_key)

        try:
            aggregated_results_dir = project_root / RESULTS_ROOT
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
        origin_csv = None
        for i in range(prev_iteration, -1, -1):
            candidate = project_root / DATA_ROOT / f"iteration_{i}" / "data.csv"
            if candidate.exists():
                origin_csv = candidate
                break

        output_dir = project_root / DATA_ROOT / f"iteration_{iteration_num}"
        updated_path = None

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
            tracker.mark_step_completed(iteration_num, step_key)
            results["update"] = {"success": True, "path": updated_path}
            print(f"[OK] Dataset update completed: {updated_path}")
        else:
            results["update"] = {"success": False}

    print("\n" + "=" * 80)
    print(f"[OK] Iteration {iteration_num} completed.")
    print("=" * 80)
    return results


def main():
    try:
        import multiprocessing

        multiprocessing.freeze_support()
        args = parse_args()

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
        if args.top_k_screen is not None:
            config["top_k_screen"] = args.top_k_screen
        if args.num_gpus is not None:
            config["gpus"] = [f"cuda:{i}" for i in range(args.num_gpus)]
            print(f"  GPU config: {config['gpus']}")
        if args.relax_timeout_sec is not None:
            config["relax_timeout_sec"] = int(args.relax_timeout_sec)

        config["seed"] = int(args.seed)
        config["deterministic_torch"] = not bool(args.non_deterministic_torch)
        config["allow_partial_structure"] = args.allow_partial_structure

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

        tracker = ProgressTracker(base_dir=RESULTS_ROOT, steps=BO_STEPS)

        if args.reset:
            for i in range(1, args.max_iterations + 1):
                tracker.reset_round(i)
            print("[INFO] Reset progress for all iterations in range.")

        initial_samples = None
        for iteration_num in range(args.start_iteration, args.max_iterations + 1):
            if tracker.is_round_completed(iteration_num):
                print(f"\n[SKIP] Iteration {iteration_num} already completed in tracker")
                continue

            results = run_single_iteration(iteration_num, config, tracker, initial_samples)

            if not results.get("extract", {}).get("success", False):
                print(f"[WARN] Iteration {iteration_num} did not fully succeed")

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

            if (not new_initial_samples) and extract_res and (not extract_res.get("has_success")) and (
                not extract_res.get("has_stable")
            ):
                fallback_samples = load_fallback_bo_candidates()
                if fallback_samples:
                    new_initial_samples = fallback_samples
                    print("[INFO] No success/stable materials; using top5 from iteration_15 bo_candidates.json")
                else:
                    print("[WARN] No success/stable materials and fallback bo_candidates.json is missing or empty")

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
