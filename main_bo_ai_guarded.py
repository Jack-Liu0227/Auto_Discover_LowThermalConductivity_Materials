# -*- coding: utf-8 -*-
"""Independent BO-dominant + AI-supplement workflow entrypoint."""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils.bo_runtime import extract_initial_samples_from_result, load_bo_runtime_defaults
from utils.path_config import PathConfig
from utils.progress_tracker import ProgressTracker
from utils.reproducibility import setup_reproducibility
from workflow.guarded_selection import run_guarded_ai_selection, update_guarded_selection_summary
from workflow.step_bayesian_optimization import step_bayesian_optimization
from workflow.step_extract_materials import step_extract_materials
from workflow.step_merge_results import step_merge_results
from workflow.step_structure_calculation import step_structure_calculation
from workflow.step_train_model import step_train_model
from workflow.step_update_data_doc import step_update_data_and_doc

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ["PYTHONIOENCODING"] = "utf-8"

RUN_MODE = "bo_ai_guarded"
RESULTS_ROOT = f"{RUN_MODE}/results"
MODELS_ROOT = f"{RUN_MODE}/models/GPR"
DATA_ROOT = f"{RUN_MODE}/data"
DOC_ROOT = f"{RUN_MODE}/doc"

DEFAULT_CONFIG = {
    "version": 1,
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
    "phonon_imag_tol": -0.1,
    "top_k_screen": 10,
    "seed": 42,
    "seed_stride": 1000,
    "deterministic_torch": True,
    "allow_partial_structure": False,
    "skip_doc_update": False,
}

GUARDED_STEPS = [
    "train_model",
    "bayesian_optimization",
    "ai_evaluation",
    "structure_calculation",
    "merge_results",
    "success_extraction",
    "document_update",
]


def initialize_environment(path_config: PathConfig) -> None:
    path_config.create_directories()

    target_doc = path_config.doc_root / "v0.0.0" / "Theoretical_principle_document.md"
    target_doc.parent.mkdir(parents=True, exist_ok=True)
    if not target_doc.exists():
        candidate_docs = [
            path_config.init_doc_path,
            project_root / "doc" / "Theoretical_principle_document.md",
            project_root / "llm" / "doc" / "v0.0.0" / "Theoretical_principle_document.md",
        ]
        for src in candidate_docs:
            if src and Path(src).exists():
                shutil.copy2(src, target_doc)
                break

    target_data = path_config.data_root / "iteration_0" / "data.csv"
    target_data.parent.mkdir(parents=True, exist_ok=True)
    if not target_data.exists():
        candidate_data = [
            path_config.init_data_path,
            project_root / "data" / "processed_data.csv",
            project_root / "llm" / "data" / "iteration_0" / "data.csv",
            project_root / "bo" / "data" / "iteration_0" / "data.csv",
        ]
        for src in candidate_data:
            if src and Path(src).exists():
                shutil.copy2(src, target_data)
                break


def load_fallback_bo_candidates(limit: int = 5, fallback_iteration: int = 15) -> list[dict]:
    fallback_path = project_root / RESULTS_ROOT / f"iteration_{fallback_iteration}" / "selected_results" / "bo_candidates.json"
    if not fallback_path.exists():
        return []
    try:
        data = json.loads(fallback_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Failed to load fallback BO candidates: {exc}")
        return []
    if not isinstance(data, list):
        return []

    samples: list[dict] = []
    for item in data[:limit]:
        formula = item.get("formula")
        if not formula:
            continue
        entry = {"formula": formula}
        if "composition" in item:
            entry["composition"] = item["composition"]
        samples.append(entry)
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="BO-dominant + AI-supplement guarded workflow.")
    parser.add_argument("--max-iterations", type=int, default=20, help="maximum iterations")
    parser.add_argument("--samples", type=int, default=None, help="number of BO samples per iteration")
    parser.add_argument("--top-k-bayes", type=int, default=None, help="number of top BO candidates kept for screening")
    parser.add_argument("--n-top-candidates", dest="top_k_bayes", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--n-structures", type=int, default=None, help="number of generated structures per composition")
    parser.add_argument("--top-k-screen", type=int, default=None, help="number of materials sent to structure calculation")
    parser.add_argument("--n-select", dest="top_k_screen", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--num-gpus", type=int, default=None, help="number of GPUs to use")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--non-deterministic-torch", action="store_true", help="disable deterministic torch kernels")
    parser.add_argument("--start-iteration", type=int, default=1, help="start iteration")
    parser.add_argument("--reset", action="store_true", help="reset all progress and start from scratch")
    parser.add_argument("--allow-partial-structure", action="store_true", help="allow workflow to continue when some structure tasks fail")
    parser.add_argument("--skip-doc-update", action="store_true", help="reuse previous theory document instead of updating it")
    parser.add_argument("--init-data", type=str, default="data/processed_data.csv", help="path to initial dataset")
    parser.add_argument("--init-doc", type=str, default="doc/Theoretical_principle_document.md", help="path to initial theory document")
    return parser.parse_args()


def _append_iteration_summary(iteration_num: int, extract_result: dict) -> None:
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
            if not source_path or not os.path.exists(source_path):
                continue
            df_new = pd.read_csv(source_path)
            df_new.insert(0, "iteration", iteration_num)
            target_file = summary_files[key]
            if target_file.exists():
                df_existing = pd.read_csv(target_file)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(target_file, index=False)
            else:
                df_new.to_csv(target_file, index=False)
    except Exception as exc:
        print(f"[WARN] Failed to update summary files: {exc}")


def run_single_iteration(iteration_num: int, config: dict, tracker: ProgressTracker, initial_samples=None):
    print("\n" + "=" * 80)
    print(f">> Start Iteration {iteration_num} (Mode: BO-dominant + AI-supplement)")
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
        if train_result.get("success"):
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
        if save_file.exists():
            try:
                candidate_materials = json.loads(save_file.read_text(encoding="utf-8"))
                results["bayes"] = {"success": True, "top_materials": candidate_materials}
            except Exception as exc:
                print(f"[ERROR] Failed to load BO candidates: {exc}")
                return results
        else:
            print(f"[ERROR] Candidate file not found: {save_file}")
            return results
    else:
        print("\n" + "#" * 80)
        print("Step 2/6: Bayesian Optimization")
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
        if bayes_result.get("success"):
            candidate_materials = bayes_result.get("top_materials", [])
            tracker.mark_step_completed(iteration_num, step_key)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_file.write_text(json.dumps(candidate_materials, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            print(f"[ERROR] Step 2 failed: {bayes_result.get('error')}")
            return results

    print("\n" + "#" * 80)
    print("Step 3/6: Guarded AI supplement selection")
    print("#" * 80)
    guarded_result = run_guarded_ai_selection(
        iteration_num=iteration_num,
        candidate_materials=candidate_materials,
        config=config,
        tracker=tracker,
    )
    results["screen"] = guarded_result
    selected_materials = guarded_result.get("selected_materials", [])
    if not guarded_result.get("success") or not selected_materials:
        print("[ERROR] Guarded screening failed to produce selected materials.")
        return results

    step_key = "structure_calculation"
    if tracker.is_step_completed(iteration_num, step_key):
        print(f"[SKIP] Step 4 ({step_key}) already completed")
        results["structure"] = {"success": True}
    else:
        print("\n" + "#" * 80)
        print("Step 4/6: Structure generation and calculation")
        print("#" * 80)
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
            tracker=tracker,
            path_config=config.get("path_config"),
        )
        results["structure"] = structure_result
        if not structure_result.get("completed"):
            print("[INFO] Structure calculation not completed; keep progress and continue later.")
            return results

    step_key = "merge_results"
    if tracker.is_step_completed(iteration_num, step_key):
        print(f"[SKIP] Step merge_results completed")
        results["merge"] = {"success": True}
    else:
        merge_result = step_merge_results(iteration_num=iteration_num, results_root=RESULTS_ROOT, tracker=tracker)
        results["merge"] = merge_result
        if not merge_result.get("success"):
            print(f"[ERROR] merge_results failed: {merge_result.get('error')}")
            return results

    step_key = "success_extraction"
    if tracker.is_step_completed(iteration_num, step_key):
        print(f"[SKIP] Step 5 ({step_key}) already completed")
        extract_result = {"success": True, "has_success": True}
        results["extract"] = extract_result
    else:
        print("\n" + "#" * 80)
        print("Step 5/6: Extract success and stable materials")
        print("#" * 80)
        extract_result = step_extract_materials(
            iteration_num=iteration_num,
            k_threshold=config["k_threshold"],
            imag_tol=config["phonon_imag_tol"],
            results_root=RESULTS_ROOT,
        )
        results["extract"] = extract_result
        if extract_result.get("success") or extract_result.get("no_materials"):
            tracker.mark_step_completed(iteration_num, step_key)
            if extract_result.get("no_materials"):
                extract_result["success"] = True
                extract_result["has_success"] = False
                extract_result["has_stable"] = False
            _append_iteration_summary(iteration_num, extract_result)
            summary_path = update_guarded_selection_summary(iteration_num, project_root / RESULTS_ROOT, extract_result)
            if summary_path:
                print(f"[OK] Updated guarded summary: {summary_path}")
        else:
            print(f"[ERROR] Step 5 failed: {extract_result.get('error')}")
            return results

    step_key = "document_update"
    if tracker.is_step_completed(iteration_num, step_key):
        print(f"[SKIP] Step 6 ({step_key}) already completed")
        results["update"] = {"success": True}
    else:
        print("\n" + "#" * 80)
        print("Step 6/6: Update dataset and theory document")
        print("#" * 80)
        update_result = step_update_data_and_doc(
            iteration_num=iteration_num,
            extraction_result=results["extract"],
            version=config.get("version", 1),
            path_config=config.get("path_config"),
            data_root=DATA_ROOT,
            results_root=RESULTS_ROOT,
            doc_root=DOC_ROOT,
            skip_doc_update=config.get("skip_doc_update", False),
        )
        results["update"] = update_result
        if not update_result.get("success"):
            print(f"[ERROR] Step 6 failed: {update_result.get('error')}")
            return results
        tracker.mark_step_completed(
            iteration_num,
            step_key,
            metadata={
                "updated_data_path": update_result.get("updated_data_path"),
                "updated_doc_path": update_result.get("updated_doc_path"),
            },
        )

    print("\n" + "=" * 80)
    print(f"[OK] Iteration {iteration_num} completed.")
    print("=" * 80)
    return results


def main():
    try:
        results_dir = project_root / RESULTS_ROOT
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = results_dir / f"run_{timestamp}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
            force=True,
        )
        print(f"\n{'=' * 80}")
        print(f"[LOG] Log file: {log_file}")
        print(f"{'=' * 80}\n")

        args = parse_args()
        config = DEFAULT_CONFIG.copy()
        config.update(load_bo_runtime_defaults())

        if args.samples:
            config["samples"] = args.samples
        if args.top_k_bayes:
            config["top_k_bayes"] = args.top_k_bayes
        if args.n_structures:
            config["n_structures"] = args.n_structures
        if args.top_k_screen:
            config["top_k_screen"] = args.top_k_screen
        if args.num_gpus is not None:
            config["gpus"] = [f"cuda:{i}" for i in range(args.num_gpus)]
        config["seed"] = int(args.seed)
        config["deterministic_torch"] = not bool(args.non_deterministic_torch)
        config["allow_partial_structure"] = args.allow_partial_structure
        config["skip_doc_update"] = args.skip_doc_update
        config["init_doc_path"] = args.init_doc

        repro_info = setup_reproducibility(seed=config["seed"], deterministic_torch=config["deterministic_torch"])
        print(f"  Reproducibility seed: {repro_info['seed']}")
        print(f"  Deterministic torch: {repro_info['deterministic_torch']}")

        print("=" * 80)
        print("BO-dominant + AI-supplement guarded workflow")
        print(f"Data root: {DATA_ROOT}")
        print(f"Models root: {MODELS_ROOT}")
        print(f"Results root: {RESULTS_ROOT}")
        print(f"Doc root: {DOC_ROOT}")
        print("=" * 80)

        path_config = PathConfig(
            project_root=project_root,
            results_root=Path(RESULTS_ROOT),
            models_root=Path(MODELS_ROOT),
            data_root=Path(DATA_ROOT),
            doc_root=Path(DOC_ROOT),
            init_data_path=Path(args.init_data),
            init_doc_path=Path(args.init_doc),
        )
        initialize_environment(path_config)
        config["path_config"] = path_config
        config["data_root"] = DATA_ROOT
        config["models_root"] = MODELS_ROOT
        config["results_root"] = RESULTS_ROOT
        config["doc_root"] = DOC_ROOT

        tracker = ProgressTracker(base_dir=RESULTS_ROOT, steps=GUARDED_STEPS)
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
            extract_res = results.get("extract")
            new_initial_samples = None
            if extract_res:
                try:
                    new_initial_samples, source_text = extract_initial_samples_from_result(extract_res)
                    if new_initial_samples:
                        print(f"[INFO] Extracted {len(new_initial_samples)} initial samples for next iteration (source: {source_text})")
                except Exception as exc:
                    print(f"[ERROR] Failed to read initial samples: {exc}")

            if (not new_initial_samples) and extract_res and (not extract_res.get("has_success")) and (not extract_res.get("has_stable")):
                fallback_samples = load_fallback_bo_candidates()
                if fallback_samples:
                    new_initial_samples = fallback_samples
                    print("[INFO] No success/stable materials; using top5 from iteration_15 bo_candidates.json")

            if new_initial_samples:
                initial_samples = new_initial_samples
            elif initial_samples:
                initial_samples = None
                print("No new materials; force random sampling next iteration")

    except Exception as exc:
        print(f"[FATAL ERROR] {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
