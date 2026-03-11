# -*- coding: utf-8 -*-
"""ASLK entrypoint: Agno Workflow + AgentOS runtime dispatcher."""

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils.path_config import PathConfig
from utils.param_sheet import ensure_param_sheet, load_param_overrides, load_param_prefill, persist_param_values
from utils.progress_tracker import ProgressTracker
from utils.reproducibility import setup_reproducibility
from workflow.agno_pipeline import run_workflow, serve_agentos

RUN_MODE = "llm"
RESULTS_ROOT = f"{RUN_MODE}/results"
MODELS_ROOT = f"{RUN_MODE}/models/GPR"
DATA_ROOT = f"{RUN_MODE}/data"
DOC_ROOT = f"{RUN_MODE}/doc"


DEFAULT_CONFIG = {
    "version": 1,
    "samples": 100,
    "xi": 0.01,
    "n_select": 10,
    "n_structures": 5,
    "n_top_candidates": 20,
    "top_k_bayes": 20,
    "top_k_screen": 10,
    "max_workers": 4,
    "relax_workers": 1,
    "phonon_workers": 1,
    "pressure": 0.0,
    "device": "cuda",
    "gpus": ["cuda:0"],
    "k_threshold": 1.0,
    "phonon_imag_tol": -0.1,
    "seed": 42,
    "seed_stride": 1000,
    "deterministic_torch": True,
    "allow_partial_structure": False,
    "skip_doc_update": False,
    "relax_timeout_sec": 120,
    "websearch_enabled": True,
    "websearch_top_n": 5,
    "websearch_strategy": "hybrid",
    "websearch_queries_per_candidate": 2,
    "websearch_theory_template": None,
    "agentos_default_iterations": 3,
    "agentos_allow_text_iteration_override": True,
    "agentos_max_iterations_cap": 20,
    "agentos_ws_ping_interval": None,
    "agentos_ws_ping_timeout": None,
    "data_root": DATA_ROOT,
    "models_root": MODELS_ROOT,
    "results_root": RESULTS_ROOT,
    "doc_root": DOC_ROOT,
}

PARAM_MEMORY_KEYS = [
    "samples",
    "n_structures",
    "top_k_bayes",
    "top_k_screen",
    "websearch_enabled",
    "websearch_top_n",
    "phonon_imag_tol",
    "seed",
    "relax_timeout_sec",
    "skip_doc_update",
    "agentos_default_iterations",
    "agentos_ws_ping_interval",
    "agentos_ws_ping_timeout",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASLK Agno Workflow Runtime")
    parser.add_argument("--runtime", choices=["workflow", "agentos"], default="workflow")
    parser.add_argument("--websearch-top-n", type=int, default=5)
    parser.add_argument("--websearch-enabled", dest="websearch_enabled", action="store_true")
    parser.add_argument("--no-websearch-enabled", dest="websearch_enabled", action="store_false")
    parser.set_defaults(websearch_enabled=True)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--add-iterations", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--n-select", type=int, default=None)
    parser.add_argument("--n-structures", type=int, default=None)
    parser.add_argument("--phonon-imag-tol", type=float, default=None)
    parser.add_argument("--n-top-candidates", type=int, default=None)
    parser.add_argument("--top-k-bayes", type=int, default=20)
    parser.add_argument("--top-k-screen", type=int, default=10)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--non-deterministic-torch", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--allow-partial-structure", action="store_true")
    parser.add_argument("--init-data", type=str, default="data/processed_data.csv")
    parser.add_argument("--init-doc", type=str, default="doc/Theoretical_principle_document.md")
    parser.add_argument("--agentos-host", type=str, default="127.0.0.1")
    parser.add_argument("--agentos-port", type=int, default=7777)
    parser.add_argument(
        "--params-csv",
        type=str,
        default="config/agentos_params.csv",
        help="Editable parameter sheet (CSV) used to override workflow config",
    )
    args = parser.parse_args()
    if args.add_iterations is not None and args.add_iterations <= 0:
        parser.error("--add-iterations must be > 0")
    if args.max_iterations is not None and args.max_iterations <= 0:
        parser.error("--max-iterations must be > 0")
    if args.websearch_top_n < 0:
        parser.error("--websearch-top-n must be >= 0")
    if args.add_iterations is not None and args.max_iterations is not None:
        parser.error("Use either --max-iterations or --add-iterations, not both")
    return args


def initialize_environment(path_config: PathConfig) -> None:
    path_config.create_directories()

    target_doc = path_config.doc_root / "v0.0.0" / "Theoretical_principle_document.md"
    target_doc.parent.mkdir(parents=True, exist_ok=True)
    if not target_doc.exists():
        legacy_doc_name = "\u7406\u8bba\u539f\u7406\u6587\u6863.md"
        candidate_docs = [
            path_config.init_doc_path,
            project_root / "doc" / "Theoretical_principle_document.md",
            project_root / "llm" / "doc" / "v0.0.0" / "Theoretical_principle_document.md",
            project_root / "llm" / "doc" / "v0.0.0" / legacy_doc_name,
            project_root / "llm_first_version" / "doc" / "v0.0.0" / "Theoretical_principle_document.md",
            project_root / "llm_first_version" / "doc" / "v0.0.0" / legacy_doc_name,
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
        ]
        for src in candidate_data:
            if src and Path(src).exists():
                shutil.copy2(src, target_data)
                break


def build_config(args: argparse.Namespace) -> dict:
    config = DEFAULT_CONFIG.copy()
    if args.samples is not None:
        config["samples"] = args.samples
    if args.n_select is not None:
        config["n_select"] = args.n_select
    if args.n_structures is not None:
        config["n_structures"] = args.n_structures
    if args.phonon_imag_tol is not None:
        config["phonon_imag_tol"] = args.phonon_imag_tol
    if args.n_top_candidates is not None:
        config["n_top_candidates"] = args.n_top_candidates

    config["top_k_bayes"] = args.n_top_candidates if args.n_top_candidates is not None else args.top_k_bayes
    config["top_k_screen"] = args.n_select if args.n_select is not None else args.top_k_screen

    if args.num_gpus is not None:
        config["gpus"] = [f"cuda:{i}" for i in range(args.num_gpus)]
    config["allow_partial_structure"] = args.allow_partial_structure
    config["seed"] = int(args.seed)
    config["deterministic_torch"] = not bool(args.non_deterministic_torch)
    config["websearch_enabled"] = args.websearch_enabled
    config["websearch_top_n"] = args.websearch_top_n
    config["websearch_strategy"] = "hybrid"
    config["websearch_queries_per_candidate"] = 2
    config["websearch_theory_template"] = None
    config["init_data_path"] = args.init_data
    config["init_doc_path"] = args.init_doc
    return config


def setup_logging() -> Path:
    os.environ["PYTHONIOENCODING"] = "utf-8"
    results_dir = project_root / RESULTS_ROOT
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = results_dir / f"run_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return log_file


def _get_all_recorded_rounds(tracker: ProgressTracker) -> list[int]:
    rounds: list[int] = []
    for key in tracker.progress.keys():
        if key.startswith("iteration_"):
            try:
                rounds.append(int(key.split("_")[1]))
            except (ValueError, IndexError):
                continue
    return sorted(set(rounds))


def _reset_all_rounds(tracker: ProgressTracker) -> None:
    for round_num in _get_all_recorded_rounds(tracker):
        tracker.reset_round(round_num)


def resolve_iteration_window(
    tracker: ProgressTracker,
    max_iterations: int | None,
    add_iterations: int | None,
) -> tuple[list[int], int, int, int]:
    completed_rounds = tracker.get_completed_rounds()
    last_completed = max(completed_rounds) if completed_rounds else 0

    if add_iterations is not None:
        effective_target = last_completed + add_iterations
    else:
        effective_target = max_iterations if max_iterations is not None else 3

    start_iteration = max(1, last_completed + 1)
    return completed_rounds, last_completed, start_iteration, effective_target


def main() -> None:
    log_file = setup_logging()
    args = parse_args()
    config = build_config(args)
    param_sheet_path = ensure_param_sheet(project_root / args.params_csv)
    prefill_values, prefill_warnings = load_param_prefill(param_sheet_path, config)
    sheet_overrides, sheet_warnings = load_param_overrides(param_sheet_path, config)
    config.update(sheet_overrides)
    config["agentos_ui_defaults"] = prefill_values
    config["params_csv_path"] = str(param_sheet_path)
    if args.max_iterations is not None:
        config["agentos_default_iterations"] = args.max_iterations
    repro_info = setup_reproducibility(
        seed=int(config.get("seed", 42)),
        deterministic_torch=bool(config.get("deterministic_torch", True)),
    )
    path_config = PathConfig.from_run_mode(
        project_root=project_root,
        run_mode=RUN_MODE,
        init_data_path=args.init_data,
        init_doc_path=args.init_doc,
    )
    config["path_config"] = path_config

    initialize_environment(path_config)

    tracker = ProgressTracker(base_dir=RESULTS_ROOT)
    if args.reset:
        _reset_all_rounds(tracker)

    completed_rounds, last_completed, start_iteration, effective_target = resolve_iteration_window(
        tracker=tracker,
        max_iterations=args.max_iterations,
        add_iterations=args.add_iterations,
    )

    print("=" * 80)
    print("ASLK Agno Workflow Runtime")
    print("=" * 80)
    print(f"log file: {log_file}")
    print(f"params sheet: {param_sheet_path}")
    print(f"params overrides applied: {len(sheet_overrides)}")
    if sheet_overrides:
        print(f"overrides: {sheet_overrides}")
    if prefill_values:
        print(f"params prefill loaded: {len(prefill_values)}")
    if prefill_warnings:
        print(f"params prefill warnings: {prefill_warnings}")
    if sheet_warnings:
        print(f"params warnings: {sheet_warnings}")
    print(f"runtime: {args.runtime}")
    print(f"seed: {repro_info['seed']}")
    print(f"deterministic_torch: {repro_info['deterministic_torch']}")
    print(f"websearch enabled: {args.websearch_enabled}")
    print(f"websearch top-n: {args.websearch_top_n}")
    print(f"requested max-iterations: {args.max_iterations}")
    print(f"requested add-iterations: {args.add_iterations}")
    print(f"completed_rounds={completed_rounds}")
    print(f"last_completed_round={last_completed}")
    print(f"effective_target_iterations={effective_target}")
    print(f"execution_range=[{start_iteration} .. {effective_target}]")
    print(f"top-k: bayes={config['top_k_bayes']}, screen={config['top_k_screen']}")
    if args.runtime == "agentos":
        print(f"agentos endpoint: http://{args.agentos_host}:{args.agentos_port}")
        print("workflow id: aslk-agentos-workflow")
        print(f"agentos default iterations: {config['agentos_default_iterations']}")

    if args.runtime == "agentos":
        serve_agentos(
            config=config,
            tracker=tracker,
            host=args.agentos_host,
            port=args.agentos_port,
        )
        return

    if effective_target < start_iteration:
        print("=" * 80)
        print("No new iterations to run: target already reached.")
        print("=" * 80)
        return

    results = run_workflow(
        config=config,
        tracker=tracker,
        start_iteration=start_iteration,
        max_iterations=effective_target,
    )
    updated_count, persist_warnings = persist_param_values(
        param_sheet_path,
        config,
        keys=PARAM_MEMORY_KEYS,
        enable_for_new_keys=False,
    )
    all_success = all(r.get("success") for r in results)
    print("=" * 80)
    print(f"completed runs: {len(results)}")
    print(f"all success: {all_success}")
    print(f"params remembered to csv: {updated_count}")
    if persist_warnings:
        print(f"params persist warnings: {persist_warnings}")
    print("=" * 80)


if __name__ == "__main__":
    main()
# python main.py --runtime agentos --agentos-host 0.0.0.0 --agentos-port 8000
