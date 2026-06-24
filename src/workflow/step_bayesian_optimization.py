# -*- coding: utf-8 -*-
"""
Workflow Step 2: Bayesian Optimization.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from generators.acquisition_ei import main as run_acquisition
from utils.config_loader import ensure_theory_doc_sync, get_workflow_search_prior
from utils.path_config import PathConfig


def step_bayesian_optimization(
    iteration_num: int,
    xi: float = 0.01,
    n_samples: int = 100,
    n_top: int = 10,
    initial_samples: list | None = None,
    seed: int | None = None,
    seed_stride: int = 1000,
    path_config: Optional[PathConfig] = None,
    models_root: str = "models/GPR",
    results_root: str = "results",
):
    """
    Run the Bayesian optimization step with the trained GPR model.

    Args:
        iteration_num: Current iteration number.
        xi: EI exploration parameter.
        n_samples: Number of compositions sampled this round.
        n_top: Number of top candidates to keep.
        initial_samples: Optional warm-start samples from previous success cases.
        seed: Optional base random seed.
        seed_stride: Iteration-specific offset applied to the seed.
        path_config: Preferred path configuration object.
        models_root: Backward-compatible models root.
        results_root: Backward-compatible results root.

    Returns:
        dict: Candidate selection result.
    """
    print("=" * 80)
    print(f"Step 2: Bayesian Optimization (Iteration {iteration_num})")
    print("=" * 80)

    try:
        ensure_theory_doc_sync()
        search_prior = get_workflow_search_prior()
    except Exception as exc:
        print(f"[ERROR] Theory/config sync check failed: {exc}")
        return {
            "success": False,
            "error": f"Theory/config sync check failed: {exc}",
        }

    if path_config:
        model_path = path_config.get_model_file_path(iteration_num - 1)
        resolved_models_root = str(path_config.models_root.relative_to(path_config.project_root))
        resolved_results_root = str(path_config.results_root.relative_to(path_config.project_root))
    else:
        prev_iteration = iteration_num - 1
        model_dir = project_root / models_root / f"iteration_{prev_iteration}"
        model_path = model_dir / "gpr_thermal_conductivity.joblib"
        resolved_models_root = models_root
        resolved_results_root = results_root

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return {
            "success": False,
            "error": f"Model file not found: {model_path}",
        }

    print(f"Using model: {model_path}")
    print(f"   - EI exploration parameter: {xi}")
    print(f"   - Sampling count: {n_samples}")
    print(f"   - Top candidates kept: {n_top}")
    print(
        "   - Search prior: "
        f"{search_prior['schema_type']} | "
        f"max_atoms={search_prior['max_atoms']} | "
        f"elements={len(search_prior['allowed_elements'])}"
    )
    if initial_samples:
        print(f"   - Warm-start samples: {len(initial_samples)}")

    try:
        results = run_acquisition(
            xi=xi,
            n_samples=n_samples,
            iteration_num=iteration_num,
            model_path=str(model_path),
            initial_samples=initial_samples,
            n_top=n_top,
            seed=(int(seed) + int(seed_stride) * int(iteration_num)) if seed is not None else None,
            results_root=resolved_results_root,
            models_root=resolved_models_root,
        )

        print(f"[OK] Bayesian optimization finished with {len(results)} candidates.")
        top_materials = results[:n_top]
        return {
            "success": True,
            "n_materials": len(results),
            "n_top": n_top,
            "top_materials": top_materials,
            "top10_materials": results[:10],
            "all_materials": results,
        }
    except Exception as exc:
        print(f"[ERROR] Bayesian optimization failed: {exc}")
        return {
            "success": False,
            "error": str(exc),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--xi", type=float, default=0.01)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    result = step_bayesian_optimization(args.iteration, args.xi, args.samples)
    print(f"\nResult: {result}")
