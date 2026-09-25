# -*- coding: utf-8 -*-
"""
Workflow Step 5: Extract Success and Stable Materials
Extract success and stable materials, and deduplicate.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from tools.success_extractor import extract_success_materials
from utils.config_loader import ensure_theory_doc_sync, get_effective_thresholds
from utils.workflow_resume import load_active_material_formulas
# from agents.deduplicate_success import deduplicate_success_materials - 移至函数内部导入


def step_extract_materials(
    iteration_num: int,
    k_threshold: float | None = None,
    imag_tol: float | None = None,
    results_root: str = "results",
    path_config=None,
    postprocess_workers: int = 1,
    novelty_workers: int = 1,
):
    """
    Step 5: Extract success and stable materials, and deduplicate.
    
    Args:
        iteration_num: Current iteration number
        k_threshold: Thermal conductivity threshold
        imag_tol: Dynamic stability threshold on Min_Frequency (THz)
        results_root: Root directory for results
        
    Returns:
        dict: Information about success and stable materials
    """
    print("=" * 80)
    print(f"Step 5: Extract Success and Stable Materials (Iteration {iteration_num})")
    print("=" * 80)

    try:
        ensure_theory_doc_sync()
        thresholds = get_effective_thresholds()
    except Exception as exc:
        print(f"[ERROR] Theory/config sync check failed: {exc}")
        return {
            'success': False,
            'error': f'Theory/config sync check failed: {exc}'
        }

    if k_threshold is None:
        k_threshold = thresholds["thermal_conductivity"]
    if imag_tol is None:
        imag_tol = thresholds["dynamic_min_frequency"]
    
    # 延迟导入以避免模块冲突
    from agents.deduplicate_success import deduplicate_success_materials
    
    if path_config is not None:
        iteration_results_dir = path_config.get_iteration_results_path(iteration_num)
        relax_dir = iteration_results_dir / "MyRelaxStructure"
        output_dir = iteration_results_dir / "success_examples"
    else:
        relax_dir = project_root / results_root / f"iteration_{iteration_num}" / "MyRelaxStructure"
        output_dir = project_root / results_root / f"iteration_{iteration_num}" / "success_examples"

    def _write_extraction_status(payload: dict) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        status_path = output_dir / "extraction_status.json"
        temp_path = status_path.with_name(f".{status_path.name}.{os.getpid()}.tmp")
        try:
            temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(temp_path, status_path)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
    
    if not relax_dir.exists():
        print(f"[ERROR] Relaxation directory not found: {relax_dir}")
        return {
            'success': False,
            'error': 'Relaxation directory not found'
        }
    
    print(f"Relaxation Dir: {relax_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"K Threshold: {k_threshold} W/(m-K)")
    print(f"Imaginary tolerance (Min_Frequency >=): {imag_tol} THz")
    
    try:
        # Extract materials
        active_formulas = load_active_material_formulas(
            path_config.results_root if path_config is not None else project_root / results_root,
            iteration_num,
        )
        success_csv = extract_success_materials(
            myrelax_dir=str(relax_dir),
            output_dir=str(output_dir),
            k_threshold=k_threshold,
            imag_tol=imag_tol,
            allowed_formulas=active_formulas,
            max_workers=max(1, int(postprocess_workers)),
        )
        
        if not success_csv:
            print("[WARN] No success materials file generated.")
            _write_extraction_status(
                {
                    "iteration": iteration_num,
                    "status": "no_materials",
                    "has_success": False,
                    "has_stable": False,
                }
            )
            # 返回 no_materials=True 表示没有材料但不是错误
            return {
                'success': False,
                'no_materials': True,  # 特殊标志：没有材料但不是错误
                'has_success': False,
                'has_stable': False
            }
        
        print(f"[OK] Materials extracted to: {success_csv}")
        
        # Check files
        success_file = output_dir / "success_materials.csv"
        stable_file = output_dir / "stable_materials.csv"
        
        has_success = success_file.exists()
        has_stable = stable_file.exists()
        
        deduped_file = None
        deduped_stable_file = None

        dedup_jobs: list[tuple[str, Path, str]] = []
        if has_success:
            print("\nProcessing success materials...")
            df_success = pd.read_csv(success_file)
            print(f"   Original count: {len(df_success)}")
            dedup_jobs.append(
                ("success", success_file, str(success_file).replace(".csv", "_deduped.csv"))
            )
        if has_stable:
            print("\nProcessing stable materials...")
            df_stable = pd.read_csv(stable_file)
            print(f"   Original count: {len(df_stable)}")
            dedup_jobs.append(
                ("stable", stable_file, str(stable_file).replace(".csv", "_deduped.csv"))
            )

        dedup_results: dict[str, str | None] = {}
        worker_count = max(1, min(int(postprocess_workers), len(dedup_jobs))) if dedup_jobs else 1
        if worker_count == 1:
            for kind, input_file, output_file in dedup_jobs:
                dedup_results[kind] = deduplicate_success_materials(str(input_file), output_file)
        else:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="dedup-material-set",
            ) as executor:
                futures = {
                    executor.submit(deduplicate_success_materials, str(input_file), output_file): kind
                    for kind, input_file, output_file in dedup_jobs
                }
                for future in as_completed(futures):
                    kind = futures[future]
                    try:
                        dedup_results[kind] = future.result()
                    except Exception as exc:
                        print(f"   [WARN] {kind} deduplication failed: {exc}")
                        dedup_results[kind] = None

        for kind in ("success", "stable"):
            result_path = dedup_results.get(kind)
            source_file = success_file if kind == "success" else stable_file
            if not source_file.exists():
                continue
            if result_path:
                deduped_df = pd.read_csv(result_path)
                print(f"   {kind} deduped count: {len(deduped_df)}")
                print(f"   [OK] Deduped file: {result_path}")
            else:
                print(f"   [WARN] {kind} deduplication failed, using original file.")
                result_path = str(source_file)
            if kind == "success":
                deduped_file = result_path
            else:
                deduped_stable_file = result_path

        novelty_result = {}
        try:
            from agents.final_structure_novelty import compare_final_materials_to_databases

            novelty_result = compare_final_materials_to_databases(
                iteration_num=iteration_num,
                results_root=str(project_root / results_root),
                success_deduped_file=deduped_file if has_success else None,
                stable_deduped_file=deduped_stable_file if has_stable else None,
                limit_per_db=5,
                max_workers=max(1, int(novelty_workers)),
            )
            print(f"[OK] Final DB novelty file: {novelty_result.get('final_db_novelty_file')}")
            print(f"[OK] Final DB novelty summary: {novelty_result.get('novelty_summary')}")
        except Exception as exc:
            print(f"[WARN] Final DB novelty comparison failed: {exc}")

        _write_extraction_status(
            {
                "iteration": iteration_num,
                "status": "completed",
                "has_success": has_success,
                "has_stable": has_stable,
                "success_file": str(success_file) if has_success else None,
                "stable_file": str(stable_file) if has_stable else None,
            }
        )
        return {
            'success': True,
            'has_success': has_success,
            'has_stable': has_stable,
            'success_file': str(success_file) if has_success else None,
            'stable_file': str(stable_file) if has_stable else None,
            'success_deduped_file': deduped_file if has_success else None,
            'stable_deduped_file': deduped_stable_file if has_stable else None,
            'final_db_novelty_file': novelty_result.get('final_db_novelty_file'),
            'final_db_novelty_json': novelty_result.get('final_db_novelty_json'),
            'final_db_novelty_summary_file': novelty_result.get('final_db_novelty_summary_file'),
            'novelty_summary': novelty_result.get('novelty_summary', {}),
        }
        
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=1)
    parser.add_argument('--k-threshold', type=float, default=None)
    parser.add_argument('--imag-tol', type=float, default=None)
    args = parser.parse_args()
    
    result = step_extract_materials(args.iteration, args.k_threshold, args.imag_tol)
    print(f"\nResult: {result}")
