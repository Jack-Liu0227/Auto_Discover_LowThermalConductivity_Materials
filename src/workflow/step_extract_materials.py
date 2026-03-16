# -*- coding: utf-8 -*-
"""
Workflow Step 5: Extract Success and Stable Materials
Extract success and stable materials, and deduplicate.
"""
import os
import sys
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from tools.success_extractor import extract_success_materials
from utils.config_loader import ensure_theory_doc_sync, get_effective_thresholds
# from agents.deduplicate_success import deduplicate_success_materials - 移至函数内部导入


def step_extract_materials(
    iteration_num: int,
    k_threshold: float | None = None,
    imag_tol: float | None = None,
    results_root: str = "results",
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
    
    relax_dir = project_root / results_root / f"iteration_{iteration_num}" / "MyRelaxStructure"
    output_dir = project_root / results_root / f"iteration_{iteration_num}" / "success_examples"
    
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
        success_csv = extract_success_materials(
            myrelax_dir=str(relax_dir),
            output_dir=str(output_dir),
            k_threshold=k_threshold,
            imag_tol=imag_tol,
        )
        
        if not success_csv:
            print("[WARN] No success materials file generated.")
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

        # Deduplicate success materials
        if has_success:
            print(f"\nProcessing success materials...")
            df_success = pd.read_csv(success_file)
            print(f"   Original count: {len(df_success)}")
            
            # Deduplicate
            output_file = str(success_file).replace('.csv', '_deduped.csv')
            deduped_file = deduplicate_success_materials(str(success_file), output_file)
            if deduped_file:
                df_deduped = pd.read_csv(deduped_file)
                print(f"   Deduped count: {len(df_deduped)}")
                print(f"   [OK] Deduped file: {deduped_file}")
            else:
                print("   [WARN] Deduplication failed, using original file.")
                deduped_file = str(success_file)
        
        # Deduplicate stable materials
        if has_stable:
            print(f"\nProcessing stable materials...")
            df_stable = pd.read_csv(stable_file)
            print(f"   Original count: {len(df_stable)}")
            
            # Deduplicate
            output_file = str(stable_file).replace('.csv', '_deduped.csv')
            deduped_stable_file = deduplicate_success_materials(str(stable_file), output_file)
            if deduped_stable_file:
                df_deduped_stable = pd.read_csv(deduped_stable_file)
                print(f"   Deduped count: {len(df_deduped_stable)}")
                print(f"   [OK] Deduped file: {deduped_stable_file}")
            else:
                print("   [WARN] Deduplication failed, using original file.")
                deduped_stable_file = str(stable_file)

        novelty_result = {}
        try:
            from agents.final_structure_novelty import compare_final_materials_to_databases

            novelty_result = compare_final_materials_to_databases(
                iteration_num=iteration_num,
                results_root=str(project_root / results_root),
                success_deduped_file=deduped_file if has_success else None,
                stable_deduped_file=deduped_stable_file if has_stable else None,
                limit_per_db=5,
            )
            print(f"[OK] Final DB novelty file: {novelty_result.get('final_db_novelty_file')}")
            print(f"[OK] Final DB novelty summary: {novelty_result.get('novelty_summary')}")
        except Exception as exc:
            print(f"[WARN] Final DB novelty comparison failed: {exc}")

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
