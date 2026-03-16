# -*- coding: utf-8 -*-
"""
Workflow Step 6: Update Dataset and Document
"""
import os
import sys
import shutil
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.path_config import PathConfig

# Lazy imports to prevent startup crashes
# from utils.update_dataset import update_dataset
# from agents.success_learner import analyze_success_and_update_theory

THEORY_DOC_NAME = "Theoretical_principle_document.md"

def step_update_data_and_doc(
    iteration_num: int, 
    extraction_result: dict, 
    version: int = 1,
    path_config: Optional[PathConfig] = None,
    # 向后兼容参数
    data_root: str = "llm/data", 
    results_root: str = "llm/results", 
    doc_root: str = "llm/doc", 
    skip_doc_update: bool = False
):
    """
    Step 6: Update dataset and theoretical document.
    
    Args:
        iteration_num: Current iteration number
        extraction_result: Result from step 5
        version: Document version
        data_root: Root directory for data storage
        results_root: Root directory for results storage
        doc_root: Root directory for doc storage
        skip_doc_update: If True, skip LLM-based document update and just copy previous doc
        
    Returns:
        dict: Update status info
    """
    print("=" * 80)
    print(f"Step 6: Update Dataset and Document (Iteration {iteration_num})")
    print("=" * 80)

    def _write_doc_update_error(output_path: Path, message: str) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(message)

    def _write_doc_update_report(output_path: Path, title: str, body: str) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(f"# {title}\n\n{body}\n")
    
    has_success = extraction_result.get('has_success', False)
    has_stable = extraction_result.get('has_stable', False)
    
    updated_data_path = None
    updated_doc_path = None
    
    # === 6.1 Update Dataset ===
    print(f"\n{'='*80}")
    print("6.1 Update Dataset")
    print(f"{'='*80}")
    
    # Lazy Import
    try:
        from utils.update_dataset import update_dataset
    except ImportError as e:
        print(f"[ERROR] Failed to import update_dataset: {e}")
        return {'success': False, 'error': str(e)}

    if has_success:
        print("[INFO] Updating dataset with success examples...")
        success_csv = extraction_result.get('success_deduped_file') or extraction_result.get('success_file')
        
        # 检查 success_csv 是否有效
        if not success_csv:
            print("[WARN] success_csv is None, falling back to copy previous data")
            has_success = False  # 重置标志，让后续逻辑处理
        else:
            # 使用上一轮的data.csv作为基础，查找最近存在的数据
            prev_iteration = iteration_num - 1
            origin_csv = None
            
            # 使用 PathConfig 确定数据路径（如果可用）
            if path_config:
                data_dir_base = path_config.data_root
            else:
                data_dir_base = project_root / data_root
            
            for i in range(prev_iteration, -1, -1):
                candidate = data_dir_base / f"iteration_{i}" / "data.csv"
                if candidate.exists():
                    origin_csv = candidate
                    if i != prev_iteration:
                        print(f"[WARN] iteration_{prev_iteration} data not found, using iteration_{i}/data.csv")
                    break
            
            if origin_csv is None:
                print("[ERROR] No existing data.csv found in any iteration")
                return {'success': False, 'error': 'No data source found'}
            
            # 使用 PathConfig 确定输出目录（如果可用）
            if path_config:
                output_path = path_config.get_iteration_data_path(iteration_num)
                output_dir = output_path.parent  # 获取目录部分
            else:
                output_dir = project_root / data_root / f"iteration_{iteration_num}"
            
            updated_data_path = update_dataset(
                success_csv=success_csv,
                origin_csv=str(origin_csv),
                output_dir=str(output_dir)
            )
            
            if updated_data_path:
                print(f"[SUCCESS] Dataset updated: {updated_data_path}")
            else:
                print("[WARN] Dataset update failed")
            
    if has_stable and not has_success:
        print("[INFO] Updating dataset with stable materials...")
        stable_csv = extraction_result.get('stable_deduped_file') or extraction_result.get('stable_file')
        
        # 检查 stable_csv 是否有效
        if not stable_csv:
            print("[WARN] stable_csv is None, falling back to copy previous data")
            has_stable = False  # 重置标志，让后续逻辑处理
        else:
            # 使用上一轮的data.csv作为基础，查找最近存在的数据
            prev_iteration = iteration_num - 1
            origin_csv = None
            
            # 使用 PathConfig 确定数据路径（如果可用）
            if path_config:
                data_dir_base = path_config.data_root
            else:
                data_dir_base = project_root / data_root
            
            for i in range(prev_iteration, -1, -1):
                candidate = data_dir_base / f"iteration_{i}" / "data.csv"
                if candidate.exists():
                    origin_csv = candidate
                    if i != prev_iteration:
                        print(f"[WARN] iteration_{prev_iteration} data not found, using iteration_{i}/data.csv")
                    break
            
            if origin_csv is None:
                print("[ERROR] No existing data.csv found in any iteration")
                return {'success': False, 'error': 'No data source found'}
            
            # 使用 PathConfig 确定输出目录（如果可用）
            if path_config:
                output_path = path_config.get_iteration_data_path(iteration_num)
                output_dir = output_path.parent  # 获取目录部分
            else:
                output_dir = project_root / data_root / f"iteration_{iteration_num}"
            
            updated_data_path = update_dataset(
                success_csv=stable_csv,
                origin_csv=str(origin_csv),
                output_dir=str(output_dir)
            )
            
            if updated_data_path:
                print(f"[SUCCESS] Dataset updated: {updated_data_path}")
            else:
                print("[WARN] Dataset update failed")
    
    if not has_success and not has_stable:
        print("[INFO] No success or stable materials, copying previous iteration's data...")
        # 即使没有新材料，也需要确保当前iteration有data.csv（复制最近存在的）
        prev_iteration = iteration_num - 1
        
        # 使用 PathConfig 确定路径（如果可用）
        if path_config:
            data_dir_base = path_config.data_root
            output_path = path_config.get_iteration_data_path(iteration_num)
            output_dir = output_path.parent  # 获取目录部分
        else:
            data_dir_base = project_root / data_root
            output_dir = project_root / data_root / f"iteration_{iteration_num}"
            output_path = output_dir / "data.csv"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找最近存在的数据
        source_data = None
        for i in range(prev_iteration, -1, -1):
            candidate = data_dir_base / f"iteration_{i}" / "data.csv"
            if candidate.exists():
                source_data = candidate
                break
        
        if source_data:
            shutil.copy2(source_data, output_path)
            updated_data_path = str(output_path)
            print(f"[SUCCESS] Copied data: {source_data} -> {output_path}")
        else:
            print("[WARN] No data source available to copy")
    
    # === 6.2 Update Theoretical Document ===
    print(f"\n{'='*80}")
    print("6.2 Update Theoretical Document")
    print(f"{'='*80}")
    
    # Lazy Import
    try:
        from agents.success_learner import analyze_success_and_update_theory
    except ImportError as e:
        print(f"[ERROR] Failed to import analyze_success_and_update_theory: {e}")
        # Not fatal for whole step, can fallback
        pass
    
    # 使用 PathConfig 确定输出路径（如果可用）
    if path_config:
        doc_output_dir = path_config.get_iteration_results_path(iteration_num) / "reports"
    else:
        doc_output_dir = project_root / results_root / f"iteration_{iteration_num}" / "reports"
    doc_output_path = doc_output_dir / "llm_theory_update_output.md"
    
    # 优先使用success材料，其次stable材料
    materials_csv = None
    
    print(f"[DEBUG] has_success={has_success}, has_stable={has_stable}")
    print(f"[DEBUG] extraction_result keys: {list(extraction_result.keys())}")
    
    if skip_doc_update:
        print("[INFO] skip_doc_update is True, skipping LLM theory update...")
    elif has_success:
        materials_csv = extraction_result.get('success_deduped_file') or extraction_result.get('success_file')
        print(f"[INFO] Updating theory with success examples...")
        print(f"[DEBUG] success_deduped_file={extraction_result.get('success_deduped_file')}")
        print(f"[DEBUG] success_file={extraction_result.get('success_file')}")
        print(f"[DEBUG] materials_csv={materials_csv}")
    elif has_stable:
        materials_csv = extraction_result.get('stable_deduped_file') or extraction_result.get('stable_file')
        print("[INFO] Updating theory with stable materials...")
        print(f"[DEBUG] stable材料不更新理论文档，仅用于数据集更新")
        materials_csv = None  # 稳定材料不更新理论文档
    
    if materials_csv:
        success_csv = materials_csv
        print(f"\n{'='*80}")
        print(f"[DEBUG] 准备调用LLM更新理论文档")
        print(f"[DEBUG] 成功案例文件: {success_csv}")
        print(f"{'='*80}\n")
        
        try:
            # Need to re-import locally if not imported above? No, analyze_success_and_update_theory is lazily imported.
            # But wait, I put the import inside a try block but if it fails, the variable won't be defined.
            # Let's handle that properly. 
            if 'analyze_success_and_update_theory' not in locals():
                 raise ImportError("Module not loaded")

            result = analyze_success_and_update_theory(
                success_csv_path=success_csv,
                iteration_num=iteration_num,
                version=version,
                results_root=results_root,
                doc_root=doc_root
            )
            
            if result:
                print(f"[SUCCESS] Theory document updated")
                print(f"   Report: {result.get('analysis_report', 'N/A')}")
                print(f"   Doc: {result.get('updated_doc', 'N/A')}")
                updated_doc_path = result.get('updated_doc')
                materials_csv = None  # 标记已处理
            else:
                print("\n" + "="*80)
                print("⚠️  严重警告: 有成功案例但理论文档未能更新！")
                print("="*80)
                print(f"成功案例文件: {success_csv}")
                print("LLM调用返回空结果")
                print("="*80 + "\n")
                error_message = (
                    "LLM theory update returned empty result.\n"
                    f"success_csv: {success_csv}\n"
                )
                print("[ERROR] Theory update failed, stopping iteration")
                _write_doc_update_error(doc_output_path, error_message)
                return {'success': False, 'error': error_message}
                
        except Exception as e:
            print("\n" + "="*80)
            print("❌ 严重错误: 有成功案例但理论文档更新异常！")
            print("="*80)
            print(f"成功案例文件: {success_csv}")
            print(f"异常: {e}")
            import traceback
            traceback.print_exc()
            print("="*80 + "\n")
            print(f"[ERROR] Theory update exception: {e}")
            error_message = (
                "LLM theory update raised exception.\n"
                f"success_csv: {success_csv}\n"
                f"exception: {e}\n"
            )
            _write_doc_update_error(doc_output_path, error_message)
            return {'success': False, 'error': error_message}
    
    if not materials_csv and not updated_doc_path:
        print("[INFO] Copying previous theory document...")
        
        # 使用 PathConfig 确定文档路径（如果可用）
        if path_config:
            doc_dir_base = path_config.doc_root
        else:
            doc_dir_base = project_root / doc_root
        
        # 优先使用上一轮results中的文档
        if iteration_num == 1:
            prev_doc = doc_dir_base / "v0.0.0" / THEORY_DOC_NAME
        else:
            # 优先从results目录查找
            prev_doc_dir = doc_dir_base / f"v0.0.{iteration_num-1}" / THEORY_DOC_NAME
            
            if prev_doc_dir.exists():
                prev_doc = prev_doc_dir
                print(f"[INFO] Using previous theory document: {prev_doc_dir}")
            else:
                # 回退到v0.0.0
                prev_doc = doc_dir_base / "v0.0.0" / THEORY_DOC_NAME
                print(f"[WARN] No previous doc found, falling back to v0.0.0")
        
        if prev_doc.exists():
            print(f"[SUCCESS] Reusing previous doc: {prev_doc}")
            updated_doc_path = str(prev_doc)
            _write_doc_update_report(
                doc_output_path,
                "Theory Document Update Skipped",
                (
                    f"Iteration {iteration_num} reused the previous theory document.\n\n"
                    f"Previous doc: `{prev_doc}`\n\n"
                    f"Reason: no success materials were available for LLM-based theory update."
                ),
            )
        else:
            print(f"[WARN] Previous doc not found: {prev_doc}")
            fallback_doc = project_root / "assets" / "theory.md"
            if fallback_doc.exists():
                print(f"[SUCCESS] Reusing fallback doc: {fallback_doc}")
                updated_doc_path = str(fallback_doc)
                _write_doc_update_report(
                    doc_output_path,
                    "Theory Document Update Fallback",
                    (
                        f"Iteration {iteration_num} fell back to the bundled theory template.\n\n"
                        f"Fallback doc: `{fallback_doc}`\n\n"
                        f"Reason: previous versioned theory document was not found."
                    ),
                )
    
    # === 6.3 Sync to doc/v0.0.{iteration_num} ===
    if updated_doc_path:
        # 使用 PathConfig 确定文档路径（如果可用）
        if path_config:
            versioned_doc_dir = path_config.doc_root / f"v0.0.{iteration_num}"
        else:
            versioned_doc_dir = project_root / doc_root / f"v0.0.{iteration_num}"
        
        versioned_doc_path = versioned_doc_dir / THEORY_DOC_NAME
        versioned_doc_dir.mkdir(parents=True, exist_ok=True)
        # 检查是否是同一个文件（避免 SameFileError）
        src_path = Path(updated_doc_path).resolve()
        dst_path = versioned_doc_path.resolve()
        if src_path != dst_path:
            shutil.copy2(updated_doc_path, versioned_doc_path)
            print(f"[SUCCESS] Synced to versioned doc: {versioned_doc_path}")
        else:
            print(f"[INFO] Doc already at versioned location: {versioned_doc_path}")
        updated_doc_path = str(versioned_doc_path)
    
    return {
        'success': True,
        'has_success_materials': has_success,
        'updated_data_path': updated_data_path,
        'updated_doc_path': updated_doc_path
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=1)
    args = parser.parse_args()
    
    mock_extraction_result = {
        'success': True,
        'has_success': False,
        'has_stable': True,
        'stable_file': 'results/iteration_1/success_examples/stable_materials.csv'
    }
    
    result = step_update_data_and_doc(args.iteration, mock_extraction_result)
    print(f"\nResult: {result}")
