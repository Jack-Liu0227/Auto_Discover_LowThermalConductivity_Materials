# -*- coding: utf-8 -*-
"""
数据回滚工具
用于将失败的iteration材料返回到数据集以便重新训练
"""

import pandas as pd
from pathlib import Path
import logging
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)

def rollback_failed_iteration(
    iteration_num: int,
    project_root: str = ".",
    results_root: str = "results",
    data_root: str = "data"
) -> dict:
    """
    检查iteration是否失败，并将ai_selected_materials.csv回滚到数据集
    
    失败条件:
    - stable.csv为空或不存在
    - success.csv为空或不存在
    
    Args:
        iteration_num: 当前迭代轮次
        project_root: 项目根目录
        results_root: 结果存储根目录
        data_root: 数据存储根目录
        
    Returns:
        dict: 包含操作结果的字典
    """
    project_root = Path(project_root)
    iteration_dir = project_root / results_root / f"iteration_{iteration_num}"
    success_dir = iteration_dir / "success_examples"
    selected_dir = iteration_dir / "selected_results"
    data_dir = project_root / data_root
    
    # 检查关键文件
    stable_csv = success_dir / "stable_materials.csv"
    success_csv = success_dir / "success_materials.csv"
    selected_csv = selected_dir / "ai_selected_materials.csv"
    
    # 去重后的文件路径
    stable_csv_dedup = success_dir / "stable_materials_deduped.csv"
    success_csv_dedup = success_dir / "success_materials_deduped.csv"
    
    # 判断是否失败
    is_failed = True
    failure_reasons = []
    
    # 检查并去重 stable.csv
    if stable_csv.exists():
        df_stable = pd.read_csv(stable_csv)
        if len(df_stable) > 0:
            logger.info(f"📊 stable.csv 存在: {len(df_stable)} 条原始记录")
            
            # 执行去重
            try:
                from agents.deduplicate_success import deduplicate_success_materials
                logger.info("🔄 正在对 stable_materials.csv 进行去重...")
                deduplicate_success_materials(str(stable_csv), str(stable_csv_dedup))
                
                if stable_csv_dedup.exists():
                    df_stable_dedup = pd.read_csv(stable_csv_dedup)
                    logger.info(f"✅ stable.csv 去重完成: {len(df_stable)} -> {len(df_stable_dedup)} 条记录")
                    
                    if len(df_stable_dedup) > 0:
                        is_failed = False
                    else:
                        failure_reasons.append("stable.csv去重后为空")
                else:
                    logger.warning("⚠️ 去重文件未生成，使用原始文件")
                    if len(df_stable) > 0:
                        is_failed = False
                    else:
                        failure_reasons.append("stable.csv为空")
            except Exception as e:
                logger.warning(f"⚠️ stable.csv 去重失败: {e}，使用原始文件")
                if len(df_stable) > 0:
                    is_failed = False
                else:
                    failure_reasons.append("stable.csv为空")
        else:
            failure_reasons.append("stable.csv为空")
    else:
        failure_reasons.append("stable.csv不存在")
    
    # 检查并去重 success.csv
    if success_csv.exists():
        df_success = pd.read_csv(success_csv)
        if len(df_success) > 0:
            logger.info(f"📊 success.csv 存在: {len(df_success)} 条原始记录")
            
            # 执行去重
            try:
                from agents.deduplicate_success import deduplicate_success_materials
                logger.info("🔄 正在对 success_materials.csv 进行去重...")
                deduplicate_success_materials(str(success_csv), str(success_csv_dedup))
                
                if success_csv_dedup.exists():
                    df_success_dedup = pd.read_csv(success_csv_dedup)
                    logger.info(f"✅ success.csv 去重完成: {len(df_success)} -> {len(df_success_dedup)} 条记录")
                    
                    if len(df_success_dedup) > 0:
                        is_failed = False
                    else:
                        failure_reasons.append("success.csv去重后为空")
                else:
                    logger.warning("⚠️ 去重文件未生成，使用原始文件")
                    if len(df_success) > 0:
                        is_failed = False
                    else:
                        failure_reasons.append("success.csv为空")
            except Exception as e:
                logger.warning(f"⚠️ success.csv 去重失败: {e}，使用原始文件")
                if len(df_success) > 0:
                    is_failed = False
                else:
                    failure_reasons.append("success.csv为空")
        else:
            failure_reasons.append("success.csv为空")
    else:
        failure_reasons.append("success.csv不存在")
    
    result = {
        "iteration": iteration_num,
        "is_failed": is_failed,
        "failure_reasons": failure_reasons,
        "rollback_performed": False,
        "backup_path": None,
        "new_data_csv": None
    }
    
    if not is_failed:
        logger.info(f"🎉 Iteration {iteration_num} 成功，无需回滚")
        return result
    
    # 失败时进行回滚
    logger.warning(f"❌ Iteration {iteration_num} 失败: {', '.join(failure_reasons)}")
    
    if not selected_csv.exists():
        logger.error(f"❌ 找不到 ai_selected_materials.csv: {selected_csv}")
        result["error"] = "ai_selected_materials.csv不存在"
        return result
    
    # 读取要回滚的数据
    df_selected = pd.read_csv(selected_csv)
    logger.info(f"📊 从 ai_selected_materials.csv 读取了 {len(df_selected)} 条材料")
    
    # 确定原始数据源 - 查找最近存在的数据文件
    prev_iteration = iteration_num - 1
    origin_data_csv = None
    for i in range(prev_iteration, -1, -1):
        candidate = data_dir / f"iteration_{i}" / "data.csv"
        if candidate.exists():
            origin_data_csv = candidate
            if i != prev_iteration:
                logger.warning(f"⚠️ iteration_{prev_iteration} 数据不存在，使用 iteration_{i}/data.csv")
            break
    
    if origin_data_csv is None:
        logger.error("❌ 找不到任何可用的数据文件")
        result["error"] = "找不到任何iteration的data.csv"
        return result
    
    # 读取原始数据
    df_origin = pd.read_csv(origin_data_csv)
    logger.info(f"📊 从 {origin_data_csv} 读取了 {len(df_origin)} 条原始数据")
    
    # 合并数据（去重）
    # 检查列名并统一
    formula_col_selected = None
    formula_col_origin = None
    
    # 查找公式列（支持 'formula' 或 'Formula'）
    for col in df_selected.columns:
        if col.lower() == 'formula':
            formula_col_selected = col
            break
    
    for col in df_origin.columns:
        if col.lower() == 'formula':
            formula_col_origin = col
            break
    
    if formula_col_selected is None:
        logger.error("❌ ai_selected_materials.csv 缺少 'formula' 列")
        result["error"] = "缺少formula列"
        return result
    
    if formula_col_origin is None:
        logger.error("❌ 数据文件缺少 'Formula' 列")
        result["error"] = "缺少Formula列"
        return result
    
    # 统一列名为 'Formula'
    if formula_col_selected != 'Formula':
        df_selected = df_selected.rename(columns={formula_col_selected: 'Formula'})
    if formula_col_origin != 'Formula':
        df_origin = df_origin.rename(columns={formula_col_origin: 'Formula'})
    
    # 确保列一致 - 只保留df_origin中存在的列，但保留Formula
    common_cols = ['Formula']  # 必须保留Formula列
    for col in df_selected.columns:
        if col != 'Formula' and col in df_origin.columns:
            common_cols.append(col)
    
    # 如果df_selected只有Formula列，填充其他必需的列为NaN
    if len(common_cols) == 1:
        logger.warning("⚠️ ai_selected_materials只有Formula列，将填充其他列为NaN")
        for col in df_origin.columns:
            if col not in df_selected.columns:
                df_selected[col] = None
        common_cols = df_origin.columns.tolist()
    
    df_selected = df_selected[common_cols]
    
    # 合并并去重
    df_combined = pd.concat([df_origin, df_selected], ignore_index=True)
    
    # 按Formula去重，保留第一次出现的
    original_count = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['Formula'], keep='first')
    removed_count = original_count - len(df_combined)
    
    logger.info(f"📊 合并后总计 {len(df_combined)} 条数据 (去除 {removed_count} 条重复)")
    
    # 备份原始数据文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = origin_data_csv.with_name(f"{origin_data_csv.stem}_backup_{timestamp}.csv")
    shutil.copy2(origin_data_csv, backup_path)
    logger.info(f"💾 已备份原始数据到: {backup_path}")
    result["backup_path"] = str(backup_path)
    
    # 保存新的 data.csv 到 data/iteration_{iteration_num}/data.csv
    # 注意: data_dir 已经是 project_root / data_root
    new_data_dir = data_dir / f"iteration_{iteration_num}"
    new_data_dir.mkdir(parents=True, exist_ok=True)
    new_data_csv = new_data_dir / "data.csv"
    
    df_combined.to_csv(new_data_csv, index=False, encoding='utf-8-sig')
    logger.info(f"✅ 已保存新数据到: {new_data_csv}")
    result["new_data_csv"] = str(new_data_csv)
    result["rollback_performed"] = True
    
    # 同时更新来源数据文件
    df_combined.to_csv(origin_data_csv, index=False, encoding='utf-8-sig')
    logger.info(f"✅ 已更新数据文件: {origin_data_csv}")
    
    return result


def check_and_rollback_all_iterations(
    max_iteration: int,
    project_root: str = ".",
    results_root: str = "results",
    data_root: str = "data"
) -> list:
    """
    检查所有iteration并回滚失败的
    
    Args:
        max_iteration: 最大迭代轮次
        project_root: 项目根目录
        results_root: 结果存储根目录
        data_root: 数据存储根目录
        
    Returns:
        list: 所有回滚结果
    """
    results = []
    
    for i in range(1, max_iteration + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"检查 Iteration {i}")
        logger.info(f"{'='*80}")
        
        result = rollback_failed_iteration(
            iteration_num=i,
            project_root=project_root,
            results_root=results_root,
            data_root=data_root
        )
        results.append(result)
        
        if result.get("rollback_performed"):
            logger.info(f"✅ Iteration {i} 已回滚")
        elif result.get("is_failed"):
            logger.error(f"❌ Iteration {i} 回滚失败: {result.get('error', 'Unknown error')}")
        else:
            logger.info(f"✨ Iteration {i} 成功，无需回滚")
    
    return results


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 示例使用
    import sys
    
    if len(sys.argv) > 1:
        iteration = int(sys.argv[1])
        result = rollback_failed_iteration(iteration)
        print("\n回滚结果:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print("用法: python rollback_data.py <iteration_number>")
        print("示例: python rollback_data.py 1")
