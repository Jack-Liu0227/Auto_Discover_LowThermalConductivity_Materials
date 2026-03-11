"""
为已有的 success/stable materials CSV 文件添加 formula 列

从 Structure 列中提取 Full Formula (例如 "Ag5 Sb6 As4")
"""

import os
import re
import pandas as pd
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def extract_formula_from_structure(structure_str: str) -> str:
    """
    从 Structure 字符串中提取 Full Formula
    
    示例输入:
        "Full Formula (Ag5 Sb6 As4)\nReduced Formula: ..."
    返回:
        "Ag5 Sb6 As4"
    """
    if not structure_str or pd.isna(structure_str) or structure_str == "N/A":
        return ""
    
    # 匹配 "Full Formula (XXX)" 格式
    match = re.search(r'Full Formula \(([^)]+)\)', str(structure_str))
    if match:
        return match.group(1).strip()
    return ""


def add_formula_column_to_csv(csv_path: str, output_path: Optional[str] = None) -> str:
    """
    为 CSV 文件添加 formula 列
    
    Args:
        csv_path: 输入 CSV 文件路径
        output_path: 输出 CSV 文件路径，如果为 None 则覆盖原文件
        
    Returns:
        输出文件路径
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")
    
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    if 'Structure' not in df.columns:
        logger.warning(f"CSV 文件缺少 'Structure' 列: {csv_path}")
        return csv_path
    
    # 提取 formula
    df['formula'] = df['Structure'].apply(extract_formula_from_structure)
    
    # 调整列顺序：将 formula 放在 '组分' 后面
    cols = list(df.columns)
    if '组分' in cols and 'formula' in cols:
        cols.remove('formula')
        idx = cols.index('组分') + 1
        cols.insert(idx, 'formula')
        df = df[cols]
    
    # 保存
    if output_path is None:
        output_path = csv_path
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"已添加 formula 列: {output_path}")
    
    return output_path


def process_iteration_success_files(iteration: int = 1, base_dir: str = "results"):
    """
    处理指定迭代的所有 success/stable CSV 文件
    
    Args:
        iteration: 迭代编号
        base_dir: 结果目录基础路径
    """
    success_dir = Path(base_dir) / f"iteration_{iteration}" / "success_examples"
    
    if not success_dir.exists():
        logger.warning(f"目录不存在: {success_dir}")
        return
    
    # 要处理的文件
    target_files = [
        "success_materials.csv",
        "success_materials_deduped.csv",
        "stable_materials.csv",
        "stable_materials_deduped.csv"
    ]
    
    for filename in target_files:
        filepath = success_dir / filename
        if filepath.exists():
            try:
                add_formula_column_to_csv(str(filepath))
                print(f"✅ 处理完成: {filepath}")
            except Exception as e:
                print(f"❌ 处理失败: {filepath} - {e}")
        else:
            print(f"⚠️  文件不存在: {filepath}")


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        # 处理指定的 CSV 文件
        for csv_path in sys.argv[1:]:
            try:
                add_formula_column_to_csv(csv_path)
                print(f"✅ 处理完成: {csv_path}")
            except Exception as e:
                print(f"❌ 处理失败: {csv_path} - {e}")
    else:
        # 默认处理 iteration_1 和 iteration_2
        print("处理 iteration_1...")
        process_iteration_success_files(1)
        print("\n处理 iteration_2...")
        process_iteration_success_files(2)
