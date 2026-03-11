# -*- coding: utf-8 -*-
"""
Workflow Step 1: Train GPR Model
使用上一轮数据训练模型,保存到 models/GPR/iteration_{iteration_num-1}
"""
import os
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from models.train_gpr_model import train_gpr_model
from utils.path_config import PathConfig


def step_train_model(
    iteration_num: int, 
    path_config: Optional[PathConfig] = None,
    # 向后兼容参数
    data_root: str = "data_llm", 
    models_root: str = "models/GPR"
):
    """
    步骤1: 使用上一轮数据训练GPR模型
    
    Args:
        iteration_num: 当前迭代轮次
        path_config: PathConfig对象，用于统一路径管理（优先使用）
        data_root: 数据存储根目录 (相对于项目根目录，向后兼容)
        models_root: 模型存储根目录 (相对于项目根目录，向后兼容)
        
    Returns:
        dict: 包含模型路径和训练状态的信息
    """
    print("=" * 80)
    print(f"步骤 1: 训练 GPR 模型 (Iteration {iteration_num})")
    print("=" * 80)
    
    # 确定数据路径 - 统一使用 iteration_N/data.csv 格式
    # 查找最近存在的数据文件
    prev_iteration = iteration_num - 1
    data_path = None
    
    # 使用 PathConfig 确定基础路径（如果可用）
    if path_config:
        data_dir_base = path_config.data_root
    else:
        data_dir_base = project_root / data_root
    
    for i in range(prev_iteration, -1, -1):
        candidate = data_dir_base / f"iteration_{i}" / "data.csv"
        if candidate.exists():
            data_path = candidate
            if i != prev_iteration:
                print(f"⚠️ iteration_{prev_iteration} 数据不存在，使用 iteration_{i}/data.csv")
            break
    
    if data_path is None:
        print("❌ 找不到任何可用的数据文件")
        return {
            'success': False,
            'error': 'No data file found in any iteration'
        }
    
    # 确定模型保存路径 (使用当前迭代前一轮的数据训练)
    if path_config:
        model_dir = path_config.get_iteration_model_path(prev_iteration)
    else:
        model_dir = project_root / models_root / f"iteration_{prev_iteration}"
    
    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        return {
            'success': False,
            'error': f'Data file not found: {data_path}'
        }
    
    print(f"📖 使用数据: {data_path}")
    print(f"💾 模型保存到: {model_dir}")
    
    try:
        # 训练模型
        train_gpr_model(str(data_path), str(model_dir))
        
        # 验证模型文件
        model_file = model_dir / "gpr_thermal_conductivity.joblib"
        scaler_file = model_dir / "gpr_scaler.joblib"
        
        if model_file.exists() and scaler_file.exists():
            print(f"✅ 模型训练成功!")
            return {
                'success': True,
                'model_dir': str(model_dir),
                'model_file': str(model_file),
                'scaler_file': str(scaler_file)
            }
        else:
            print(f"❌ 训练完成但文件不存在")
            return {
                'success': False,
                'error': 'Model files not created'
            }
            
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # 测试
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=1)
    args = parser.parse_args()
    
    result = step_train_model(args.iteration)
    print(f"\n结果: {result}")
