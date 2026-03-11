# -*- coding: utf-8 -*-
"""
Workflow Step 2: Bayesian Optimization
使用训练好的模型进行贝叶斯优化
"""
import os
import sys
from pathlib import Path
from typing import Union, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from generators.acquisition_ei import main as run_acquisition
from utils.path_config import PathConfig


def step_bayesian_optimization(
    iteration_num: int, 
    xi: float = 0.01, 
    n_samples: int = 100, 
    n_top: int = 10, 
    initial_samples: list = None,
    seed: int | None = None,
    seed_stride: int = 1000,
    path_config: Optional[PathConfig] = None,
    # 向后兼容参数
    models_root: str = "models/GPR", 
    results_root: str = "results"
):
    """
    步骤2: 使用GPR模型进行贝叶斯优化
    
    Args:
        iteration_num: 当前迭代轮次
        xi: EI探索参数
        n_samples: 采样数量
        n_top: 选取的候选材料数量
        initial_samples: 初始采样点 (上一轮的成功案例)
        path_config: 路径配置对象 (推荐)
        models_root: 模型根目录 (向后兼容)
        results_root: 结果根目录 (向后兼容)
        
    Returns:
        dict: 包含候选材料列表的信息
    """
    print("=" * 80)
    print(f"步骤 2: 贝叶斯优化 (Iteration {iteration_num})")
    print("=" * 80)
    
    # 使用 PathConfig 或向后兼容的字符串路径
    if path_config:
        model_path = path_config.get_model_file_path(iteration_num - 1)
        _models_root = str(path_config.models_root.relative_to(path_config.project_root))
        _results_root = str(path_config.results_root.relative_to(path_config.project_root))
    else:
        prev_iteration = iteration_num - 1
        model_dir = project_root / models_root / f"iteration_{prev_iteration}"
        model_path = model_dir / "gpr_thermal_conductivity.joblib"
        _models_root = models_root
        _results_root = results_root
    
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return {
            'success': False,
            'error': f'Model file not found: {model_path}'
        }

    print(f"Using model: {model_path}")
    print(f"   - EI探索参数 (ξ): {xi}")
    print(f"   - 采样数量: {n_samples}")
    print(f"   - 候选材料数量: {n_top}")
    if initial_samples:
        print(f"   - 初始采样点: {len(initial_samples)} 个")
    
    try:
        # 执行贝叶斯优化
        results = run_acquisition(
            xi=xi, 
            n_samples=n_samples, 
            iteration_num=iteration_num, 
            model_path=str(model_path),
            initial_samples=initial_samples,
            n_top=n_top,
            seed=(int(seed) + int(seed_stride) * int(iteration_num)) if seed is not None else None,
            results_root=_results_root,
            models_root=_models_root
        )
        
        print(f"✅ 贝叶斯优化完成! 筛选出 {len(results)} 个候选材料")
        
        # 返回 TOP N 用于后续 AI 评估
        top_materials = results[:n_top]
        
        return {
            'success': True,
            'n_materials': len(results),
            'n_top': n_top,
            'top_materials': top_materials,
            'top10_materials': results[:10],  # 保留向后兼容
            'all_materials': results
        }
        
    except Exception as e:
        print(f"❌ 贝叶斯优化失败: {e}")
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # 测试
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=1)
    parser.add_argument('--xi', type=float, default=0.01)
    parser.add_argument('--samples', type=int, default=100)
    args = parser.parse_args()
    
    result = step_bayesian_optimization(args.iteration, args.xi, args.samples)
    print(f"\n结果: {result}")
