# -*- coding: utf-8 -*-
"""
贝叶斯优化运行器 - 封装贝叶斯优化材料发现流程
"""

import sys
import os



# 导入训练模块
from models.train_gpr_model import train_gpr_model
from utils.update_dataset import update_dataset

def run_bayesian_optimization(xi=0.01, n_samples=100, iteration_num=1, 
                              results_root="results", data_root="data", models_root="models/GPR"):
    """
    运行贝叶斯优化材料发现流程
    
    流程:
        1. 更新数据集 (如果轮次 > 1)
        2. 重新训练 GPR 模型 (如果轮次 > 1)
        3. 蒙特卡洛采样生成候选材料
        4. GPR模型预测热导率 (均值μ + 标准差σ)
        5. 计算EI采集函数值，平衡低热导率和高不确定性
    
    Args:
        xi: EI探索参数 (默认0.01，用于数值稳定性)
        n_samples: 采样数量
        iteration_num: 当前迭代轮次
        results_root: 结果存储根目录
        data_root: 数据存储根目录
        models_root: 模型存储根目录
    
    Returns:
        采样结果列表
    """
    print("=" * 80)
    print("模块 1: 贝叶斯优化材料发现 (Expected Improvement)")
    print("=" * 80)
    print(f"参数设置:")
    print(f"  - 当前轮次: Iteration {iteration_num}")
    print(f"  - EI探索参数 (ξ): {xi}")
    print(f"  - 采样数量: {n_samples}")
    print("=" * 80)
    
    # 默认模型路径 (注意：这里假设模型保存在 iteration_X 类似结构下)
    # 修正路径命名: iteration{round_num-1} -> iteration_{iteration_num-1}
    model_dir = os.path.join(models_root, f'iteration_{iteration_num-1}')
    model_path = os.path.join(model_dir, 'gpr_thermal_conductivity.joblib')
    
    # -------------------------------------------------------------------------
    # 数据更新与模型重训练 (仅在 Iteration > 1 时执行)
    # -------------------------------------------------------------------------
    if iteration_num > 1:
        print(f"\n[数据更新] 正在合并上一轮成功案例...")
        prev_iteration = iteration_num - 1
        success_csv = f"{results_root}/iteration_{prev_iteration}/success_examples/success_materials_deduped.csv"
        
        # 兼容未去重的情况
        # 兼容未去重的情况
        if not os.path.exists(success_csv):
             success_csv = f"{results_root}/iteration_{prev_iteration}/success_examples/success_materials.csv"
        
        # 兼容稳定材料 (Fallback)
        if not os.path.exists(success_csv):
             success_csv = f"{results_root}/iteration_{prev_iteration}/success_examples/stable_materials.csv"
        
        # 使用上一轮的data.csv作为基础
        # 查找最近存在的数据文件
        prev_prev_iteration = prev_iteration - 1
        origin_csv = None
        for i in range(prev_prev_iteration, -1, -1):
            candidate = f"{data_root}/iteration_{i}/data.csv"
            if os.path.exists(candidate):
                origin_csv = candidate
                if i != prev_prev_iteration:
                    print(f"[WARN] iteration_{prev_prev_iteration} 数据不存在，使用 iteration_{i}/data.csv")
                break
        
        if origin_csv is None:
            print("[ERROR] 找不到任何可用的数据文件")
            return None
        
        updated_data_dir = f"{data_root}/iteration_{prev_iteration}"
        updated_data_path = update_dataset(
            success_csv=success_csv,
            origin_csv=origin_csv,
            output_dir=updated_data_dir
        )
        
        if updated_data_path:
            print(f"✅ 数据集已更新: {updated_data_path}")
            
            # 重训练模型
            print(f"\n[模型训练] 使用更新后的数据重新训练 GPR 模型...")
            
            # 使用 update_dataset 返回的新数据路径进行训练
            train_gpr_model(updated_data_path, model_dir)
            
            # 确认模型文件存在
            if not os.path.exists(model_path):
                print(f"❌ 警告: 训练似乎未生成预期模型文件 {model_path}，尝试搜索...")
                pass
            else:
                 print(f"✅ 模型重训练完成: {model_path}")
        else:
             print("⚠️ 数据更新未生成新文件 (可能无新的成功案例)")
             pass
    else:
        # Iteration 1，使用预训练的初始模型 (models/GPR_0)
         model_path = os.path.join('models', 'GPR_0', 'gpr_thermal_conductivity.joblib')
         if not os.path.exists(model_path):
             print(f"⚠️ 初始模型不存在: {model_path}，尝试使用 {models_root}/iteration0")
             model_path = os.path.join(models_root, 'iteration0', 'gpr_thermal_conductivity.joblib')
             # Or try iteration_0 if consistent with rebuild script
             if not os.path.exists(model_path):
                 model_path = os.path.join(models_root, 'iteration_0', 'gpr_thermal_conductivity.joblib')

    # -------------------------------------------------------------------------
    # 贝叶斯优化采样
    # -------------------------------------------------------------------------
    
    # 导入采样模块
    generators_path = os.path.join(os.path.dirname(__file__), '..', 'generators')
    if generators_path not in sys.path:
        sys.path.insert(0, generators_path)
    
    try:
        from acquisition_ei import main as run_acquisition
    except ImportError:
        # 尝试相对导入
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from generators.acquisition_ei import main as run_acquisition
    
    print(f"\n[采样] 开始执行贝叶斯优化采样 (使用模型: {model_path})...")
    
    # 执行采样流程 (Args updated: round_num -> iteration_num)
    results = run_acquisition(
        xi=xi, 
        n_samples=n_samples, 
        iteration_num=iteration_num, 
        model_path=model_path,
        results_root=results_root
    )

    print(f"\n✅ 采样完成! 共评估 {len(results)} 个候选材料")
    print(f"📁 详细日志已保存到 {results_root}/iteration_{iteration_num}/logs/ 目录")
    print(f"📊 采样结果已保存到 {results_root}/iteration_{iteration_num}/selected_results/ 目录")

    # 返回 TOP 10 用于后续评估
    top10 = results[:10]
    return top10


if __name__ == '__main__':
    # 测试
    run_bayesian_optimization(xi=0.01, n_samples=100, iteration_num=1)


