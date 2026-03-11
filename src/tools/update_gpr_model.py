# -*- coding: utf-8 -*-
"""
GPR模型增量更新工具 - 使用历史成功数据增强训练

该脚本在每轮迭代后，将新的成功材料数据添加到GPR模型的训练集中，
从而使模型在后续轮次中能够更准确地预测材料的热导率。
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# 特征元素顺序 (与训练时一致)
FEATURE_ELEMENTS = ['Ag', 'As', 'Bi', 'Cu', 'Ge', 'In', 'Pb', 'S', 'Sb', 'Se', 'Sn', 'Te', 'Ti', 'V']


def composition_to_features(formula_str):
    """
    将化学式字符串转换为特征向量（归一化元素比例）
    
    Args:
        formula_str: 化学式字符串，如 "Ag9In5Bi6"
        
    Returns:
        特征向量（归一化的元素比例）
    """
    from pymatgen.core import Composition
    
    try:
        comp = Composition(formula_str)
        comp_dict = comp.get_el_amt_dict()
        
        total = sum(comp_dict.values())
        features = []
        for elem in FEATURE_ELEMENTS:
            ratio = comp_dict.get(elem, 0) / total if total > 0 else 0
            features.append(ratio)
        
        return features
    except Exception as e:
        logger.error(f"无法解析化学式 {formula_str}: {e}")
        return None


def load_original_training_data(data_path="data/training_data.csv"):
    """
    加载原始训练数据
    
    Args:
        data_path: 原始训练数据路径
        
    Returns:
        X, y (特征矩阵和标签)
    """
    if not Path(data_path).exists():
        logger.warning(f"原始训练数据不存在: {data_path}")
        return None, None
    
    try:
        df = pd.read_csv(data_path, encoding='utf-8-sig')
        
        # 假设列名为 'formula' 和 'thermal_conductivity'
        if 'formula' not in df.columns or 'thermal_conductivity' not in df.columns:
            logger.error("训练数据缺少必需列")
            return None, None
        
        X_list = []
        y_list = []
        
        for _, row in df.iterrows():
            features = composition_to_features(row['formula'])
            if features is not None:
                X_list.append(features)
                y_list.append(row['thermal_conductivity'])
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"原始训练数据: {len(X)} 个样本")
        return X, y
        
    except Exception as e:
        logger.error(f"加载原始训练数据失败: {e}")
        return None, None


def load_success_materials(iteration_num, project_root=".", results_root="results"):
    """
    加载指定轮次及之前的所有成功材料数据
    
    Args:
        iteration_num: 当前轮次
        project_root: 项目根目录
        results_root: 结果存储根目录
        
    Returns:
        X, y (成功材料的特征和热导率)
    """
    X_list = []
    y_list = []
    
    for r in range(1, iteration_num + 1):
        success_csv = Path(project_root) / results_root / f"iteration_{r}" / "success_examples" / "success_materials.csv"
        stable_csv = Path(project_root) / results_root / f"iteration_{r}" / "success_examples" / "stable_materials.csv"
        
        target_csv = None
        if success_csv.exists():
            target_csv = success_csv
        elif stable_csv.exists():
            target_csv = stable_csv
            logger.info(f"Iteration {r}: 使用稳定材料 (stable_materials.csv) 进行模型更新")
            
        if target_csv:
            try:
                df = pd.read_csv(target_csv, encoding='utf-8-sig')
                
                if '组分' in df.columns and '热导率 (W/m·K)' in df.columns:
                    for _, row in df.iterrows():
                        features = composition_to_features(row['组分'])
                        if features is not None:
                            X_list.append(features)
                            y_list.append(row['热导率 (W/m·K)'])
                    
                    logger.info(f"加载 Iteration {r} 成功材料: {len(df)} 个")
            except Exception as e:
                logger.warning(f"无法加载 Iteration {r} 成功材料: {e}")
    
    if X_list:
        X = np.array(X_list)
        y = np.array(y_list)
        logger.info(f"总计成功材料: {len(X)} 个")
        return X, y
    else:
        logger.warning("未找到成功材料数据")
        return None, None


def update_gpr_model(iteration_num, project_root=".", original_data_path=None, results_root="results", models_root="models/GPR"):
    """
    使用历史成功数据更新 GPR 模型
    
    Args:
        iteration_num: 当前轮次
        project_root: 项目根目录
        original_data_path: 原始训练数据路径（可选）
        results_root: 结果存储根目录
        models_root: 模型存储根目录
        
    Returns:
        更新后的模型保存路径，失败则返回 None
    """
    logger.info("=" * 80)
    logger.info(f"更新 GPR 模型 (Iteration {iteration_num})")
    logger.info("=" * 80)
    
    project_root = Path(project_root)
    model_dir = project_root / models_root
    
    # 1. 加载原始模型
    original_model_path = model_dir / "gpr_thermal_conductivity.joblib"
    original_scaler_path = model_dir / "gpr_scaler.joblib"
    
    if not original_model_path.exists():
        logger.error(f"原始模型不存在: {original_model_path}")
        return None
    
    model = joblib.load(original_model_path)
    scaler = joblib.load(original_scaler_path)
    logger.info("✅ 加载原始模型")
    
    # 2. 加载原始训练数据（如果提供）
    X_original, y_original = None, None
    if original_data_path and Path(original_data_path).exists():
        X_original, y_original = load_original_training_data(original_data_path)
    
    # 3. 加载成功材料数据
    X_success, y_success = load_success_materials(iteration_num, project_root, results_root)
    
    if X_success is None or len(X_success) == 0:
        logger.warning("没有成功材料数据可用于更新模型")
        return None
    
    # 4. 合并数据
    if X_original is not None and y_original is not None:
        X_combined = np.vstack([X_original, X_success])
        y_combined = np.concatenate([y_original, y_success])
        logger.info(f"合并数据: 原始 {len(X_original)} + 成功 {len(X_success)} = 总计 {len(X_combined)}")
    else:
        X_combined = X_success
        y_combined = y_success
        logger.info(f"仅使用成功材料数据: {len(X_combined)} 个样本")
    
    # 5. 重新训练模型
    logger.info("正在重新训练 GPR 模型...")
    
    # 标准化特征
    scaler_new = StandardScaler()
    X_scaled = scaler_new.fit_transform(X_combined)
    
    # 对数变换热导率（与原始训练一致）
    y_log = np.log(y_combined)
    
    # 训练新模型（使用原模型的超参数）
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    
    # 创建新的GPR（可以继承原模型的核函数配置）
    kernel = model.kernel_  # 使用原模型的核函数
    gpr_new = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        random_state=42,
        normalize_y=True
    )
    
    gpr_new.fit(X_scaled, y_log)
    logger.info("✅ 模型训练完成")
    
    # 6. 保存更新后的模型
    # 备份原模型
    backup_dir = model_dir / "backups"
    backup_dir.mkdir(exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    backup_model = backup_dir / f"gpr_thermal_conductivity_iteration{iteration_num-1}_{timestamp}.joblib"
    backup_scaler = backup_dir / f"gpr_scaler_iteration{iteration_num-1}_{timestamp}.joblib"
    
    joblib.dump(model, backup_model)
    joblib.dump(scaler, backup_scaler)
    logger.info(f"✅ 原模型已备份: {backup_model}")
    
    # 保存新模型
    joblib.dump(gpr_new, original_model_path)
    joblib.dump(scaler_new, original_scaler_path)
    logger.info(f"✅ 新模型已保存: {original_model_path}")
    
    logger.info("=" * 80)
    logger.info("GPR 模型更新完成!")
    logger.info("=" * 80)
    
    return str(original_model_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 测试
    import argparse
    parser = argparse.ArgumentParser(description='更新 GPR 模型')
    parser.add_argument('--round', type=int, default=2, help='当前轮次')
    parser.add_argument('--project-root', type=str, default='.', help='项目根目录')
    parser.add_argument('--original-data', type=str, default=None, help='原始训练数据路径（可选）')
    
    parser.add_argument('--results-root', type=str, default='results', help='结果存储根目录')
    parser.add_argument('--models-root', type=str, default='models/GPR', help='模型存储根目录')
    
    args = parser.parse_args()
    
    result = update_gpr_model(
        iteration_num=args.round,
        project_root=args.project_root,
        original_data_path=args.original_data,
        results_root=args.results_root,
        models_root=args.models_root
    )
    
    if result:
        print(f"\n✅ 模型更新成功: {result}")
    else:
        print("\n❌ 模型更新失败")
