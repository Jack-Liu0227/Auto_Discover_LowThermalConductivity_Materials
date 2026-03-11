# -*- coding: utf-8 -*-
"""
配置文件读取工具
"""
import yaml
import os
from pathlib import Path


def load_config(config_path=None):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，默认为项目根目录下的 config/config.yaml
        
    Returns:
        配置字典
    """
    if config_path is None:
        # 自动查找项目根目录
        current_dir = Path(__file__).parent.parent.parent
        config_path = current_dir / 'config' / 'config.yaml'
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_bayesian_config(config=None):
    """
    获取贝叶斯优化配置
    
    Args:
        config: 已加载的配置字典，如果为None则自动加载
        
    Returns:
        贝叶斯优化配置字典
    """
    if config is None:
        config = load_config()
    
    return config.get('bayesian_optimization', {})


def get_acquisition_params(config=None):
    """
    获取采集函数参数
    
    Args:
        config: 已加载的配置字典
        
    Returns:
        dict: {'function': 'EI', 'xi': 0.01}
    """
    bo_config = get_bayesian_config(config)
    return bo_config.get('acquisition', {'function': 'EI', 'xi': 0.01})


def get_sampling_params(config=None):
    """
    获取采样参数
    
    Args:
        config: 已加载的配置字典
        
    Returns:
        dict: {'n_samples': 100, 'max_atoms': 20, 'allowed_elements': [...]}
    """
    bo_config = get_bayesian_config(config)
    return bo_config.get('sampling', {
        'n_samples': 100,
        'max_atoms': 20,
        'allowed_elements': []
    })


def get_model_config(config=None):
    """
    获取模型配置
    
    Args:
        config: 已加载的配置字典
        
    Returns:
        dict: 模型配置
    """
    bo_config = get_bayesian_config(config)
    return bo_config.get('model', {
        'initial_model': 'models/GPR_0/gpr_thermal_conductivity.joblib',
        'retrain_from_round': 2
    })


if __name__ == '__main__':
    # 测试
    config = load_config()
    print("贝叶斯优化配置:")
    print(get_bayesian_config(config))
    print("\n采集函数参数:")
    print(get_acquisition_params(config))
    print("\n采样参数:")
    print(get_sampling_params(config))
