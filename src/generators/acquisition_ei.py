# -*- coding: utf-8 -*-
"""
贝叶斯优化采样函数 - Expected Improvement (EI)

流程:
    1. 调用蒙特卡洛采样器随机生成候选材料
    2. 使用GPR模型预测每个材料的热导率 (均值μ和标准差σ)
    3. 计算EI采集函数值，平衡低热导率和高不确定性

采样函数 (Expected Improvement):
    EI(x) = (f_min - μ(x)) * Φ(Z) + σ(x) * φ(Z)
    
    其中:
        Z = (f_min - μ(x)) / σ(x)
        Φ(Z) - 标准正态分布累积分布函数 (CDF)
        φ(Z) - 标准正态分布概率密度函数 (PDF)
        f_min - 当前已知最佳值 (最低热导率，log空间)
        μ(x) - 预测均值 (热导率, log空间)
        σ(x) - 预测标准差 (不确定性)

EI值越大 → 既有可能比当前最佳更好，又有探索价值 → 最优材料
自动平衡 Exploitation (低预测值) 和 Exploration (高不确定性)
"""
import os
import sys
import logging
import random
import numpy as np
import joblib
import json
import pandas as pd
from datetime import datetime
from scipy.stats import norm  # 用于计算正态分布的 CDF 和 PDF

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# 导入蒙特卡洛采样器
from monte_carlo_sampler import monte_carlo_sampling, mcmc_sampling, ELEMENTS, MAX_ATOMS

# 特征元素顺序 (与训练时一致)
FEATURE_ELEMENTS = ['Ag', 'As', 'Bi', 'Cu', 'Ge', 'In', 'Pb', 'S', 'Sb', 'Se', 'Sn', 'Te', 'Ti', 'V']

# 同族/性质相近元素替换表 (用于突变)
ELEMENT_SUBSTITUTION = {
    'S': ['Se', 'Te'],
    'Se': ['S', 'Te'],
    'Te': ['S', 'Se'],
    'Ag': ['Cu'],
    'Cu': ['Ag'],
    'Ge': ['Sn', 'Pb'],
    'Sn': ['Ge', 'Pb'],
    'Pb': ['Ge', 'Sn'],
    'As': ['Sb', 'Bi'],
    'Sb': ['As', 'Bi'],
    'Bi': ['As', 'Sb'],
    'Ti': ['V'], # 简单邻近替换
    'V': ['Ti']
}

# 注意: 移除了 numpy._core 的兼容性修复代码
# 因为它会干扰 pymatgen 和其他使用 numpy 的库的导入
# 如果遇到模型加载的 numpy 版本不兼容问题，请确保环境中 numpy 版本一致




def setup_logger(log_dir, name='ei_acquisition'):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)

    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') # 移除时间戳
    log_file = os.path.join(log_dir, 'ei_acquisition.log')

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, log_file



def composition_to_features(composition):
    """将组分字典转换为特征向量（请注意：此处使用原始原子数，不归一化，与模型训练数据保持一致）"""
    # total = sum(composition.values())
    features = []
    for elem in FEATURE_ELEMENTS:
        # ratio = composition.get(elem, 0) / total if total > 0 else 0
        # features.append(ratio)
        val = composition.get(elem, 0)
        features.append(val)
    return features


def calculate_acquisition(mu, sigma, f_min, xi=0.01):
    """
    计算Expected Improvement (EI) 采集函数值

    EI(x) = (f_min - μ(x)) * Φ(Z) + σ(x) * φ(Z)
    
    其中 Z = (f_min - μ(x)) / σ(x)

    Args:
        mu: 预测均值 (log空间)
        sigma: 预测标准差 (log空间)
        f_min: 当前已知最佳值 (最低热导率，log空间)
        xi: 探索参数 (默认0.01，用于数值稳定性)

    Returns:
        EI值 (越大越好: 平衡低预测值和高不确定性)
    """
    # 避免除以零
    sigma = np.maximum(sigma, 1e-9)
    
    # 计算 Z 值
    Z = (f_min - mu - xi) / sigma
    
    # 计算 EI
    ei = (f_min - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    # 确保 EI 非负
    ei = np.maximum(ei, 0.0)
    
    return ei


def load_model_and_scaler(model_path, project_root, models_root):
    """Load model and scaler for GPR predictions."""
    if model_path:
        if not os.path.isabs(model_path):
            model_path = os.path.join(project_root, model_path)
        model_dir = os.path.dirname(model_path)
        scaler_path = os.path.join(model_dir, 'gpr_scaler.joblib')
    else:
        model_dir = os.path.join(project_root, models_root)
        model_path = os.path.join(model_dir, 'gpr_thermal_conductivity.joblib')
        scaler_path = os.path.join(model_dir, 'gpr_scaler.joblib')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler, model_path, scaler_path


def get_f_min(iteration_num, project_root, results_root):
    """Get f_min from history or default."""
    if iteration_num > 1:
        df_history = load_historical_success_data(iteration_num, project_root, results_root)
        if df_history is not None and len(df_history) > 0 and 'thermal_conductivity' in df_history.columns:
            return np.log(df_history['thermal_conductivity'].min())
    return np.log(1.0)


def load_historical_success_data(iteration_num, project_root, results_root="results"):
    """
    加载历史轮次的成功材料数据，用于增强训练数据
    
    Args:
        iteration_num: 当前轮次
        project_root: 项目根目录
        results_root: 结果存储根目录
        
    Returns:
        历史成功材料的DataFrame，如果没有则返回None
    """
    if iteration_num <= 1:
        return None
    
    all_success_data = []
    
    # 遍历之前所有轮次
    for prev_iteration in range(1, iteration_num):
        success_csv = os.path.join(
            project_root, 
            results_root, 
            f'iteration_{prev_iteration}', 
            'success_examples', 
            'success_materials.csv'
        )
        
        if os.path.exists(success_csv):
            try:
                df_success = pd.read_csv(success_csv, encoding='utf-8-sig')
                # 只保留需要的列：formula + thermal_conductivity（兼容中英文表头）
                formula_col = None
                for c in ["formula", "composition", "组分"]:
                    if c in df_success.columns:
                        formula_col = c
                        break
                k_col = None
                for c in ["thermal_conductivity_w_mk", "热导率(W/m·K)", "热导率 (W/m·K)"]:
                    if c in df_success.columns:
                        k_col = c
                        break
                if formula_col and k_col:
                    df_success_clean = df_success[[formula_col, k_col]].copy()
                    df_success_clean.columns = ["formula", "thermal_conductivity"]
                    all_success_data.append(df_success_clean)
                    print(f"  ✅ 加载 Iteration {prev_iteration} 成功材料: {len(df_success_clean)} 个")
            except Exception as e:
                print(f"  ⚠️ 无法加载 Iteration {prev_iteration} 数据: {e}")
    
    if all_success_data:
        df_combined = pd.concat(all_success_data, ignore_index=True)
        return df_combined
    else:
        return None


def mutate_sampling(initial_samples, n_mutations, logger):
    """
    基于初始样本进行突变采样 (局部搜索)
    
    策略:
    1. 元素替换: 将一个元素替换为同族元素
    2. 配比微调: 随机改变原子数 (+/- 1)
    
    Args:
        initial_samples: 初始样本列表
        n_mutations: 需要生成的突变样本数量
        logger: 日志记录器
        
    Returns:
        mutated_samples: 突变后的样本列表
    """
    if not initial_samples:
        return []
        
    mutated_samples = []
    generated_formulas = set()
    
    # 将初始样本的公式加入已生成集合，避免生成自己
    for s in initial_samples:
        generated_formulas.add(s.get('formula'))
    
    logger.info(f"开始突变采样... (目标: {n_mutations} 个)")
    
    import random
    from pymatgen.core import Composition
    
    attempts = 0
    max_attempts = n_mutations * 20
    
    while len(mutated_samples) < n_mutations and attempts < max_attempts:
        attempts += 1
        
        # 随机选择一个种子样本
        seed = random.choice(initial_samples)
        seed_formula = seed.get('formula')
        
        try:
            # 解析种子组分
            # 注意: 如果 initial_samples 来自 DataFrame，可能没有 composition 字典，只有 formula
            if 'composition' in seed:
                comp_dict = seed['composition'].copy()
            else:
                comp = Composition(seed_formula)
                comp_dict = {str(el): int(amt) for el, amt in comp.get_el_amt_dict().items()}
            
            elements = list(comp_dict.keys())
            
            # 随机选择突变类型: 0=元素替换, 1=配比微调
            mutation_type = random.choice([0, 1])
            new_comp_dict = comp_dict.copy()
            
            if mutation_type == 0: # 元素替换
                # 随机选择一个元素进行替换
                el_to_replace = random.choice(elements)
                if el_to_replace in ELEMENT_SUBSTITUTION:
                    substitutes = ELEMENT_SUBSTITUTION[el_to_replace]
                    # 过滤掉已存在的元素
                    valid_subs = [sub for sub in substitutes if sub not in elements]
                    if valid_subs:
                        new_el = random.choice(valid_subs)
                        # 替换
                        amt = new_comp_dict.pop(el_to_replace)
                        new_comp_dict[new_el] = amt
            
            else: # 配比微调
                # 随机选择一个元素改变数量
                el_to_change = random.choice(elements)
                change = random.choice([-1, 1])
                new_amt = new_comp_dict[el_to_change] + change
                if new_amt >= 1: # 保证至少1个原子
                    new_comp_dict[el_to_change] = new_amt
            
            # 过滤掉总原子数过多的 (小于等于 MAX_ATOMS = 20)
            if sum(new_comp_dict.values()) > MAX_ATOMS:
                continue
                
            # 构建新公式字符串
            # 简单的构建方式，不保证标准化顺序，但足够用来查重
            # 为了更好的展示，按字母顺序排序
            sorted_els = sorted(new_comp_dict.keys())
            formula_parts = []
            for el in sorted_els:
                amt = int(new_comp_dict[el])
                if amt == 1:
                    formula_parts.append(el)
                else:
                    formula_parts.append(f"{el}{amt}")
            new_formula = ''.join(formula_parts)
            
            # 查重
            if new_formula not in generated_formulas:
                generated_formulas.add(new_formula)
                
                n_elements = len(new_comp_dict)
                total_atoms = int(sum(new_comp_dict.values()))
                
                mutated_samples.append({
                    'formula': new_formula,
                    'composition': new_comp_dict,
                    'n_elements': n_elements,
                    'total_atoms': total_atoms,
                    'is_mutation': True,
                    'seed_formula': seed_formula
                })
                # logger.info(f"  [突变] {seed_formula} -> {new_formula}")

        except Exception as e:
            # logger.warning(f"突变失败 {seed_formula}: {e}")
            pass

    logger.info(f"突变采样完成! 生成 {len(mutated_samples)} 个新候选材料")
    return mutated_samples


def main(
    xi=0.01,
    n_samples=100,
    iteration_num=1,
    model_path=None,
    initial_samples=None,
    n_top=10,
    seed=None,
    results_root="results",
    models_root="models/GPR",
    sampling_method=None,
    mcmc_params=None
):
    """
    主函数

    Args:
        xi: EI探索参数 (默认0.01，用于数值稳定性)
        n_samples: 采样数量
        iteration_num: 当前轮次（用于路径和历史数据加载）
        model_path: GPR模型路径 (可选, 默认为 None 则使用默认路径)
        initial_samples: 初始采样点 (上一轮的成功案例)，可选
        n_top: 选取的候选材料数量 (默认10)
        results_root: 结果输出根目录 (默认 "results")
        models_root: 模型存储根目录 (默认 "models/GPR")
    """
    # 路径设置（支持动态轮次）
    project_root = os.path.join(script_dir, '..', '..')
    log_dir = os.path.join(project_root, results_root, f'iteration_{iteration_num}', 'logs')
    # model_dir = os.path.join(project_root, 'src', 'models') # Deprecated if model_path is used

    # 设置日志
    logger, log_file = setup_logger(log_dir)
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        logger.info(f"Reproducibility seed: {seed}")

    logger.info("=" * 90)
    logger.info("贝叶斯优化采样函数 - Expected Improvement (EI)")
    logger.info("=" * 90)
    logger.info(f"当前轮次: Iteration {iteration_num}")
    logger.info(f"目标: 平衡低热导率 (Exploitation) 和高不确定性 (Exploration)")
    logger.info(f"EI(x) = (f_min - μ) * Φ(Z) + σ * φ(Z)  (越大越好)")
    logger.info(f"探索参数 ξ = {xi}")

    hard_constraints = None
    if sampling_method is None or mcmc_params is None or hard_constraints is None:
        try:
            from utils.config_loader import get_sampling_params
            samp_params = get_sampling_params()
            if sampling_method is None:
                sampling_method = samp_params.get('method', 'random')
            if mcmc_params is None:
                mcmc_params = samp_params.get('mcmc', {})
            hard_constraints = samp_params.get('hard_constraints')
        except Exception:
            if sampling_method is None:
                sampling_method = 'random'
            if mcmc_params is None:
                mcmc_params = {}
            hard_constraints = None

    sampling_method = (sampling_method or 'random').lower()
    mcmc_params = mcmc_params or {}
    #=========================================================================
    # 步骤1: 智能采样策略 (70% 突变采样 + 30% 随机探索)
    #=========================================================================
    logger.info("")
    logger.info("=" * 80)
    
    # 创建一个静默的logger用于采样
    silent_logger = logging.getLogger('silent_sampler')
    silent_logger.setLevel(logging.WARNING)  # 只显示警告以上
    
    # 检查是否有历史成功案例
    if sampling_method == "mcmc":
        logger.info("Step1: MCMC sampling (importance-weighted)")
        model, scaler, model_path_resolved, scaler_path = load_model_and_scaler(
            model_path, project_root, models_root
        )
        f_min_for_mcmc = get_f_min(iteration_num, project_root, results_root)
        w_ei = float(mcmc_params.get("w_ei", 1.0))
        w_k = float(mcmc_params.get("w_k", 1.0))
        w_sigma = float(mcmc_params.get("w_sigma", 0.5))
        temperature = float(mcmc_params.get("temperature", 1.0))
        burn_in = int(mcmc_params.get("burn_in", 200))
        thin = int(mcmc_params.get("thin", 5))
        max_steps = mcmc_params.get("max_steps", None)

        n_mcmc = n_samples

        start_comp = None
        if initial_samples:
            try:
                from pymatgen.core import Composition
                seed = random.choice(initial_samples)
                if isinstance(seed, dict) and "composition" in seed:
                    start_comp = seed["composition"]
                elif isinstance(seed, dict) and seed.get("formula"):
                    comp = Composition(seed.get("formula"))
                    start_comp = {str(el): int(amt) for el, amt in comp.get_el_amt_dict().items()}
            except Exception:
                start_comp = None

        def score_fn(comp):
            features = composition_to_features(comp)
            X = np.array([features])
            X_scaled = scaler.transform(X)
            mu_log, sigma_log = model.predict(X_scaled, return_std=True)
            mu_val = float(mu_log[0])
            sigma_val = float(sigma_log[0])
            k_pred_val = float(np.exp(mu_val))
            ei_val = float(calculate_acquisition(np.array([mu_val]), np.array([sigma_val]), f_min_for_mcmc, xi)[0])
            score = (
                w_ei * np.log1p(ei_val)
                + w_sigma * np.log1p(sigma_val)
                - w_k * np.log(k_pred_val)
            )
            meta = {
                "k_pred": k_pred_val,
                "mu_log": mu_val,
                "sigma_log": sigma_val,
                "ei": ei_val,
                "sampling_method": "mcmc"
            }
            return float(score), meta

        samples = mcmc_sampling(
            n_mcmc,
            MAX_ATOMS,
            logger,
            score_fn,
            start_composition=start_comp,
            burn_in=burn_in,
            thin=thin,
            max_steps=max_steps,
            temperature=temperature,
            hard_constraints=hard_constraints,
        )
    else:
        if initial_samples and len(initial_samples) > 0:
            logger.info("步骤1: 混合采样策略 (基于历史成功案例)")
            logger.info("=" * 80)
            logger.info(f"📊 可用历史成功案例: {len(initial_samples)} 个")
            
            # 70% 基于历史案例的突变采样 (重要性采样)
            n_mutations = int(n_samples * 0.70)
            # 30% 随机探索 (保持全局探索能力)
            n_random = n_samples - n_mutations
            
            logger.info(f"采样策略: {n_mutations} 个突变采样 (70%) + {n_random} 个随机探索 (30%)")
            logger.info("")
            
            # 1. 随机探索
            logger.info(f"[1/2] 随机探索采样 ({n_random} 个)...")
            samples = monte_carlo_sampling(
                n_random,
                MAX_ATOMS,
                silent_logger,
                hard_constraints=hard_constraints,
            )
            logger.info(f"  ✅ 随机采样完成: {len(samples)} 个")
            
            # 2. 突变采样 (局部搜索)
            logger.info("")
            logger.info(f"[2/2] 基于成功案例的突变采样 ({n_mutations} 个)...")
            logger.info("  策略: 元素替换 + 配比微调")
            mutated_samples = mutate_sampling(initial_samples, n_mutations, logger)
            
            if mutated_samples:
                samples.extend(mutated_samples)
                logger.info(f"  ✅ 突变采样完成: {len(mutated_samples)} 个")
            
            logger.info("")
            logger.info(f"📦 总采样数: {len(samples)} 个 (随机: {n_random}, 突变: {len(mutated_samples)})")
            
        else:
            # 第一轮迭代：100% 随机采样
            logger.info("步骤1: 蒙特卡洛采样生成候选材料 (无历史数据)")
            logger.info("=" * 80)
            logger.info("首轮迭代，采用 100% 随机采样策略")
            
            samples = monte_carlo_sampling(
                n_samples,
                MAX_ATOMS,
                silent_logger,
                hard_constraints=hard_constraints,
            )
            
            logger.info(f"采样完成! 生成 {len(samples)} 个候选材料")
        
    logger.info(f"元素空间: {ELEMENTS}")
    logger.info(f"化合物类型: 二元、三元")
    logger.info(f"最大原子数: {MAX_ATOMS}")

    # 显示部分采样结果
    logger.info("-" * 80)
    logger.info("采样结果示例 (前10个):")
    for i, s in enumerate(samples[:10]):
        logger.info(f"  [{i+1:2d}] {s['formula']:15} | {s['n_elements']}元 | 原子数: {s['total_atoms']:2d}")
    logger.info("  ...")

    #=========================================================================
    # 步骤2: 使用GPR模型预测热导率
    #=========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("步骤2: 使用GPR模型预测热导率")
    logger.info("=" * 80)

    # 加载模型
    if model_path:
        # 如果提供了具体路径，则使用该路径
        if not os.path.isabs(model_path):
            model_path = os.path.join(project_root, model_path)
        
        model_dir = os.path.dirname(model_path)
        scaler_path = os.path.join(model_dir, 'gpr_scaler.joblib')
    else:
        # 默认路径
        model_dir = os.path.join(project_root, models_root)
        model_path = os.path.join(model_dir, 'gpr_thermal_conductivity.joblib')
        scaler_path = os.path.join(model_dir, 'gpr_scaler.joblib')

    logger.info(f"加载GPR模型: {os.path.basename(model_path)}")
    logger.info(f"模型完整路径: {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"❌ 模型文件不存在: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    if not os.path.exists(scaler_path):
        logger.error(f"❌ Scaler文件不存在: {scaler_path}")
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info("模型加载成功!")

    # 转换特征
    logger.info("转换化学式为特征向量...")
    X = np.array([composition_to_features(s['composition']) for s in samples])
    X_scaled = scaler.transform(X)
    logger.info(f"特征矩阵形状: {X.shape}")

    # GPR预测 (返回均值和标准差)
    logger.info("GPR预测中...")
    mu_log, sigma_log = model.predict(X_scaled, return_std=True)

    # 转换到原始空间
    k_pred = np.exp(mu_log)

    # 将预测结果添加到samples
    for i, s in enumerate(samples):
        s['k_pred'] = k_pred[i]
        s['mu_log'] = mu_log[i]
        s['sigma_log'] = sigma_log[i]
        # 计算95%置信区间
        s['k_lower'] = np.exp(mu_log[i] - 1.96 * sigma_log[i])
        s['k_upper'] = np.exp(mu_log[i] + 1.96 * sigma_log[i])

    logger.info("预测完成!")

    #=========================================================================
    # 步骤3: 获取当前最佳值 f_min
    #=========================================================================
    logger.info("")
    logger.info("=" * 90)
    logger.info("步骤3: 获取当前最佳值 (f_min)")
    logger.info("=" * 90)
    
    # 尝试从历史数据获取最佳值
    if iteration_num > 1:
        df_history = load_historical_success_data(iteration_num, project_root, results_root)
        if df_history is not None and len(df_history) > 0 and 'thermal_conductivity' in df_history.columns:
            # 获取历史最低热导率 (log空间)
            f_min = np.log(df_history['thermal_conductivity'].min())
            logger.info(f"从历史数据获取 f_min = {f_min:.4f} (log空间)")
            logger.info(f"对应热导率 = {np.exp(f_min):.4f} W/m·K")
        else:
            # 如果没有历史数据，使用当前预测的最小值
            f_min = mu_log.min()
            logger.info(f"无历史数据，使用当前预测最小值 f_min = {f_min:.4f} (log空间)")
    else:
        # Iteration 1，从原始训练数据估算
        # 假设目标是找到 < 1.0 W/m·K 的材料，设置 f_min = log(1.0) = 0
        f_min = np.log(1.0)
        logger.info(f"Iteration 1，设置目标 f_min = {f_min:.4f} (log(1.0 W/m·K))")
    
    #=========================================================================
    # 步骤4: 计算 EI 采集函数值
    #=========================================================================
    logger.info("")
    logger.info("=" * 90)
    logger.info("步骤4: 计算 Expected Improvement (EI) 值")
    logger.info("=" * 90)
    logger.info(f"EI(x) = (f_min - μ) * Φ(Z) + σ * φ(Z), ξ = {xi}")
    logger.info("EI越大 → 既有潜力超越当前最佳，又有探索价值 → 最优材料")
    logger.info("自动平衡 Exploitation (低预测值) 和 Exploration (高不确定性)")

    # 计算 EI 值
    ei_values = calculate_acquisition(mu_log, sigma_log, f_min, xi)

    for i, s in enumerate(samples):
        s['ei'] = ei_values[i]

    # 按 EI 排序 (越大越好)
    samples_sorted = sorted(samples, key=lambda x: x['ei'], reverse=True)

    #=========================================================================
    # 输出全部100个样本的详细信息
    #=========================================================================
    logger.info("")
    logger.info("=" * 90)
    logger.info("全部采样结果 (按EI降序排列)")
    logger.info("=" * 90)
    logger.info(f"{'排名':>4} | {'化学式':^15} | {'k (W/Km)':^10} | {'μ_log':^8} | {'σ_log':^8} | {'EI':^10}")
    logger.info("-" * 75)

    for rank, s in enumerate(samples_sorted, 1):
        logger.info(f"{rank:>4} | {s['formula']:^15} | {s['k_pred']:^10.4f} | {s['mu_log']:^8.4f} | {s['sigma_log']:^8.4f} | {s['ei']:^10.6f}")

    # 统计分析
    logger.info("")
    logger.info("=" * 90)
    logger.info("统计分析")
    logger.info("=" * 90)
    logger.info(f"EI 范围: {ei_values.min():.6f} ~ {ei_values.max():.6f}")
    logger.info(f"EI 均值: {ei_values.mean():.6f}")
    logger.info(f"热导率范围: {k_pred.min():.4f} ~ {k_pred.max():.4f} W/m·K")
    logger.info(f"热导率均值: {k_pred.mean():.4f} W/m·K")
    logger.info(f"不确定性范围: {sigma_log.min():.4f} ~ {sigma_log.max():.4f}")
    logger.info(f"不确定性均值: {sigma_log.mean():.4f}")
    logger.info(f"当前最佳值 f_min: {f_min:.4f} (log空间) = {np.exp(f_min):.4f} W/m·K")

    # 推荐材料 TOP N
    logger.info("")
    logger.info("=" * 90)
    logger.info(f"🏆 TOP {n_top} 推荐材料 (最大 EI 值 = 最佳探索-利用平衡)")
    logger.info("=" * 90)

    for rank, s in enumerate(samples_sorted[:n_top], 1):
        k_lower = np.exp(s['mu_log'] - 1.96 * s['sigma_log'])
        k_upper = np.exp(s['mu_log'] + 1.96 * s['sigma_log'])
        logger.info(f"  {rank:>2}. {s['formula']:15} | k = {s['k_pred']:.4f} W/m·K | σ = {s['sigma_log']:.4f} | 95% CI: [{k_lower:.3f}, {k_upper:.3f}] | EI = {s['ei']:.6f}")

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"日志已保存: {log_file}")

    # 保存完整结果到 results/iteration_X/selected_results/ 目录（支持动态轮次）
    results_dir = os.path.join(project_root, results_root, f'iteration_{iteration_num}', 'selected_results')
    os.makedirs(results_dir, exist_ok=True)

    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') # 移除时间戳

    # 保存 TOP N 到 JSON
    top_file = os.path.join(results_dir, f'top{n_top}_materials.json')
    top_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'acquisition_function': 'Expected Improvement (EI)',
        'xi': xi,
        'seed': seed,
        'f_min': float(f_min),
        'f_min_real_space': float(np.exp(f_min)),
        'n_samples': n_samples,
        'n_top': n_top,
        f'top{n_top}': []
    }

    for rank, s in enumerate(samples_sorted[:n_top], 1):
        k_lower = np.exp(s['mu_log'] - 1.96 * s['sigma_log'])
        k_upper = np.exp(s['mu_log'] + 1.96 * s['sigma_log'])
        top_data[f'top{n_top}'].append({
            'rank': rank,
            'formula': s['formula'],
            'k_pred': float(s['k_pred']),
            'mu_log': float(s['mu_log']),
            'sigma_log': float(s['sigma_log']),
            'ei': float(s['ei']),
            'k_lower': float(k_lower),
            'k_upper': float(k_upper),
            'elements': ', '.join(s.get('elements', [])),
            'n_elements': s.get('n_elements', 0),
            'total_atoms': s.get('total_atoms', 0)
        })

    with open(top_file, 'w', encoding='utf-8') as f:
        json.dump(top_data, f, ensure_ascii=False, indent=2)

    logger.info(f"TOP {n_top} 结果已保存: {top_file}")

    # 保存全部结果到 CSV
    all_results_file = os.path.join(results_dir, f'all_samples.csv')
    df_data = []
    for rank, s in enumerate(samples_sorted, 1):
        k_lower = np.exp(s['mu_log'] - 1.96 * s['sigma_log'])
        k_upper = np.exp(s['mu_log'] + 1.96 * s['sigma_log'])
        df_data.append({
            'rank': rank,
            'formula': s['formula'],
            'k_pred': s['k_pred'],
            'mu_log': s['mu_log'],
            'sigma_log': s['sigma_log'],
            'ei': s['ei'],
            'k_lower': k_lower,
            'k_upper': k_upper,
            'elements': ', '.join(s.get('elements', [])),
            'n_elements': s.get('n_elements', 0),
            'total_atoms': s.get('total_atoms', 0)
        })

    df = pd.DataFrame(df_data)
    df.to_csv(all_results_file, index=False, encoding='utf-8-sig')

    logger.info(f"全部采样结果已保存: {all_results_file}")
    logger.info("")

    return samples_sorted


if __name__ == '__main__':
    # 从配置文件读取参数
    try:
        from utils.config_loader import get_acquisition_params, get_sampling_params
        
        acq_params = get_acquisition_params()
        samp_params = get_sampling_params()
        
        XI = acq_params.get('xi', 0.01)
        N_SAMPLES = samp_params.get('n_samples', 100)
        
        print(f"从配置文件加载参数: XI={XI}, N_SAMPLES={N_SAMPLES}")
    except Exception as e:
        # 如果读取配置失败，使用默认值
        print(f"配置文件读取失败，使用默认参数: {e}")
        XI = 0.01
        N_SAMPLES = 100

    main(xi=XI, n_samples=N_SAMPLES)

