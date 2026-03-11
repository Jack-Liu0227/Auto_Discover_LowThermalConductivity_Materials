# -*- coding: utf-8 -*-
"""
Bayesian Optimization material discovery pipeline (BO-only).

Workflow:
1. Train/update the model from previous-iteration data.
2. Run BO acquisition and select top candidates.
3. Generate structures and run relaxation/phonon/thermal calculations.
4. Extract success/stable materials and deduplicate.
5. Update dataset CSV for the next iteration.
6. Continue to the next iteration.
"""

import argparse
import sys
import os
import logging
from datetime import datetime
import shutil
from pathlib import Path

# Ensure UTF-8 output to avoid Windows console encoding issues.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 璁剧疆鏍囧噯杈撳嚭涓篣TF-8缂栫爜
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 添加项目路径
project_root = Path(__file__).parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# 导入工作流步骤
try:
    from workflow.step_train_model import step_train_model
    from workflow.step_bayesian_optimization import step_bayesian_optimization
    from workflow.step_structure_calculation import step_structure_calculation
    from workflow.step_merge_results import step_merge_results
    from workflow.step_extract_materials import step_extract_materials
    from utils.update_dataset import update_dataset
    from utils.progress_tracker import ProgressTracker
    from utils.path_config import PathConfig
    from utils.reproducibility import setup_reproducibility
except Exception as e:
    print(f"[FATAL] Failed to import workflow steps: {e}")
    sys.exit(1)

import pandas as pd
import json

# ============== 路径配置 ==============
# 配置选项 1: BO 模式（仅贝叶斯优化，无 AI 评估）- 使用统一父目录
RUN_MODE = "bo_new"  # 运行模式：bo 或 llm
RESULTS_ROOT = f"{RUN_MODE}/results"
MODELS_ROOT = f"{RUN_MODE}/models/GPR"
DATA_ROOT = f"{RUN_MODE}/data"

# 配置选项 2: LLM 模式（带 AI 评估和理论文档，参考 main.py）
# RUN_MODE = "llm"
# RESULTS_ROOT = f"{RUN_MODE}/results"
# MODELS_ROOT = f"{RUN_MODE}/models/GPR"
# DATA_ROOT = f"{RUN_MODE}/data"
# DOC_ROOT = f"{RUN_MODE}/doc"  # BO 模式不需要

# 使用命令行参数可以指定初始数据来源
# --init-data: 初始数据文件路径

# ============== 默认配置 ==============
DEFAULT_CONFIG = {
    'samples': 100,         # 贝叶斯采样数量
    'xi': 0.01,             # EI 探索参数
    'n_top_candidates': 20, # 贝叶斯优化筛出的候选材料数量（作为后续计算输入）
    'n_structures': 5,      # 每个组分生成的结构数量
    'max_workers': 4,       # 结构生成并行数
    'relax_workers': 1,     # 弛豫并行数
    'phonon_workers': 1,    # 声子计算并行数
    'pressure': 0.0,        # 弛豫压力 (GPa)
    'device': 'cuda',       # 计算设备（向后兼容）
    'gpus': ['cuda:0'],     # GPU 列表，默认单 GPU
    'k_threshold': 1.0,     # 热导率阈值 (W/mK)
    'n_select': 10,         # 绛涢€夎繘鍏ヨ绠楃殑鏉愭枡鏁伴噺 (鐢ㄤ簬妯℃嫙 LLM 绛涢€夌殑婕忔枟)
    'seed': 42,             # 全局随机种子
    'seed_stride': 1000,    # 每轮派生种子跨度
    'deterministic_torch': True,
    'allow_partial_structure': False,  # 是否允许跳过未完成的结构计算
}

# BO 流程的步骤列表（与 main.py 不同，没有 ai_evaluation 和 document_update）
BO_STEPS = [
    "train_model",
    "bayesian_optimization",
    "structure_calculation",
    "merge_results",
    "success_extraction",
    "data_update"
]


def load_fallback_bo_candidates(limit=5, fallback_iteration=15):
    fallback_path = project_root / RESULTS_ROOT / f"iteration_{fallback_iteration}" / "selected_results" / "bo_candidates.json"
    if not fallback_path.exists():
        return []
    try:
        with open(fallback_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to load fallback BO candidates: {exc}")
        return []
    if not isinstance(data, list):
        return []
    samples = []
    for item in data[:limit]:
        formula = item.get("formula")
        if not formula:
            continue
        entry = {"formula": formula}
        if "composition" in item:
            entry["composition"] = item["composition"]
        samples.append(entry)
    return samples


def prepare_initial_data(init_data_path=None):
    """Prepare iteration_0 data for BO workflow."""
    bo_data_root = project_root / DATA_ROOT
    
    # 检查 iteration_0 是否存在
    bo_iter0 = bo_data_root / "iteration_0" / "data.csv"
    if not bo_iter0.exists():
        print(f"[INFO] Initial data not found: {bo_iter0}")
        
        # 尝试来源列表
        sources = []
        
        # 如果提供了自定义路径，优先使用
        if init_data_path:
            custom_data = Path(init_data_path)
            if not custom_data.is_absolute():
                custom_data = project_root / custom_data
            sources.append(custom_data)
            print(f"[INFO] Using custom initial data: {custom_data}")
        
        # 默认来源
        sources.extend([
            project_root / "llm" / "data" / "iteration_0" / "data.csv",
            project_root / "data" / "processed_data.csv",
            project_root / "data" / "iteration_0" / "data.csv"
        ])
        
        found = False
        for src in sources:
            if src.exists():
                print(f"[INFO] Found source data: {src}")
                bo_iter0.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, bo_iter0)
                print(f"[INFO] Copied to: {bo_iter0}")
                found = True
                break
        
        if not found:
            print("[WARN] No valid initial data source found; first iteration may fail.")
    else:
        print(f"\n{'#'*80}")
        print(f"{'#'*80}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Bayesian Optimization materials discovery (BO-only).")
    
    # 核心参数
    parser.add_argument('--max-iterations', type=int, default=20,
                        help='maximum iterations (default: 5)')
    parser.add_argument('--samples', type=int, default=None,
                        help='number of BO samples per iteration')
    parser.add_argument('--n-top-candidates', type=int, default=None,
                        help='number of top BO candidates kept for downstream calculation')
    parser.add_argument('--n-structures', type=int, default=None,
                        help='number of generated structures per composition')
    parser.add_argument('--n-select', type=int, default=None,
                        help='number of materials selected for structure calculation')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='number of GPUs to use (e.g. 3 -> cuda:0,1,2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility')
    parser.add_argument('--non-deterministic-torch', action='store_true',
                        help='disable deterministic torch kernels')
    parser.add_argument('--start-iteration', type=int, default=1,
                        help='start iteration (default: 1)')
    parser.add_argument('--reset', action='store_true',
                        help='Reset all progress and start from scratch')
    parser.add_argument('--allow-partial-structure', action='store_true',
                        help='allow workflow to continue when some structure tasks fail')
    # 路径参数 - 控制初始数据来源
    parser.add_argument('--init-data', type=str, default='data/processed_data.csv',
                        help='path to initial dataset (default: data/processed_data.csv)')
    
    args = parser.parse_args()
    return args


def run_single_iteration(iteration_num: int, config: dict, tracker: ProgressTracker, initial_samples=None):
    """Run one BO-only iteration."""
    print("\n" + "=" * 80)
    print(f">> Start Iteration {iteration_num} (Mode: BO-only)")
    print("=" * 80)
    sys.stdout.flush()
    
    results = {}
    
    # ========== 步骤 1: 训练 GPR 模型 ==========
    step_key = "train_model"
    if tracker.is_step_completed(iteration_num, step_key):
        print(f"[SKIP] Step 1 ({step_key}) already completed")
        results['train'] = {'success': True}
    else:
        train_result = step_train_model(
            iteration_num=iteration_num,
            data_root=DATA_ROOT,
            models_root=MODELS_ROOT,
            path_config=config.get('path_config')
        )
        results['train'] = train_result
        if train_result['success']:
            tracker.mark_step_completed(iteration_num, step_key)
        else:
            print(f"[ERROR] Step 1 failed: {train_result.get('error')}")
            return results
    
    # ========== 步骤 2: 贝叶斯优化 ==========
    # ========== Step 2: Bayesian Optimization ==========
    step_key = "bayesian_optimization"
    candidate_materials = []
    
    # 检查是否已完成，若已完成则从文件加载
    save_dir = project_root / RESULTS_ROOT / f'iteration_{iteration_num}' / 'selected_results'
    save_file = save_dir / 'bo_candidates.json'
    
    if tracker.is_step_completed(iteration_num, step_key):
        print(f"[SKIP] Step 2 ({step_key}) already completed")
        # 从保存的文件加载候选材料
        if save_file.exists():
            try:
                with open(save_file, 'r', encoding='utf-8') as f:
                    candidate_materials = json.load(f)
                print(f"[INFO] Loaded {len(candidate_materials)} BO candidates from file: {save_file}")
                results['bayes'] = {'success': True, 'top_materials': candidate_materials}
            except Exception as e:
                print(f"[ERROR] Failed to load BO candidates: {e}")
                return results
        else:
            print(f"[ERROR] Candidate file not found: {save_file}")
            return results
    else:
        print("")
        print("#"*80)
        print("Step 2/5: Bayesian Optimization")
        print("#"*80)
        bayes_result = step_bayesian_optimization(
            iteration_num=iteration_num,
            xi=config['xi'],
            n_samples=config['samples'],
            n_top=config['n_top_candidates'],
            initial_samples=initial_samples,
            seed=config.get('seed'),
            seed_stride=config.get('seed_stride', 1000),
            models_root=MODELS_ROOT,
            results_root=RESULTS_ROOT,
            path_config=config.get('path_config')
        )
        results['bayes'] = bayes_result
        
        if bayes_result['success']:
            candidate_materials = bayes_result.get('top_materials', [])
            tracker.mark_step_completed(iteration_num, step_key)
            
            # Save candidates for reuse
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(candidate_materials, f, indent=2, ensure_ascii=False)
            print(f"[OK] Saved candidates: {save_file}")
        else:
            print(f"[ERROR] Step 2 failed: {bayes_result.get('error')}")
            return results

    step_key_select = "selection_bo"
    selected_materials = []
    
    n_select = config.get('n_select', 5)
    if candidate_materials:
        selected_materials = candidate_materials[:n_select]
        print(f"[SELECT] BO candidates {len(candidate_materials)} -> top {len(selected_materials)} for calculation")

    # ========== 步骤 3: 结构生成与计算（对应原步骤 4） ==========
    step_key = "structure_calculation"
    if tracker.is_step_completed(iteration_num, step_key):
        print(f"[SKIP] Step 3 ({step_key}) already completed")
        results['structure'] = {'success': True}
    else:
        print(f"\n{'#'*80}")
        print("Step 3/5: Structure generation and calculation")
        print(f"{'#'*80}")
        
        if not selected_materials:
            print("[ERROR] No candidate materials available for structure generation.")
            return results
            
        structure_result = step_structure_calculation(
            iteration_num=iteration_num,
            materials=selected_materials,  # 使用截取后的结果
            n_structures=config['n_structures'],
            max_workers=config['max_workers'],
            relax_workers=config['relax_workers'],
            phonon_workers=config['phonon_workers'],
            pressure=config['pressure'],
            device=config['device'],
            gpus=config.get('gpus', ['cuda:0']),  # 浼犻€扜PU鍒楄〃
            allow_partial_completion=config.get('allow_partial_structure', False),
            results_root=RESULTS_ROOT,
            tracker=tracker,  # 浼犻€抰racker浠ユ敮鎸佸瓙姝ラ璺熻釜
            path_config=config.get('path_config')
        )
        results['structure'] = structure_result
        
        if structure_result.get('completed'):
            tracker.mark_step_completed(iteration_num, step_key)
        else:
            print("[INFO] Structure calculation not completed; keep progress and continue later.")
            return results

    # ========== 步骤 4: 提取成功和稳定材料（对应原步骤 5） ==========
    # ========== Step: Merge Results ==========
    step_key = "merge_results"
    merge_result = {"success": False}
    if tracker.is_step_completed(iteration_num, step_key):
        print(f"[SKIP] Step merge_results completed")
        merge_result = {"success": True}
        results["merge"] = merge_result
    else:
        merge_result = step_merge_results(
            iteration_num=iteration_num,
            results_root=RESULTS_ROOT,
            tracker=tracker
        )
        results["merge"] = merge_result
        if not merge_result.get("success"):
            print(f"[ERROR] merge_results failed: {merge_result.get('error')}")
            return results

    step_key = "success_extraction"
    extract_result = {'success': False}
    
    if tracker.is_step_completed(iteration_num, step_key):
        print(f"[SKIP] Step 4 ({step_key}) already completed")
        extract_result = {'success': True, 'has_success': True} # 假定成功以继续流程
        results['extract'] = extract_result
    else:
        print(f"\n{'#'*80}")
        print("Step 4/5: Extract success and stable materials")
        print(f"{'#'*80}")
        
        extract_result = step_extract_materials(
            iteration_num=iteration_num,
            k_threshold=config['k_threshold'],
            results_root=RESULTS_ROOT
        )
        results['extract'] = extract_result
        
        if extract_result.get('success') or extract_result.get('no_materials'):
            tracker.mark_step_completed(iteration_num, step_key)
            
            # --- 新增：汇总结果到 results 根目录（追加模式） ---
            try:
                aggregated_results_dir = project_root / RESULTS_ROOT
                aggregated_results_dir.mkdir(exist_ok=True, parents=True)
                
                # 定义汇总文件
                summary_files = {
                    'success': aggregated_results_dir / 'success_materials.csv',
                    'stable': aggregated_results_dir / 'stable_materials.csv'
                }
                
                # 确定本轮的源文件
                source_files = {}
                if extract_result.get('success_deduped_file'): 
                    source_files['success'] = extract_result['success_deduped_file']
                elif extract_result.get('success_file'):
                    source_files['success'] = extract_result['success_file']
                    
                if extract_result.get('stable_deduped_file'): 
                    source_files['stable'] = extract_result['stable_deduped_file']
                elif extract_result.get('stable_file'):
                    source_files['stable'] = extract_result['stable_file']
                
                # 执行追加逻辑
                for key, source_path in source_files.items():
                    if source_path and os.path.exists(source_path):
                        # 读取新增数据
                        df_new = pd.read_csv(source_path)
                        # 添加 iteration 列
                        df_new.insert(0, 'iteration', iteration_num)
                        
                        target_file = summary_files[key]
                        
                        if target_file.exists():
                            # 读取现有汇总
                            df_existing = pd.read_csv(target_file)
                            # 合并（保留所有历史记录，仅追加）
                            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                            df_combined.to_csv(target_file, index=False)
                            print(f"  [LOG] Appended to summary: {RESULTS_ROOT}/{target_file.name} (total: {len(df_combined)})")
                        else:
                            # 创建新表
                            df_new.to_csv(target_file, index=False)
                            print(f"  [LOG] Created summary: {RESULTS_ROOT}/{target_file.name} (rows: {len(df_new)})")

            except Exception as e:
                print(f"[WARN] Failed to update summary files: {e}")
            # -----------------------------------------------------

            if extract_result.get('no_materials'):
                print("[INFO] No materials met success/stability criteria in this iteration.")
                extract_result['success'] = True
                extract_result['has_success'] = False
                extract_result['has_stable'] = False
        else:
            print(f"[ERROR] Step 4 failed: {extract_result.get('error')}")
            return results

    # ========== 步骤 5: 更新数据集（对应原步骤 6，但去除文档更新） ==========
    step_key = "data_update"
    if tracker.is_step_completed(iteration_num, step_key):
        print(f"[SKIP] Step 5 ({step_key}) already completed")
        results['update'] = {'success': True}
    else:
        print(f"\n{'#'*80}")
        print("Step 5/5: Update dataset (CSV only)")
        print(f"{'#'*80}")
        
        has_success = extract_result.get('has_success', False)
        has_stable = extract_result.get('has_stable', False)
        
        success_csv = extract_result.get('success_deduped_file') or extract_result.get('success_file')
        stable_csv = extract_result.get('stable_deduped_file') or extract_result.get('stable_file')
        
        target_csv = None
        if has_success and success_csv:
            target_csv = success_csv
            print(f"Prepare to merge success materials: {target_csv}")
        elif has_stable and stable_csv:
            target_csv = stable_csv
            print(f"Prepare to merge stable materials: {target_csv}")
        
        updated_path = None
        
        prev_iteration = iteration_num - 1
        origin_csv = None
        # 查找上一轮数据
        for i in range(prev_iteration, -1, -1):
            candidate = project_root / DATA_ROOT / f"iteration_{i}" / "data.csv"
            if candidate.exists():
                origin_csv = candidate
                break
        
        output_dir = project_root / DATA_ROOT / f"iteration_{iteration_num}"
        
        if target_csv and origin_csv:
            print(f"Merge source: {target_csv}")
            print(f"Base dataset: {origin_csv}")
            
            try:
                # 直接调用 update_dataset 工具函数
                updated_path = update_dataset(
                    success_csv=str(target_csv),
                    origin_csv=str(origin_csv),
                    output_dir=str(output_dir)
                )
            except Exception as e:
                print(f"[ERROR] Failed to update dataset: {e}")
        else:
            if not origin_csv:
                print(f"[ERROR] Previous dataset not found (searched iteration 0..{prev_iteration}).")
        
        # 如果没有更新（没有新材料或更新失败），直接复制上一轮数据
        if not updated_path:
            print("No new materials or update failed; copy previous dataset for continuity.")
            if origin_csv:
                output_dir.mkdir(parents=True, exist_ok=True)
                dest = output_dir / "data.csv"
                shutil.copy2(origin_csv, dest)
                updated_path = str(dest)
                print(f"Copied dataset: {dest}")
            else:
                print("[ERROR] No historical dataset available to copy.")
        
        if updated_path:
            tracker.mark_step_completed(iteration_num, step_key)
            results['update'] = {'success': True, 'path': updated_path}
            print(f"[OK] Dataset update completed: {updated_path}")
        else:
            results['update'] = {'success': False}
    
    print("\n" + "=" * 80)
    print(f"[OK] Iteration {iteration_num} completed.")
    print("=" * 80)
    
    return results


def main():
    try:
        # ========== 配置日志系统 ==========
        # 创建 results 目录
        results_dir = project_root / RESULTS_ROOT
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成日志文件名（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = results_dir / f"run_{timestamp}.log"
        
        # 配置日志格式
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # 配置日志处理器
        handlers = [
            logging.FileHandler(log_file, encoding='utf-8'),  # 文件输出
            logging.StreamHandler(sys.stdout)  # 控制台输出
        ]
        
        # 閰嶇疆鏍规棩蹇楄褰曞櫒
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=handlers,
            force=True  # 强制重新配置
        )
        
        logger = logging.getLogger(__name__)
        logger.info("=" * 80)
        logger.info(f"Program started. Log file: {log_file}")
        logger.info("=" * 80)
        
        # 强制刷新日志到文件
        for handler in logging.root.handlers:
            handler.flush()
        
        # 控制台也打印日志位置
        print(f"\n{'='*80}")
        print(f"[LOG] Log file: {log_file}")
        print(f"{'='*80}\n")
        
        args = parse_args()
        config = DEFAULT_CONFIG.copy()
        
        if args.samples: config['samples'] = args.samples
        if args.n_top_candidates: config['n_top_candidates'] = args.n_top_candidates
        if args.n_structures: config['n_structures'] = args.n_structures
        if args.n_select: config['n_select'] = args.n_select
        if args.num_gpus is not None:
            config['gpus'] = [f'cuda:{i}' for i in range(args.num_gpus)]
            print(f"  GPU config: {config['gpus']}")
        config['seed'] = int(args.seed)
        config['deterministic_torch'] = not bool(args.non_deterministic_torch)
        repro_info = setup_reproducibility(
            seed=config['seed'],
            deterministic_torch=config['deterministic_torch'],
        )
        print(f"  Reproducibility seed: {repro_info['seed']}")
        print(f"  Deterministic torch: {repro_info['deterministic_torch']}")
        config['allow_partial_structure'] = args.allow_partial_structure
        
        print("=" * 80)
        print("Bayesian Optimization Materials Discovery")
        print(f"Data root: {DATA_ROOT}")
        print(f"Models root: {MODELS_ROOT}")
        print(f"Results root: {RESULTS_ROOT}")
        print("=" * 80)
        
        # 初始化数据（传入自定义路径）
        prepare_initial_data(init_data_path=args.init_data)
        
        # 创建 PathConfig 对象
        path_config = PathConfig.from_run_mode(
            project_root=project_root,
            run_mode=RUN_MODE,
            init_data_path=args.init_data,
            init_doc_path=None  # BO 模式不使用文档
        )
        config['path_config'] = path_config
        
        # BO 流程使用自定义步骤列表（没有 ai_evaluation 和 document_update）
        tracker = ProgressTracker(base_dir=RESULTS_ROOT, steps=BO_STEPS)
        
        
        if args.reset:
            for i in range(1, args.max_iterations + 1):
                tracker.reset_round(i)
            print("[INFO] Reset progress for all iterations in range.")
        
        initial_samples = None
        
        for iteration_num in range(args.start_iteration, args.max_iterations + 1):
            if tracker.is_round_completed(iteration_num):
                 # 这里有一个小问题：progress.json 是否是共享的？
                 # 如果 main.py 和 main_bo.py 共用同一个 progress.json，可能会冲突。
                 # 需要确认 ProgressTracker 的默认路径是否固定。
                print(f"\n[SKIP] Iteration {iteration_num} already completed in tracker")
                continue
                
            results = run_single_iteration(iteration_num, config, tracker, initial_samples)
            
             # 检查是否成功
             # 注意：这里的 success keys 和 main.py 不完全一致，依赖 run_single_iteration 的返回值
            if not results.get('extract', {}).get('success', False):
                 print(f"[WARN] Iteration {iteration_num} did not fully succeed")
            
            # 准备下一轮 Initial Samples
            new_initial_samples = None
            extract_res = None
            if 'extract' in results:
                extract_res = results['extract']
                # 优先取 success
                sample_file = None
                is_stable_fallback = False
                
                if extract_res.get('has_success'):
                     sample_file = extract_res.get('success_deduped_file') or extract_res.get('success_file')
                elif extract_res.get('has_stable'):
                     sample_file = extract_res.get('stable_deduped_file') or extract_res.get('stable_file')
                     is_stable_fallback = True
                
                if sample_file and Path(sample_file).exists():
                    try:
                        df = pd.read_csv(sample_file)
                        new_initial_samples = []
                        for _, row in df.iterrows():
                             # 处理各种可能的列名
                            formula = row.get('formula') or row.get('Formula') or row.get('缁勫垎')
                            kappa = (
                                row.get('thermal_conductivity')
                                or row.get('kappa')
                                or row.get('热导率(W/m·K)')
                                or row.get('Thermal_Conductivity')
                                or row.get('鐑鐜?(W/m路K)')  # 兼容历史乱码列名
                            )
                            
                            if formula and kappa is not None:
                                # 如果是 stable fallback，需要额外过滤 k < 5
                                if is_stable_fallback and float(kappa) >= 5.0:
                                    continue
                                new_initial_samples.append({'formula': formula, 'thermal_conductivity': kappa})
                                
                        if new_initial_samples:
                            source_text = "stable materials (k<5)" if is_stable_fallback else "success materials"
                            print(f"[INFO] Extracted {len(new_initial_samples)} initial samples for next iteration (source: {source_text})")
                        else:
                            print("[WARN] Extracted initial samples are empty (possibly filtered by K threshold).")
                            
                    except Exception as e:
                        print(f"[ERROR] Failed to read initial samples: {e}")

            if (not new_initial_samples) and extract_res and (not extract_res.get('has_success')) and (not extract_res.get('has_stable')):
                fallback_samples = load_fallback_bo_candidates()
                if fallback_samples:
                    new_initial_samples = fallback_samples
                    print("[INFO] No success/stable materials; using top5 from iteration_15 bo_candidates.json")
                else:
                    print("[WARN] No success/stable materials and fallback bo_candidates.json missing or empty")

            if new_initial_samples:
                initial_samples = new_initial_samples
            elif initial_samples:
                initial_samples = None
                print('No new materials; force random sampling next iteration')
                
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

