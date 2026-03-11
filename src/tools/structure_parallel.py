"""
并行晶体结构生成模块
使用多进程并行处理多个组分的结构生成
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import as_completed
import os
import sys

import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

# Limit BLAS/OMP threads before heavy imports in worker processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# 延迟导入，避免模块级别卡住
# 这些导入会在函数内部进行
_imports_done = False
PMGComposition = None
CrystaLLMWrapper = None
Composition = None

def _ensure_imports():
    """确保必要的导入完成"""
    global _imports_done, PMGComposition, CrystaLLMWrapper, Composition
    if _imports_done:
        return
    
    from pymatgen.core import Composition as PMGComp
    PMGComposition = PMGComp
    
    try:
        from tools.crystallm_wrapper import CrystaLLMWrapper as Wrapper
    except ImportError:
        try:
            from crystallm_wrapper import CrystaLLMWrapper as Wrapper
        except ImportError:
            from src.tools.crystallm_wrapper import CrystaLLMWrapper as Wrapper
    CrystaLLMWrapper = Wrapper
    
    try:
        from utils.types import Composition as Comp
    except ImportError:
        try:
            from tools.types import Composition as Comp
        except ImportError:
            from src.utils.types import Composition as Comp
    Composition = Comp
    
    _imports_done = True


def generate_single_composition_worker(args: tuple) -> Dict[str, Any]:
    """
    为单个组分生成结构
    
    Args:
        args: (index, material, wrapper_config, gen_config)
    
    Returns:
        包含生成结果的字典
    """
    # 在worker中确保导入完成
    _ensure_imports()
    
    i, material, wrapper_config, gen_config = args
    
    try:
        # 使用模块级 logger，避免在主进程中重复调用 basicConfig
        formula = material.get('formula', '')
        if not formula:
            return {
                'index': i,
                'formula': 'Unknown',
                'success': False,
                'error': 'No formula provided'
            }
        
        import sys
        logger.info(f"[Task {i+1}] 开始生成 {formula} 的结构...")
        
        # 解析元素组成（使用模块级别的PMGComposition）
        pmg_comp = PMGComposition(formula)
        elements = {str(el): amt for el, amt in pmg_comp.get_el_amt_dict().items()}
        composition = Composition(formula=formula, elements=elements)
        logger.info(f"[Task {i+1}] 元素组成解析完成: {elements}")
        
        # 初始化 wrapper
        try:
            wrapper = CrystaLLMWrapper(**wrapper_config)
            logger.info(f"[Task {i+1}] Wrapper 初始化成功")
        except Exception as init_error:
            logger.error(f"[Task {i+1}] Wrapper 初始化失败: {init_error}")
            import traceback
            traceback.print_exc()
            raise
        
        # 生成结构
        try:
            result = wrapper.run(
                composition,
                **gen_config
            )
            logger.info(f"[Task {i+1}] wrapper.run() 完成")
        except Exception as run_error:
            print(f"[DEBUG Task {i+1}] ❌ wrapper.run() 失败: {run_error}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        
        if result.is_success():
            n_structures = len(result.result)
            n_relaxed = result.metadata.get('n_relaxed', 0)
            logger.info(f"[{i+1}] {formula}: 成功生成 {n_structures} 个结构")
            
            return {
                'index': i,
                'formula': formula,
                'success': True,
                'n_structures': n_structures,
                'n_relaxed': n_relaxed
            }
        else:
            logger.warning(f"[{i+1}] {formula}: 生成失败 - {result.error}")
            return {
                'index': i,
                'formula': formula,
                'success': False,
                'error': result.error
            }
            
    except Exception as e:
        import traceback
        err_msg = f"组分 {i+1} ({material.get('formula', 'Unknown')}) 生成异常: {e}"
        logger.error(err_msg)
        logger.debug(traceback.format_exc())
        return {
            'index': i,
            'formula': material.get('formula', 'Unknown'),
            'success': False,
            'error': str(e)
        }


def generate_structures_parallel(
    materials: List[Dict],
    device: str = "cuda",
    output_dir: str = None,
    n_structures: int = 5,
    relax_structures: bool = True,
    pressure: float = 0.0,
    relax_output_dir: str = None,
    max_workers: int = 4,
    gpus: List[str] = None
) -> List[Dict[str, Any]]:
    """
    并行生成多个组分的结构
    
    Args:
        materials: 材料列表
        device: 默认设备（向后兼容）
        output_dir: 输出目录
        n_structures: 每个材料生成的结构数
        relax_structures: 是否弛豫
        pressure: 弛豫压力
        relax_output_dir: 弛豫输出目录
        max_workers: 最大并行数
        gpus: GPU列表，如果为None则使用device参数
    
    Returns:
        结果列表
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

    # GPU配置
    if gpus is None:
        gpus = [device]  # 向后兼容
    
    logger.info(f"📊 GPU配置: {gpus} ({len(gpus)}个)")
    logger.info(f"准备生成 {len(materials)} 个材料的结构 (max_workers={max_workers})")

    # 准备任务，为每个任务分配GPU
    tasks = []
    for i, m in enumerate(materials):
        # 循环分配GPU
        assigned_gpu = gpus[i % len(gpus)]
        wrapper_config = {'device': assigned_gpu, 'output_dir': output_dir}
        gen_config = {
            'n_structures': n_structures,
            'relax_structures': relax_structures,
            'pressure': pressure,
            'relax_output_dir': relax_output_dir,
            'calculate_properties': False  # 生成阶段仅保存结构，后续流程统一计算热导率/声子谱
        }
        tasks.append((i, m, wrapper_config, gen_config))
        logger.info(f"  任务 {i+1}: {m.get('formula', 'Unknown')} -> {assigned_gpu}")
    
    results = [None] * len(materials)

    # 根据GPU数量和max_workers决定并行模式
    if len(gpus) > 1 and max_workers > 1:
        # 多GPU并行模式：每个GPU运行max_workers个任务
        # 总并行数 = GPU数量 × max_workers
        actual_workers = len(gpus) * max_workers
        logger.info(f"🚀 使用多GPU并行模式")
        logger.info(f"  - GPU数量: {len(gpus)}")
        logger.info(f"  - 每GPU并行数: {max_workers}")
        logger.info(f"  - 总并行数: {actual_workers}")
        
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            future_to_idx = {
                executor.submit(generate_single_composition_worker, task): task[0]
                for task in tasks
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                    formula = tasks[idx][1].get('formula', 'Unknown')
                    assigned_gpu = tasks[idx][2]['device']
                    if result['success']:
                        print(f"[{idx+1}/{len(tasks)}] ✅ {formula} ({assigned_gpu}) 生成成功")
                    else:
                        print(f"[{idx+1}/{len(tasks)}] ❌ {formula} ({assigned_gpu}) 生成失败: {result.get('error')}")
                except Exception as e:
                    print(f"[{idx+1}/{len(tasks)}] ❌ 任务异常: {e}")
                    results[idx] = {'index': idx, 'success': False, 'error': str(e)}
    else:
        # 单GPU顺序模式
        logger.info(f"使用顺序模式执行 {len(materials)} 个任务...")
        for task in tasks:
            idx = task[0]
            formula = task[1].get('formula', 'Unknown')
            assigned_gpu = task[2]['device']
            print(f"[{idx+1}/{len(tasks)}] 处理组分: {formula} ({assigned_gpu})")
            results[idx] = generate_single_composition_worker(task)
            if results[idx]['success']:
                print(f"[{idx+1}/{len(tasks)}] {formula} 生成成功")
            else:
                print(f"[{idx+1}/{len(tasks)}] {formula} 生成失败: {results[idx].get('error')}")

    return [r for r in results if r is not None]
