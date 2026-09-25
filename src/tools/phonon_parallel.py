"""
并行声子谱计算模块
使用多进程并行计算多个结构的声子谱以提升效率
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import gc

import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


def _cleanup_gpu_memory():
    """清理 GPU 显存"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"GPU 清理警告: {e}")


def calculate_single_phonon_worker(args: tuple) -> Dict[str, Any]:
    """
    计算单个结构的声子谱（在子进程中运行）
    
    Args:
        args: (index, structure, comp_formula, comp_dir_str, gpu_device)
    
    Returns:
        包含计算结果的字典
    """
    i, structure, comp_formula, comp_dir_str, gpu_device = args
    
    import os

    # Keep the logical CUDA index unchanged. Mutating CUDA_VISIBLE_DEVICES here
    # is unsafe when sibling tasks are coordinated by parent threads and also
    # remaps cuda:N to local cuda:0 inside a spawned process.
    worker_device = str(gpu_device or "cuda").strip().lower()
    if worker_device.startswith("cuda"):
        try:
            from utils.gpu_parallel import set_process_cuda_device
        except ImportError:
            from src.utils.gpu_parallel import set_process_cuda_device
        set_process_cuda_device(worker_device)

    try:
        # 在子进程中重新导入必要的模块
        from pathlib import Path
        import logging
        
        #  配置子进程日志
        logging.basicConfig(level=logging.INFO)
        sub_logger = logging.getLogger(__name__)
        
        try:
            from mattersim_wrapper import MattersimWrapper
        except ImportError:
            try:
                from tools.mattersim_wrapper import MattersimWrapper
            except ImportError:
                from .mattersim_wrapper import MattersimWrapper
        
        comp_dir = Path(comp_dir_str)
        
        # 为每个结构创建独立的声子谱文件夹
        phonon_dir = comp_dir / f"{comp_formula}_sample_{i+1}_phonon"
        phonon_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置声子谱图像保存路径
        plot_path = phonon_dir / "phonon_spectrum.png"
        
        sub_logger.info(f"  [进程 {os.getpid()}] [GPU: {gpu_device}] 开始计算结构 {i+1} 的声子谱...")
        
        # Initialize MatterSim with the explicit worker device.
        mattersim = MattersimWrapper(config={"device": worker_device})
        
        # 计算声子谱
        response = mattersim.run(
            structure,
            calculate_phonon=True,
            save_plot=True,
            plot_path=str(plot_path)
        )
        
        if response.is_success():
            result = response.result
            has_imaginary = "是" if result.has_imaginary_freq else "否"
            min_frequency = getattr(result, 'min_frequency', None)
            gamma_min_optical = getattr(result, 'gamma_min_optical', None)
            gamma_max_acoustic = getattr(result, 'gamma_max_acoustic', None)
            
            # 清理 GPU 缓存
            _cleanup_gpu_memory()
            
            return {
                'index': i,
                'success': True,
                'has_imaginary': has_imaginary,
                'min_frequency': min_frequency,
                'gamma_min_optical': gamma_min_optical,
                'gamma_max_acoustic': gamma_max_acoustic,
                'phonon_dir': str(phonon_dir)
            }
        else:
            sub_logger.warning(f"  结构 {i+1} 声子谱计算失败: {response.error}")
            # 即使失败也清理缓存
            _cleanup_gpu_memory()
            return {
                'index': i,
                'success': False,
                'has_imaginary': "未知",
                'error': response.error,
                'min_frequency': None,
                'gamma_min_optical': None,
                'gamma_max_acoustic': None
            }
            
    except Exception as e:
        import traceback
        err_msg = f"结构 {i+1} 计算异常: {e}\n{traceback.format_exc()}"
        logger.warning(err_msg)
        return {
            'index': i,
            'success': False,
            'has_imaginary': "未知",
            'error': str(e),
            'min_frequency': None,
            'gamma_min_optical': None,
            'gamma_max_acoustic': None
        }


def calculate_phonons_parallel(
    structures: List,
    composition,
    comp_dir: Path,
    max_workers: int = 4,
    gpus: List[str] = None
) -> List[Dict[str, Any]]:
    """
    并行计算多个结构的声子谱
    
    Args:
        structures: 结构列表
        composition: 组分
        comp_dir: 组分目录
        max_workers: 每个GPU的并行数
        gpus: GPU列表，如果为None则使用单个cuda设备
    
    Returns:
        声子谱计算结果列表（按结构顺序）
    """
    try:
        from utils.gpu_parallel import (
            normalize_gpu_list,
            run_spawn_task,
            run_tasks_by_gpu,
            validate_gpu_devices,
        )
    except ImportError:
        from src.utils.gpu_parallel import (
            normalize_gpu_list,
            run_spawn_task,
            run_tasks_by_gpu,
            validate_gpu_devices,
        )

    gpus = normalize_gpu_list(gpus, device="cuda")
    validate_gpu_devices(gpus)
    logger.info(f"📊 GPU配置: {gpus} ({len(gpus)}个)")

    tasks = []
    for i, structure in enumerate(structures):
        assigned_gpu = gpus[i % len(gpus)]
        tasks.append((i, structure, composition.formula, str(comp_dir), assigned_gpu))

    logger.info(f"准备计算 {len(structures)} 个结构的声子谱 (max_workers={max_workers})")
    workers_per_gpu = max(1, int(max_workers)) if len(gpus) > 1 else 1
    actual_workers = len(gpus) * workers_per_gpu
    if len(gpus) > 1:
        logger.info("🚀 使用多GPU并行模式")
    else:
        logger.info("使用单GPU顺序模式")
    logger.info(f"  - GPU数量: {len(gpus)}")
    logger.info(f"  - 每GPU并行数: {workers_per_gpu}")
    logger.info(f"  - 总并行数: {actual_workers}")

    def _run_phonon(task):
        try:
            return run_spawn_task(calculate_single_phonon_worker, task)
        except Exception as exc:
            return {
                'index': task[0],
                'success': False,
                'has_imaginary': "未知",
                'error': str(exc),
                'min_frequency': None,
                'gamma_min_optical': None,
                'gamma_max_acoustic': None,
            }

    results = run_tasks_by_gpu(
        tasks,
        gpus=gpus,
        workers_per_gpu=workers_per_gpu,
        task_device=lambda task: task[4],
        run_task=_run_phonon,
    )
    for idx, result in enumerate(results):
        assigned_gpu = tasks[idx][4]
        if result.get('success'):
            print(
                f"  [{idx+1}/{len(tasks)}] 结构 {idx+1} "
                f"({assigned_gpu}): 虚频={result['has_imaginary']}"
            )
        else:
            print(
                f"  [{idx+1}/{len(tasks)}] 结构 {idx+1} "
                f"({assigned_gpu}): 计算失败 - {result.get('error', 'Unknown')}"
            )

    return results
