"""
并行声子谱计算模块
使用多进程并行计算多个结构的声子谱以提升效率
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import as_completed
import os
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
    
    # ============ 关键：在导入任何模块之前设置GPU环境变量 ============
    import os
    
    # 设置 PyTorch CUDA 内存优化
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 设置当前任务使用的GPU（必须在import torch之前）
    if gpu_device and ':' in gpu_device:
        gpu_id = gpu_device.split(':')[-1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        # print(f"  [Worker {os.getpid()}] 设置GPU: {gpu_device} (CUDA_VISIBLE_DEVICES={gpu_id})")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
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
        
        # 初始化 Mattersim（在子进程中）
        mattersim = MattersimWrapper()
        
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
                'gamma_min_optical': None
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
            'gamma_min_optical': None
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
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # GPU配置
    if gpus is None:
        gpus = ['cuda']
    
    logger.info(f"📊 GPU配置: {gpus} ({len(gpus)}个)")
    
    # 准备任务，为每个任务分配GPU
    tasks = []
    for i, structure in enumerate(structures):
        assigned_gpu = gpus[i % len(gpus)]
        tasks.append((i, structure, composition.formula, str(comp_dir), assigned_gpu))
    
    logger.info(f"准备计算 {len(structures)} 个结构的声子谱 (max_workers={max_workers})")
    
    results = [None] * len(structures)
    
    # 根据GPU数量和max_workers决定并行模式
    if len(gpus) > 1 and max_workers > 1:
        # 多GPU并行模式：总并行数 = GPU数量 × max_workers
        actual_workers = len(gpus) * max_workers
        logger.info(f"🚀 使用多GPU并行模式")
        logger.info(f"  - GPU数量: {len(gpus)}")
        logger.info(f"  - 每GPU并行数: {max_workers}")
        logger.info(f"  - 总并行数: {actual_workers}")
        
        try:
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                future_to_idx = {
                    executor.submit(calculate_single_phonon_worker, task): task[0]
                    for task in tasks
                }
                
                completed_count = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    completed_count += 1
                    try:
                        result = future.result()
                        results[idx] = result
                        assigned_gpu = tasks[idx][4]
                        
                        if result['success']:
                            print(f"  [{completed_count}/{len(tasks)}] 结构 {idx+1} ({assigned_gpu}): 虚频={result['has_imaginary']}")
                        else:
                            print(f"  [{completed_count}/{len(tasks)}] 结构 {idx+1} ({assigned_gpu}): 计算失败 - {result.get('error', 'Unknown')}")
                    except Exception as e:
                        print(f"  [{completed_count}/{len(tasks)}] 结构 {idx+1} 运行异常: {e}")
                        results[idx] = {'index': idx, 'success': False, 'has_imaginary': "未知"}
        except Exception as e:
            logger.error(f"并行计算发生严重错误: {e}")
            return ["未知"] * len(structures)
    else:
        # 单GPU顺序模式
        logger.info(f"使用顺序模式执行 {len(tasks)} 个任务...")
        for i, task in enumerate(tasks):
            idx = task[0]
            assigned_gpu = task[4]
            result = calculate_single_phonon_worker(task)
            results[idx] = result
            if result['success']:
                print(f"  [{i+1}/{len(tasks)}] 结构 {idx+1} ({assigned_gpu}): 虚频={result['has_imaginary']}")
            else:
                print(f"  [{i+1}/{len(tasks)}] 结构 {idx+1} ({assigned_gpu}): 计算失败")
    
    # 按顺序返回结果，缺失的补充为未知
    normalized_results = []
    for idx, result in enumerate(results):
        if result:
            normalized_results.append(result)
        else:
            normalized_results.append({
                'index': idx,
                'success': False,
                'has_imaginary': "未知",
                'error': "missing_result",
                'min_frequency': None,
                'gamma_min_optical': None
                'gamma_max_acoustic': None
            })
    
    return normalized_results
