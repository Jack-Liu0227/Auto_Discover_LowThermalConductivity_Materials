"""
并行计算优化

提供并行计算功能：
- 候选材料并行评估
- 多GPU支持
- 进程池管理
- 任务队列
"""

import logging
import multiprocessing as mp
from typing import List, Callable, Any, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import torch

logger = logging.getLogger(__name__)


class ParallelExecutor:
    """并行执行器"""
    
    def __init__(
        self,
        n_workers: Optional[int] = None,
        use_threads: bool = False,
        gpu_ids: Optional[List[int]] = None
    ):
        """
        初始化并行执行器
        
        Args:
            n_workers: 工作进程/线程数，默认为CPU核心数
            use_threads: 是否使用线程池（默认使用进程池）
            gpu_ids: 可用的GPU ID列表
        """
        if n_workers is None:
            n_workers = mp.cpu_count()
        
        self.n_workers = n_workers
        self.use_threads = use_threads
        self.gpu_ids = gpu_ids or []
        
        # 检测可用GPU
        if torch.cuda.is_available() and not self.gpu_ids:
            self.gpu_ids = list(range(torch.cuda.device_count()))
        
        logger.info(
            f"ParallelExecutor initialized: "
            f"n_workers={n_workers}, use_threads={use_threads}, "
            f"gpu_ids={self.gpu_ids}"
        )
    
    def map(
        self,
        func: Callable,
        items: List[Any],
        show_progress: bool = False
    ) -> List[Any]:
        """
        并行映射函数
        
        Args:
            func: 要并行执行的函数
            items: 输入项列表
            show_progress: 是否显示进度条
            
        Returns:
            结果列表
        """
        if len(items) == 0:
            return []
        
        # 选择执行器
        executor_class = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
        
        results = []
        
        with executor_class(max_workers=self.n_workers) as executor:
            # 提交任务
            futures = {executor.submit(func, item): i for i, item in enumerate(items)}
            
            # 收集结果
            completed = 0
            total = len(futures)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    idx = futures[future]
                    results.append((idx, result))
                    
                    completed += 1
                    if show_progress and completed % 10 == 0:
                        logger.info(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
                
                except Exception as e:
                    idx = futures[future]
                    logger.error(f"Task {idx} failed: {e}")
                    results.append((idx, None))
        
        # 按原始顺序排序
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def map_with_gpu(
        self,
        func: Callable,
        items: List[Any],
        show_progress: bool = False
    ) -> List[Any]:
        """
        使用GPU并行映射函数
        
        Args:
            func: 要并行执行的函数（需要接受gpu_id参数）
            items: 输入项列表
            show_progress: 是否显示进度条
            
        Returns:
            结果列表
        """
        if len(items) == 0:
            return []
        
        if not self.gpu_ids:
            logger.warning("No GPUs available, falling back to CPU")
            return self.map(func, items, show_progress)
        
        # 为每个任务分配GPU
        tasks = []
        for i, item in enumerate(items):
            gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
            tasks.append((item, gpu_id))
        
        # 包装函数以传递GPU ID
        def wrapped_func(task):
            item, gpu_id = task
            return func(item, gpu_id=gpu_id)
        
        return self.map(wrapped_func, tasks, show_progress)


class TaskQueue:
    """任务队列（用于异步任务管理）"""
    
    def __init__(self, max_size: int = 1000):
        """
        初始化任务队列
        
        Args:
            max_size: 队列最大大小
        """
        self.queue = mp.Queue(maxsize=max_size)
        self.results = mp.Queue()
        self.max_size = max_size
        
        logger.info(f"TaskQueue initialized with max_size={max_size}")
    
    def put(self, task: Any):
        """
        添加任务到队列
        
        Args:
            task: 任务对象
        """
        self.queue.put(task)
    
    def get(self, timeout: Optional[float] = None) -> Any:
        """
        从队列获取任务
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            任务对象
        """
        return self.queue.get(timeout=timeout)
    
    def put_result(self, result: Any):
        """
        添加结果到结果队列
        
        Args:
            result: 结果对象
        """
        self.results.put(result)
    
    def get_result(self, timeout: Optional[float] = None) -> Any:
        """
        从结果队列获取结果
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            结果对象
        """
        return self.results.get(timeout=timeout)

    def is_empty(self) -> bool:
        """
        检查队列是否为空

        Returns:
            是否为空
        """
        return self.queue.empty()

    def size(self) -> int:
        """
        获取队列大小

        Returns:
            队列大小
        """
        return self.queue.qsize()


def parallel_evaluate_candidates(
    candidates: List[Any],
    evaluate_func: Callable,
    n_workers: Optional[int] = None,
    use_gpu: bool = True,
    gpu_ids: Optional[List[int]] = None
) -> List[Any]:
    """
    并行评估候选材料

    Args:
        candidates: 候选材料列表
        evaluate_func: 评估函数
        n_workers: 工作进程数
        use_gpu: 是否使用GPU
        gpu_ids: 可用的GPU ID列表

    Returns:
        评估结果列表
    """
    executor = ParallelExecutor(
        n_workers=n_workers,
        use_threads=False,
        gpu_ids=gpu_ids
    )

    if use_gpu and executor.gpu_ids:
        logger.info(f"Evaluating {len(candidates)} candidates using {len(executor.gpu_ids)} GPUs")
        return executor.map_with_gpu(evaluate_func, candidates, show_progress=True)
    else:
        logger.info(f"Evaluating {len(candidates)} candidates using {executor.n_workers} CPUs")
        return executor.map(evaluate_func, candidates, show_progress=True)


def get_optimal_worker_count(task_type: str = "cpu") -> int:
    """
    获取最优工作进程数

    Args:
        task_type: 任务类型（"cpu", "gpu", "io"）

    Returns:
        最优工作进程数
    """
    if task_type == "cpu":
        # CPU密集型任务：使用CPU核心数
        return mp.cpu_count()
    elif task_type == "gpu":
        # GPU密集型任务：使用GPU数量
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 1
    elif task_type == "io":
        # IO密集型任务：使用CPU核心数的2-4倍
        return mp.cpu_count() * 2
    else:
        return mp.cpu_count()


def distribute_tasks_to_gpus(
    tasks: List[Any],
    gpu_ids: Optional[List[int]] = None
) -> Dict[int, List[Any]]:
    """
    将任务分配到多个GPU

    Args:
        tasks: 任务列表
        gpu_ids: GPU ID列表

    Returns:
        GPU ID到任务列表的映射
    """
    if gpu_ids is None:
        if torch.cuda.is_available():
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            gpu_ids = [-1]  # CPU

    # 均匀分配任务
    distribution = {gpu_id: [] for gpu_id in gpu_ids}

    for i, task in enumerate(tasks):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        distribution[gpu_id].append(task)

    logger.info(f"Distributed {len(tasks)} tasks to {len(gpu_ids)} GPUs")
    for gpu_id, gpu_tasks in distribution.items():
        logger.info(f"  GPU {gpu_id}: {len(gpu_tasks)} tasks")

    return distribution

