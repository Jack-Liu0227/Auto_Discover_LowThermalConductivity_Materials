"""
进度管理器 - 支持断点续传
使用JSON文件跟踪任务进度，支持恢复中断的任务
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


class ProgressManager:
    """
    进度管理器 - 跟踪和恢复任务进度
    
    进度文件格式：
    {
        "stage": "generation|relaxation|phonon|complete",
        "materials": {
            "formula1": {
                "generation": "pending|running|complete|failed",
                "relaxation": "pending|running|complete|failed",
                "phonon": "pending|running|complete|failed",
                "structures": ["struct1.cif", "struct2.cif", ...],
                "error": "error message if failed"
            },
            ...
        },
        "started_at": "2025-12-23 19:30:00",
        "updated_at": "2025-12-23 19:35:00"
    }
    """
    
    def __init__(self, progress_file: str = "results/iteration_1/progress.json"):
        """
        初始化进度管理器
        
        Args:
            progress_file: 进度文件路径
        """
        self.progress_file = Path(progress_file)
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.progress = self._load_progress()
    
    def _load_progress(self) -> Dict[str, Any]:
        """加载进度文件"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                logger.info(f"✅ 从进度文件恢复: {self.progress_file}")
                return progress
            except Exception as e:
                logger.warning(f"无法加载进度文件: {e}，创建新进度")
        
        # 创建新进度
        return {
            "stage": "generation",
            "materials": {},
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _save_progress(self):
        """保存进度文件"""
        with self.lock:
            try:
                self.progress["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(self.progress_file, 'w', encoding='utf-8') as f:
                    json.dump(self.progress, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"保存进度文件失败: {e}")
    
    def initialize_materials(self, formulas: List[str]):
        """
        初始化材料列表
        
        Args:
            formulas: 材料化学式列表
        """
        with self.lock:
            for formula in formulas:
                if formula not in self.progress["materials"]:
                    self.progress["materials"][formula] = {
                        "generation": "pending",
                        "relaxation": "pending",
                        "phonon": "pending",
                        "structures": [],
                        "error": None
                    }
        self._save_progress()
    
    def update_material_status(
        self,
        formula: str,
        stage: str,
        status: str,
        structures: Optional[List[str]] = None,
        error: Optional[str] = None
    ):
        """
        更新材料状态
        
        Args:
            formula: 化学式
            stage: 阶段 (generation/relaxation/phonon)
            status: 状态 (pending/running/complete/failed)
            structures: 结构文件列表
            error: 错误信息
        """
        with self.lock:
            if formula not in self.progress["materials"]:
                self.progress["materials"][formula] = {
                    "generation": "pending",
                    "relaxation": "pending",
                    "phonon": "pending",
                    "structures": [],
                    "error": None
                }
            
            self.progress["materials"][formula][stage] = status
            
            if structures:
                self.progress["materials"][formula]["structures"] = structures
            
            if error:
                self.progress["materials"][formula]["error"] = error
        
        self._save_progress()
    
    def set_stage(self, stage: str):
        """
        设置当前阶段
        
        Args:
            stage: 阶段名称
        """
        with self.lock:
            self.progress["stage"] = stage
        self._save_progress()
    
    def get_pending_materials(self, stage: str) -> List[str]:
        """
        获取指定阶段待处理的材料
        
        Args:
            stage: 阶段名称
            
        Returns:
            待处理材料的化学式列表
        """
        with self.lock:
            pending = []
            for formula, status in self.progress["materials"].items():
                # 如果该阶段是 pending 或 failed（允许重试）
                if status.get(stage) in ["pending", "failed"]:
                    pending.append(formula)
            return pending
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计字典
        """
        with self.lock:
            stats = {
                "total": len(self.progress["materials"]),
                "generation": {"complete": 0, "failed": 0, "pending": 0, "running": 0},
                "relaxation": {"complete": 0, "failed": 0, "pending": 0, "running": 0},
                "phonon": {"complete": 0, "failed": 0, "pending": 0, "running": 0}
            }
            
            for material in self.progress["materials"].values():
                for stage in ["generation", "relaxation", "phonon"]:
                    status = material.get(stage, "pending")
                    stats[stage][status] = stats[stage].get(status, 0) + 1
            
            return stats
    
    def print_summary(self):
        """打印进度摘要"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 80)
        print("📊 任务进度汇总")
        print("=" * 80)
        print(f"当前阶段: {self.progress['stage']}")
        print(f"总材料数: {stats['total']}")
        print(f"开始时间: {self.progress.get('started_at', 'N/A')}")
        print(f"更新时间: {self.progress.get('updated_at', 'N/A')}")
        print()
        
        for stage in ["generation", "relaxation", "phonon"]:
            stage_stats = stats[stage]
            print(f"{stage.title()}:")
            print(f"  ✅ 完成: {stage_stats['complete']}")
            print(f"  ❌ 失败: {stage_stats['failed']}")
            print(f"  ⏳ 待处理: {stage_stats['pending']}")
            print(f"  🔄 进行中: {stage_stats['running']}")
        
        print("=" * 80)
    
    def resume_or_start(self) -> bool:
        """
        判断是恢复任务还是新任务
        
        Returns:
            True if resuming, False if starting new
        """
        return self.progress_file.exists() and len(self.progress["materials"]) > 0
    
    def clear_progress(self):
        """清空进度（开始新任务）"""
        if self.progress_file.exists():
            self.progress_file.unlink()
        self.progress = self._load_progress()
        logger.info("进度已清空，开始新任务")
