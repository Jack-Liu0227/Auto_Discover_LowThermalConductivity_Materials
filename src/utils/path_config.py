# -*- coding: utf-8 -*-
"""
路径配置管理
统一管理项目中所有路径，支持 LLM 和 BO 两种模式
"""
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class PathConfig:
    """路径配置类 - 使用 Path 对象"""
    
    # 项目根目录
    project_root: Path
    
    # 运行模式相关路径
    results_root: Path      # 结果存储目录
    models_root: Path       # 模型存储目录
    data_root: Path         # 数据存储目录
    doc_root: Optional[Path] = None  # 文档存储目录 (BO模式不需要)
    
    # 初始文件路径
    init_data_path: Optional[Path] = None
    init_doc_path: Optional[Path] = None
    
    def __post_init__(self):
        """初始化后处理，确保所有路径都是 Path 对象"""
        # 转换为绝对路径
        if not self.project_root.is_absolute():
            self.project_root = self.project_root.resolve()
        
        # 确保相对路径基于 project_root
        if not self.results_root.is_absolute():
            self.results_root = self.project_root / self.results_root
        if not self.models_root.is_absolute():
            self.models_root = self.project_root / self.models_root
        if not self.data_root.is_absolute():
            self.data_root = self.project_root / self.data_root
        
        if self.doc_root and not self.doc_root.is_absolute():
            self.doc_root = self.project_root / self.doc_root
        
        if self.init_data_path and not self.init_data_path.is_absolute():
            self.init_data_path = self.project_root / self.init_data_path
        
        if self.init_doc_path and not self.init_doc_path.is_absolute():
            self.init_doc_path = self.project_root / self.init_doc_path
    
    def get_iteration_data_path(self, iteration: int) -> Path:
        """获取指定迭代的数据文件路径"""
        return self.data_root / f"iteration_{iteration}" / "data.csv"
    
    def get_iteration_model_path(self, iteration: int) -> Path:
        """获取指定迭代的模型目录路径"""
        return self.models_root / f"iteration_{iteration}"
    
    def get_iteration_results_path(self, iteration: int) -> Path:
        """获取指定迭代的结果目录路径"""
        return self.results_root / f"iteration_{iteration}"
    
    def get_model_file_path(self, iteration: int, model_name: str = "gpr_thermal_conductivity.joblib") -> Path:
        """获取指定迭代的模型文件路径"""
        return self.get_iteration_model_path(iteration) / model_name
    
    def get_theory_doc_path(self, iteration: int) -> Path:
        """
        获取理论文档路径
        - iteration 1: 使用初始文档
        - iteration 2+: 使用上一轮更新的文档
        """
        if iteration == 1:
            if self.init_doc_path:
                return self.init_doc_path
            elif self.doc_root:
                return self.doc_root / "v0.0.0" / "Theoretical_principle_document.md"
            else:
                raise ValueError("No document root configured for BO mode")
        else:
            # 使用上一轮 LLM 更新的文档
            if not self.doc_root:
                raise ValueError("No document root configured for theory documents")
            return self.doc_root / f"v0.0.{iteration - 1}" / "Theoretical_principle_document.md"
    
    def create_directories(self):
        """创建必要的目录结构"""
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.models_root.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)
        if self.doc_root:
            self.doc_root.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_run_mode(cls, project_root: Path, run_mode: str, 
                     init_data_path: Optional[str] = None,
                     init_doc_path: Optional[str] = None) -> 'PathConfig':
        """
        根据运行模式创建路径配置
        
        Args:
            project_root: 项目根目录
            run_mode: 运行模式 ('llm' 或 'bo')
            init_data_path: 初始数据文件路径
            init_doc_path: 初始文档路径
        
        Returns:
            PathConfig 实例
        """
        project_root = Path(project_root)
        
        # 基础路径
        results_root = Path(f"{run_mode}/results")
        models_root = Path(f"{run_mode}/models/GPR")
        data_root = Path(f"{run_mode}/data")
        doc_root = Path(f"{run_mode}/doc") if run_mode == "llm" else None
        
        # 初始文件路径
        init_data = Path(init_data_path) if init_data_path else None
        init_doc = Path(init_doc_path) if init_doc_path else None
        
        return cls(
            project_root=project_root,
            results_root=results_root,
            models_root=models_root,
            data_root=data_root,
            doc_root=doc_root,
            init_data_path=init_data,
            init_doc_path=init_doc
        )
    
    def to_dict(self) -> dict:
        """转换为字典格式，用于传递给旧代码"""
        return {
            'project_root': str(self.project_root),
            'results_root': str(self.results_root),
            'models_root': str(self.models_root),
            'data_root': str(self.data_root),
            'doc_root': str(self.doc_root) if self.doc_root else None,
            'init_data_path': str(self.init_data_path) if self.init_data_path else None,
            'init_doc_path': str(self.init_doc_path) if self.init_doc_path else None,
        }
