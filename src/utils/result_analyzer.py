"""
结果分析工具

用于分析ASLK系统的运行结果，生成统计报告和可视化图表。

Author: ASLK Team
Date: 2025-11-19
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """
    结果分析器
    
    分析自进化循环的运行结果，生成统计报告。
    """
    
    def __init__(self, output_dir: str = "results/iteration_1/reports"):
        """
        初始化结果分析器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_loop_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析循环结果
        
        Args:
            results: 循环结果字典
        
        Returns:
            分析报告字典
        """
        self.logger.info("开始分析循环结果...")
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "summary": self._analyze_summary(results),
            "iterations": self._analyze_iterations(results),
            "materials": self._analyze_materials(results),
            "performance": self._analyze_performance(results),
        }
        
        self.logger.info("结果分析完成")
        return analysis
    
    def _analyze_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析总体统计"""
        iterations = results.get("iterations", [])
        success_materials = results.get("success_materials", [])
        
        total_candidates = sum(len(it.get("candidates", [])) for it in iterations)
        total_structures = sum(len(it.get("structures", [])) for it in iterations)
        
        return {
            "total_iterations": len(iterations),
            "total_candidates": total_candidates,
            "total_structures": total_structures,
            "success_materials": len(success_materials),
            "success_rate": len(success_materials) / total_structures if total_structures > 0 else 0,
        }
    
    def _analyze_iterations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析每次迭代"""
        iterations = results.get("iterations", [])
        
        iteration_stats = []
        for i, it in enumerate(iterations):
            stats = {
                "iteration": i + 1,
                "candidates": len(it.get("candidates", [])),
                "structures": len(it.get("structures", [])),
                "successes": it.get("successes", 0),
                "failures": it.get("failures", 0),
                "avg_k": it.get("avg_k", 0),
                "min_k": it.get("min_k", 0),
            }
            iteration_stats.append(stats)
        
        return iteration_stats
    
    def _analyze_materials(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析发现的材料"""
        materials = results.get("success_materials", [])
        
        if not materials:
            return {"count": 0, "materials": []}
        
        # 提取热导率值
        k_values = [m.get("k", 0) for m in materials]
        
        return {
            "count": len(materials),
            "avg_k": np.mean(k_values) if k_values else 0,
            "min_k": np.min(k_values) if k_values else 0,
            "max_k": np.max(k_values) if k_values else 0,
            "materials": [
                {
                    "formula": m.get("formula", ""),
                    "k": m.get("k", 0),
                    "iteration": m.get("iteration", 0),
                }
                for m in materials
            ]
        }
    
    def _analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能指标"""
        return {
            "total_time": results.get("total_time", 0),
            "avg_iteration_time": results.get("avg_iteration_time", 0),
            "model_retrains": results.get("model_retrains", 0),
        }
    
    def save_report(self, analysis: Dict[str, Any], filename: str = "analysis_report.json"):
        """
        保存分析报告
        
        Args:
            analysis: 分析结果
            filename: 文件名
        """
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"分析报告已保存到: {filepath}")
    
    def generate_summary_text(self, analysis: Dict[str, Any]) -> str:
        """
        生成文本摘要
        
        Args:
            analysis: 分析结果
        
        Returns:
            文本摘要
        """
        summary = analysis["summary"]
        materials = analysis["materials"]
        
        text = f"""
# ASLK 运行结果摘要

**生成时间**: {analysis['timestamp']}

## 总体统计
- 总迭代次数: {summary['total_iterations']}
- 总候选组分: {summary['total_candidates']}
- 总生成结构: {summary['total_structures']}
- 成功材料数: {summary['success_materials']}
- 成功率: {summary['success_rate']:.2%}

## 发现的材料
- 材料数量: {materials['count']}
- 平均热导率: {materials['avg_k']:.3f} W/(m·K)
- 最低热导率: {materials['min_k']:.3f} W/(m·K)
- 最高热导率: {materials['max_k']:.3f} W/(m·K)

## 性能指标
- 总运行时间: {analysis['performance']['total_time']:.2f} 秒
- 平均迭代时间: {analysis['performance']['avg_iteration_time']:.2f} 秒
- 模型重训练次数: {analysis['performance']['model_retrains']}
"""
        return text.strip()

    def plot_iteration_progress(self, analysis: Dict[str, Any], save_path: Optional[str] = None):
        """
        绘制迭代进度图

        Args:
            analysis: 分析结果
            save_path: 保存路径（可选）
        """
        try:
            import matplotlib.pyplot as plt

            iterations = analysis["iterations"]
            if not iterations:
                self.logger.warning("没有迭代数据可绘制")
                return

            iteration_nums = [it["iteration"] for it in iterations]
            successes = [it["successes"] for it in iterations]
            failures = [it["failures"] for it in iterations]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(iteration_nums, successes, marker='o', label='成功', color='green')
            ax.plot(iteration_nums, failures, marker='x', label='失败', color='red')
            ax.set_xlabel('迭代次数')
            ax.set_ylabel('材料数量')
            ax.set_title('ASLK 迭代进度')
            ax.legend()
            ax.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"迭代进度图已保存到: {save_path}")
            else:
                plt.savefig(self.output_dir / "iteration_progress.png", dpi=300, bbox_inches='tight')

            plt.close()
        except ImportError:
            self.logger.warning("matplotlib未安装，跳过绘图")

    def plot_k_distribution(self, analysis: Dict[str, Any], save_path: Optional[str] = None):
        """
        绘制热导率分布图

        Args:
            analysis: 分析结果
            save_path: 保存路径（可选）
        """
        try:
            import matplotlib.pyplot as plt

            materials = analysis["materials"]["materials"]
            if not materials:
                self.logger.warning("没有材料数据可绘制")
                return

            k_values = [m["k"] for m in materials]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(k_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            ax.axvline(x=1.0, color='red', linestyle='--', label='目标阈值 (k=1.0)')
            ax.set_xlabel('热导率 k (W/(m·K))')
            ax.set_ylabel('材料数量')
            ax.set_title('发现材料的热导率分布')
            ax.legend()
            ax.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"热导率分布图已保存到: {save_path}")
            else:
                plt.savefig(self.output_dir / "k_distribution.png", dpi=300, bbox_inches='tight')

            plt.close()
        except ImportError:
            self.logger.warning("matplotlib未安装，跳过绘图")

    def export_to_csv(self, analysis: Dict[str, Any], filename: str = "materials.csv"):
        """
        导出材料列表到CSV

        Args:
            analysis: 分析结果
            filename: 文件名
        """
        materials = analysis["materials"]["materials"]
        if not materials:
            self.logger.warning("没有材料数据可导出")
            return

        df = pd.DataFrame(materials)
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False, encoding='utf-8')
        self.logger.info(f"材料列表已导出到: {filepath}")

