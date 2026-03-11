"""
高级分析功能模块

提供数据分析、趋势分析、相关性分析等高级功能。

Author: ASLK Team
Date: 2025-11-19
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class AdvancedAnalyzer:
    """
    高级分析器
    
    提供数据统计、趋势分析、相关性分析等功能。
    """
    
    def __init__(self, output_dir: str = "results/iteration_1/reports/analysis"):
        """
        初始化高级分析器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_data_distribution(self, data: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        分析数据分布
        
        Args:
            data: 数据矩阵
            feature_names: 特征名称列表
        
        Returns:
            分布统计字典
        """
        self.logger.info("分析数据分布...")
        
        df = pd.DataFrame(data, columns=feature_names)
        
        distribution_stats = {
            "n_samples": len(df),
            "n_features": len(feature_names),
            "features": []
        }
        
        for col in df.columns:
            stats_dict = {
                "name": col,
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
                "q25": float(df[col].quantile(0.25)),
                "q75": float(df[col].quantile(0.75)),
                "skewness": float(df[col].skew()),
                "kurtosis": float(df[col].kurtosis()),
            }
            distribution_stats["features"].append(stats_dict)
        
        self.logger.info("数据分布分析完成")
        return distribution_stats
    
    def analyze_correlation(
        self, 
        data: np.ndarray, 
        feature_names: List[str],
        method: str = 'pearson'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        分析特征相关性
        
        Args:
            data: 数据矩阵
            feature_names: 特征名称列表
            method: 相关性方法 ('pearson', 'spearman', 'kendall')
        
        Returns:
            相关性矩阵和统计信息
        """
        self.logger.info(f"分析特征相关性（方法: {method}）...")
        
        df = pd.DataFrame(data, columns=feature_names)
        corr_matrix = df.corr(method=method)
        
        # 找出高相关性特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # 阈值0.7
                    high_corr_pairs.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": float(corr_value)
                    })
        
        # 按相关性绝对值排序
        high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        stats = {
            "method": method,
            "n_features": len(feature_names),
            "high_correlation_pairs": high_corr_pairs,
            "avg_correlation": float(corr_matrix.abs().mean().mean()),
        }
        
        self.logger.info(f"发现{len(high_corr_pairs)}对高相关性特征")
        return corr_matrix, stats
    
    def analyze_trend(
        self,
        iterations: List[Dict[str, Any]],
        metric: str = 'successes'
    ) -> Dict[str, Any]:
        """
        分析迭代趋势
        
        Args:
            iterations: 迭代数据列表
            metric: 分析的指标名称
        
        Returns:
            趋势分析结果
        """
        self.logger.info(f"分析{metric}趋势...")
        
        if not iterations:
            return {"error": "没有迭代数据"}
        
        values = [it.get(metric, 0) for it in iterations]
        x = np.arange(len(values))
        
        # 线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # 移动平均
        window_size = min(5, len(values))
        moving_avg = pd.Series(values).rolling(window=window_size).mean().tolist()
        
        trend_analysis = {
            "metric": metric,
            "n_iterations": len(values),
            "trend": "increasing" if slope > 0 else "decreasing",
            "slope": float(slope),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "is_significant": p_value < 0.05,
            "moving_average": moving_avg,
            "total_change": float(values[-1] - values[0]) if values else 0,
            "percent_change": float((values[-1] - values[0]) / values[0] * 100) if values and values[0] != 0 else 0,
        }
        
        self.logger.info(f"趋势: {trend_analysis['trend']}, R²={trend_analysis['r_squared']:.3f}")
        return trend_analysis

    def plot_correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        绘制相关性热图

        Args:
            corr_matrix: 相关性矩阵
            save_path: 保存路径（可选）
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                corr_matrix,
                annot=False,
                cmap='coolwarm',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                ax=ax
            )
            ax.set_title('特征相关性热图')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"相关性热图已保存到: {save_path}")
            else:
                plt.savefig(self.output_dir / "correlation_heatmap.png", dpi=300, bbox_inches='tight')

            plt.close()

        except ImportError:
            self.logger.warning("matplotlib或seaborn未安装，跳过绘图")

    def plot_trend_analysis(
        self,
        iterations: List[Dict[str, Any]],
        metrics: List[str],
        save_path: Optional[str] = None
    ):
        """
        绘制趋势分析图

        Args:
            iterations: 迭代数据列表
            metrics: 要绘制的指标列表
            save_path: 保存路径（可选）
        """
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
            if len(metrics) == 1:
                axes = [axes]

            for ax, metric in zip(axes, metrics):
                values = [it.get(metric, 0) for it in iterations]
                x = range(1, len(values) + 1)

                ax.plot(x, values, marker='o', label=f'{metric}（实际值）')

                # 添加趋势线
                z = np.polyfit(x, values, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.8, label='趋势线')

                ax.set_xlabel('迭代次数')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} 趋势分析')
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"趋势分析图已保存到: {save_path}")
            else:
                plt.savefig(self.output_dir / "trend_analysis.png", dpi=300, bbox_inches='tight')

            plt.close()

        except ImportError:
            self.logger.warning("matplotlib未安装，跳过绘图")

    def generate_comprehensive_report(
        self,
        data: np.ndarray,
        feature_names: List[str],
        iterations: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        生成综合分析报告

        Args:
            data: 数据矩阵
            feature_names: 特征名称列表
            iterations: 迭代数据（可选）

        Returns:
            综合分析报告
        """
        self.logger.info("生成综合分析报告...")

        report = {}

        # 数据分布分析
        report["distribution"] = self.analyze_data_distribution(data, feature_names)

        # 相关性分析
        corr_matrix, corr_stats = self.analyze_correlation(data, feature_names)
        report["correlation"] = corr_stats

        # 绘制相关性热图
        self.plot_correlation_heatmap(corr_matrix)

        # 趋势分析（如果提供了迭代数据）
        if iterations:
            report["trends"] = {}
            for metric in ['successes', 'failures', 'avg_k']:
                if any(metric in it for it in iterations):
                    report["trends"][metric] = self.analyze_trend(iterations, metric)

            # 绘制趋势图
            self.plot_trend_analysis(iterations, list(report["trends"].keys()))

        report["output_dir"] = str(self.output_dir)

        self.logger.info("综合分析报告生成完成")
        return report

