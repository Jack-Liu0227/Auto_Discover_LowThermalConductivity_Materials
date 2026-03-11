"""
模型解释性模块

使用SHAP值分析模型预测，提供特征重要性可视化。

Author: ASLK Team
Date: 2025-11-19
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    模型解释器
    
    使用SHAP分析模型预测，生成特征重要性报告。
    """
    
    def __init__(self, model, feature_names: List[str], output_dir: str = "results/iteration_1/reports/explainability"):
        """
        初始化模型解释器
        
        Args:
            model: 训练好的模型（支持predict方法）
            feature_names: 特征名称列表
            output_dir: 输出目录
        """
        self.model = model
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.explainer = None
        self.shap_values = None
    
    def initialize_explainer(self, background_data: np.ndarray, explainer_type: str = "tree"):
        """
        初始化SHAP解释器
        
        Args:
            background_data: 背景数据集（用于计算SHAP值）
            explainer_type: 解释器类型 ("tree", "kernel", "linear")
        """
        try:
            import shap
            
            if explainer_type == "tree":
                # 适用于树模型（RandomForest, XGBoost）
                self.explainer = shap.TreeExplainer(self.model)
                self.logger.info("TreeExplainer初始化成功")
            elif explainer_type == "kernel":
                # 通用解释器（较慢）
                self.explainer = shap.KernelExplainer(self.model.predict, background_data)
                self.logger.info("KernelExplainer初始化成功")
            elif explainer_type == "linear":
                # 适用于线性模型
                self.explainer = shap.LinearExplainer(self.model, background_data)
                self.logger.info("LinearExplainer初始化成功")
            else:
                raise ValueError(f"不支持的解释器类型: {explainer_type}")
        
        except ImportError:
            self.logger.error("SHAP库未安装，请运行: pip install shap")
            raise
    
    def explain_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        计算SHAP值
        
        Args:
            X: 输入特征矩阵
        
        Returns:
            SHAP值矩阵
        """
        if self.explainer is None:
            raise ValueError("请先调用initialize_explainer初始化解释器")
        
        self.logger.info(f"计算{len(X)}个样本的SHAP值...")
        self.shap_values = self.explainer.shap_values(X)
        self.logger.info("SHAP值计算完成")
        
        return self.shap_values
    
    def get_feature_importance(self, shap_values: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            shap_values: SHAP值矩阵（可选，默认使用最近计算的）
        
        Returns:
            特征重要性DataFrame
        """
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("请先调用explain_predictions计算SHAP值")
        
        # 计算平均绝对SHAP值
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # 创建DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        })
        
        # 按重要性排序
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
        """
        绘制特征重要性图
        
        Args:
            top_n: 显示前N个重要特征
            save_path: 保存路径（可选）
        """
        try:
            import matplotlib.pyplot as plt
            
            importance_df = self.get_feature_importance()
            top_features = importance_df.head(top_n)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(top_features)), top_features['importance'], color='skyblue')
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('平均|SHAP值|')
            ax.set_title(f'Top {top_n} 特征重要性')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"特征重要性图已保存到: {save_path}")
            else:
                plt.savefig(self.output_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
            
            plt.close()
        
        except ImportError:
            self.logger.warning("matplotlib未安装，跳过绘图")

    def plot_shap_summary(self, X: np.ndarray, save_path: Optional[str] = None):
        """
        绘制SHAP摘要图

        Args:
            X: 输入特征矩阵
            save_path: 保存路径（可选）
        """
        try:
            import shap
            import matplotlib.pyplot as plt

            if self.shap_values is None:
                raise ValueError("请先调用explain_predictions计算SHAP值")

            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values,
                X,
                feature_names=self.feature_names,
                show=False
            )

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"SHAP摘要图已保存到: {save_path}")
            else:
                plt.savefig(self.output_dir / "shap_summary.png", dpi=300, bbox_inches='tight')

            plt.close()

        except ImportError:
            self.logger.warning("SHAP或matplotlib未安装，跳过绘图")

    def plot_shap_waterfall(self, sample_idx: int, X: np.ndarray, save_path: Optional[str] = None):
        """
        绘制单个样本的SHAP瀑布图

        Args:
            sample_idx: 样本索引
            X: 输入特征矩阵
            save_path: 保存路径（可选）
        """
        try:
            import shap
            import matplotlib.pyplot as plt

            if self.shap_values is None:
                raise ValueError("请先调用explain_predictions计算SHAP值")

            # 创建Explanation对象
            explanation = shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                data=X[sample_idx],
                feature_names=self.feature_names
            )

            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(explanation, show=False)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"SHAP瀑布图已保存到: {save_path}")
            else:
                plt.savefig(self.output_dir / f"shap_waterfall_sample_{sample_idx}.png", dpi=300, bbox_inches='tight')

            plt.close()

        except ImportError:
            self.logger.warning("SHAP或matplotlib未安装，跳过绘图")

    def export_importance_report(self, filename: str = "feature_importance.csv"):
        """
        导出特征重要性报告

        Args:
            filename: 文件名
        """
        importance_df = self.get_feature_importance()
        filepath = self.output_dir / filename
        importance_df.to_csv(filepath, index=False, encoding='utf-8')
        self.logger.info(f"特征重要性报告已导出到: {filepath}")

        return importance_df

    def generate_explanation_report(self, X: np.ndarray, top_n: int = 20) -> Dict[str, Any]:
        """
        生成完整的解释性报告

        Args:
            X: 输入特征矩阵
            top_n: 显示前N个重要特征

        Returns:
            解释性报告字典
        """
        self.logger.info("生成解释性报告...")

        # 计算SHAP值
        if self.shap_values is None:
            self.explain_predictions(X)

        # 获取特征重要性
        importance_df = self.get_feature_importance()

        # 生成图表
        self.plot_feature_importance(top_n=top_n)
        self.plot_shap_summary(X)

        # 导出报告
        self.export_importance_report()

        report = {
            "total_features": len(self.feature_names),
            "samples_analyzed": len(X),
            "top_features": importance_df.head(top_n).to_dict('records'),
            "output_dir": str(self.output_dir),
        }

        self.logger.info("解释性报告生成完成")
        return report

