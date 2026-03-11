"""
超参数优化模块

使用Optuna进行自动超参数优化。

Author: ASLK Team
Date: 2025-11-19
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import json
import numpy as np
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    超参数优化器
    
    使用Optuna进行贝叶斯优化，自动搜索最优超参数。
    """
    
    def __init__(
        self, 
        model_class,
        param_space: Dict[str, Any],
        output_dir: str = "results/iteration_1/reports/optimization"
    ):
        """
        初始化超参数优化器
        
        Args:
            model_class: 模型类（需要支持fit和predict方法）
            param_space: 参数搜索空间
            output_dir: 输出目录
        """
        self.model_class = model_class
        self.param_space = param_space
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.study = None
        self.best_params = None
    
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 100,
        cv: int = 5,
        scoring: str = 'accuracy',
        direction: str = 'maximize'
    ) -> Dict[str, Any]:
        """
        执行超参数优化
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            n_trials: 优化试验次数
            cv: 交叉验证折数
            scoring: 评分指标
            direction: 优化方向 ('maximize' 或 'minimize')
        
        Returns:
            最优参数字典
        """
        try:
            import optuna
            
            self.logger.info(f"开始超参数优化，试验次数: {n_trials}")
            
            def objective(trial):
                """Optuna目标函数"""
                # 根据参数空间建议参数
                params = {}
                for param_name, param_config in self.param_space.items():
                    param_type = param_config['type']
                    
                    if param_type == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                    elif param_type == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_type == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )
                
                # 创建模型
                model = self.model_class(**params)
                
                # 交叉验证评分
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring=scoring, n_jobs=-1
                )
                
                return scores.mean()
            
            # 创建Optuna study
            self.study = optuna.create_study(direction=direction)
            self.study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            # 获取最优参数
            self.best_params = self.study.best_params
            
            self.logger.info(f"优化完成！最优得分: {self.study.best_value:.4f}")
            self.logger.info(f"最优参数: {self.best_params}")
            
            # 保存结果
            self._save_optimization_results()
            
            return self.best_params
        
        except ImportError:
            self.logger.error("Optuna库未安装，请运行: pip install optuna")
            raise
    
    def _save_optimization_results(self):
        """保存优化结果"""
        if self.study is None:
            return
        
        results = {
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials),
            "trials": [
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params
                }
                for trial in self.study.trials
            ]
        }
        
        filepath = self.output_dir / "optimization_results.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"优化结果已保存到: {filepath}")
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        绘制优化历史

        Args:
            save_path: 保存路径（可选）
        """
        try:
            import optuna
            import matplotlib.pyplot as plt

            if self.study is None:
                raise ValueError("请先运行optimize方法")

            fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"优化历史图已保存到: {save_path}")
            else:
                plt.savefig(self.output_dir / "optimization_history.png", dpi=300, bbox_inches='tight')

            plt.close()

        except ImportError:
            self.logger.warning("Optuna或matplotlib未安装，跳过绘图")

    def plot_param_importances(self, save_path: Optional[str] = None):
        """
        绘制参数重要性

        Args:
            save_path: 保存路径（可选）
        """
        try:
            import optuna
            import matplotlib.pyplot as plt

            if self.study is None:
                raise ValueError("请先运行optimize方法")

            fig = optuna.visualization.matplotlib.plot_param_importances(self.study)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"参数重要性图已保存到: {save_path}")
            else:
                plt.savefig(self.output_dir / "param_importances.png", dpi=300, bbox_inches='tight')

            plt.close()

        except ImportError:
            self.logger.warning("Optuna或matplotlib未安装，跳过绘图")

    def get_best_model(self, **additional_params):
        """
        使用最优参数创建模型

        Args:
            **additional_params: 额外的模型参数

        Returns:
            使用最优参数的模型实例
        """
        if self.best_params is None:
            raise ValueError("请先运行optimize方法")

        params = {**self.best_params, **additional_params}
        model = self.model_class(**params)

        self.logger.info(f"使用最优参数创建模型: {params}")
        return model

    def generate_optimization_report(self) -> Dict[str, Any]:
        """
        生成优化报告

        Returns:
            优化报告字典
        """
        if self.study is None:
            raise ValueError("请先运行optimize方法")

        self.logger.info("生成优化报告...")

        # 生成图表
        self.plot_optimization_history()
        self.plot_param_importances()

        # 获取前10个试验
        top_trials = sorted(self.study.trials, key=lambda t: t.value, reverse=True)[:10]

        report = {
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials),
            "top_10_trials": [
                {
                    "trial_number": trial.number,
                    "value": trial.value,
                    "params": trial.params
                }
                for trial in top_trials
            ],
            "output_dir": str(self.output_dir),
        }

        self.logger.info("优化报告生成完成")
        return report


def create_rf_param_space() -> Dict[str, Any]:
    """创建RandomForest参数空间"""
    return {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
        'max_depth': {'type': 'int', 'low': 3, 'high': 20},
        'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
        'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
        'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
    }


def create_xgb_param_space() -> Dict[str, Any]:
    """创建XGBoost参数空间"""
    return {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
        'max_depth': {'type': 'int', 'low': 3, 'high': 15},
        'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.3, 'log': True},
        'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0},
        'gamma': {'type': 'float', 'low': 0, 'high': 5},
        'reg_alpha': {'type': 'float', 'low': 0, 'high': 1},
        'reg_lambda': {'type': 'float', 'low': 0, 'high': 1},
    }

