# -*- coding: utf-8 -*-
"""
进度追踪模块
用于跟踪每个 round 的执行进度，避免重复执行
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    进度追踪器
    保存每个 round 的各个步骤执行状态
    """

    DEFAULT_STEPS = [
        "train_model",            # 训练模型
        "bayesian_optimization",  # 贝叶斯优化
        "ai_evaluation",          # AI 评估
        "structure_calculation",  # 结构生成和计算 (涵盖子步骤)
        "merge_results",          # 合并声子与热导率结果
        "success_extraction",     # 成功材料提取
        "document_update",        # 数据和文档更新
    ]

    def __init__(self, base_dir: str = "results", steps: List[str] = None):
        """
        初始化进度追踪器

        Args:
            base_dir: 结果根目录
            steps: 自定义步骤列表，如果为 None 则使用默认列表
        """
        self.base_dir = Path(base_dir)
        self.progress_file = self.base_dir / "progress.json"
        self.steps = steps if steps is not None else self.DEFAULT_STEPS
        self.progress = self._load_progress()

    def _load_progress(self) -> Dict:
        """加载进度文件"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Backfill steps that may be missing in older progress files or manual rollbacks.
                changed = False
                for round_key, round_data in data.items():
                    if not round_key.startswith("iteration_"):
                        continue
                    match = re.match(r"iteration_(\d+)", round_key)
                    iteration_num = int(match.group(1)) if match else None
                    reports_dir = self.base_dir / round_key / "reports"
                    llm_eval_report = reports_dir / "llm_evaluation_output.md"
                    theory_report = reports_dir / "llm_theory_update_output.md"

                    if "ai_evaluation" not in round_data and llm_eval_report.exists():
                        round_data["ai_evaluation"] = {
                            "completed": True,
                            "timestamp": round_data.get("bayesian_optimization", {}).get("timestamp", ""),
                            "metadata": {"backfilled": True, "report": str(llm_eval_report)},
                        }
                        changed = True

                    if "document_update" not in round_data:
                        doc_candidates = [theory_report]
                        if iteration_num is not None:
                            doc_candidates.append(
                                self.base_dir.parent / "doc" / f"v0.0.{iteration_num}" / "Theoretical_principle_document.md"
                            )
                        if any(path.exists() for path in doc_candidates):
                            round_data["document_update"] = {
                                "completed": True,
                                "timestamp": round_data.get("success_extraction", {}).get("timestamp", ""),
                                "metadata": {"backfilled": True},
                            }
                            changed = True

                    if "merge_results" not in round_data:
                        success_done = round_data.get("success_extraction", {}).get("completed", False)
                        update_done = round_data.get("document_update", {}).get("completed", False) or \
                                      round_data.get("data_update", {}).get("completed", False)
                        if success_done or update_done:
                            ts = round_data.get("success_extraction", {}).get("timestamp") or \
                                 round_data.get("document_update", {}).get("timestamp") or \
                                 round_data.get("data_update", {}).get("timestamp") or ""
                            round_data["merge_results"] = {
                                "completed": True,
                                "timestamp": ts,
                                "metadata": {"backfilled": True}
                            }
                            changed = True
                if changed:
                    self.progress = data
                    self._save_progress()
                return data
            except Exception as e:
                logger.warning(f"加载进度文件失败: {e}")
                return {}
        return {}

    def _save_progress(self):
        """保存进度文件"""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(self.progress, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存进度文件失败: {e}")

    def is_step_completed(self, iteration_num: int, step: str) -> bool:
        """
        检查某个步骤是否已完成

        Args:
            iteration_num: 轮次
            step: 步骤名称

        Returns:
            是否已完成
        """
        round_key = f"iteration_{iteration_num}"
        if round_key not in self.progress:
            return False
        step_data = self.progress[round_key].get(step, {})
        completed = step_data.get("completed", False)

        if step == "structure_calculation":
            if completed:
                return True
            substeps = step_data.get("substeps", {})
            expected = [
                "generation",
                "relaxation",
                "thermal_conductivity",
                "deduplication",
                "phonon_spectrum",
            ]
            if not substeps:
                return False
            if not all(substeps.get(s, {}).get("completed", False) for s in expected):
                return False
            if "merge_results" in substeps:
                return substeps.get("merge_results", {}).get("completed", False)
            return True

        return completed

    def mark_step_completed(self, iteration_num: int, step: str, metadata: Optional[Dict] = None):
        """
        标记某个步骤已完成

        Args:
            iteration_num: 轮次
            step: 步骤名称
            metadata: 附加元数据（如文件路径、数量等）
        """
        round_key = f"iteration_{iteration_num}"
        if round_key not in self.progress:
            self.progress[round_key] = {}

        existing_step = self.progress[round_key].get(step, {})
        substeps = existing_step.get("substeps", {})
        existing_metadata = existing_step.get("metadata", {})

        merged_metadata = existing_metadata.copy()
        if metadata:
            merged_metadata.update(metadata)

        step_entry = {
            "completed": True,
            "timestamp": datetime.now().isoformat(),
            "metadata": merged_metadata,
        }
        if substeps:
            step_entry["substeps"] = substeps

        self.progress[round_key][step] = step_entry
        self._save_progress()
        logger.info(f"✅ 标记步骤完成: Iteration {iteration_num} - {step}")

    def is_substep_completed(self, iteration_num: int, step: str, substep: str) -> bool:
        """
        检查某个子步骤是否已完成

        Args:
            iteration_num: 轮次
            step: 主步骤名称
            substep: 子步骤名称

        Returns:
            是否已完成
        """
        round_key = f"iteration_{iteration_num}"
        if round_key not in self.progress:
            return False

        step_data = self.progress[round_key].get(step, {})
        substeps = step_data.get("substeps", {})
        return substeps.get(substep, {}).get("completed", False)

    def mark_substep_completed(self, iteration_num: int, step: str, substep: str,
                               metadata: Optional[Dict] = None):
        """
        标记某个子步骤已完成

        Args:
            iteration_num: 轮次
            step: 主步骤名称
            substep: 子步骤名称
            metadata: 附加元数据
        """
        round_key = f"iteration_{iteration_num}"
        if round_key not in self.progress:
            self.progress[round_key] = {}

        if step not in self.progress[round_key]:
            self.progress[round_key][step] = {
                "completed": False,
                "timestamp": "",
                "metadata": {},
                "substeps": {},
            }

        if "substeps" not in self.progress[round_key][step]:
            self.progress[round_key][step]["substeps"] = {}

        self.progress[round_key][step]["substeps"][substep] = {
            "completed": True,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        self._save_progress()
        logger.info(f"✅ 标记子步骤完成: Iteration {iteration_num} - {step}.{substep}")

    def update_substep(self, iteration_num: int, step: str, substep: str,
                       metadata: Optional[Dict] = None, completed: Optional[bool] = None):
        """
        更新子步骤进度信息（不强制标记完成）

        Args:
            iteration_num: 轮次
            step: 主步骤名称
            substep: 子步骤名称
            metadata: 追加的元数据
            completed: 如提供则更新完成状态
        """
        round_key = f"iteration_{iteration_num}"
        if round_key not in self.progress:
            self.progress[round_key] = {}

        if step not in self.progress[round_key]:
            self.progress[round_key][step] = {
                "completed": False,
                "timestamp": "",
                "metadata": {},
                "substeps": {},
            }

        if "substeps" not in self.progress[round_key][step]:
            self.progress[round_key][step]["substeps"] = {}

        substeps = self.progress[round_key][step]["substeps"]
        existing = substeps.get(substep, {})
        existing_metadata = existing.get("metadata", {})

        merged_metadata = existing_metadata.copy()
        if metadata:
            merged_metadata.update(metadata)

        substeps[substep] = {
            "completed": existing.get("completed", False) if completed is None else completed,
            "timestamp": datetime.now().isoformat(),
            "metadata": merged_metadata,
        }

        self._save_progress()
        logger.info(f"[PROGRESS] Update substep Iteration {iteration_num} - {step}.{substep}")

    def get_substep_metadata(self, iteration_num: int, step: str, substep: str) -> Optional[Dict]:
        """
        获取子步骤的元数据

        Args:
            iteration_num: 轮次
            step: 主步骤名称
            substep: 子步骤名称

        Returns:
            元数据字典，如果不存在则返回 None
        """
        round_key = f"iteration_{iteration_num}"
        if round_key not in self.progress:
            return None

        step_data = self.progress[round_key].get(step, {})
        substeps = step_data.get("substeps", {})
        substep_data = substeps.get(substep, {})
        return substep_data.get("metadata")

    def reset_substep(self, iteration_num: int, step: str, substep: str):
        """
        重置某个子步骤的进度

        Args:
            iteration_num: 轮次
            step: 主步骤名称
            substep: 子步骤名称
        """
        round_key = f"iteration_{iteration_num}"
        if round_key in self.progress and step in self.progress[round_key]:
            substeps = self.progress[round_key][step].get("substeps", {})
            if substep in substeps:
                del substeps[substep]
                self._save_progress()
                logger.info(f"🔄 已重置 Iteration {iteration_num} - {step}.{substep} 的进度")

    def get_round_progress(self, iteration_num: int) -> Dict[str, bool]:
        """
        获取某轮的所有步骤完成状态

        Args:
            iteration_num: 轮次

        Returns:
            步骤完成状态字典
        """
        result = {}
        for step in self.steps:
            result[step] = self.is_step_completed(iteration_num, step)
        return result

    def is_round_completed(self, iteration_num: int) -> bool:
        """
        检查某轮是否所有步骤都完成

        Args:
            iteration_num: 轮次

        Returns:
            是否全部完成
        """
        progress = self.get_round_progress(iteration_num)
        return all(progress.values())

    def get_next_incomplete_step(self, iteration_num: int) -> Optional[str]:
        """
        获取下一个未完成的步骤

        Args:
            iteration_num: 轮次

        Returns:
            未完成的步骤名称，如果全部完成则返回 None
        """
        for step in self.steps:
            if not self.is_step_completed(iteration_num, step):
                return step
        return None

    def reset_round(self, iteration_num: int):
        """
        重置某轮的进度（用于重新执行）

        Args:
            iteration_num: 轮次
        """
        round_key = f"iteration_{iteration_num}"
        if round_key in self.progress:
            del self.progress[round_key]
            self._save_progress()
            logger.info(f"🔄 已重置 Iteration {iteration_num} 的进度")

    def reset_step(self, iteration_num: int, step: str):
        """
        重置某个步骤的进度（用于重新执行）

        Args:
            iteration_num: 轮次
            step: 步骤名称
        """
        round_key = f"iteration_{iteration_num}"
        if round_key in self.progress and step in self.progress[round_key]:
            del self.progress[round_key][step]
            self._save_progress()
            logger.info(f"🔄 已重置 Iteration {iteration_num} - {step} 的进度")

    def print_progress(self, iteration_num: int):
        """
        打印某轮的进度状态

        Args:
            iteration_num: 轮次
        """
        print(f"\n📊 Iteration {iteration_num} 进度:")
        print("=" * 60)
        progress = self.get_round_progress(iteration_num)
        for step in self.steps:
            status = "✅" if progress[step] else "⏳"
            print(f"  {status} {step}")
        print("=" * 60)

        if self.is_round_completed(iteration_num):
            print(f"✨ Iteration {iteration_num} 所有步骤已完成!\n")
        else:
            next_step = self.get_next_incomplete_step(iteration_num)
            print(f"➡️  下一步: {next_step}\n")

    def get_completed_rounds(self) -> List[int]:
        """
        获取所有已完成的轮次列表

        Returns:
            已完成的轮次编号列表
        """
        completed = []
        for key in self.progress.keys():
            if key.startswith("iteration_"):
                iteration_num = int(key.split("_")[1])
                if self.is_round_completed(iteration_num):
                    completed.append(iteration_num)
        return sorted(completed)


# 测试代码
if __name__ == "__main__":
    # 测试用例
    tracker = ProgressTracker(base_dir="test_results")

    # 模拟 round 1 的执行
    print("模拟 Iteration 1 执行:")
    for i, step in enumerate(tracker.steps[:3]):
        tracker.mark_step_completed(1, step, metadata={"test": i})
        tracker.print_progress(1)

    # 检查状态
    print("\n检查 Iteration 1 是否完成:", tracker.is_round_completed(1))
    print("下一步:", tracker.get_next_incomplete_step(1))
