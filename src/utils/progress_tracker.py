# -*- coding: utf-8 -*-
"""
Progress tracking utilities for iterative workflows.

This module intentionally keeps log and console output ASCII-safe so it can run
cleanly on Windows terminals configured with GBK.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Persist step-level and substep-level progress for each iteration."""

    @staticmethod
    def _is_valid_llm_evaluation_report(path: Path) -> bool:
        """Do not infer completion from a truncated or malformed report."""
        try:
            from .workflow_resume import is_valid_llm_evaluation_report
        except ImportError:
            from utils.workflow_resume import is_valid_llm_evaluation_report
        return is_valid_llm_evaluation_report(path)

    DEFAULT_STEPS = [
        "train_model",
        "bayesian_optimization",
        "ai_evaluation",
        "structure_calculation",
        "merge_results",
        "success_extraction",
        "document_update",
    ]

    def __init__(self, base_dir: str = "results", steps: Optional[List[str]] = None):
        self.base_dir = Path(base_dir)
        self.progress_file = self.base_dir / "progress.json"
        self.steps = steps if steps is not None else self.DEFAULT_STEPS
        self.progress = self._load_progress()

    def _load_progress(self) -> Dict:
        """Load saved progress data if available."""
        if not self.progress_file.exists():
            return {}

        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("Failed to load progress file: %s", exc)
            return {}

        if not isinstance(data, dict):
            logger.warning("Ignoring malformed progress file (expected an object): %s", self.progress_file)
            return {}

        changed = False
        for round_key, round_data in data.items():
            if not round_key.startswith("iteration_") or not isinstance(round_data, dict):
                continue

            reports_dir = self.base_dir / round_key / "reports"
            llm_eval_report = reports_dir / "llm_evaluation_output.md"
            theory_report = reports_dir / "llm_theory_update_output.md"
            bayes_data = round_data.get("bayesian_optimization")
            success_data = round_data.get("success_extraction")
            bayes_timestamp = bayes_data.get("timestamp", "") if isinstance(bayes_data, dict) else ""
            success_timestamp = success_data.get("timestamp", "") if isinstance(success_data, dict) else ""

            if "ai_evaluation" not in round_data and self._is_valid_llm_evaluation_report(llm_eval_report):
                round_data["ai_evaluation"] = {
                    "completed": True,
                    "timestamp": bayes_timestamp,
                    "metadata": {
                        "backfilled": True,
                        "report": str(llm_eval_report),
                    },
                }
                changed = True

            try:
                from .workflow_resume import is_valid_theory_document_artifact
            except ImportError:
                from utils.workflow_resume import is_valid_theory_document_artifact
            doc_root = self.base_dir.parent / "doc"
            doc_valid = is_valid_theory_document_artifact(
                doc_root,
                int(round_key.split("_", 1)[1]),
            )
            if "document_update" not in round_data and theory_report.exists() and doc_valid:
                round_data["document_update"] = {
                    "completed": True,
                    "timestamp": success_timestamp,
                    "metadata": {"backfilled": True, "validated_document": True},
                }
                changed = True

        if changed:
            self.progress = data
            self._save_progress()

        return data

    def _save_progress(self):
        """Atomically write progress state so interruption cannot corrupt it."""
        temp_file = self.progress_file.with_name(f".{self.progress_file.name}.{os.getpid()}.tmp")
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.progress, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_file, self.progress_file)
        except Exception as exc:
            logger.error("Failed to save progress file: %s", exc)
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    def is_step_completed(self, iteration_num: int, step: str) -> bool:
        """Return whether the given step is complete for this iteration."""
        round_key = f"iteration_{iteration_num}"
        if round_key not in self.progress:
            return False

        step_data = self.progress[round_key].get(step, {})
        if not isinstance(step_data, dict):
            return False
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
            if not isinstance(substeps, dict) or not substeps:
                return False
            if not all(
                isinstance(substeps.get(name), dict)
                and substeps[name].get("completed", False)
                for name in expected
            ):
                return False
            if "merge_results" in substeps:
                merge_data = substeps.get("merge_results")
                return isinstance(merge_data, dict) and merge_data.get("completed", False)
            return True

        return completed

    def mark_step_started(self, iteration_num: int, step: str, metadata: Optional[Dict] = None):
        """Persist an in-progress marker before executing a step."""
        round_key = f"iteration_{iteration_num}"
        if round_key not in self.progress:
            self.progress[round_key] = {}

        existing = self.progress[round_key].get(step)
        if not isinstance(existing, dict):
            existing = {}
        existing_metadata = existing.get("metadata", {})
        if not isinstance(existing_metadata, dict):
            existing_metadata = {}
        if metadata:
            existing_metadata = {**existing_metadata, **metadata}

        step_entry = {
            "completed": False,
            "timestamp": datetime.now().isoformat(),
            "metadata": existing_metadata,
        }
        substeps = existing.get("substeps")
        if isinstance(substeps, dict):
            step_entry["substeps"] = substeps
        self.progress[round_key][step] = step_entry
        self._save_progress()
        logger.info("Marked step started: iteration=%s step=%s", iteration_num, step)

    def mark_step_completed(self, iteration_num: int, step: str, metadata: Optional[Dict] = None):
        """Mark a step as complete and merge any metadata."""
        round_key = f"iteration_{iteration_num}"
        if round_key not in self.progress:
            self.progress[round_key] = {}

        existing_step = self.progress[round_key].get(step, {})
        if not isinstance(existing_step, dict):
            existing_step = {}
        substeps = existing_step.get("substeps", {})
        if not isinstance(substeps, dict):
            substeps = {}
        existing_metadata = existing_step.get("metadata", {})
        if not isinstance(existing_metadata, dict):
            existing_metadata = {}

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
        logger.info("Marked step completed: iteration=%s step=%s", iteration_num, step)

    def is_substep_completed(self, iteration_num: int, step: str, substep: str) -> bool:
        """Return whether the given substep is complete."""
        round_key = f"iteration_{iteration_num}"
        if round_key not in self.progress:
            return False

        step_data = self.progress[round_key].get(step, {})
        if not isinstance(step_data, dict):
            return False
        substeps = step_data.get("substeps", {})
        if not isinstance(substeps, dict):
            return False
        substep_data = substeps.get(substep, {})
        return isinstance(substep_data, dict) and substep_data.get("completed", False)

    def mark_substep_completed(
        self,
        iteration_num: int,
        step: str,
        substep: str,
        metadata: Optional[Dict] = None,
    ):
        """Mark a substep as complete."""
        round_key = f"iteration_{iteration_num}"
        if round_key not in self.progress:
            self.progress[round_key] = {}

        step_data = self.progress[round_key].get(step)
        if not isinstance(step_data, dict):
            step_data = {
                "completed": False,
                "timestamp": "",
                "metadata": {},
                "substeps": {},
            }
            self.progress[round_key][step] = step_data

        if not isinstance(step_data.get("substeps"), dict):
            step_data["substeps"] = {}

        step_data["substeps"][substep] = {
            "completed": True,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        self._save_progress()
        logger.info(
            "Marked substep completed: iteration=%s step=%s substep=%s",
            iteration_num,
            step,
            substep,
        )

    def update_substep(
        self,
        iteration_num: int,
        step: str,
        substep: str,
        metadata: Optional[Dict] = None,
        completed: Optional[bool] = None,
    ):
        """Update a substep without discarding prior metadata."""
        round_key = f"iteration_{iteration_num}"
        if round_key not in self.progress:
            self.progress[round_key] = {}

        step_data = self.progress[round_key].get(step)
        if not isinstance(step_data, dict):
            step_data = {
                "completed": False,
                "timestamp": "",
                "metadata": {},
                "substeps": {},
            }
            self.progress[round_key][step] = step_data

        if not isinstance(step_data.get("substeps"), dict):
            step_data["substeps"] = {}

        substeps = step_data["substeps"]
        existing = substeps.get(substep, {})
        if not isinstance(existing, dict):
            existing = {}
        existing_metadata = existing.get("metadata", {})
        if not isinstance(existing_metadata, dict):
            existing_metadata = {}

        merged_metadata = existing_metadata.copy()
        if metadata:
            merged_metadata.update(metadata)

        substeps[substep] = {
            "completed": existing.get("completed", False) if completed is None else completed,
            "timestamp": datetime.now().isoformat(),
            "metadata": merged_metadata,
        }

        self._save_progress()
        logger.info(
            "Updated substep: iteration=%s step=%s substep=%s completed=%s",
            iteration_num,
            step,
            substep,
            substeps[substep]["completed"],
        )

    def get_substep_metadata(self, iteration_num: int, step: str, substep: str) -> Optional[Dict]:
        """Return metadata for a substep, if available."""
        round_key = f"iteration_{iteration_num}"
        if round_key not in self.progress:
            return None

        step_data = self.progress[round_key].get(step, {})
        if not isinstance(step_data, dict):
            return None
        substeps = step_data.get("substeps", {})
        if not isinstance(substeps, dict):
            return None
        substep_data = substeps.get(substep, {})
        metadata = substep_data.get("metadata") if isinstance(substep_data, dict) else None
        return metadata if isinstance(metadata, dict) else None

    def reset_substep(self, iteration_num: int, step: str, substep: str):
        """Clear a substep from saved progress."""
        round_key = f"iteration_{iteration_num}"
        if round_key in self.progress and step in self.progress[round_key]:
            step_data = self.progress[round_key].get(step)
            if not isinstance(step_data, dict):
                return
            substeps = step_data.get("substeps", {})
            if not isinstance(substeps, dict):
                return
            if substep in substeps:
                del substeps[substep]
                self._save_progress()
                logger.info(
                    "Reset substep: iteration=%s step=%s substep=%s",
                    iteration_num,
                    step,
                    substep,
                )

    def get_round_progress(self, iteration_num: int) -> Dict[str, bool]:
        """Return completion state for all steps in one iteration."""
        result = {}
        for step in self.steps:
            result[step] = self.is_step_completed(iteration_num, step)
        return result

    def is_round_completed(self, iteration_num: int) -> bool:
        """Return whether every step in the iteration is complete."""
        progress = self.get_round_progress(iteration_num)
        return all(progress.values())

    def get_next_incomplete_step(self, iteration_num: int) -> Optional[str]:
        """Return the next incomplete step in the configured step order."""
        for step in self.steps:
            if not self.is_step_completed(iteration_num, step):
                return step
        return None

    def reset_round(self, iteration_num: int):
        """Remove all saved progress for one iteration."""
        round_key = f"iteration_{iteration_num}"
        if round_key in self.progress:
            del self.progress[round_key]
            self._save_progress()
            logger.info("Reset iteration progress: iteration=%s", iteration_num)

    def reset_step(self, iteration_num: int, step: str):
        """Remove one step from saved progress."""
        round_key = f"iteration_{iteration_num}"
        if round_key in self.progress and step in self.progress[round_key]:
            del self.progress[round_key][step]
            self._save_progress()
            logger.info("Reset step progress: iteration=%s step=%s", iteration_num, step)

    def print_progress(self, iteration_num: int):
        """Print a compact progress summary for one iteration."""
        print(f"\nIteration {iteration_num} progress:")
        print("=" * 60)
        progress = self.get_round_progress(iteration_num)
        for step in self.steps:
            status = "[done]" if progress[step] else "[todo]"
            print(f"  {status} {step}")
        print("=" * 60)

        if self.is_round_completed(iteration_num):
            print(f"Iteration {iteration_num} is complete.\n")
        else:
            next_step = self.get_next_incomplete_step(iteration_num)
            print(f"Next step: {next_step}\n")

    def get_completed_rounds(self) -> List[int]:
        """Return all iterations whose steps are fully complete."""
        completed = []
        for key in self.progress.keys():
            if not key.startswith("iteration_"):
                continue
            match = re.match(r"iteration_(\d+)", key)
            if not match:
                continue
            iteration_num = int(match.group(1))
            if self.is_round_completed(iteration_num):
                completed.append(iteration_num)
        return sorted(completed)


if __name__ == "__main__":
    tracker = ProgressTracker(base_dir="test_results")
    print("Simulate iteration 1:")
    for i, step in enumerate(tracker.steps[:3]):
        tracker.mark_step_completed(1, step, metadata={"test": i})
        tracker.print_progress(1)

    print("Iteration 1 complete:", tracker.is_round_completed(1))
    print("Next step:", tracker.get_next_incomplete_step(1))
