# -*- coding: utf-8 -*-
"""
Progress tracking utilities for iterative workflows.

This module intentionally keeps log and console output ASCII-safe so it can run
cleanly on Windows terminals configured with GBK.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Persist step-level and substep-level progress for each iteration."""

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

        changed = False
        for round_key, round_data in data.items():
            if not round_key.startswith("iteration_"):
                continue

            reports_dir = self.base_dir / round_key / "reports"
            llm_eval_report = reports_dir / "llm_evaluation_output.md"
            theory_report = reports_dir / "llm_theory_update_output.md"

            if "ai_evaluation" not in round_data and llm_eval_report.exists():
                round_data["ai_evaluation"] = {
                    "completed": True,
                    "timestamp": round_data.get("bayesian_optimization", {}).get("timestamp", ""),
                    "metadata": {
                        "backfilled": True,
                        "report": str(llm_eval_report),
                    },
                }
                changed = True

            if "document_update" not in round_data and theory_report.exists():
                round_data["document_update"] = {
                    "completed": True,
                    "timestamp": round_data.get("success_extraction", {}).get("timestamp", ""),
                    "metadata": {"backfilled": True},
                }
                changed = True

        if changed:
            self.progress = data
            self._save_progress()

        return data

    def _save_progress(self):
        """Write progress state to disk."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(self.progress, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error("Failed to save progress file: %s", exc)

    def is_step_completed(self, iteration_num: int, step: str) -> bool:
        """Return whether the given step is complete for this iteration."""
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
            if not all(substeps.get(name, {}).get("completed", False) for name in expected):
                return False
            if "merge_results" in substeps:
                return substeps.get("merge_results", {}).get("completed", False)
            return True

        return completed

    def mark_step_completed(self, iteration_num: int, step: str, metadata: Optional[Dict] = None):
        """Mark a step as complete and merge any metadata."""
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
        logger.info("Marked step completed: iteration=%s step=%s", iteration_num, step)

    def is_substep_completed(self, iteration_num: int, step: str, substep: str) -> bool:
        """Return whether the given substep is complete."""
        round_key = f"iteration_{iteration_num}"
        if round_key not in self.progress:
            return False

        step_data = self.progress[round_key].get(step, {})
        substeps = step_data.get("substeps", {})
        return substeps.get(substep, {}).get("completed", False)

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
        substeps = step_data.get("substeps", {})
        substep_data = substeps.get(substep, {})
        return substep_data.get("metadata")

    def reset_substep(self, iteration_num: int, step: str, substep: str):
        """Clear a substep from saved progress."""
        round_key = f"iteration_{iteration_num}"
        if round_key in self.progress and step in self.progress[round_key]:
            substeps = self.progress[round_key][step].get("substeps", {})
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
