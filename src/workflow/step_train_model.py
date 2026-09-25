# -*- coding: utf-8 -*-
"""Workflow Step 1: train the GPR model for the current iteration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from models.train_gpr_model import train_gpr_model
from utils.path_config import PathConfig
from utils.workflow_resume import file_sha256


def step_train_model(
    iteration_num: int,
    path_config: Optional[PathConfig] = None,
    data_root: str = "data_llm",
    models_root: str = "models/GPR",
):
    """Train the GPR model using the immediately preceding iteration data."""
    print("=" * 80)
    print(f"Step 1: Train GPR model (Iteration {iteration_num})")
    print("=" * 80)

    prev_iteration = iteration_num - 1
    data_path = None

    if path_config:
        data_dir_base = path_config.data_root
    else:
        data_dir_base = project_root / data_root

    candidate = data_dir_base / f"iteration_{prev_iteration}" / "data.csv"
    if candidate.exists():
        data_path = candidate

    if data_path is None:
        required_path = data_dir_base / f"iteration_{prev_iteration}" / "data.csv"
        print(f"[ERROR] Required previous-iteration data is missing: {required_path}")
        return {
            "success": False,
            "error": f"Required previous-iteration data not found: {required_path}",
        }

    if path_config:
        model_dir = path_config.get_iteration_model_path(prev_iteration)
    else:
        model_dir = project_root / models_root / f"iteration_{prev_iteration}"

    if not data_path.exists():
        print(f"[ERROR] Data file missing: {data_path}")
        return {"success": False, "error": f"Data file not found: {data_path}"}

    print(f"[INFO] Training data: {data_path}")
    print(f"[INFO] Model output dir: {model_dir}")

    try:
        train_gpr_model(str(data_path), str(model_dir))

        model_file = model_dir / "gpr_thermal_conductivity.joblib"
        scaler_file = model_dir / "gpr_scaler.joblib"

        if model_file.exists() and scaler_file.exists():
            print("[OK] Model training completed")
            return {
                "success": True,
                "model_dir": str(model_dir),
                "model_file": str(model_file),
                "scaler_file": str(scaler_file),
                "training_data": str(data_path),
                "training_data_sha256": file_sha256(data_path),
            }

        print("[ERROR] Training finished but expected model files were not created")
        return {"success": False, "error": "Model files not created"}

    except Exception as exc:
        print(f"[ERROR] Training failed: {exc}")
        return {"success": False, "error": str(exc)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=1)
    args = parser.parse_args()

    result = step_train_model(args.iteration)
    print(f"\nResult: {result}")
