from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from utils.config_loader import get_acquisition_params, get_effective_thresholds, get_sampling_params


def load_bo_runtime_defaults() -> dict[str, Any]:
    acquisition = get_acquisition_params()
    sampling = get_sampling_params()
    thresholds = get_effective_thresholds()
    return {
        "xi": float(acquisition.get("xi", 0.01)),
        "samples": int(sampling.get("n_samples", 100)),
        "k_threshold": float(thresholds.get("thermal_conductivity", 1.0)),
        "phonon_imag_tol": float(thresholds.get("dynamic_min_frequency", -0.1)),
    }


def extract_initial_samples_from_result(
    extract_result: dict[str, Any] | None,
    stable_kappa_limit: float = 5.0,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not extract_result:
        return None, None

    sample_file = None
    is_stable_fallback = False
    if extract_result.get("has_success"):
        sample_file = extract_result.get("success_deduped_file") or extract_result.get("success_file")
    elif extract_result.get("has_stable"):
        sample_file = extract_result.get("stable_deduped_file") or extract_result.get("stable_file")
        is_stable_fallback = True

    if not sample_file or not Path(sample_file).exists():
        return None, None

    df = pd.read_csv(sample_file)
    initial_samples: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        formula = row.get("formula") or row.get("Formula") or row.get("缁勫垎")
        kappa = (
            row.get("thermal_conductivity")
            or row.get("kappa")
            or row.get("热导率(W/m·K)")
            or row.get("Thermal_Conductivity")
            or row.get("鐑鐜?(W/m路K)")
        )
        if not formula or kappa is None:
            continue
        if is_stable_fallback and float(kappa) >= float(stable_kappa_limit):
            continue
        initial_samples.append({"formula": formula, "thermal_conductivity": kappa})

    if not initial_samples:
        return None, None
    source = "stable materials (k<5)" if is_stable_fallback else "success materials"
    return initial_samples, source
