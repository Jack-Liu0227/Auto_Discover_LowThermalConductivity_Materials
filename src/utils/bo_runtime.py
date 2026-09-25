from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd

CRYSTAL_SYSTEM_ORDER = ("triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic")

def crystal_system_from_spacegroup(spacegroup: Any = None, number: Any = None) -> str | None:
    """Resolve a crystal system from a space-group number or symbol."""
    # Pymatgen provides the authoritative International Tables mapping.  Keep
    # the range and symbol fallbacks below for lightweight environments or
    # incomplete historical metadata.
    try:
        from pymatgen.symmetry.groups import SpaceGroup

        if number is not None and str(number).strip():
            try:
                resolved = SpaceGroup.from_int_number(int(float(number)))
                return str(resolved.crystal_system).lower()
            except (TypeError, ValueError, AttributeError):
                pass
        symbol = str(spacegroup or "").strip()
        if symbol:
            try:
                return str(SpaceGroup(symbol).crystal_system).lower()
            except (TypeError, ValueError, AttributeError):
                # CIF symbols may use a subscript separator (for example
                # ``P 21/c``); pymatgen accepts the normalized ``P2_1/c`` form.
                normalized_symbol = symbol.replace(" ", "")
                normalized_symbol = re.sub(r"([23456])([123456])", r"\1_\2", normalized_symbol)
                try:
                    return str(SpaceGroup(normalized_symbol).crystal_system).lower()
                except (TypeError, ValueError, AttributeError):
                    pass
    except ImportError:
        pass

    try:
        n = int(float(number)) if number is not None and str(number).strip() else None
    except (TypeError, ValueError):
        n = None
    if n is not None and 1 <= n <= 230:
        if n <= 2: return "triclinic"
        if n <= 15: return "monoclinic"
        if n <= 74: return "orthorhombic"
        if n <= 142: return "tetragonal"
        if n <= 167: return "trigonal"
        if n <= 194: return "hexagonal"
        return "cubic"
    symbol = str(spacegroup or "").strip().replace("'", "").replace('"', '').upper().replace(" ", "")
    if not symbol: return None
    if symbol.startswith("P1") or symbol.startswith("P-1"): return "triclinic"
    if symbol[0] in "ABC" and ("2" in symbol or "/C" in symbol): return "monoclinic"
    if any(token in symbol for token in ("M-3", "-43", "23")): return "cubic"
    if any(token in symbol for token in ("6", "-6")): return "hexagonal"
    if any(token in symbol for token in ("3", "-3")): return "trigonal"
    if any(token in symbol for token in ("4", "-4")): return "tetragonal"
    if any(token in symbol for token in ("222", "MM2")): return "orthorhombic"
    return None


def crystal_system_is_high_symmetry(crystal_system: str | None, minimum: str = "orthorhombic") -> bool:
    if not crystal_system or crystal_system.lower() not in CRYSTAL_SYSTEM_ORDER:
        return False
    try:
        return CRYSTAL_SYSTEM_ORDER.index(crystal_system.lower()) >= CRYSTAL_SYSTEM_ORDER.index(minimum.lower())
    except ValueError:
        return False


def filter_high_symmetry_parents(rows: list[dict[str, Any]], minimum: str = "orthorhombic") -> tuple[list[dict[str, Any]], dict[str, int]]:
    accepted, stats = [], {name: 0 for name in CRYSTAL_SYSTEM_ORDER}
    stats["unresolved"] = 0
    for row in rows:
        system = str(row.get("crystal_system", "")).strip().lower()
        if system not in CRYSTAL_SYSTEM_ORDER:
            stats["unresolved"] += 1
            continue
        stats[system] += 1
        if crystal_system_is_high_symmetry(system, minimum):
            accepted.append(row)
    return accepted, stats


def _first_present(row: pd.Series, keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        try:
            missing = pd.isna(value)
            if not hasattr(missing, "__len__") and bool(missing):
                continue
        except (TypeError, ValueError):
            pass
        return value
    return None

from utils.config_loader import get_acquisition_params, get_effective_thresholds, get_sampling_params, get_high_symmetry_config, get_llm_formula_generation_config, get_screening_config


def load_bo_runtime_defaults() -> dict[str, Any]:
    acquisition = get_acquisition_params()
    sampling = get_sampling_params()
    thresholds = get_effective_thresholds()
    screening = get_screening_config()
    high_symmetry = get_high_symmetry_config()
    generation = get_llm_formula_generation_config()
    return {
        "xi": float(acquisition.get("xi", 0.01)),
        "samples": int(sampling.get("n_samples", 100)),
        "sampling_params": sampling,
        "k_threshold": float(thresholds.get("thermal_conductivity", 1.0)),
        "phonon_imag_tol": float(thresholds.get("dynamic_min_frequency", -0.1)),
        "high_symmetry": high_symmetry,
        "high_symmetry_parent_enabled": bool(high_symmetry.get("enabled", True)),
        "high_symmetry_min_crystal_system": str(high_symmetry.get("min_crystal_system", "orthorhombic")),
        "high_symmetry_parent_scope": str(high_symmetry.get("scope", "historical_seed_only")),
        "llm_formula_generation": generation,
        "llm_formula_generation_enabled": bool(generation.get("enabled", True)),
        "llm_formula_generation_start_iteration": int(generation.get("start_iteration", 2)),
        "llm_formula_proposal_count": int(generation.get("proposal_count", 20)),
        "llm_formula_parent_count": int(generation.get("parent_count", 10)),
        "llm_formula_generation_max_tokens": int(generation.get("max_tokens", 12000)),
        "llm_formula_generation_max_retries": int(generation.get("max_retries", 3)),
        "screening": screening,
        **screening,
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

    try:
        df = pd.read_csv(sample_file, encoding="utf-8-sig")
    except (OSError, ValueError, pd.errors.ParserError):
        return None, None

    initial_samples: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        formula = _first_present(row, ("formula", "Formula", "composition"))
        kappa = _first_present(
            row,
            (
                "thermal_conductivity",
                "thermal_conductivity_w_mk",
                "kappa",
                "Kappa_Slack (W m-1 K-1)",
                "热导率(W/m·K)",
                "Thermal_Conductivity",
            ),
        )
        if formula is None or kappa is None:
            continue
        try:
            kappa_value = float(kappa)
        except (TypeError, ValueError):
            continue
        if not pd.notna(kappa_value):
            continue
        if is_stable_fallback and kappa_value >= float(stable_kappa_limit):
            continue
        initial_samples.append({"formula": str(formula).strip(), "thermal_conductivity": kappa_value})

    if not initial_samples:
        return None, None
    source = "stable materials (k<5)" if is_stable_fallback else "success materials"
    return initial_samples, source
