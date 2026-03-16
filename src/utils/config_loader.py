# -*- coding: utf-8 -*-
"""Load workflow config and validate it against the theory seed document."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_THEORY_DOC_PATH = PROJECT_ROOT / "doc" / "Theoretical_principle_document.md"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the project YAML config."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def get_bayesian_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    return dict(config.get("bayesian_optimization", {}))


def get_acquisition_params(config: dict[str, Any] | None = None) -> dict[str, Any]:
    bo_config = get_bayesian_config(config)
    return dict(bo_config.get("acquisition", {"function": "EI", "xi": 0.01}))


def get_sampling_params(config: dict[str, Any] | None = None) -> dict[str, Any]:
    bo_config = get_bayesian_config(config)
    default_sampling = {
        "n_samples": 100,
        "max_atoms": 20,
        "allowed_elements": [],
        "hard_constraints": {},
    }
    sampling = default_sampling | dict(bo_config.get("sampling", {}))
    sampling["allowed_elements"] = [str(item) for item in sampling.get("allowed_elements", [])]
    return sampling


def get_effective_thresholds(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the canonical runtime thresholds used by the workflow."""
    config = config or load_config()
    thresholds = dict(config.get("thresholds", {}))
    tools = dict(config.get("tools", {}))
    ai4kappa = dict(tools.get("ai4kappa", {}))
    mattersim = dict(tools.get("mattersim", {}))

    thermal_conductivity = thresholds.get(
        "thermal_conductivity",
        ai4kappa.get("k_threshold", 1.0),
    )
    dynamic_min_frequency = thresholds.get(
        "dynamic_min_frequency",
        mattersim.get("imaginary_freq_threshold", -0.1),
    )

    return {
        "thermal_conductivity": float(thermal_conductivity),
        "dynamic_min_frequency": float(dynamic_min_frequency),
        "stability": str(thresholds.get("stability", "stable")),
    }


def get_workflow_search_prior(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the active workflow search prior from config."""
    sampling = get_sampling_params(config)
    thresholds = get_effective_thresholds(config)
    hard_constraints = dict(sampling.get("hard_constraints", {}))
    schema = dict(hard_constraints.get("schema", {}))

    return {
        "allowed_elements": list(sampling.get("allowed_elements", [])),
        "max_atoms": int(sampling.get("max_atoms", 20)),
        "hard_constraints": hard_constraints,
        "schema_type": str(schema.get("type", "")),
        "thermal_conductivity": thresholds["thermal_conductivity"],
        "dynamic_min_frequency": thresholds["dynamic_min_frequency"],
    }


def read_theory_doc(doc_path: str | Path | None = None) -> str:
    path = Path(doc_path) if doc_path is not None else DEFAULT_THEORY_DOC_PATH
    if not path.exists():
        raise FileNotFoundError(f"Theory document not found: {path}")
    return path.read_text(encoding="utf-8")


def extract_doc_element_library(doc_text: str) -> list[str]:
    """Parse the element library table in Section 1.1."""
    elements: list[str] = []
    seen: set[str] = set()

    for line in doc_text.splitlines():
        if not line.startswith("|") or "**" not in line:
            continue
        columns = [segment.strip() for segment in line.split("|")]
        if len(columns) < 4:
            continue
        raw_elements = columns[2]
        for item in raw_elements.split(","):
            token = item.strip()
            if re.fullmatch(r"[A-Z][a-z]?", token) and token not in seen:
                seen.add(token)
                elements.append(token)
    return elements


def extract_doc_workflow_prior(doc_text: str) -> dict[str, Any]:
    """Parse the workflow note that mirrors active search settings."""
    patterns = {
        "composition_prior": r"\*\*Composition prior\*\*:\s*(?:ternary\s*)?`([^`]+)`",
        "max_atoms": r"\*\*Max atoms\*\*:\s*`?(\d+)`?",
        "thermal_conductivity": r"\*\*Success threshold\*\*:\s*`?k\s*<\s*([-+]?\d*\.?\d+)\s*W/\(m-K\)`?",
        "dynamic_min_frequency": r"\*\*Dynamical stability threshold\*\*:\s*`?Min_Frequency\s*>=\s*([-+]?\d*\.?\d+)\s*THz`?",
    }

    parsed: dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, doc_text)
        if not match:
            continue
        value = match.group(1)
        if key == "max_atoms":
            parsed[key] = int(value)
        elif key in {"thermal_conductivity", "dynamic_min_frequency"}:
            parsed[key] = float(value)
        else:
            parsed[key] = value
    return parsed


def validate_theory_doc_sync(
    config: dict[str, Any] | None = None,
    doc_path: str | Path | None = None,
) -> list[str]:
    """Return a list of sync issues between config and the theory document."""
    config = config or load_config()
    doc_text = read_theory_doc(doc_path)
    prior = get_workflow_search_prior(config)

    issues: list[str] = []

    doc_elements = extract_doc_element_library(doc_text)
    if doc_elements and doc_elements != prior["allowed_elements"]:
        issues.append(
            "Element library mismatch: "
            f"doc={doc_elements} vs config={prior['allowed_elements']}"
        )

    doc_prior = extract_doc_workflow_prior(doc_text)
    doc_schema_type = doc_prior.get("composition_prior")
    if doc_schema_type and doc_schema_type != prior["schema_type"]:
        issues.append(
            "Composition prior mismatch: "
            f"doc={doc_schema_type} vs config={prior['schema_type']}"
        )

    if "max_atoms" in doc_prior and int(doc_prior["max_atoms"]) != int(prior["max_atoms"]):
        issues.append(
            "Max-atoms mismatch: "
            f"doc={doc_prior['max_atoms']} vs config={prior['max_atoms']}"
        )

    if "thermal_conductivity" in doc_prior:
        doc_k = float(doc_prior["thermal_conductivity"])
        cfg_k = float(prior["thermal_conductivity"])
        if abs(doc_k - cfg_k) > 1e-9:
            issues.append(
                "Success-threshold mismatch: "
                f"doc={doc_k} vs config={cfg_k}"
            )

    if "dynamic_min_frequency" in doc_prior:
        doc_imag = float(doc_prior["dynamic_min_frequency"])
        cfg_imag = float(prior["dynamic_min_frequency"])
        if abs(doc_imag - cfg_imag) > 1e-9:
            issues.append(
                "Dynamical-stability threshold mismatch: "
                f"doc={doc_imag} vs config={cfg_imag}"
            )

    return issues


def ensure_theory_doc_sync(
    config: dict[str, Any] | None = None,
    doc_path: str | Path | None = None,
) -> None:
    """Raise when the theory document and active config drift apart."""
    issues = validate_theory_doc_sync(config=config, doc_path=doc_path)
    if issues:
        raise ValueError("; ".join(issues))


if __name__ == "__main__":
    current_config = load_config()
    print("Bayesian optimization config:")
    print(get_bayesian_config(current_config))
    print("\nAcquisition params:")
    print(get_acquisition_params(current_config))
    print("\nSampling params:")
    print(get_sampling_params(current_config))
    print("\nThresholds:")
    print(get_effective_thresholds(current_config))
    print("\nSync issues:")
    print(validate_theory_doc_sync(current_config))
