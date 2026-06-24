from __future__ import annotations

from typing import Any

from .models import DatabaseRecord


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_mp_doc(doc: Any) -> DatabaseRecord:
    symmetry = None
    if getattr(doc, "symmetry", None) is not None:
        symbol = getattr(doc.symmetry, "symbol", None)
        crystal = getattr(getattr(doc.symmetry, "crystal_system", None), "value", None)
        if symbol and crystal:
            symmetry = f"{crystal} / {symbol}"
        else:
            symmetry = symbol or crystal

    material_id = str(getattr(doc, "material_id", "unknown"))
    return DatabaseRecord(
        material_id=material_id,
        formula=str(getattr(doc, "formula_pretty", "")),
        source_url=f"https://next-gen.materialsproject.org/materials/{material_id}",
        symmetry=symmetry,
        stability=_to_float(getattr(doc, "formation_energy_per_atom", None)),
        band_gap=_to_float(getattr(doc, "band_gap", None)),
        energy_above_hull=_to_float(getattr(doc, "energy_above_hull", None)),
        metadata={
            "density": _to_float(getattr(doc, "density", None)),
            "is_stable": getattr(doc, "is_stable", None),
        },
        structure_available=False,
    )


def normalize_oqmd_entry(entry: dict[str, Any]) -> DatabaseRecord:
    material_id = str(entry.get("entry_id", "unknown"))
    return DatabaseRecord(
        material_id=material_id,
        formula=str(entry.get("name") or entry.get("composition") or ""),
        source_url=f"https://oqmd.org/materials/entry/{material_id}",
        symmetry=entry.get("spacegroup"),
        stability=_to_float(entry.get("stability")),
        metadata={
            "icsd_id": entry.get("icsd_id"),
            "volume": _to_float(entry.get("volume")),
            "unit_cell": entry.get("unit_cell"),
            "sites": entry.get("sites"),
            "composition": entry.get("composition"),
        },
        structure_available=bool(entry.get("unit_cell") and entry.get("sites")),
    )


def normalize_aflow_result(result: Any) -> DatabaseRecord:
    auid = str(getattr(result, "auid", "unknown"))
    compound = str(getattr(result, "compound", ""))
    return DatabaseRecord(
        material_id=auid,
        formula=compound,
        source_url=f"http://aflowlib.org/material.php?id={auid}",
        symmetry=str(getattr(result, "spacegroup_relax", "") or ""),
        stability=_to_float(getattr(result, "enthalpy_formation_atom", None)),
        band_gap=_to_float(getattr(result, "Egap", None)),
        energy_above_hull=_to_float(getattr(result, "energy_cell", None)),
        metadata={
            "species": getattr(result, "species", None),
            "stoichiometry": getattr(result, "stoichiometry", None),
        },
        structure_available=False,
        structure_error="AFLOW structure retrieval not available in current query path.",
    )
