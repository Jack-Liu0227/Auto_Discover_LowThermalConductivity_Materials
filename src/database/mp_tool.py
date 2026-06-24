from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from .models import DatabaseQueryResult
from .normalizers import normalize_mp_doc

MP_FIELDS = [
    "material_id",
    "formula_pretty",
    "symmetry",
    "formation_energy_per_atom",
    "energy_above_hull",
    "band_gap",
    "density",
    "is_stable",
]


def _fetch_mp_structure_cif(mpr: Any, material_id: str) -> tuple[str | None, str | None]:
    try:
        structure = mpr.get_structure_by_material_id(material_id)
        if structure is not None:
            return structure.to(fmt="cif"), None
    except Exception:
        pass

    try:
        materials_api = getattr(mpr, "materials", None)
        if materials_api is not None and hasattr(materials_api, "get_structure_by_material_id"):
            structure = materials_api.get_structure_by_material_id(material_id)
            if structure is not None:
                return structure.to(fmt="cif"), None
    except Exception:
        pass

    try:
        docs = mpr.materials.summary.search(material_ids=[material_id], fields=["structure"])
        if docs:
            struct = getattr(docs[0], "structure", None)
            if struct is not None:
                return struct.to(fmt="cif"), None
    except Exception as exc:
        return None, str(exc)
    return None, "structure not available from MP API response"


def query_materials_project(formula: str, limit: int = 3, include_structure: bool = False) -> dict:
    load_dotenv(override=False)
    api_key = os.getenv("MP_API_KEY")
    if not api_key:
        return DatabaseQueryResult(
            success=False,
            database="MP",
            query={"formula": formula, "limit": limit, "include_structure": include_structure},
            error="MP_API_KEY is not set.",
        ).to_dict()

    try:
        from mp_api.client import MPRester
    except Exception as exc:
        return DatabaseQueryResult(
            success=False,
            database="MP",
            query={"formula": formula, "limit": limit, "include_structure": include_structure},
            error=f"mp-api import failed: {exc}",
        ).to_dict()

    try:
        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(formula=formula, fields=MP_FIELDS)
            records = [normalize_mp_doc(doc) for doc in docs[:limit]]
            if include_structure:
                for record in records:
                    cif_content, structure_error = _fetch_mp_structure_cif(mpr, record.material_id)
                    record.cif_content = cif_content
                    record.structure_available = bool(cif_content)
                    record.structure_error = None if cif_content else structure_error
        return DatabaseQueryResult(
            success=True,
            database="MP",
            query={"formula": formula, "limit": limit, "include_structure": include_structure},
            records=records,
            error=None,
        ).to_dict()
    except Exception as exc:
        return DatabaseQueryResult(
            success=False,
            database="MP",
            query={"formula": formula, "limit": limit, "include_structure": include_structure},
            error=str(exc),
        ).to_dict()
