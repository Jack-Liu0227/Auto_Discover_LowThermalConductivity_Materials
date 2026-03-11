from __future__ import annotations

import os
import re
from typing import Any

import requests

from .models import DatabaseRecord
from .models import DatabaseQueryResult
from .normalizers import normalize_aflow_result

try:
    from pymatgen.core import Composition
except Exception:
    Composition = None


def _formula_to_species(formula: str) -> str:
    tokens = re.findall(r"[A-Z][a-z]?", formula or "")
    return ",".join(sorted(set(tokens)))


def _normalize_formula_fallback(formula: str) -> str:
    token_pattern = r"([A-Z][a-z]*)(\d*(?:\.\d+)?)"
    tokens = re.findall(token_pattern, (formula or "").replace(" ", ""))
    if not tokens:
        return (formula or "").replace(" ", "")

    normalized = []
    for el, count in tokens:
        normalized.append((el, float(count) if count else 1.0))
    normalized.sort(key=lambda x: x[0])

    parts: list[str] = []
    for el, cnt in normalized:
        if abs(cnt - round(cnt)) < 1e-8:
            cnt_str = str(int(round(cnt)))
        else:
            cnt_str = f"{cnt:g}"
        parts.append(f"{el}{'' if cnt_str == '1' else cnt_str}")
    return "".join(parts)


def _normalize_formula_reduced(formula: str) -> str:
    compact = (formula or "").strip().replace(" ", "")
    if not compact:
        return ""
    if Composition is not None:
        try:
            return Composition(compact).reduced_formula
        except Exception:
            pass
    return _normalize_formula_fallback(compact)


def _filter_exact_formula_records(candidate_formula: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target = _normalize_formula_reduced(candidate_formula)
    if not target:
        return []
    exact: list[dict[str, Any]] = []
    for record in records:
        record_formula = str(record.get("formula") or "")
        if _normalize_formula_reduced(record_formula) == target:
            exact.append(record)
    return exact


def _normalize_aflux_entry(entry: dict[str, Any]) -> DatabaseRecord:
    auid = str(entry.get("auid", "unknown"))
    compound = str(entry.get("compound", "") or "")
    source_url = entry.get("aurl")
    if source_url and not str(source_url).startswith("http"):
        source_url = f"https://{source_url}"
    return DatabaseRecord(
        material_id=auid,
        formula=compound,
        source_url=source_url or f"https://aflowlib.org/material.php?id={auid}",
        symmetry=str(entry.get("spacegroup_relax", "") or ""),
        stability=_to_float(entry.get("enthalpy_formation_atom")),
        band_gap=_to_float(entry.get("Egap")),
        energy_above_hull=_to_float(entry.get("energy_cell")),
        metadata={
            "species": entry.get("species"),
            "stoichiometry": entry.get("stoichiometry"),
            "prototype": entry.get("prototype"),
        },
        structure_available=False,
        structure_error="AFLOW structure retrieval not available in current query path.",
    )


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _query_aflux_http(species: str, limit: int) -> dict:
    base_url = os.getenv("AFLOW_BASE_URL", "https://aflowlib.org/API/aflux/").strip()
    if not base_url:
        base_url = "https://aflowlib.org/API/aflux/"
    if not base_url.endswith("/"):
        base_url += "/"

    # AFLUX grammar: ?species(A,B),paging(page,size),format(json)
    query_url = f"{base_url}?species({species}),paging(1,{max(1, int(limit))}),format(json)"
    try:
        resp = requests.get(query_url, timeout=25)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return DatabaseQueryResult(
            success=False,
            database="AFLOW",
            query={"species": species, "limit": limit, "url": query_url},
            error=f"AFLUX http query failed: {exc}",
        ).to_dict()

    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, dict) and ("compound" in value or "auid" in value):
                rows.append(value)
    records = [_normalize_aflux_entry(row) for row in rows[:limit]]
    return DatabaseQueryResult(
        success=True,
        database="AFLOW",
        query={"species": species, "limit": limit, "url": query_url},
        records=records,
        error=None,
    ).to_dict()


def query_aflow(formula: str, limit: int = 3, include_structure: bool = False) -> dict:
    _ = include_structure
    species = _formula_to_species(formula)
    if not species:
        return DatabaseQueryResult(
            success=False,
            database="AFLOW",
            query={"formula": formula, "limit": limit, "include_structure": include_structure},
            error="Invalid formula for AFLOW query.",
        ).to_dict()

    # First try python-aflow package (legacy path in existing code).
    try:
        from aflow import K, search

        results = search().filter(K.species == species)
        rows: list[dict[str, Any]] = []
        for idx, result in enumerate(results):
            if idx >= limit:
                break
            rows.append(normalize_aflow_result(result))
        exact_rows = _filter_exact_formula_records(formula, rows)

        return DatabaseQueryResult(
            success=True,
            database="AFLOW",
            query={
                "formula": formula,
                "limit": limit,
                "species": species,
                "include_structure": include_structure,
                "raw_records_count": len(rows),
                "exact_records_count": len(exact_rows),
                "exact_match_mode": "reduced_formula",
            },
            records=exact_rows,
            error=None,
        ).to_dict()
    except Exception as exc:
        # Fallback to AFLUX HTTP endpoint, because many python-aflow installs
        # still point to deprecated /search/API path that now returns 404/502.
        fallback = _query_aflux_http(species=species, limit=limit)
        if fallback.get("success"):
            raw_rows = fallback.get("records", [])
            exact_rows = _filter_exact_formula_records(formula, raw_rows)
            fallback_query = dict(fallback.get("query") or {})
            fallback_query.update(
                {
                    "formula": formula,
                    "include_structure": include_structure,
                    "raw_records_count": len(raw_rows),
                    "exact_records_count": len(exact_rows),
                    "exact_match_mode": "reduced_formula",
                }
            )
            fallback["query"] = fallback_query
            fallback["records"] = exact_rows
            return fallback
        return DatabaseQueryResult(
            success=False,
            database="AFLOW",
            query={"formula": formula, "limit": limit, "species": species, "include_structure": include_structure},
            error=f"python-aflow failed: {exc}; fallback failed: {fallback.get('error')}",
        ).to_dict()
