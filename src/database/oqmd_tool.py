from __future__ import annotations

import requests

try:
    from pymatgen.core import Composition
except Exception:
    Composition = None

from .models import DatabaseQueryResult
from .normalizers import normalize_oqmd_entry


def _canonical_formulas(formula: str) -> list[str]:
    formula = (formula or "").strip()
    if not formula:
        return []

    candidates = [formula]
    if Composition is not None:
        try:
            comp = Composition(formula.replace(" ", ""))
            integer_formula, _ = comp.get_integer_formula_and_factor()
            reduced_formula = comp.reduced_formula
            el_amt = comp.get_el_amt_dict()
            parts: list[str] = []
            for el, amt in sorted(el_amt.items()):
                amt_f = float(amt)
                count = str(int(amt_f)) if amt_f.is_integer() else f"{amt_f:g}"
                parts.append(f"{el}{count}")
            expanded = "".join(parts)
            for item in (integer_formula, reduced_formula):
                if item and item not in candidates:
                    candidates.append(item)
            if expanded and expanded not in candidates:
                candidates.append(expanded)
        except Exception:
            pass
    return candidates


def _query_oqmd_http(formula: str, limit: int) -> dict:
    url = "https://oqmd.org/oqmdapi/formationenergy"
    try:
        resp = requests.get(url, params={"composition": formula, "limit": limit}, timeout=25)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return DatabaseQueryResult(
            success=False,
            database="OQMD",
            query={"formula": formula, "limit": limit, "source": "http"},
            error=str(exc),
        ).to_dict()

    rows = payload.get("data", []) if isinstance(payload, dict) else []
    records = [normalize_oqmd_entry(row) for row in rows[:limit]]
    return DatabaseQueryResult(
        success=True,
        database="OQMD",
        query={"formula": formula, "limit": limit, "source": "http"},
        records=records,
        error=None,
    ).to_dict()


def _query_oqmd_qmpy(formula: str, limit: int) -> dict:
    import qmpy_rester as qr
    with qr.QMPYRester() as q:
        payload = q.get_oqmd_phases(
            composition=formula,
            limit=limit,
            verbose=False,
        )
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    records = [normalize_oqmd_entry(row) for row in rows[:limit]]
    return DatabaseQueryResult(
        success=True,
        database="OQMD",
        query={"formula": formula, "limit": limit, "source": "qmpy_rester"},
        records=records,
        error=None,
    ).to_dict()


def query_oqmd(formula: str, limit: int = 3, include_structure: bool = False) -> dict:
    _ = include_structure  # structure payload already included by OQMD response when available
    formulas = _canonical_formulas(formula)
    if not formulas:
        return DatabaseQueryResult(
            success=False,
            database="OQMD",
            query={"formula": formula, "limit": limit, "include_structure": include_structure},
            error="Invalid formula.",
        ).to_dict()

    qmpy_error: str | None = None
    try:
        import qmpy_rester  # noqa: F401
        qmpy_available = True
    except Exception as exc:
        qmpy_available = False
        qmpy_error = str(exc)

    last_http_error: str | None = None
    http_success_seen = False

    for candidate_formula in formulas:
        if qmpy_available:
            try:
                result = _query_oqmd_qmpy(candidate_formula, limit)
                result["query"]["include_structure"] = include_structure
                result["query"]["input_formula"] = formula
                if result.get("records"):
                    return result
                if not result.get("error"):
                    # Try another canonical formula before fallback.
                    continue
            except Exception as exc:
                qmpy_error = str(exc)
                qmpy_available = False

        result = _query_oqmd_http(candidate_formula, limit)
        result["query"]["include_structure"] = include_structure
        result["query"]["input_formula"] = formula
        if result.get("success") and result.get("records"):
            return result
        if result.get("success"):
            http_success_seen = True
        if not result.get("success"):
            last_http_error = str(result.get("error") or "")
            continue

    # Keep successful-but-empty behavior if APIs are reachable.
    if qmpy_available or qmpy_error is None or http_success_seen:
        return DatabaseQueryResult(
            success=True,
            database="OQMD",
            query={"formula": formula, "limit": limit, "include_structure": include_structure},
            records=[],
            error=None,
        ).to_dict()

    return DatabaseQueryResult(
        success=False,
        database="OQMD",
        query={"formula": formula, "limit": limit, "include_structure": include_structure},
        error=f"qmpy_rester failed: {qmpy_error}; http fallback failed: {last_http_error}",
    ).to_dict()
