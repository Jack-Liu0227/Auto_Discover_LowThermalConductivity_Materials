"""Composition identity and source-independent chemical selection."""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from functools import reduce
from typing import Any, Iterable

from pymatgen.core import Composition

# Unicode subscript digits commonly appear in CIF/database formulas.  Keep the
# normalization here shared by validation, IDs, matching, and persisted traces.
SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SCORE_FIELDS = ("chemical_plausibility_score", "low_kappa_mechanism_score", "stability_risk_score")
DEFAULT_GROUPS = {
    "A": ["Ag", "Cu", "In", "Sn", "Pb"],
    "B": ["As", "Sb", "Ge", "Bi", "Ti", "V"],
    "Ch": ["S", "Se", "Te"],
}


def normalize_formula_text(value: Any) -> str:
    return "".join(str(value or "").translate(SUBSCRIPTS).split())


def formula_amounts(value: Any) -> dict[str, int]:
    amounts = Composition(normalize_formula_text(value), strict=True).get_el_amt_dict()
    if not amounts or any(not math.isfinite(v) or v <= 0 or not math.isclose(v, round(v), abs_tol=1e-8) for v in amounts.values()):
        raise ValueError("formula must contain positive integer coefficients")
    return {e: int(round(v)) for e, v in amounts.items()}


def canonical_formula(value: Any) -> str:
    text = normalize_formula_text(value)
    if not text:
        return ""
    try:
        return normalize_formula_text(Composition(text, strict=True).reduced_formula)
    except (ValueError, TypeError):
        return ""


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def add_candidate_ids(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonical deduplication and opaque IDs in a source-independent display order."""
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = canonical_formula(candidate.get("formula"))
        if not key:
            continue
        row = dict(candidate)
        row["canonical_formula"] = key
        source = str(row.get("candidate_source") or "bo")
        sources = row.get("candidate_sources")
        row["candidate_sources"] = sorted(set(sources if isinstance(sources, list) else [source]))
        row["candidate_source"] = source
        if key in unique:
            kept = unique[key]
            kept["candidate_sources"] = sorted(set(kept["candidate_sources"] + row["candidate_sources"]))
            kept["candidate_source"] = "both" if len(kept["candidate_sources"]) > 1 else kept["candidate_source"]
            kept.setdefault("additional_lineage", []).append({k: row.get(k) for k in ("formula", "parent_formula", "generation_mode", "substitution_rule")})
            continue
        row["candidate_id"] = "cand_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        if source != "bo" and row.get("original_bo_rank") is None:
            row["original_bo_rank"] = None
        else:
            row["original_bo_rank"] = row.get("original_bo_rank") or row.get("rank")
        unique[key] = row
    return sorted(unique.values(), key=lambda r: r["candidate_id"])


def chemistry_family_key(candidate: dict[str, Any], groups: dict | None = None) -> tuple[str, ...]:
    groups = groups or DEFAULT_GROUPS
    amounts = formula_amounts(candidate.get("formula"))
    return tuple("+".join(sorted(e for e in amounts if e in groups.get(g, []))) for g in ("A", "B", "Ch"))


def stoichiometry_family_key(candidate: dict[str, Any], groups: dict | None = None) -> tuple[int, ...]:
    groups = groups or DEFAULT_GROUPS
    amounts = formula_amounts(candidate.get("formula"))
    values = tuple(sum(v for e, v in amounts.items() if e in groups.get(g, [])) for g in ("A", "B", "Ch"))
    divisor = reduce(math.gcd, values) or 1
    return tuple(v // divisor for v in values)


def valid_chemical_score(row: dict[str, Any]) -> bool:
    return all(type(row.get(k)) is int and 0 <= row[k] <= 10 for k in SCORE_FIELDS)


def select_diverse_candidates(candidates, scores, n_select, *, max_same_element_tuple=2, max_same_stoichiometry_pattern=4, groups=None):
    """No numerical weights or source quotas: chemical plausibility, risk, mechanism.

    Valid chemical scores precede the unscored fallback pool. Fallback uses real
    EI if available; absent predictions are never represented as zero.
    """
    if min(max_same_element_tuple, max_same_stoichiometry_pattern) < 1:
        raise ValueError("diversity caps must be positive")
    rows = add_candidate_ids(candidates)
    by_id = {r["candidate_id"]: r for r in rows}
    by_formula = {r["canonical_formula"]: r for r in rows}
    scored = {}
    for item in scores or []:
        if not isinstance(item, dict) or not valid_chemical_score(item):
            continue
        target = by_id.get(item.get("candidate_id")) if item.get("candidate_id") else by_formula.get(canonical_formula(item.get("formula")))
        if target is None:
            continue
        if item.get("formula") and canonical_formula(item["formula"]) != target["canonical_formula"]:
            continue
        normalized_score = dict(item)
        # This is an audit/display composite only.  Selection still compares
        # the three chemistry fields separately and applies the diversity caps;
        # no BO/LLM source weight is introduced here.
        normalized_score["chemical_score"] = (
            normalized_score["chemical_plausibility_score"]
            + normalized_score["low_kappa_mechanism_score"]
            - normalized_score["stability_risk_score"]
        )
        scored[target["candidate_id"]] = normalized_score

    def order(row):
        item = scored.get(row["candidate_id"])
        if item is not None:
            return (0, -item["chemical_plausibility_score"], item["stability_risk_score"], -item["low_kappa_mechanism_score"], row["candidate_id"])
        ei = finite_number(row.get("ei"))
        return (1, ei is None, -(ei or 0.0), 0, row["candidate_id"])

    ordered = sorted(rows, key=order)
    target_count = min(max(int(n_select), 0), len(rows))
    selected, seen, reasons = [], set(), {}
    element_counts, stoich_counts = Counter(), Counter()
    relaxations = []
    level = 0
    while len(selected) < target_count:
        element_cap, stoich_cap = max_same_element_tuple + level, max_same_stoichiometry_pattern + level
        for row in ordered:
            cid = row["candidate_id"]
            if cid in seen:
                continue
            try:
                ek = chemistry_family_key(row, groups)
            except (TypeError, ValueError, KeyError):
                ek = ("invalid", cid)
            try:
                sk = stoichiometry_family_key(row, groups)
            except (TypeError, ValueError, KeyError):
                sk = ("invalid", cid)
            if element_counts[ek] >= element_cap or stoich_counts[sk] >= stoich_cap:
                continue
            selected.append(row)
            seen.add(cid)
            element_counts[ek] += 1
            stoich_counts[sk] += 1
            reasons[cid] = {"selection_reason": "chemical_scores" if cid in scored else "deterministic_ei_fallback", "cap_relaxation_level": level}
            if len(selected) == target_count:
                break
        if len(selected) < target_count:
            level += 1
            relaxations.append({"level": level, "element_tuple_cap": max_same_element_tuple + level, "stoichiometry_cap": max_same_stoichiometry_pattern + level, "reason": "insufficient eligible candidates under previous caps"})
    selections = [dict(row, final_rank=i, **reasons[row["candidate_id"]]) for i, row in enumerate(selected, 1)]
    return selections, {
        "selected_count": len(selections), "requested_count": int(n_select),
        "scored_count": len(scored), "fallback_count": sum(r["candidate_id"] not in scored for r in selections),
        "relaxed_caps": bool(relaxations), "relaxations": relaxations,
        "element_tuple_counts": {"/".join(k): v for k, v in element_counts.items()},
        "stoichiometry_pattern_counts": {":".join(map(str, k)): v for k, v in stoich_counts.items()},
        "score_order": ["chemical_plausibility_score descending", "stability_risk_score ascending", "low_kappa_mechanism_score descending", "candidate_id"],
    }
