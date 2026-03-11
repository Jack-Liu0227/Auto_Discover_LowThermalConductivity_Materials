from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from typing import Any

from agno.agent import Agent

from database import query_aflow, query_materials_project, query_oqmd
from workflow.step_ai_evaluation import step_ai_evaluation

try:
    from pymatgen.core import Composition
except Exception:
    Composition = None

try:
    from agno.tools.websearch import WebSearchTools
except Exception:
    WebSearchTools = None


DEFAULT_GENERIC_THEORY_QUERY = (
    "low lattice thermal conductivity mechanisms phonon scattering "
    "anharmonicity mass disorder lone pair rattling review"
)


def _extract_formula(material: dict[str, Any]) -> str:
    return str(material.get("formula") or material.get("Formula") or material.get("composition") or "").strip()


def _normalize_formula_fallback(formula: str) -> str:
    token_pattern = r"([A-Z][a-z]*)(\d*(?:\.\d+)?)"
    tokens = re.findall(token_pattern, formula.replace(" ", ""))
    if not tokens:
        return formula.replace(" ", "")

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


def normalize_formula_reduced(formula: str) -> str:
    formula = (formula or "").strip()
    if not formula:
        return ""

    compact = formula.replace(" ", "")
    if Composition is not None:
        try:
            return Composition(compact).reduced_formula
        except Exception:
            pass

    return _normalize_formula_fallback(compact)


def _extract_elements_from_formula(formula: str) -> list[str]:
    tokens = re.findall(r"[A-Z][a-z]?", formula or "")
    return sorted(set(tokens))


def _query_evidence_parallel(formula: str, limit: int = 3) -> dict[str, Any]:
    tools = {
        "MP": lambda: query_materials_project(formula, limit),
        "OQMD": lambda: query_oqmd(formula, limit),
        "AFLOW": lambda: query_aflow(formula, limit),
    }
    evidence: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fn): name for name, fn in tools.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                evidence[name] = future.result()
            except Exception as exc:
                evidence[name] = {
                    "success": False,
                    "database": name,
                    "query": {"formula": formula},
                    "records": [],
                    "error": str(exc),
                }
    return evidence


def _query_evidence_tiered(formula: str, limit: int = 3) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for name, fn in [
        ("MP", lambda: query_materials_project(formula, limit)),
        ("OQMD", lambda: query_oqmd(formula, limit)),
        ("AFLOW", lambda: query_aflow(formula, limit)),
    ]:
        result = fn()
        evidence[name] = result
        if result.get("success") and result.get("records"):
            break
    return evidence


def _query_evidence_mp_only(formula: str, limit: int = 3) -> dict[str, Any]:
    return {"MP": query_materials_project(formula, limit)}


def query_evidence(formula: str, strategy: str, limit: int = 3) -> dict[str, Any]:
    if strategy == "tiered":
        return _query_evidence_tiered(formula, limit)
    if strategy == "mp_only":
        return _query_evidence_mp_only(formula, limit)
    return _query_evidence_parallel(formula, limit)


def is_exact_formula_hit(candidate_formula: str, records: list[dict[str, Any]]) -> bool:
    candidate_reduced = normalize_formula_reduced(candidate_formula)
    if not candidate_reduced:
        return False

    for record in records:
        record_formula = str(record.get("formula") or "")
        if not record_formula:
            continue
        if normalize_formula_reduced(record_formula) == candidate_reduced:
            return True
    return False


def deduplicate_candidates(
    candidates: list[dict[str, Any]],
    db_strategy: str = "parallel_all",
    mode: str = "exact_reduced_formula",
    limit: int = 5,
    max_candidate_workers: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    if mode != "exact_reduced_formula":
        raise ValueError(f"Unsupported dedup mode: {mode}")

    def _process_one(material: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        formula = _extract_formula(material)
        evidence = query_evidence(formula, db_strategy, limit=limit) if formula else {}

        item = dict(material)
        item["db_evidence"] = evidence
        item["candidate_reduced_formula"] = normalize_formula_reduced(formula)

        matched_databases: list[str] = []
        local_db_success = False
        for db_name, result in evidence.items():
            if result.get("success"):
                local_db_success = True
            records = result.get("records", [])
            if is_exact_formula_hit(formula, records):
                matched_databases.append(db_name)

        item["db_exact_match_databases"] = matched_databases
        db_status = {
            db_name: {
                "success": bool(result.get("success")),
                "records": len(result.get("records", [])),
            }
            for db_name, result in evidence.items()
        }
        item["db_status"] = db_status

        item["is_existing_material"] = bool(matched_databases)

        print(
            f"[db-dedup] formula={formula}, reduced={item['candidate_reduced_formula']}, "
            f"matched={matched_databases}, status={db_status}"
        )
        return item, local_db_success

    workers = max(1, int(max_candidate_workers))
    ordered_items: list[dict[str, Any] | None] = [None] * len(candidates)
    any_db_success = False

    if workers == 1 or len(candidates) <= 1:
        for idx, material in enumerate(candidates):
            item, local_ok = _process_one(material)
            ordered_items[idx] = item
            any_db_success = any_db_success or local_ok
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, material): idx for idx, material in enumerate(candidates)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    item, local_ok = future.result()
                except Exception as exc:
                    material = candidates[idx]
                    formula = _extract_formula(material)
                    item = dict(material)
                    item["db_evidence"] = {}
                    item["candidate_reduced_formula"] = normalize_formula_reduced(formula)
                    item["db_exact_match_databases"] = []
                    item["db_status"] = {"DEDUP_WORKER": {"success": False, "records": 0}}
                    item["is_existing_material"] = False
                    item["dedup_error"] = str(exc)
                    local_ok = False
                    print(f"[db-dedup] worker failed idx={idx}, formula={formula}, error={exc}")
                ordered_items[idx] = item
                any_db_success = any_db_success or local_ok

    novel_candidates: list[dict[str, Any]] = []
    filtered_out: list[dict[str, Any]] = []
    for item in ordered_items:
        if not item:
            continue
        if item.get("is_existing_material"):
            filtered_out.append(item)
        else:
            novel_candidates.append(item)

    db_unavailable = not any_db_success
    print(
        f"[db-dedup] summary total={len(candidates)}, filtered={len(filtered_out)}, "
        f"novel={len(novel_candidates)}, db_unavailable={db_unavailable}, workers={workers}"
    )
    return novel_candidates, filtered_out, db_unavailable


def build_generic_theory_query(theory_template: str | None = None) -> str:
    return (theory_template or DEFAULT_GENERIC_THEORY_QUERY).strip()


def build_element_theory_query(material: dict[str, Any]) -> str:
    formula = _extract_formula(material)
    elements = _extract_elements_from_formula(formula)
    if not elements:
        return build_generic_theory_query()
    elem_str = " ".join(elements)
    return (
        f"{elem_str} low lattice thermal conductivity mechanism phonon anharmonicity "
        "mass disorder lone pair rattling"
    )


def build_material_theory_query(material: dict[str, Any]) -> str:
    formula = _extract_formula(material)
    elements = _extract_elements_from_formula(formula)
    elem_str = " ".join(elements)
    if formula and elem_str:
        return (
            f"{formula} {elem_str} low thermal conductivity mechanism phonon anharmonicity "
            "disorder lone pair rattling"
        )
    if formula:
        return f"{formula} low thermal conductivity mechanism phonon anharmonicity disorder"
    return build_generic_theory_query()


def build_websearch_queries(
    material: dict[str, Any],
    strategy: str = "hybrid",
    max_queries: int = 2,
    theory_template: str | None = None,
) -> list[str]:
    generic = build_generic_theory_query(theory_template)
    element = build_element_theory_query(material)
    material_query = build_material_theory_query(material)

    if strategy == "generic":
        ordered = [generic]
    elif strategy == "element":
        ordered = [element, generic]
    elif strategy == "material":
        ordered = [material_query, generic]
    else:
        # hybrid
        ordered = [generic, material_query]

    deduped: list[str] = []
    seen = set()
    for query in ordered:
        q = (query or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        deduped.append(q)

    return deduped[: max(1, max_queries)]


def _invoke_websearch(query: str) -> tuple[str, list[str], str | None]:
    if not query:
        return "", [], None

    if WebSearchTools is None:
        return "", [], "WebSearchTools is unavailable"

    try:
        tool = WebSearchTools()
    except Exception as exc:
        return "", [], f"WebSearchTools init failed: {exc}"

    result: Any = None
    call_error: str | None = None

    for method_name in ("web_search", "search_web", "search", "duckduckgo_search"):
        method = getattr(tool, method_name, None)
        if callable(method):
            try:
                result = method(query=query)
                call_error = None
                break
            except TypeError:
                try:
                    result = method(query)
                    call_error = None
                    break
                except Exception as exc:
                    call_error = str(exc)
            except Exception as exc:
                call_error = str(exc)

    if result is None:
        return "", [], call_error or "No callable websearch method found"

    summary = str(result)
    urls = sorted(set(re.findall(r"https?://[^\s\]\)\}\>\"']+", summary)))
    return summary[:4000], urls[:10], None


def _extract_body_texts(raw_summary: str, max_items: int = 12) -> list[str]:
    if not raw_summary:
        return []
    bodies: list[str] = []
    pattern = re.compile(r'"body"\s*:\s*"((?:[^"\\]|\\.)*)"')
    for match in pattern.finditer(raw_summary):
        token = match.group(1)
        try:
            text = json.loads(f"\"{token}\"")
        except Exception:
            text = token.encode("utf-8", "ignore").decode("unicode_escape", "ignore")
        normalized = re.sub(r"\s+", " ", str(text)).strip()
        if normalized:
            bodies.append(normalized)
        if len(bodies) >= max_items:
            break
    return bodies


def _normalize_search_snippet(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    normalized = normalized.replace("\\n", " ").replace("\u00a0", " ")
    return normalized.strip(" -.;,")


def _score_mechanism_snippet(text: str) -> int:
    lowered = text.lower()
    score = 0
    keywords = {
        "anharm": 3,
        "lone pair": 3,
        "rattling": 3,
        "resonant": 2,
        "localized": 2,
        "mass disorder": 3,
        "disorder": 2,
        "distortion": 2,
        "group veloc": 2,
        "flatten": 2,
        "mean free path": 2,
        "boundary scattering": 2,
        "phonon scattering": 2,
        "thermal conductivity": 1,
        "defect": 1,
    }
    for token, weight in keywords.items():
        if token in lowered:
            score += weight
    if re.search(r"\b(19|20)\d{2}\b", text):
        score += 1
    return score


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _select_mechanism_evidence(bodies: list[str], max_points: int = 8) -> list[str]:
    scored: list[tuple[int, int, str]] = []
    for idx, body in enumerate(bodies):
        snippet = _normalize_search_snippet(body)
        if len(snippet) < 30:
            continue
        scored.append((_score_mechanism_snippet(snippet), idx, snippet[:280]))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [snippet for _, _, snippet in scored[: max_points * 2]]
    return _dedupe_preserve_order(selected)[:max_points]


def _condense_body_texts(raw_summary: str, max_points: int = 8) -> str:
    bodies = _extract_body_texts(raw_summary)
    if not bodies:
        return ""

    selected = _select_mechanism_evidence(bodies, max_points=max_points)
    points = [f"- {snippet}" for snippet in selected]
    return "\n".join(points)


def build_aggregated_websearch_query(
    candidates: list[dict[str, Any]],
    top_n: int = 5,
    theory_template: str | None = None,
) -> str:
    effective = candidates[: max(1, top_n)]
    formulas = [f for f in (_extract_formula(item) for item in effective) if f]
    elements: set[str] = set()
    for formula in formulas:
        for elem in _extract_elements_from_formula(formula):
            elements.add(elem)
    formula_part = ", ".join(formulas[:8])
    element_part = " ".join(sorted(elements)[:10])
    generic = build_generic_theory_query(theory_template)
    return (
        f"{generic}; candidate formulas: {formula_part}; "
        f"elements: {element_part}; identify common low thermal conductivity mechanisms"
    ).strip()


def enrich_topn_with_websearch(
    candidates: list[dict[str, Any]],
    top_n: int = 5,
    enabled: bool = True,
    theory_query: str | None = None,
    strategy: str = "hybrid",
    queries_per_candidate: int = 2,
    theory_template: str | None = None,
) -> list[dict[str, Any]]:
    if theory_query:
        # Backward-compatible override maps to generic custom template.
        theory_template = theory_query

    if not enabled or top_n <= 0:
        enriched = [dict(item) for item in candidates]
        for item in enriched:
            item.setdefault("websearch_queries", [])
            item.setdefault("websearch_summary", "")
            item.setdefault("websearch_sources", [])
            item.setdefault("websearch_errors", [])
            item.setdefault("websearch_success_count", 0)
        return enriched

    enriched = [dict(item) for item in candidates]
    effective_top_n = min(len(enriched), max(1, top_n))
    aggregate_query = build_aggregated_websearch_query(
        candidates=enriched,
        top_n=effective_top_n,
        theory_template=theory_template,
    )
    print(f"[websearch] aggregate mode, top_n={effective_top_n}, query={aggregate_query}")
    raw_summary, sources, err = _invoke_websearch(aggregate_query)
    condensed_summary = _condense_body_texts(raw_summary)
    success_count = 0 if err else 1
    errors = [err] if err else []

    for idx, item in enumerate(enriched):
        if idx == 0 and idx < top_n:
            item["websearch_queries"] = [aggregate_query]
            item["websearch_summary"] = condensed_summary[:3000]
            item["websearch_sources"] = sorted(set(sources))[:10]
            item["websearch_errors"] = errors
            item["websearch_success_count"] = success_count
            item["websearch_mode"] = "aggregated_single_query"
            if raw_summary and not condensed_summary:
                item["websearch_summary"] = raw_summary[:1000]
        else:
            item["websearch_queries"] = []
            item["websearch_summary"] = ""
            item["websearch_sources"] = []
            item["websearch_errors"] = []
            item["websearch_success_count"] = 0
            item["websearch_mode"] = "aggregated_single_query"

    return enriched


def rank_by_ei(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda x: -float(x.get("ei", x.get("score", 0))))


_screening_tools: list[Any] = [
    query_materials_project,
    query_oqmd,
    query_aflow,
]
if WebSearchTools is not None:
    _screening_tools.append(WebSearchTools())
_screening_tools.append(step_ai_evaluation)

screening_agent = Agent(
    id="aslk-screening-agent",
    name="ASLK Screening Agent",
    role="Deduplicate known materials and screen novel candidates with theory-guided AI evaluation.",
    instructions=[
        "Query MP/OQMD/AFLOW and filter out known formulas by exact reduced-formula match.",
        "For remaining novel candidates, enrich only top-N with web search evidence.",
        "Use theory document + web evidence for final AI screening.",
    ],
    tools=_screening_tools,
)

__all__ = [
    "screening_agent",
    "normalize_formula_reduced",
    "is_exact_formula_hit",
    "deduplicate_candidates",
    "build_generic_theory_query",
    "build_element_theory_query",
    "build_material_theory_query",
    "build_websearch_queries",
    "enrich_topn_with_websearch",
    "rank_by_ei",
]
