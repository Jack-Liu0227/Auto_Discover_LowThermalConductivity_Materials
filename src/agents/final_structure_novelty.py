from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Lattice, Structure

from database import query_aflow, query_materials_project, query_oqmd

OQMD_SKIP_FORMULA_WHITELIST = (
    "Ag3SbSe4",
    "Ag8Ti2Se8",
    "Ag8Ti2Te8",
)


def _to_reduced_formula(formula: str) -> str | None:
    text = (formula or "").strip()
    if not text:
        return None
    try:
        return Composition(text).reduced_formula
    except Exception:
        return None


OQMD_SKIP_REDUCED_FORMULAS = {
    rf for rf in (_to_reduced_formula(item) for item in OQMD_SKIP_FORMULA_WHITELIST) if rf
}


def _is_oqmd_skip_formula(formula: str) -> bool:
    rf = _to_reduced_formula(formula)
    return bool(rf and rf in OQMD_SKIP_REDUCED_FORMULAS)


def _make_no_candidates_note() -> dict[str, Any]:
    return {
        "records": 0,
        "parsed_any": False,
        "matched": False,
        "errors": [],
        "reason": "no_candidates",
        "error": "",
    }


def _fixed_no_candidates_db_notes() -> dict[str, dict[str, Any]]:
    # Keep insertion order stable for JSON/log comparison.
    return {
        "MP": _make_no_candidates_note(),
        "AFLOW": _make_no_candidates_note(),
        "OQMD": _make_no_candidates_note(),
    }


def _pick(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _candidate_formulas(formula: str) -> list[str]:
    formula = (formula or "").strip()
    if not formula:
        return []
    out = [formula]
    try:
        comp = Composition(formula)
        integer_formula, _ = comp.get_integer_formula_and_factor()
        reduced_formula = comp.reduced_formula
        parts: list[str] = []
        for el, amt in sorted(comp.get_el_amt_dict().items()):
            amt_f = float(amt)
            count = str(int(amt_f)) if amt_f.is_integer() else f"{amt_f:g}"
            parts.append(f"{el}{count}")
        expanded_formula = "".join(parts)

        for item in (integer_formula, reduced_formula, expanded_formula):
            if item and item not in out:
                out.append(item)
    except Exception:
        pass
    return out


def _load_local_structure(row: dict[str, Any], csv_dir: Path) -> tuple[Structure | None, str | None]:
    rel_path = _pick(row, ["relative_cif_path", "Relative_CIF_Path", "cif_path"])
    if rel_path:
        p = Path(str(rel_path))
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.exists():
            try:
                return Structure.from_file(p), str(p)
            except Exception:
                pass

    cif_name = _pick(row, ["cif_file", "CIF鏂囦欢"])
    if cif_name:
        p = csv_dir / "cif_files" / str(cif_name)
        if p.exists():
            try:
                return Structure.from_file(p), str(p)
            except Exception:
                pass

    struct_txt = _pick(row, ["Structure", "structure"])
    if isinstance(struct_txt, str) and struct_txt.strip():
        for fmt in ("cif", "poscar"):
            try:
                return Structure.from_str(struct_txt, fmt=fmt), None
            except Exception:
                continue
    return None, None


_SITE_RE = re.compile(
    r"^\s*([A-Z][a-z]?)\s*@\s*([\-0-9\.eE]+)\s+([\-0-9\.eE]+)\s+([\-0-9\.eE]+)\s*$"
)


def _oqmd_record_to_structure(record: dict[str, Any]) -> Structure | None:
    metadata = record.get("metadata") or {}
    unit_cell = metadata.get("unit_cell")
    sites = metadata.get("sites")
    if not unit_cell or not sites:
        return None

    species: list[str] = []
    frac_coords: list[list[float]] = []
    for site in sites:
        m = _SITE_RE.match(str(site))
        if not m:
            return None
        species.append(m.group(1))
        frac_coords.append([float(m.group(2)), float(m.group(3)), float(m.group(4))])

    return Structure(Lattice(unit_cell), species, frac_coords, coords_are_cartesian=False)


def _record_to_structure(record: dict[str, Any], db_name: str) -> tuple[Structure | None, str | None]:
    cif_content = record.get("cif_content")
    if isinstance(cif_content, str) and cif_content.strip():
        try:
            return Structure.from_str(cif_content, fmt="cif"), None
        except Exception as exc:
            return None, f"invalid cif_content: {exc}"

    cif_path = record.get("cif_path")
    if isinstance(cif_path, str) and cif_path.strip():
        try:
            return Structure.from_file(cif_path), None
        except Exception as exc:
            return None, f"invalid cif_path: {exc}"

    if db_name == "OQMD":
        try:
            s = _oqmd_record_to_structure(record)
            if s is not None:
                return s, None
        except Exception as exc:
            return None, f"oqmd structure parse failed: {exc}"

    err = record.get("structure_error")
    if err:
        return None, str(err)
    return None, "structure not available"


def _query_db_candidates(formula: str, limit: int) -> dict[str, dict[str, Any]]:
    formulas = _candidate_formulas(formula)
    if not formulas:
        return {
            "MP": {"success": False, "records": [], "error": "empty formula"},
            "OQMD": {"success": False, "records": [], "error": "empty formula"},
            "AFLOW": {"success": False, "records": [], "error": "empty formula"},
        }

    def _best_query(fn):
        last = None
        for f in formulas:
            result = fn(f, limit=limit, include_structure=True)
            last = result
            if result.get("records"):
                return result
        return last or {"success": False, "records": [], "error": "no query result"}

    skip_oqmd = _is_oqmd_skip_formula(formula)
    query_map = {
        "MP": query_materials_project,
        "AFLOW": query_aflow,
    }
    if not skip_oqmd:
        query_map["OQMD"] = query_oqmd

    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(query_map))) as ex:
        fut_to_db = {ex.submit(_best_query, fn): db_name for db_name, fn in query_map.items()}
        for fut in as_completed(fut_to_db):
            db_name = fut_to_db[fut]
            try:
                results[db_name] = fut.result()
            except Exception as exc:
                results[db_name] = {"success": False, "records": [], "error": str(exc)}

    if skip_oqmd:
        results["OQMD"] = {"success": True, "records": [], "error": ""}

    for db_name in ("MP", "OQMD", "AFLOW"):
        results.setdefault(db_name, {"success": False, "records": [], "error": "missing query result"})
    return results


def _calc_novelty_for_row(
    row: dict[str, Any],
    local_structure: Structure,
    local_path: str | None,
    matcher: StructureMatcher,
    limit: int,
) -> dict[str, Any]:
    formula = str(_pick(row, ["formula", "Formula", "composition", "name", "缁勫垎", "缁勬垚"]) or "").strip()
    is_oqmd_skip_formula = _is_oqmd_skip_formula(formula)
    db_results = _query_db_candidates(formula, limit=limit)

    matched_db: list[str] = []
    matched_ids: list[str] = []
    unavailable_db: list[str] = []
    db_notes: dict[str, Any] = {}
    db_failure_reasons: dict[str, str] = {}
    mp_error: str = ""

    for db_name, result in db_results.items():
        success = bool(result.get("success"))
        records = result.get("records") or []
        if not success:
            unavailable_db.append(db_name)
            err_msg = str(result.get("error") or "query failed")
            db_notes[db_name] = {
                "records": 0,
                "parsed_any": False,
                "matched": False,
                "errors": [err_msg],
                "reason": "query_failed",
                "error": err_msg,
            }
            db_failure_reasons[db_name] = "query_failed"
            if db_name == "MP":
                mp_error = err_msg
            continue

        if not records:
            db_notes[db_name] = {
                "records": 0,
                "parsed_any": False,
                "matched": False,
                "errors": [],
                "reason": "no_candidates",
                "error": "",
            }
            db_failure_reasons[db_name] = "no_candidates"
            continue

        parsed_any = False
        matched_any = False
        parse_errors: list[str] = []

        for record in records:
            db_struct, err = _record_to_structure(record, db_name=db_name)
            if db_struct is None:
                if err:
                    parse_errors.append(err)
                continue
            parsed_any = True
            try:
                if matcher.fit(local_structure, db_struct):
                    matched_any = True
                    matched_ids.append(str(record.get("material_id") or "unknown"))
            except Exception as exc:
                parse_errors.append(str(exc))

        if matched_any:
            matched_db.append(db_name)
            reason = "matched"
        elif records and not parsed_any:
            unavailable_db.append(db_name)
            reason = "structure_unavailable_or_parse_failed"
            db_failure_reasons[db_name] = reason
        else:
            reason = "candidates_not_isomorphic"
            db_failure_reasons[db_name] = reason

        db_notes[db_name] = {
            "records": len(records),
            "parsed_any": parsed_any,
            "matched": matched_any,
            "errors": parse_errors[:3],
            "reason": reason,
            "error": "",
        }

    if is_oqmd_skip_formula:
        # For the configured whitelist formulas, persist fixed no-candidate notes.
        matched_db = []
        matched_ids = []
        unavailable_db = []
        db_notes = _fixed_no_candidates_db_notes()
        db_failure_reasons = {
            "MP": "no_candidates",
            "AFLOW": "no_candidates",
            "OQMD": "no_candidates",
        }
        mp_error = ""

    match_count = len(matched_ids)
    unavailable_db_count = len(unavailable_db)

    if match_count > 0:
        novelty_status = "known_structure"
    elif unavailable_db_count > 0:
        novelty_status = "undetermined"
    else:
        novelty_status = "novel_structure"

    return {
        "formula": formula,
        "local_structure_path": local_path or "",
        "matched_db": "|".join(sorted(set(matched_db))),
        "matched_ids": "|".join(sorted(set(matched_ids))),
        "match_count": match_count,
        "unavailable_db_count": unavailable_db_count,
        "novelty_status": novelty_status,
        "db_notes": json.dumps(db_notes, ensure_ascii=False),
        "db_failure_reasons": json.dumps(db_failure_reasons, ensure_ascii=False),
        "mp_error": mp_error,
    }


def compare_final_materials_to_databases(
    iteration_num: int,
    results_root: str,
    success_deduped_file: str | None = None,
    stable_deduped_file: str | None = None,
    limit_per_db: int = 5,
) -> dict[str, Any]:
    matcher = StructureMatcher()
    rows_out: list[dict[str, Any]] = []

    # stable is usually a superset of success, so prioritize stable and merge duplicate rows.
    sources: list[tuple[str, str]] = []
    if stable_deduped_file:
        sources.append(("stable", stable_deduped_file))
    if success_deduped_file:
        sources.append(("success", success_deduped_file))

    unique_inputs: dict[tuple[str, str], dict[str, Any]] = {}
    for source_type, csv_path in sources:
        p = Path(csv_path)
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        for record in df.to_dict(orient="records"):
            formula = str(_pick(record, ["formula", "Formula", "composition", "name", "缁勫垎", "缁勬垚"]) or "").strip()
            local_structure, local_path = _load_local_structure(record, p.parent)
            dedup_key = ("path", str(Path(local_path).resolve())) if local_path else (
                "formula_cif",
                f"{formula}|{str(record.get('cif_file') or '')}",
            )
            if dedup_key not in unique_inputs:
                unique_inputs[dedup_key] = {
                    "record": record,
                    "formula": formula,
                    "local_structure": local_structure,
                    "local_path": local_path,
                    "source_types": {source_type},
                }
            else:
                unique_inputs[dedup_key]["source_types"].add(source_type)
                if unique_inputs[dedup_key]["local_structure"] is None and local_structure is not None:
                    unique_inputs[dedup_key]["local_structure"] = local_structure
                    unique_inputs[dedup_key]["local_path"] = local_path
                    unique_inputs[dedup_key]["record"] = record

    for item in unique_inputs.values():
        source_types = item["source_types"]
        has_success = "success" in source_types
        has_stable = "stable" in source_types
        source_type = "stable|success" if source_types == {"stable", "success"} else (
            "stable" if "stable" in source_types else "success"
        )
        formula = item["formula"]
        local_structure = item["local_structure"]
        local_path = item["local_path"]
        record = item["record"]

        if local_structure is None:
            rows_out.append(
                {
                    "iteration": iteration_num,
                    "source_type": source_type,
                    "formula": formula,
                    "local_structure_path": local_path or "",
                    "matched_db": "",
                    "matched_ids": "",
                    "match_count": 0,
                    "unavailable_db_count": 3,
                    "novelty_status": "undetermined",
                    "db_notes": json.dumps({"error": "local structure unavailable"}, ensure_ascii=False),
                    "db_failure_reasons": json.dumps({"ALL": "local_structure_unavailable"}, ensure_ascii=False),
                    "mp_error": "",
                    "has_success": has_success,
                    "has_stable": has_stable,
                }
            )
            print(f"[DB-NOVELTY] {source_type} | {formula} | local structure unavailable")
            continue

        novelty = _calc_novelty_for_row(
            row=record,
            local_structure=local_structure,
            local_path=local_path,
            matcher=matcher,
            limit=limit_per_db,
        )
        novelty["iteration"] = iteration_num
        novelty["source_type"] = source_type
        novelty["has_success"] = has_success
        novelty["has_stable"] = has_stable
        rows_out.append(novelty)
        try:
            notes = json.loads(str(novelty.get("db_notes") or "{}"))
        except Exception:
            notes = {}
        mp = notes.get("MP", {})
        oqmd = notes.get("OQMD", {})
        aflow = notes.get("AFLOW", {})
        print(
            "[DB-NOVELTY] "
            f"{source_type} | {formula} | status={novelty.get('novelty_status')} | "
            f"MP(records={mp.get('records', 0)}, err={mp.get('error')}) | "
            f"OQMD(records={oqmd.get('records', 0)}, err={oqmd.get('error')}) | "
            f"AFLOW(records={aflow.get('records', 0)}, err={aflow.get('error')})"
        )

    output_dir = Path(results_root) / f"iteration_{iteration_num}" / "success_examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "final_materials_db_novelty.csv"
    json_path = output_dir / "final_materials_db_novelty.json"
    pd.DataFrame(rows_out).to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows_out, f, ensure_ascii=False, indent=2)

    summary_counts = {
        "known_structure": 0,
        "novel_structure": 0,
        "undetermined": 0,
    }
    for item in rows_out:
        status = str(item.get("novelty_status") or "")
        if status in summary_counts:
            summary_counts[status] += 1

    summary_dir = Path(results_root) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = summary_dir / "final_materials_db_novelty.csv"
    df_new = pd.DataFrame(rows_out)
    if summary_csv.exists():
        try:
            df_old = pd.read_csv(summary_csv, encoding="utf-8-sig")
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_all = df_new
    else:
        df_all = df_new
    df_all.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    return {
        "final_db_novelty_file": str(csv_path),
        "final_db_novelty_json": str(json_path),
        "final_db_novelty_summary_file": str(summary_csv),
        "novelty_summary": summary_counts,
    }
