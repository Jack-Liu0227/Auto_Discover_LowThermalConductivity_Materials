from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any

import pandas as pd
from pydantic import ValidationError

try:
    from pymatgen.core import Composition
except ImportError:  # pragma: no cover - runtime dependency in production
    Composition = None

from agents.screening_agent import enrich_topn_with_websearch, rank_by_ei
from schemas import WorkflowInput
from utils.bo_runtime import extract_initial_samples_from_result
from utils.bo_runtime import crystal_system_from_spacegroup, crystal_system_is_high_symmetry, filter_high_symmetry_parents
from utils.config_loader import get_high_symmetry_config
from agents.ai_client import AIClient
from utils.candidate_identity import add_candidate_ids, canonical_formula, select_diverse_candidates
from tools.success_extractor import enrich_space_group_number
from utils.config_loader import get_effective_thresholds
from utils.param_sheet import persist_param_values
from utils.theory_doc_context import build_websearch_theory_context
from utils.workflow_resume import (
    file_sha256,
    load_saved_ai_evaluation_result,
    load_saved_bayesian_result,
    load_saved_document_update_result,
    load_saved_extract_result,
    is_valid_structure_artifacts,
    _is_valid_merge_artifacts,
    reset_steps_from,
)
from workflow.agno_state import AGNO_SESSION_STATE_DEFAULT, AGNO_STATE_KEYS
from workflow.step_ai_evaluation import step_ai_evaluation
from workflow.step_bayesian_optimization import step_bayesian_optimization
from workflow.step_extract_materials import step_extract_materials
from workflow.step_merge_results import step_merge_results
from workflow.step_structure_calculation import step_structure_calculation
from workflow.step_train_model import step_train_model
from workflow.step_update_data_doc import step_update_data_and_doc

SCREENING_MODE = "llm_full_rerank"
CHEMISTRY_DIVERSITY_SCREENING_MODE = "chemistry_diverse_rerank"


def _load_high_symmetry_parents(config: dict[str, Any], iteration_num: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load prior stable examples and retain orthorhombic-or-higher parents."""
    symmetry_cfg = get_high_symmetry_config(config)
    if "high_symmetry_parent_enabled" in config:
        symmetry_cfg["enabled"] = config["high_symmetry_parent_enabled"]
    if "high_symmetry_min_crystal_system" in config:
        symmetry_cfg["min_crystal_system"] = config["high_symmetry_min_crystal_system"]
    if "high_symmetry_parent_scope" in config:
        symmetry_cfg["scope"] = config["high_symmetry_parent_scope"]
    if iteration_num < 2 or not bool(symmetry_cfg.get("enabled", True)):
        return [], {}
    root = Path(config["results_root"])
    rows: list[dict[str, Any]] = []
    for number in range(iteration_num - 1, 0, -1):
        directory = root / f"iteration_{number}" / "success_examples"
        for filename in ("stable_materials_deduped.csv", "stable_materials.csv", "success_materials.csv"):
            path = directory / filename
            if not path.exists():
                continue
            try:
                frame = pd.read_csv(path, encoding="utf-8-sig")
            except Exception:
                continue
            for _, row in frame.iterrows():
                item = row.to_dict()
                formula = _extract_formula(item)
                if not formula:
                    continue
                system = str(item.get("crystal_system") or "").strip().lower() or crystal_system_from_spacegroup(
                    item.get("space_group") or item.get("Space_Group"),
                    item.get("space_group_number") or item.get("Space Group Number"),
                )
                minimum = str(symmetry_cfg.get("min_crystal_system", "orthorhombic"))
                filter_status = "eligible" if crystal_system_is_high_symmetry(system, minimum) else "excluded"
                filter_reason = (
                    f"crystal system {system} is below minimum {minimum}"
                    if system and filter_status == "excluded"
                    else "crystal system unresolved"
                    if not system
                    else "meets configured minimum crystal system"
                )
                item.update({
                    "formula": formula,
                    "crystal_system": system or "",
                    "parent_crystal_system": system or "",
                    "parent_spacegroup": item.get("space_group") or item.get("Space_Group"),
                    "parent_symmetry_filter_status": filter_status,
                    "parent_symmetry_filter_reason": filter_reason,
                })
                rows.append(item)
            break
    unique = {}
    for row in rows:
        key = canonical_formula(row.get("formula")) or str(row["formula"]).strip()
        unique.setdefault(key, row)
    filtered, stats = filter_high_symmetry_parents(
        list(unique.values()),
        str(symmetry_cfg.get("min_crystal_system", "orthorhombic")),
    )
    stats["high_symmetry_parent_count"] = len(filtered)
    stats["excluded_monoclinic_parent_count"] = stats.get("monoclinic", 0)
    stats["excluded_triclinic_parent_count"] = stats.get("triclinic", 0)
    stats["unresolved_symmetry_parent_count"] = stats.get("unresolved", 0)
    return filtered, stats


def _validate_abch_formula(
    formula: str,
    groups: dict[str, Any],
    *,
    hard_constraints: dict[str, Any] | None = None,
    max_atoms: int | None = None,
) -> tuple[bool, str]:
    """Validate that a formula contains exactly one A, one B, and one Ch element."""
    if Composition is None:
        return False, "pymatgen is unavailable; cannot validate formula composition"
    try:
        composition = Composition(formula, strict=True)
        elements = {str(element) for element in composition.elements}
    except Exception as exc:
        return False, f"invalid chemical formula: {exc}"

    normalized_groups = {
        name: {str(element) for element in values}
        for name, values in groups.items()
        if name in {"A", "B", "Ch"}
    }
    missing_groups = {name for name in ("A", "B", "Ch") if name not in normalized_groups}
    if missing_groups:
        return False, f"missing configured groups: {sorted(missing_groups)}"
    if not elements:
        return False, "formula contains no elements"

    unexpected = elements - set().union(*normalized_groups.values())
    if unexpected:
        return False, f"elements outside A-B-Ch groups: {sorted(unexpected)}"

    memberships = {
        element: [name for name, allowed in normalized_groups.items() if element in allowed]
        for element in elements
    }
    ambiguous = sorted(element for element, matches in memberships.items() if len(matches) != 1)
    if ambiguous:
        return False, f"elements have ambiguous or missing group membership: {ambiguous}"

    occupied = {matches[0] for matches in memberships.values()}
    missing = {name for name in ("A", "B", "Ch") if name not in occupied}
    if missing:
        return False, f"formula is not A-B-Ch; missing groups: {sorted(missing)}"
    if len(elements) != 3:
        return False, f"formula must contain exactly three elements, found: {sorted(elements)}"
    amounts = composition.get_el_amt_dict()
    if any(abs(float(value) - round(float(value))) > 1e-8 or float(value) < 1 for value in amounts.values()):
        return False, "formula must use positive integer stoichiometric coefficients"
    if max_atoms is not None and sum(float(value) for value in amounts.values()) > int(max_atoms):
        return False, f"formula exceeds max_atoms={max_atoms}"
    if hard_constraints:
        bounds = dict(hard_constraints.get("stoichiometry", {}))
        for group_name, members in normalized_groups.items():
            group_total = sum(float(amounts.get(element, 0)) for element in members)
            bound = dict(bounds.get(group_name, {}))
            if bound and not (float(bound.get("min", 1)) <= group_total <= float(bound.get("max", 10))):
                return False, f"{group_name} stoichiometry is outside configured bounds"
    return True, "valid A-B-Ch formula"


def _formula_group_signature(formula: str, groups: dict[str, Any]) -> dict[str, tuple[str, int]] | None:
    if Composition is None:
        return None
    try:
        composition = Composition(formula, strict=True)
    except Exception:
        return None
    normalized_groups = {
        name: {str(element) for element in values}
        for name, values in groups.items()
        if name in {"A", "B", "Ch"}
    }
    signature: dict[str, tuple[str, int]] = {}
    for element, amount in composition.get_el_amt_dict().items():
        matches = [name for name, allowed in normalized_groups.items() if str(element) in allowed]
        if len(matches) != 1 or not math.isclose(float(amount), round(float(amount)), abs_tol=1e-8):
            return None
        signature[matches[0]] = (str(element), int(round(float(amount))))
    if set(signature) != {"A", "B", "Ch"}:
        return None
    return signature


def _validate_parent_substitution(
    formula: str,
    parent_formula: str,
    parent_formulas: set[str],
    *,
    groups: dict[str, Any] | None = None,
    mode: str | None = None,
) -> tuple[bool, str]:
    """Validate a parent-bound proposal and its declared substitution mode."""
    if not parent_formula or canonical_formula(parent_formula) not in {canonical_formula(item) for item in parent_formulas}:
        return False, "parent_formula is not one of the supplied parent structures"
    if Composition is None:
        return False, "pymatgen is unavailable; cannot validate parent stoichiometry"
    try:
        parent = Composition(parent_formula)
        proposal = Composition(formula)
    except Exception as exc:
        return False, f"cannot parse parent/proposal composition: {exc}"
    if groups:
        parent_signature = _formula_group_signature(parent_formula, groups)
        proposal_signature = _formula_group_signature(formula, groups)
        if parent_signature is None or proposal_signature is None:
            return False, "cannot resolve parent/proposal A-B-Ch site signatures"
        if any(parent_signature[group][1] != proposal_signature[group][1] for group in ("A", "B", "Ch")):
            return False, "proposal does not preserve parent stoichiometric coefficients by site"
    else:
        parent_amounts = sorted(round(float(value), 8) for value in parent.get_el_amt_dict().values())
        proposal_amounts = sorted(round(float(value), 8) for value in proposal.get_el_amt_dict().values())
        if parent_amounts != proposal_amounts:
            return False, "proposal does not preserve parent stoichiometric coefficients"
    if canonical_formula(formula) == canonical_formula(parent_formula):
        return False, "proposal is identical to the parent formula"
    if groups:
        parent_signature = _formula_group_signature(parent_formula, groups)
        proposal_signature = _formula_group_signature(formula, groups)
        if parent_signature is None or proposal_signature is None:
            return False, "cannot resolve parent/proposal A-B-Ch site signatures"
        changed_sites = sum(
            parent_signature[group][0] != proposal_signature[group][0]
            for group in ("A", "B", "Ch")
        )
        if mode == "single_site_substitution" and changed_sites != 1:
            return False, "single-site mode must change exactly one A, B, or Ch element"
        if mode == "multi_site_substitution" and changed_sites < 2:
            return False, "multi-site mode must change at least two A, B, or Ch elements"
        if changed_sites < 1:
            return False, "substitution must change at least one site element"
    return True, "valid parent-preserving substitution"


def _validate_parent_stoichiometry_variant(
    formula: str,
    parent_formula: str,
    parent_formulas: set[str],
    groups: dict[str, Any],
) -> tuple[bool, str]:
    """Validate a same-element parent template with changed integer amounts."""
    if canonical_formula(parent_formula) not in {canonical_formula(item) for item in parent_formulas}:
        return False, "parent_formula is not one of the supplied parent structures"
    parent_signature = _formula_group_signature(parent_formula, groups)
    proposal_signature = _formula_group_signature(formula, groups)
    if parent_signature is None or proposal_signature is None:
        return False, "cannot resolve parent/proposal A-B-Ch site signatures"
    if tuple(parent_signature[group][0] for group in ("A", "B", "Ch")) != tuple(
        proposal_signature[group][0] for group in ("A", "B", "Ch")
    ):
        return False, "stoichiometry variant must preserve the parent's element tuple"
    if tuple(parent_signature[group][1] for group in ("A", "B", "Ch")) == tuple(
        proposal_signature[group][1] for group in ("A", "B", "Ch")
    ):
        return False, "stoichiometry variant must change at least one integer coefficient"
    return True, "valid parent stoichiometry variant"


def _generate_llm_formula_proposals(parents: list[dict[str, Any]], config: dict[str, Any], iteration_num: int, existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    generation_cfg = dict(config.get("llm_formula_generation", {}))
    enabled = bool(config.get("llm_formula_generation_enabled", generation_cfg.get("enabled", False)))
    start_iteration = int(config.get("llm_formula_generation_start_iteration", generation_cfg.get("start_iteration", 2)))
    report_dir = Path(config["results_root"]) / f"iteration_{iteration_num}" / "reports"
    input_report = report_dir / "llm_formula_generation_input.md"
    output_report = report_dir / "llm_formula_generation_output.md"
    report_dir.mkdir(parents=True, exist_ok=True)

    if not parents or not enabled or iteration_num < start_iteration:
        reason = (
            "skipped: no eligible high-symmetry parents"
            if not parents
            else "skipped: LLM formula generation is disabled"
            if not enabled
            else f"skipped: iteration {iteration_num} is before start_iteration {start_iteration}"
        )
        input_report.write_text(
            f"# LLM Formula Generation Input\n\n- Iteration: `{iteration_num}`\n- Status: `{reason}`\n",
            encoding="utf-8",
        )
        output_report.write_text(
            f"# LLM Formula Generation Output\n\n- Iteration: `{iteration_num}`\n- Status: `{reason}`\n- Proposals: `[]`\n",
            encoding="utf-8",
        )
        return []

    count = max(1, int(config.get("llm_formula_proposal_count", generation_cfg.get("proposal_count", 20))))
    selected_parents = parents[:int(config.get("llm_formula_parent_count", generation_cfg.get("parent_count", 10)))]
    sampling_params = dict(config.get("sampling_params", {}))
    hard_constraints = dict(sampling_params.get("hard_constraints", {}))
    schema = dict(hard_constraints.get("schema", {}))
    groups = dict(schema.get("groups", {}))
    groups.setdefault("A", ["Ag", "Cu", "In", "Sn", "Pb"])
    groups.setdefault("B", ["As", "Sb", "Ge", "Bi", "Ti", "V"])
    groups.setdefault("Ch", ["S", "Se", "Te"])
    group_text = "A = [%s]; B = [%s]; Ch = [%s]" % (
        ", ".join(map(str, groups["A"])),
        ", ".join(map(str, groups["B"])),
        ", ".join(map(str, groups["Ch"])),
    )
    system_prompt = (
        "You are a materials-science formula design assistant. "
        "Return valid JSON only, with no Markdown or commentary. "
        "Design chemically plausible parent-preserving substitutions for stable, low-thermal-conductivity materials. "
        "Do not claim that stability or low thermal conductivity is proven without calculations."
    )
    prompt = (
        "Generate exactly %d candidate formulas under the JSON key `proposals`.\n"
        "The objective is to obtain materials that are likely to preserve the stability of the supplied high-symmetry "
        "parent structures while offering plausible mechanisms for low lattice thermal conductivity (κ_L).\n\n"
        "This is a constrained parent-structure substitution task, not free-form formula invention.\n\n"
        "A-B-Ch CHEMICAL SPACE:\n"
        "- A-site elements: %s\n"
        "- B-site elements: %s\n"
        "- Ch-site elements: %s\n"
        "- Every formula must contain exactly one A element, one B element, and one Ch element.\n"
        "- Only same-site substitutions are allowed: A→A, B→B, and Ch→Ch. Never exchange elements between groups.\n\n"
        "PARENT-PRESERVING SUBSTITUTION RULES:\n"
        "1. Select one supplied high-symmetry parent for every proposal.\n"
        "2. Set parent_formula to the exact supplied parent formula.\n"
        "3. Change element symbols only; preserve all stoichiometric coefficients and the total atom count exactly.\n"
        "4. Do not introduce vacancies, interstitials, fractional occupancies, extra elements, or new coefficients.\n"
        "5. Preserve the parent prototype and space-group family as the structural template.\n"
        "6. The formula must be different from its parent, but directly obtainable by the listed substitutions.\n"
        "7. Multiple same-site substitutions are allowed; list every replacement explicitly.\n\n"
        "STABILITY DESIGN OBJECTIVE:\n"
        "Prefer chemically compatible substitutions that preserve site roles, coordination, charge balance, bonding "
        "environment, and structural topology. Avoid extreme size mismatch and substitutions likely to destabilize the parent.\n\n"
        "LOW-κ_L DESIGN OBJECTIVE:\n"
        "Prefer plausible mechanisms such as heavy-atom mass contrast, mass disorder, softer bonding, lower vibrational "
        "frequencies, stronger anharmonicity, or lone-pair activity, but do not sacrifice chemical plausibility. "
        "Do not invent numerical formation energies, phonon frequencies, stability scores, or κ_L values.\n\n"
        "OUTPUT SCHEMA (keep every explanation short):\n"
        "Each item must contain: formula, parent_formula, substitution_rule, "
        "stability_rationale, low_kappa_rationale, main_risk, confidence.\n"
        "Use one concise sentence for each rationale field. confidence must be a number between 0 and 1. "
        "Rank proposals by the combined plausibility of stability and low κ_L.\n\n"
        "SUPPLIED HIGH-SYMMETRY PARENTS:\n%s"
    ) % (
        count,
        ", ".join(map(str, groups["A"])),
        ", ".join(map(str, groups["B"])),
        ", ".join(map(str, groups["Ch"])),
        json.dumps([
            {"formula": p["formula"], "crystal_system": p.get("crystal_system"), "spacegroup": p.get("parent_spacegroup")} for p in selected_parents
        ], ensure_ascii=False),
    )
    input_report.write_text(
        "# LLM Formula Generation Input\n\n"
        f"- Iteration: `{iteration_num}`\n"
        f"- Requested proposals: `{count}`\n"
        f"- Parent structures: `{len(selected_parents)}`\n"
        f"- Max output tokens: `{int(config.get('llm_formula_generation_max_tokens', generation_cfg.get('max_tokens', 12000)))}`\n"
        f"- Max retries after the initial attempt: `{int(config.get('llm_formula_generation_max_retries', generation_cfg.get('max_retries', 3)))}`\n\n"
        "## System prompt\n\n"
        f"```text\n{system_prompt}\n```\n\n"
        "## User prompt\n\n"
        f"```text\n{prompt}\n```\n",
        encoding="utf-8",
    )

    max_tokens = int(config.get("llm_formula_generation_max_tokens", generation_cfg.get("max_tokens", 12000)))
    max_retries = max(0, int(config.get("llm_formula_generation_max_retries", generation_cfg.get("max_retries", 3))))
    max_attempts = max_retries + 1
    parent_formulas = {str(parent.get("formula", "")).strip() for parent in selected_parents}
    base_seen = {_extract_formula(item) for item in existing}
    attempts: list[dict[str, Any]] = []
    result: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        raw = ""
        payload: Any = None
        proposals: list[Any] = []
        rejected: list[dict[str, str]] = []
        error: Exception | None = None
        seen = set(base_seen)
        try:
            raw = AIClient().chat(prompt, system_prompt=system_prompt, max_tokens=max_tokens)
            payload = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            proposals = payload.get("proposals", []) if isinstance(payload, dict) else []
            if not isinstance(proposals, list):
                raise ValueError("JSON field 'proposals' must be a list")
        except Exception as exc:
            error = exc
            print(f"[llm-proposal] attempt {attempt}/{max_attempts} failed: {exc}")

        attempt_result: list[dict[str, Any]] = []
        if error is None:
            for item in proposals:
                if not isinstance(item, dict):
                    rejected.append({"formula": "", "reason": "proposal item is not an object"})
                    continue
                formula = str(item.get("formula") or "").strip()
                if not formula:
                    rejected.append({"formula": "", "reason": "missing formula"})
                    continue
                valid, reason = _validate_abch_formula(formula, groups)
                if not valid:
                    rejected.append({"formula": formula, "reason": reason})
                    continue
                parent_formula = str(item.get("parent_formula") or "").strip()
                valid, reason = _validate_parent_substitution(formula, parent_formula, parent_formulas)
                if not valid:
                    rejected.append({"formula": formula, "reason": reason})
                    continue
                if formula in seen:
                    rejected.append({"formula": formula, "reason": "duplicate of an existing candidate"})
                    continue
                seen.add(formula)
                item.update({"formula": formula, "candidate_source": "llm", "original_bo_rank": None})
                attempt_result.append(item)

        attempts.append({
            "attempt": attempt,
            "raw_response": raw,
            "parsed_payload": payload,
            "parsed_proposals": len(proposals),
            "accepted_proposals": attempt_result,
            "rejected_proposals": rejected,
            "error": str(error) if error else None,
        })
        if attempt_result:
            result = attempt_result
            break

    output_report.write_text(
        "# LLM Formula Generation Output\n\n"
        f"- Iteration: `{iteration_num}`\n"
        f"- Attempts: `{len(attempts)}/{max_attempts}`\n"
        f"- Final accepted proposal count: `{len(result)}`\n"
        f"- Status: `{'success' if result else 'failed'}`\n\n"
        "## Attempt audit trail\n\n"
        f"```json\n{json.dumps(attempts, ensure_ascii=False, indent=2, default=str)}\n```\n\n"
        "## Final accepted proposals\n\n"
        f"```json\n{json.dumps(result, ensure_ascii=False, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )
    if not result:
        reasons = [str(item.get("error") or f"accepted=0, rejected={len(item.get('rejected_proposals', []))}") for item in attempts]
        raise RuntimeError(
            f"LLM formula generation failed after {len(attempts)} attempt(s): " + " | ".join(reasons)
        )
    return result


def _generate_diverse_llm_formula_proposals(
    parents: list[dict[str, Any]], config: dict[str, Any], iteration_num: int,
    existing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate proposals across several chemistry-space modes.

    This is intentionally separate from the legacy parent-preserving generator
    so old artifacts remain readable while new runs can opt into diversity.
    """
    generation_cfg = dict(config.get("llm_formula_generation", {}))
    enabled = bool(config.get("llm_formula_generation_enabled", generation_cfg.get("enabled", False)))
    start_iteration = int(config.get("llm_formula_generation_start_iteration", generation_cfg.get("start_iteration", 2)))
    report_dir = Path(config["results_root"]) / f"iteration_{iteration_num}" / "reports"
    input_report = report_dir / "llm_formula_generation_input.md"
    output_report = report_dir / "llm_formula_generation_output.md"
    report_dir.mkdir(parents=True, exist_ok=True)
    if not enabled or iteration_num < start_iteration or not parents:
        reason = "disabled" if not enabled else "before_start_iteration" if iteration_num < start_iteration else "no_high_symmetry_parent"
        input_report.write_text(f"# LLM Formula Generation Input\n\n- Status: `{reason}`\n", encoding="utf-8")
        output_report.write_text(f"# LLM Formula Generation Output\n\n- Status: `{reason}`\n- Proposals: `[]`\n", encoding="utf-8")
        return []

    count = max(4, int(config.get("llm_formula_proposal_count", generation_cfg.get("proposal_count", 20))))
    parent_count = int(config.get("llm_formula_parent_count", generation_cfg.get("parent_count", 10)))
    selected_parents = parents[:parent_count]
    sampling = dict(config.get("sampling_params", {}))
    hard_constraints = dict(sampling.get("hard_constraints", {}))
    groups = dict(dict(hard_constraints.get("schema", {})).get("groups", {}))
    groups.setdefault("A", ["Ag", "Cu", "In", "Sn", "Pb"])
    groups.setdefault("B", ["As", "Sb", "Ge", "Bi", "Ti", "V"])
    groups.setdefault("Ch", ["S", "Se", "Te"])
    stoich = dict(hard_constraints.get("stoichiometry", {}))
    max_atoms = int(sampling.get("max_atoms", 20))
    modes = ("single_site_substitution", "multi_site_substitution", "stoichiometry_variant", "cross_parent_or_de_novo")
    configured_weights = dict(generation_cfg.get("generation_modes", {}))
    weights = {}
    for mode in modes:
        try:
            weights[mode] = max(0.0, float(configured_weights.get(mode, 0.0)))
        except (TypeError, ValueError):
            weights[mode] = 0.0
    if sum(weights.values()) <= 0:
        weights = {mode: 1.0 for mode in modes}
    total_weight = sum(weights.values())
    raw_quotas = {mode: count * weights[mode] / total_weight for mode in modes}
    quotas = {mode: int(math.floor(raw_quotas[mode])) for mode in modes}
    remaining_quota = count - sum(quotas.values())
    for mode in sorted(modes, key=lambda name: (-(raw_quotas[name] - quotas[name]), modes.index(name)))[:remaining_quota]:
        quotas[mode] += 1
    system_prompt = (
        "You are a materials-science formula design assistant. Return JSON only. "
        "Every proposal must be a three-element A-B-Ch composition and must include generation_mode. "
        "Do not claim calculated stability or thermal conductivity."
    )
    prompt = (
        f"Generate exactly {count} proposals under `proposals`, distributed across these modes: "
        f"{json.dumps(quotas)}. A={list(groups['A'])}; B={list(groups['B'])}; Ch={list(groups['Ch'])}. "
        f"Hard bounds={json.dumps(stoich)}; max_atoms={max_atoms}.\n"
        "single_site_substitution: one site element changes and parent coefficients are preserved.\n"
        "multi_site_substitution: at least two site elements change and parent coefficients are preserved.\n"
        "stoichiometry_variant: use a supplied parent as a template but change one or more integer coefficients within bounds.\n"
        "cross_parent_or_de_novo: combine elements from the supplied parents or explore a new A-B-Ch combination; parent_formula may be null.\n"
        "Never add a fourth element, vacancy, fractional occupancy, or unsupported coefficient. "
        "Use exact parent_formula values from the supplied list for parent-bound modes.\n\n"
        "Each item must have this shape: {formula, generation_mode, parent_formula, substitution_rule, "
        "prototype_or_spacegroup, reason, confidence}. generation_mode must be one of the four names above.\n\n"
        "Supplied parents:\n" + json.dumps([
            {"formula": p.get("formula"), "crystal_system": p.get("crystal_system"), "spacegroup": p.get("parent_spacegroup")}
            for p in selected_parents
        ], ensure_ascii=False)
    )
    existing_keys = {canonical_formula(_extract_formula(item)) for item in existing if canonical_formula(_extract_formula(item))}
    historical_keys: set[str] = set()
    # Do not regenerate formulas that have already reached evaluated history,
    # even when they are absent from the current BO pool.
    for previous_iteration in range(1, iteration_num):
        history_dir = Path(config["results_root"]) / f"iteration_{previous_iteration}" / "success_examples"
        for filename in ("success_materials.csv", "stable_materials.csv", "success_materials_deduped.csv", "stable_materials_deduped.csv"):
            history_path = history_dir / filename
            if not history_path.exists():
                continue
            try:
                history_frame = pd.read_csv(history_path, encoding="utf-8-sig")
            except (OSError, ValueError, pd.errors.ParserError):
                continue
            for _, history_row in history_frame.iterrows():
                history_formula = _extract_formula(history_row.to_dict())
                history_key = canonical_formula(history_formula)
                if history_key:
                    existing_keys.add(history_key)
                    historical_keys.add(history_key)
    current_coverage = sorted({_extract_formula(item) for item in existing if _extract_formula(item)})
    prompt += (
        "\n\nCurrent BO formula coverage (avoid duplicates and explore gaps):\n"
        + json.dumps(current_coverage[:100], ensure_ascii=False)
        + "\nPreviously evaluated success/stable formulas (do not repeat):\n"
        + json.dumps(sorted(historical_keys)[:100], ensure_ascii=False)
    )
    input_report.write_text(
        f"# LLM Formula Generation Input\n\n- Iteration: `{iteration_num}`\n- Requested: `{count}`\n- Quotas: `{json.dumps(quotas)}`\n\n"
        f"```text\n{system_prompt}\n\n{prompt}\n```\n", encoding="utf-8"
    )
    parent_keys = {canonical_formula(p.get("formula")): p for p in selected_parents}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    attempts: list[dict[str, Any]] = []
    mode_counts: Counter[str] = Counter()
    max_retries = max(0, int(config.get("llm_formula_generation_max_retries", generation_cfg.get("max_retries", 3))))
    for attempt in range(max_retries + 1):
        raw = ""
        try:
            raw = AIClient().chat(prompt, system_prompt=system_prompt, max_tokens=int(config.get("llm_formula_generation_max_tokens", generation_cfg.get("max_tokens", 12000))))
            payload = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            proposals = payload.get("proposals", []) if isinstance(payload, dict) else []
        except Exception as exc:
            attempts.append({"attempt": attempt + 1, "error": str(exc), "accepted": 0, "raw_response": raw})
            continue
        if not isinstance(proposals, list):
            attempts.append({"attempt": attempt + 1, "error": "proposals is not a list", "accepted": 0, "raw_response": raw})
            continue
        attempt_accepted: list[dict[str, Any]] = []
        seen = set(existing_keys) | {canonical_formula(item.get("formula")) for item in accepted}
        for item in proposals:
            if not isinstance(item, dict):
                rejected.append({"formula": "", "reason": "proposal item is not an object"}); continue
            formula = str(item.get("formula") or "").strip()
            mode = str(item.get("generation_mode") or "").strip()
            if mode not in modes:
                rejected.append({"formula": formula, "reason": "unknown generation_mode"}); continue
            if mode_counts[mode] >= quotas[mode]:
                rejected.append({"formula": formula, "reason": f"mode quota reached for {mode}"}); continue
            valid, reason = _validate_abch_formula(formula, groups, hard_constraints=hard_constraints, max_atoms=max_atoms)
            if not valid:
                rejected.append({"formula": formula, "reason": reason}); continue
            key = canonical_formula(formula)
            if not key or key in seen:
                rejected.append({"formula": formula, "reason": "duplicate candidate"}); continue
            parent_formula = str(item.get("parent_formula") or "").strip()
            if mode != "cross_parent_or_de_novo":
                if canonical_formula(parent_formula) not in parent_keys:
                    rejected.append({"formula": formula, "reason": "parent_formula is not supplied"}); continue
                if mode in {"single_site_substitution", "multi_site_substitution"}:
                    valid, reason = _validate_parent_substitution(
                        formula,
                        parent_formula,
                        {str(p.get("formula")) for p in selected_parents},
                        groups=groups,
                        mode=mode,
                    )
                    if not valid:
                        rejected.append({"formula": formula, "reason": reason}); continue
                elif mode == "stoichiometry_variant":
                    valid, reason = _validate_parent_stoichiometry_variant(
                        formula,
                        parent_formula,
                        {str(p.get("formula")) for p in selected_parents},
                        groups,
                    )
                    if not valid:
                        rejected.append({"formula": formula, "reason": reason}); continue
            seen.add(key)
            item.update({"formula": formula, "canonical_formula": key, "generation_mode": mode, "candidate_source": "llm", "original_bo_rank": None})
            attempt_accepted.append(item)
            mode_counts[mode] += 1
        accepted.extend(attempt_accepted)
        attempts.append({"attempt": attempt + 1, "parsed": len(proposals), "accepted": len(attempt_accepted), "raw_response": raw})
        if len(accepted) >= count and all(mode_counts[mode] >= quotas[mode] for mode in modes):
            break
        # Ask the next retry specifically for modes that are still below quota.
        deficits = {mode: quotas[mode] - mode_counts[mode] for mode in modes if mode_counts[mode] < quotas[mode]}
        if deficits:
            prompt += f"\nPrevious responses still need these mode quotas: {deficits}. Return only new formulas for those deficits."
    accepted = accepted[:count]
    output_report.write_text(
        "# LLM Formula Generation Output\n\n"
        f"- Accepted: `{len(accepted)}`\n- Rejected: `{len(rejected)}`\n"
        f"- Mode counts: `{json.dumps(dict(Counter(item.get('generation_mode') for item in accepted)))}`\n\n"
        f"```json\n{json.dumps({'attempts': attempts, 'accepted': accepted, 'rejected': rejected}, ensure_ascii=False, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )
    return accepted


def _enrich_llm_proposals_with_bo_features(
    proposals: list[dict[str, Any]],
    *,
    iteration_num: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach GPR/EI audit fields without exposing them to chemistry scoring.

    LLM formulas are evaluated by the same composition feature and acquisition
    implementation as BO candidates.  Failure to load a model or parse one
    formula is recorded per proposal and never blocks chemistry screening.
    """
    if not proposals:
        return proposals
    try:
        import numpy as np
        from generators.acquisition_ei import (
            calculate_acquisition,
            composition_to_features,
            get_f_min,
            load_model_and_scaler,
        )
        from pymatgen.core import Composition as PymatgenComposition

        project_root = Path(__file__).resolve().parents[2]
        path_config = config.get("path_config")
        if path_config is not None:
            model_path = path_config.get_model_file_path(iteration_num - 1)
            models_root = str(path_config.models_root)
            results_root = str(path_config.results_root)
        else:
            models_root = str(config.get("models_root", "llm/models/GPR"))
            model_path = project_root / models_root / f"iteration_{iteration_num - 1}" / "gpr_thermal_conductivity.joblib"
            results_root = str(config.get("results_root", "llm/results"))
        model, scaler, _, _ = load_model_and_scaler(str(model_path), str(project_root), models_root)
        xi = float(config.get("xi", 0.01))
        f_min = float(get_f_min(iteration_num, str(project_root), results_root))

        for proposal in proposals:
            formula = str(proposal.get("formula") or "").strip()
            try:
                composition = PymatgenComposition(formula, strict=True)
                composition_dict = {str(element): float(amount) for element, amount in composition.get_el_amt_dict().items()}
                features = np.asarray([composition_to_features(composition_dict)], dtype=float)
                scaled = scaler.transform(features)
                mu_log, sigma_log = model.predict(scaled, return_std=True)
                mu_value = float(mu_log[0])
                sigma_value = float(sigma_log[0])
                k_pred = float(np.exp(mu_value))
                ei = float(calculate_acquisition(np.asarray([mu_value]), np.asarray([sigma_value]), f_min, xi)[0])
                proposal.update({
                    "k_pred": k_pred,
                    "mu_log": mu_value,
                    "sigma_log": sigma_value,
                    "k_lower": float(np.exp(mu_value - 1.96 * sigma_value)),
                    "k_upper": float(np.exp(mu_value + 1.96 * sigma_value)),
                    "ei": ei,
                    "bo_prediction_status": "computed",
                })
            except Exception as exc:
                proposal.update({
                    "bo_prediction_status": "unavailable",
                    "bo_prediction_error": str(exc),
                })
    except Exception as exc:
        for proposal in proposals:
            proposal.update({
                "bo_prediction_status": "unavailable",
                "bo_prediction_error": str(exc),
            })
    return proposals


def _load_saved_llm_formula_artifact(
    artifact_path: Path,
    iteration_num: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]] | None:
    """Load a completed proposal artifact for resume without another LLM call."""
    if not artifact_path.exists():
        return None
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        artifact_iteration = int(payload.get("iteration", -1))
    except (TypeError, ValueError):
        return None
    if artifact_iteration != int(iteration_num):
        return None
    proposals = payload.get("proposals", [])
    parents = payload.get("parents", [])
    stats = payload.get("parent_stats", {})
    if not isinstance(proposals, list) or not all(isinstance(item, dict) for item in proposals):
        return None
    if not isinstance(parents, list) or not all(isinstance(item, dict) for item in parents):
        parents = []
    return proposals, parents, stats if isinstance(stats, dict) else {}


def _mark_step_started(tracker, iteration_num: int, step_key: str) -> None:
    marker = getattr(tracker, "mark_step_started", None) if tracker is not None else None
    if callable(marker):
        marker(iteration_num, step_key)


def run_train_step(iteration_num: int, config: dict[str, Any], tracker=None) -> dict[str, Any]:
    step_key = "train_model"
    if tracker and tracker.is_step_completed(iteration_num, step_key):
        return {"success": True, "skipped": True}

    _mark_step_started(tracker, iteration_num, step_key)
    result = step_train_model(
        iteration_num=iteration_num,
        data_root=config["data_root"],
        models_root=config["models_root"],
        path_config=config.get("path_config"),
    )
    if result.get("success") and tracker:
        tracker.mark_step_completed(
            iteration_num,
            step_key,
            metadata={
                "training_data": result.get("training_data"),
                "training_data_sha256": result.get("training_data_sha256"),
                "model_file": result.get("model_file"),
            },
        )
    return result


def run_bayesian_step(
    iteration_num: int,
    config: dict[str, Any],
    tracker=None,
    initial_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    step_key = "bayesian_optimization"
    if tracker and tracker.is_step_completed(iteration_num, step_key):
        cached = load_saved_bayesian_result(
            results_root=config["results_root"],
            iteration_num=iteration_num,
            n_top=int(config.get("top_k_bayes", 10)),
        )
        if cached is not None:
            print(f"[resume] loaded saved BO artifacts for iteration {iteration_num}: {cached.get('artifact_path')}")
            return cached
        print(f"[resume] BO artifacts missing for iteration {iteration_num}, resetting progress from {step_key}")
        reset_steps_from(tracker, iteration_num, step_key)

    _mark_step_started(tracker, iteration_num, step_key)
    result = step_bayesian_optimization(
        iteration_num=iteration_num,
        xi=config["xi"],
        n_samples=config["samples"],
        n_top=config["top_k_bayes"],
        initial_samples=initial_samples,
        seed=config.get("seed"),
        seed_stride=int(config.get("seed_stride", 1000)),
        path_config=config.get("path_config"),
        models_root=config["models_root"],
        results_root=config["results_root"],
        sampling_params=config.get("sampling_params"),
    )
    if result.get("success") and tracker:
        tracker.mark_step_completed(
            iteration_num,
            step_key,
            {
                "status": result.get("status", "completed"),
                "sampler_generated_count": result.get("sampler_generated_count"),
                "candidate_pool_count": result.get("candidate_pool_count"),
                "candidate_dedup_removed": result.get("candidate_dedup_removed"),
                "raw_sample_count": result.get("raw_sample_count"),
                "selected_count": result.get("selected_count", len(result.get("top_materials", []))),
            },
        )
    return result


def run_extract_step(iteration_num: int, config: dict[str, Any], tracker=None) -> dict[str, Any]:
    step_key = "success_extraction"
    if tracker and tracker.is_step_completed(iteration_num, step_key):
        cached = load_saved_extract_result(config["results_root"], iteration_num)
        if cached is not None:
            print(f"[resume] loaded saved extraction artifacts for iteration {iteration_num}")
            return cached
        round_data = tracker.progress.get(f"iteration_{iteration_num}", {})
        step_data = round_data.get(step_key, {}) if isinstance(round_data, dict) else {}
        metadata = step_data.get("metadata", {}) if isinstance(step_data, dict) else {}
        if isinstance(metadata, dict) and metadata.get("no_materials") is True:
            print(f"[resume] loaded no-material extraction state for iteration {iteration_num}")
            return {
                "success": False,
                "no_materials": True,
                "has_success": False,
                "has_stable": False,
            }
        print(f"[resume] extraction artifacts missing for iteration {iteration_num}, resetting progress from {step_key}")
        reset_steps_from(tracker, iteration_num, step_key)

    _mark_step_started(tracker, iteration_num, step_key)
    thresholds = get_effective_thresholds()
    raw_k_threshold = config.get("k_threshold")
    raw_imag_tol = config.get("phonon_imag_tol")
    k_threshold = float(
        thresholds["thermal_conductivity"] if raw_k_threshold is None else raw_k_threshold
    )
    imag_tol = float(
        thresholds["dynamic_min_frequency"] if raw_imag_tol is None else raw_imag_tol
    )

    result = step_extract_materials(
        iteration_num=iteration_num,
        k_threshold=k_threshold,
        imag_tol=imag_tol,
        results_root=config["results_root"],
        path_config=config.get("path_config"),
        postprocess_workers=max(1, int(config.get("postprocess_workers", 1))),
        novelty_workers=max(1, int(config.get("novelty_workers", 1))),
    )
    if (result.get("success") or result.get("no_materials")) and tracker:
        tracker.mark_step_completed(
            iteration_num,
            step_key,
            metadata={
                "no_materials": bool(result.get("no_materials")),
                "has_success": bool(result.get("has_success")),
                "has_stable": bool(result.get("has_stable")),
            },
        )
    return result


def run_ai_evaluation_step(
    iteration_num: int,
    config: dict[str, Any],
    candidate_materials: list[dict[str, Any]],
    tracker=None,
) -> dict[str, Any]:
    step_key = "ai_evaluation"
    chemistry_diversity = bool(config.get("chemistry_diversity_enabled", False))
    if tracker and tracker.is_step_completed(iteration_num, step_key):
        cached = load_saved_ai_evaluation_result(config["results_root"], iteration_num)
        cache_mode = str(cached.get("screening_mode") or "") if isinstance(cached, dict) else ""
        cache_compatible = (not chemistry_diversity) or cache_mode == CHEMISTRY_DIVERSITY_SCREENING_MODE
        if cached is not None and cache_compatible:
            print(f"[resume] loaded saved AI screening artifacts for iteration {iteration_num}")
            return cached
        print(f"[resume] AI screening artifacts missing for iteration {iteration_num}, resetting progress from {step_key}")
        reset_steps_from(tracker, iteration_num, step_key)

    _mark_step_started(tracker, iteration_num, step_key)
    evaluation_candidates = add_candidate_ids(candidate_materials) if chemistry_diversity else list(candidate_materials)
    evaluation_mode = "candidate_scores" if chemistry_diversity else "selected_materials"
    ai_result = step_ai_evaluation(
        iteration_num=iteration_num,
        candidate_materials=evaluation_candidates,
        n_select=config["top_k_screen"],
        evaluation_mode=evaluation_mode,
        path_config=config.get("path_config"),
        results_root=config["results_root"],
        doc_root=config.get("doc_root", "llm/doc"),
        init_doc_path=config.get("init_doc_path"),
        candidate_identity_mode=chemistry_diversity,
        evaluator_evidence=str(config.get("chemistry_evaluator_evidence", "chemistry_only")) if chemistry_diversity else "full",
    )
    if chemistry_diversity:
        # A malformed LLM score response must not abort structure discovery.
        result = _build_chemistry_diverse_selection(
            iteration_num=iteration_num,
            results_root=config["results_root"],
            candidate_materials=evaluation_candidates,
            ai_result=ai_result if ai_result.get("success") else None,
            n_select=config["top_k_screen"],
            max_same_element_tuple=int(config.get("max_same_element_tuple", 2)),
            max_same_stoichiometry_pattern=int(config.get("max_same_stoichiometry_pattern", 4)),
        )
        if not result.get("success"):
            return {**result, "error": ai_result.get("error") if isinstance(ai_result, dict) else "no candidates"}
    else:
        if not ai_result.get("success"):
            # Preserve workflow progress if the legacy formula/rank evaluator
            # still returns a malformed response.  The fallback is a stable
            # BO rank cutoff over canonical-deduplicated candidates.
            fallback_candidates = add_candidate_ids(candidate_materials)

            def _fallback_rank(item: dict[str, Any]) -> float:
                try:
                    value = float(item.get("rank"))
                    return value if math.isfinite(value) else float("inf")
                except (TypeError, ValueError, OverflowError):
                    return float("inf")

            def _fallback_ei(item: dict[str, Any]) -> float:
                try:
                    value = float(item.get("ei", item.get("score")))
                    return value if math.isfinite(value) else float("-inf")
                except (TypeError, ValueError, OverflowError):
                    return float("-inf")

            fallback_candidates = sorted(
                fallback_candidates,
                key=lambda item: (
                    _fallback_rank(item),
                    -_fallback_ei(item),
                    str(item.get("candidate_id", "")),
                ),
            )
            result = _build_rank_cutoff_selection(
                iteration_num=iteration_num,
                results_root=config["results_root"],
                candidate_materials=fallback_candidates,
                n_select=config["top_k_screen"],
                screening_mode="bo_rank_fallback_after_ai_error",
            )
            result["error"] = ai_result.get("error")
        else:
            result = _build_full_rerank_selection(
                iteration_num=iteration_num,
                results_root=config["results_root"],
                candidate_materials=candidate_materials,
                ai_result=ai_result,
                screening_mode=SCREENING_MODE,
            )

    if result.get("success") and tracker:
        tracker.mark_step_completed(
            iteration_num,
            step_key,
            metadata={
                "selected_materials": len(result.get("selected_materials", [])),
                "report_file": result.get("report_path"),
                "selection_trace": result.get("trace_path"),
                "screening_mode": result.get("screening_mode"),
            },
        )
    return result


def run_structure_step(
    iteration_num: int,
    config: dict[str, Any],
    materials: list[dict[str, Any]],
    tracker=None,
) -> dict[str, Any]:
    step_key = "structure_calculation"
    if not materials:
        return {
            "success": False,
            "completed": False,
            "error": "No candidate materials available for structure calculation",
        }
    if tracker and tracker.is_step_completed(iteration_num, step_key):
        path_config = config.get("path_config")
        artifact_root = path_config.results_root if path_config is not None else Path(config["results_root"])
        iteration_root = artifact_root / f"iteration_{iteration_num}"
        processed_dir = iteration_root / "processed_structures"
        relax_dir = iteration_root / "MyRelaxStructure"
        expected_formulas = [str(material.get("formula") or "").strip() for material in materials]
        if is_valid_structure_artifacts(
            artifact_root,
            iteration_num,
            expected_formulas=expected_formulas,
        ):
            print(f"[resume] loaded validated structure artifacts for iteration {iteration_num}")
            return {
                "success": True,
                "skipped": True,
                "completed": True,
                "gen_output_dir": str(processed_dir),
                "relax_output_dir": str(relax_dir),
            }
        print(f"[resume] structure artifacts missing or invalid for iteration {iteration_num}, resetting progress from {step_key}")
        reset_steps_from(tracker, iteration_num, step_key)

    _mark_step_started(tracker, iteration_num, step_key)
    result = step_structure_calculation(
        iteration_num=iteration_num,
        materials=materials,
        n_structures=config["n_structures"],
        max_workers=config["max_workers"],
        relax_workers=config["relax_workers"],
        phonon_workers=config["phonon_workers"],
        postprocess_workers=max(1, int(config.get("postprocess_workers", 1))),
        pressure=config["pressure"],
        device=config["device"],
        gpus=config["gpus"],
        results_root=config["results_root"],
        seed=config.get("seed"),
        tracker=tracker,
        allow_partial_completion=config.get("allow_partial_structure", False),
        path_config=config.get("path_config"),
        relax_timeout_sec=config.get("relax_timeout_sec", 900),
        prefer_isolated_relax_process=config.get("prefer_isolated_relax_process", True),
        allow_in_process_relax_fallback=config.get("allow_in_process_relax_fallback", True),
    )
    if result.get("completed") and tracker:
        path_config = config.get("path_config")
        results_root = (
            path_config.results_root
            if path_config is not None
            else Path(config["results_root"])
        )
        expected_formulas = [
            str(material.get("formula") or "").strip()
            for material in materials
        ]
        if is_valid_structure_artifacts(
            results_root,
            iteration_num,
            expected_formulas=expected_formulas,
        ):
            tracker.mark_step_completed(
                iteration_num,
                step_key,
                metadata={
                    "gen_output_dir": result.get("gen_output_dir"),
                    "relax_output_dir": result.get("relax_output_dir"),
                },
            )
        else:
            result["success"] = False
            result["completed"] = False
            result["error"] = (
                "Structure completion contract failed after calculation; "
                "final artifacts are incomplete"
            )
    return result


def run_merge_step(iteration_num: int, config: dict[str, Any], tracker=None) -> dict[str, Any]:
    step_key = "merge_results"
    if tracker and tracker.is_step_completed(iteration_num, step_key):
        results_root = Path(config["results_root"])
        if _is_valid_merge_artifacts(results_root, iteration_num):
            print(f"[resume] loaded validated merge artifacts for iteration {iteration_num}")
            return {"success": True, "skipped": True}
        print(f"[resume] merge artifacts missing or invalid for iteration {iteration_num}, resetting progress from {step_key}")
        reset_steps_from(tracker, iteration_num, step_key)

    _mark_step_started(tracker, iteration_num, step_key)
    return step_merge_results(
        iteration_num=iteration_num,
        results_root=config["results_root"],
        tracker=tracker,
        path_config=config.get("path_config"),
        max_workers=max(1, int(config.get("postprocess_workers", 1))),
    )


def run_document_update_step(
    iteration_num: int,
    config: dict[str, Any],
    extraction_result: dict[str, Any],
    tracker=None,
) -> dict[str, Any]:
    step_key = "document_update"
    if tracker and tracker.is_step_completed(iteration_num, step_key):
        cached = load_saved_document_update_result(
            results_root=config["results_root"],
            data_root=config.get("data_root", "llm/data"),
            doc_root=config.get("doc_root", "llm/doc"),
            iteration_num=iteration_num,
        )
        if cached is not None:
            print(f"[resume] loaded saved document update artifacts for iteration {iteration_num}")
            return cached
        print(f"[resume] document update artifacts missing for iteration {iteration_num}, resetting progress from {step_key}")
        reset_steps_from(tracker, iteration_num, step_key)

    _mark_step_started(tracker, iteration_num, step_key)
    result = step_update_data_and_doc(
        iteration_num=iteration_num,
        extraction_result=extraction_result,
        version=config.get("version", 1),
        path_config=config.get("path_config"),
        data_root=config.get("data_root", "llm/data"),
        results_root=config.get("results_root", "llm/results"),
        doc_root=config.get("doc_root", "llm/doc"),
        skip_doc_update=config.get("skip_doc_update", False),
    )
    if result.get("success") and tracker:
        tracker.mark_step_completed(
            iteration_num,
            step_key,
            metadata={
                "updated_data_path": result.get("updated_data_path"),
                "updated_doc_path": result.get("updated_doc_path"),
                "data_sha256": (
                    file_sha256(result["updated_data_path"])
                    if result.get("updated_data_path")
                    and Path(result["updated_data_path"]).exists()
                    else None
                ),
            },
        )
    return result


def _selected_results_dir(results_root: str | Path, iteration_num: int) -> Path:
    path = Path(results_root) / f"iteration_{iteration_num}" / "selected_results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _formula_key(material: dict[str, Any]) -> str:
    value = str(material.get("formula") or material.get("composition") or material.get("name") or "").strip()
    return (
        value.replace("₀", "0")
        .replace("₁", "1")
        .replace("₂", "2")
        .replace("₃", "3")
        .replace("₄", "4")
        .replace("₅", "5")
        .replace("₆", "6")
        .replace("₇", "7")
        .replace("₈", "8")
        .replace("₉", "9")
        .replace(" ", "")
    )


def _candidate_rows(candidate_materials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, material in enumerate(candidate_materials, start=1):
        row = dict(material)
        row["formula"] = _formula_key(material)
        row["original_bo_rank"] = int(material.get("rank") or idx)
        row["k_pred"] = _safe_float(material.get("k_pred"))
        row["ei"] = _safe_float(material.get("ei", material.get("score")))
        row["sigma_log"] = _safe_float(material.get("sigma_log"))
        rows.append(row)
    return rows


def _lookup_candidates(candidate_materials: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        row["formula"]: row
        for row in _candidate_rows(candidate_materials)
        if row.get("formula")
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _persist_selection_outputs(
    *,
    iteration_num: int,
    results_root: str,
    screening_mode: str,
    selected_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    report_path: str | None = None,
    selection_audit: dict[str, Any] | None = None,
) -> dict[str, str]:
    base_dir = _selected_results_dir(results_root, iteration_num)
    selected_csv_path = base_dir / "ai_selected_materials.csv"
    trace_csv_path = base_dir / "selection_trace.csv"
    trace_json_path = base_dir / "selection_trace.json"

    _write_csv(selected_csv_path, selected_rows)
    _write_csv(trace_csv_path, trace_rows)
    _write_json(
        trace_json_path,
        {
            "iteration": iteration_num,
            "screening_mode": screening_mode,
            "selected_count": len(selected_rows),
            "selected_formulas": [row.get("formula") for row in selected_rows],
            "report_path": report_path,
            "selection_audit": selection_audit or {},
            "rows": trace_rows,
        },
    )
    return {
        "selected_csv": str(selected_csv_path),
        "trace_csv": str(trace_csv_path),
        "trace_json": str(trace_json_path),
    }


def _build_rank_cutoff_selection(
    *,
    iteration_num: int,
    results_root: str,
    candidate_materials: list[dict[str, Any]],
    n_select: int,
    screening_mode: str,
) -> dict[str, Any]:
    candidate_rows = _candidate_rows(candidate_materials)
    selected_formulas: set[str] = set()
    selected_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(candidate_rows, start=1):
        selected = idx <= n_select
        trace_row = {
            "formula": row["formula"],
            "original_bo_rank": row["original_bo_rank"],
            "k_pred": row["k_pred"],
            "ei": row["ei"],
            "sigma_log": row["sigma_log"],
            "screening_mode": screening_mode,
            "selected_for_calc": selected,
            "final_rank": idx if selected else "",
            "selection_reason": "bo_rank_cutoff" if selected else "",
        }
        trace_rows.append(trace_row)
        if selected:
            selected_formulas.add(row["formula"])
            selected_rows.append({**row, **trace_row})

    artifact_paths = _persist_selection_outputs(
        iteration_num=iteration_num,
        results_root=results_root,
        screening_mode=screening_mode,
        selected_rows=selected_rows,
        trace_rows=trace_rows,
    )
    return {
        "success": True,
        "n_selected": len(selected_rows),
        "selected_materials": selected_rows,
        "csv_path": artifact_paths["selected_csv"],
        "trace_path": artifact_paths["trace_json"],
        "trace_csv_path": artifact_paths["trace_csv"],
        "selection_trace": trace_rows,
        "screening_mode": screening_mode,
    }


def _build_full_rerank_selection(
    *,
    iteration_num: int,
    results_root: str,
    candidate_materials: list[dict[str, Any]],
    ai_result: dict[str, Any],
    screening_mode: str,
) -> dict[str, Any]:
    candidate_lookup = _lookup_candidates(candidate_materials)
    selected_rows: list[dict[str, Any]] = []
    selected_lookup: dict[str, dict[str, Any]] = {}

    for item in ai_result.get("selected_materials", []):
        formula = _formula_key(item)
        candidate = candidate_lookup.get(formula)
        if not formula or candidate is None or formula in selected_lookup:
            continue
        row = {
            **candidate,
            "formula": formula,
            "original_bo_rank": int(candidate.get("original_bo_rank", item.get("original_rank") or 0)),
            "screening_mode": screening_mode,
            "selected_for_calc": True,
            "final_rank": int(item.get("final_rank") or len(selected_rows) + 1),
            "ranking_reason": str(item.get("ranking_reason", "")).strip(),
            "main_risk": str(item.get("main_risk", "")).strip(),
        }
        selected_rows.append(row)
        selected_lookup[formula] = row

    selected_rows = sorted(selected_rows, key=lambda row: (int(row.get("final_rank", 10**9)), row.get("formula", "")))
    for idx, row in enumerate(selected_rows, start=1):
        row["final_rank"] = idx

    trace_rows: list[dict[str, Any]] = []
    for row in _candidate_rows(candidate_materials):
        selected = selected_lookup.get(row["formula"])
        trace_rows.append(
            {
                "formula": row["formula"],
                "original_bo_rank": row["original_bo_rank"],
                "k_pred": row["k_pred"],
                "ei": row["ei"],
                "sigma_log": row["sigma_log"],
                "screening_mode": screening_mode,
                "selected_for_calc": bool(selected),
                "final_rank": selected.get("final_rank") if selected else "",
                "ranking_reason": selected.get("ranking_reason", "") if selected else "",
                "main_risk": selected.get("main_risk", "") if selected else "",
            }
        )

    artifact_paths = _persist_selection_outputs(
        iteration_num=iteration_num,
        results_root=results_root,
        screening_mode=screening_mode,
        selected_rows=selected_rows,
        trace_rows=trace_rows,
        report_path=ai_result.get("report_path"),
    )
    return {
        "success": True,
        "n_selected": len(selected_rows),
        "selected_materials": selected_rows,
        "csv_path": artifact_paths["selected_csv"],
        "trace_path": artifact_paths["trace_json"],
        "trace_csv_path": artifact_paths["trace_csv"],
        "selection_trace": trace_rows,
        "report_path": ai_result.get("report_path"),
        "raw_ai_result": ai_result,
        "screening_mode": screening_mode,
    }


def _build_chemistry_diverse_selection(
    *,
    iteration_num: int,
    results_root: str,
    candidate_materials: list[dict[str, Any]],
    ai_result: dict[str, Any] | None,
    n_select: int,
    max_same_element_tuple: int = 2,
    max_same_stoichiometry_pattern: int = 4,
) -> dict[str, Any]:
    """Select scored candidates with source-independent chemistry caps."""
    rows = add_candidate_ids(candidate_materials)
    scores = (ai_result or {}).get("candidate_scores", []) if ai_result else []
    selected, audit = select_diverse_candidates(
        rows,
        scores,
        n_select,
        max_same_element_tuple=int(max_same_element_tuple),
        max_same_stoichiometry_pattern=int(max_same_stoichiometry_pattern),
    )
    score_by_id = {str(item.get("candidate_id")): item for item in scores if item.get("candidate_id")}
    score_by_formula = {canonical_formula(item.get("formula")): item for item in scores if item.get("formula")}
    selected_rows: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    for rank, row in enumerate(selected, 1):
        score_row = score_by_id.get(str(row.get("candidate_id"))) or score_by_formula.get(canonical_formula(row.get("formula")), {})
        key = canonical_formula(row.get("formula")) or str(row.get("formula", ""))
        if not key or key in selected_keys:
            continue
        selected_keys.add(key)
        selected_rows.append({
            **row,
            "final_rank": rank,
            "screening_mode": CHEMISTRY_DIVERSITY_SCREENING_MODE,
            "selected_for_calc": True,
            "chemical_score": score_row.get("chemical_score", ""),
            "chemical_plausibility_score": score_row.get("chemical_plausibility_score", ""),
            "low_kappa_mechanism_score": score_row.get("low_kappa_mechanism_score", ""),
            "stability_risk_score": score_row.get("stability_risk_score", ""),
            "ranking_reason": score_row.get("short_reason", "") or "deterministic chemistry fallback",
            "main_risk": score_row.get("main_risk", ""),
        })
    trace_rows: list[dict[str, Any]] = []
    selected_by_key = {canonical_formula(row.get("formula")): row for row in selected_rows}
    for row in rows:
        selected_row = selected_by_key.get(canonical_formula(row.get("formula")))
        score_row = score_by_id.get(str(row.get("candidate_id"))) or score_by_formula.get(canonical_formula(row.get("formula")), {})
        trace_rows.append({
            **{key: row.get(key) for key in ("candidate_id", "formula", "canonical_formula", "candidate_source", "parent_formula", "substitution_rule", "generation_mode", "original_bo_rank", "k_pred", "mu_log", "sigma_log", "ei")},
            "screening_mode": CHEMISTRY_DIVERSITY_SCREENING_MODE,
            "selected_for_calc": bool(selected_row),
            "final_rank": selected_row.get("final_rank", "") if selected_row else "",
            "chemical_score": score_row.get("chemical_score", ""),
            "chemical_plausibility_score": score_row.get("chemical_plausibility_score", ""),
            "low_kappa_mechanism_score": score_row.get("low_kappa_mechanism_score", ""),
            "stability_risk_score": score_row.get("stability_risk_score", ""),
            "short_reason": score_row.get("short_reason", "") if score_row else "",
            "main_risk": score_row.get("main_risk", "") if score_row else "",
            "selection_reason": "chemistry_score" if selected_row else "",
        })
    artifact_paths = _persist_selection_outputs(
        iteration_num=iteration_num,
        results_root=results_root,
        screening_mode=CHEMISTRY_DIVERSITY_SCREENING_MODE,
        selected_rows=selected_rows,
        trace_rows=trace_rows,
        report_path=(ai_result or {}).get("report_path"),
        selection_audit=audit,
    )
    return {
        "success": bool(selected_rows),
        "n_selected": len(selected_rows),
        "selected_materials": selected_rows,
        "csv_path": artifact_paths["selected_csv"],
        "trace_path": artifact_paths["trace_json"],
        "trace_csv_path": artifact_paths["trace_csv"],
        "selection_trace": trace_rows,
        "screening_mode": CHEMISTRY_DIVERSITY_SCREENING_MODE,
        "report_path": (ai_result or {}).get("report_path"),
        "diversity_audit": audit,
        "fallback": not bool(scores),
    }


def _load_formula_set(csv_path: str | None) -> set[str]:
    if not csv_path:
        return set()
    path = Path(csv_path)
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return set()
    formulas: set[str] = set()
    for column in ("formula", "Formula", "composition"):
        if column not in df.columns:
            continue
        formulas.update(str(value).strip() for value in df[column].tolist() if str(value).strip())
    return formulas


def _best_kappa_for_selected_success(extraction_result: dict[str, Any], selected_formulas: set[str]) -> float | None:
    success_path = extraction_result.get("success_deduped_file") or extraction_result.get("success_file")
    if not success_path or not Path(success_path).exists():
        return None
    try:
        df = pd.read_csv(success_path, encoding="utf-8-sig")
    except Exception:
        return None
    if "formula" not in df.columns:
        return None
    filtered = df[df["formula"].astype(str).isin(selected_formulas)]
    if filtered.empty:
        return None
    for column in ("thermal_conductivity_w_mk", "thermal_conductivity", "kappa", "k_pred"):
        if column in filtered.columns:
            series = pd.to_numeric(filtered[column], errors="coerce").dropna()
            if not series.empty:
                return float(series.min())
    return None


def _update_screening_summary(
    *,
    iteration_num: int,
    results_root: str,
    screen_result: dict[str, Any],
    extraction_result: dict[str, Any],
) -> str:
    summary_path = Path(results_root) / "screening_summary.csv"
    selected_rows = list(screen_result.get("selected_materials", []))
    selected_formulas = {str(row.get("formula") or "").strip() for row in selected_rows if str(row.get("formula") or "").strip()}
    success_formulas = _load_formula_set(extraction_result.get("success_deduped_file") or extraction_result.get("success_file"))
    stable_formulas = _load_formula_set(extraction_result.get("stable_deduped_file") or extraction_result.get("stable_file"))

    selected_count = len(selected_rows)
    success_count = sum(1 for row in selected_rows if str(row.get("formula") or "").strip() in success_formulas)
    stable_count = sum(1 for row in selected_rows if str(row.get("formula") or "").strip() in stable_formulas)
    def _row_bo_rank(row: dict[str, Any]) -> int:
        try:
            value = float(row.get("original_bo_rank"))
            return int(value) if math.isfinite(value) else 10**9
        except (TypeError, ValueError, OverflowError):
            return 10**9

    selected_from_bo_top3_count = sum(1 for row in selected_rows if _row_bo_rank(row) <= 3)
    selected_from_bo_13_20_count = sum(
        1 for row in selected_rows if 13 <= _row_bo_rank(row) <= 20
    )
    best_kappa = _best_kappa_for_selected_success(extraction_result, selected_formulas)
    tail_promotions_count = sum(
        1
        for row in selected_rows
        if row.get("selection_gate_status") == "tail_promotion_allowed"
    )
    tail_promotions_success_count = sum(
        1
        for row in selected_rows
        if row.get("selection_gate_status") == "tail_promotion_allowed"
        and str(row.get("formula") or "").strip() in success_formulas
    )
    protected_top3_dropped_count = sum(
        1
        for row in screen_result.get("selection_trace", [])
        if row.get("selection_gate_status") == "protected_top3" and not bool(row.get("selected_for_calc"))
    )
    trace_rows = list(screen_result.get("selection_trace", []))
    source_counts = Counter(str(row.get("candidate_source") or "unknown") for row in trace_rows)
    llm_selected_rows = [row for row in selected_rows if str(row.get("candidate_source") or "") in {"llm", "both"}]
    stable_keys = {canonical_formula(formula) for formula in stable_formulas}
    success_keys = {canonical_formula(formula) for formula in success_formulas}
    selected_p1_count = 0
    selected_low_symmetry_count = 0
    selected_high_symmetry_count = 0
    unresolved_symmetry_count = 0
    symmetry_rows = selected_rows
    final_structure_csv = extraction_result.get("stable_deduped_file") or extraction_result.get("stable_file") or extraction_result.get("success_deduped_file") or extraction_result.get("success_file")
    if final_structure_csv and Path(final_structure_csv).exists():
        try:
            final_frame = pd.read_csv(final_structure_csv, encoding="utf-8-sig")
            selected_keys = {canonical_formula(formula) for formula in selected_formulas}
            matched_rows = [
                row for row in final_frame.to_dict(orient="records")
                if canonical_formula(row.get("formula")) in selected_keys
                or canonical_formula(row.get("composition")) in selected_keys
            ]
            if matched_rows:
                symmetry_rows = matched_rows
        except (OSError, ValueError, pd.errors.ParserError):
            pass
    for row in symmetry_rows:
        system = str(row.get("crystal_system") or "").strip().lower() or crystal_system_from_spacegroup(
            row.get("space_group") or row.get("Space_Group"),
            row.get("space_group_number") or row.get("Space Group Number"),
        )
        if not system:
            unresolved_symmetry_count += 1
        elif system == "triclinic":
            try:
                if int(float(row.get("space_group_number"))) == 1:
                    selected_p1_count += 1
            except (TypeError, ValueError):
                if str(row.get("space_group") or row.get("Space_Group") or "").strip().upper().replace(" ", "") == "P1":
                    selected_p1_count += 1
            selected_low_symmetry_count += 1
        elif crystal_system_is_high_symmetry(system, "orthorhombic"):
            selected_high_symmetry_count += 1
        else:
            selected_low_symmetry_count += 1

    summary_row = {
        "iteration": iteration_num,
        "screening_mode": screen_result.get("screening_mode"),
        "selected_count": selected_count,
        "success_count": success_count,
        "stable_count": stable_count,
        "success_rate_at_k": (success_count / selected_count) if selected_count else 0.0,
        "stable_rate_at_k": (stable_count / selected_count) if selected_count else 0.0,
        "best_kappa_in_selected_success": best_kappa if best_kappa is not None else "",
        "selected_from_bo_top3_count": selected_from_bo_top3_count,
        "selected_from_bo_13_20_count": selected_from_bo_13_20_count,
        "tail_promotions_count": tail_promotions_count,
        "tail_promotions_success_count": tail_promotions_success_count,
        "protected_top3_dropped_count": protected_top3_dropped_count,
        "candidate_count": len(trace_rows),
        "bo_candidate_count": source_counts.get("bo", 0) + source_counts.get("both", 0),
        "llm_candidate_count": source_counts.get("llm", 0) + source_counts.get("both", 0),
        "llm_selected_count": len(llm_selected_rows),
        "llm_stable_count": sum(canonical_formula(row.get("formula")) in stable_keys for row in llm_selected_rows),
        "llm_success_count": sum(canonical_formula(row.get("formula")) in success_keys for row in llm_selected_rows),
        "llm_proposal_count": int(screen_result.get("llm_proposal_count", 0) or 0),
        "high_symmetry_parent_count": int(screen_result.get("high_symmetry_parent_count", 0) or 0),
        "high_symmetry_parent_stats": json.dumps(screen_result.get("high_symmetry_parent_stats", {}), ensure_ascii=False),
        "final_p1_count": selected_p1_count,
        "final_low_symmetry_count": selected_low_symmetry_count,
        "final_high_symmetry_count": selected_high_symmetry_count,
        "unresolved_symmetry_count": unresolved_symmetry_count,
        "final_structure_count_with_symmetry": len(symmetry_rows),
    }

    if summary_path.exists():
        existing = pd.read_csv(summary_path, encoding="utf-8-sig")
        if "iteration" in existing.columns:
            existing = existing[existing["iteration"] != iteration_num]
        updated = pd.concat([existing, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        updated = pd.DataFrame([summary_row])
    updated = updated.sort_values(by=["iteration"], kind="mergesort")
    updated.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return str(summary_path)


def _save_screening_artifacts(
    iteration_num: int,
    results_root: str,
    novel_pool: list[dict[str, Any]],
    websearch_enriched_candidates: list[dict[str, Any]],
) -> dict[str, str]:
    base_dir = Path(results_root) / f"iteration_{iteration_num}" / "selected_results"
    base_dir.mkdir(parents=True, exist_ok=True)

    def _to_csv(rows: list[dict[str, Any]], filename: str) -> str:
        path = base_dir / filename
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return str(path)

    return {
        "novel_pool_csv": _to_csv(novel_pool, "novel_candidates.csv"),
        "websearch_enriched_csv": _to_csv(websearch_enriched_candidates, "websearch_enriched_candidates.csv"),
        "merged_screening_candidates_csv": _to_csv(add_candidate_ids(websearch_enriched_candidates), "merged_screening_candidates.csv"),
    }


def _resolve_websearch_theory_template(iteration_num: int, config: dict[str, Any]) -> str | None:
    explicit_template = str(config.get("websearch_theory_template") or "").strip()
    if explicit_template:
        return explicit_template

    doc_path: Path | None = None
    path_config = config.get("path_config")
    if path_config:
        try:
            doc_path = path_config.get_theory_doc_path(iteration_num)
        except Exception:
            doc_path = None
    else:
        init_doc_path = str(config.get("init_doc_path") or "").strip()
        doc_root = Path(str(config.get("doc_root") or "llm/doc"))
        if iteration_num == 1 and init_doc_path:
            doc_path = Path(init_doc_path)
        elif iteration_num == 1:
            doc_path = doc_root / "v0.0.0" / "Theoretical_principle_document.md"
        else:
            doc_path = doc_root / f"v0.0.{iteration_num - 1}" / "Theoretical_principle_document.md"
        if doc_path and not doc_path.is_absolute():
            doc_path = Path.cwd() / doc_path

    if not doc_path or not doc_path.exists():
        return None

    try:
        doc_text = doc_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"[websearch] failed to read theory doc for query context: {doc_path}, error={exc}")
        return None

    theory_template = build_websearch_theory_context(doc_text, max_chars=700)
    if theory_template:
        print(f"[websearch] loaded round-aware theory context from: {doc_path}")
    return theory_template or None


def _step_input_to_text(step_input: Any) -> str:
    if step_input is None:
        return ""

    chunks: list[str] = []
    for key in ("input", "content", "message", "query", "text"):
        value = getattr(step_input, key, None)
        if value:
            chunks.append(str(value))

    if isinstance(step_input, dict):
        for key in ("input", "content", "message", "query", "text"):
            value = step_input.get(key)
            if value:
                chunks.append(str(value))

    chunks.append(str(step_input))
    return " ".join(chunks)


def _extract_requested_iterations(text: str) -> int | None:
    if not text:
        return None

    patterns = [
        r"第\s*(\d+)\s*轮",
        r"(\d+)\s*轮",
        r"(\d+)\s*iterations?",
        r"iterations?\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                value = int(match.group(1))
                if value > 0:
                    return value
            except Exception:
                continue
    return None

def _extract_step_payload(step_input: Any) -> dict[str, Any]:
    if step_input is None:
        return {}

    # Preferred path in Agno StepInput
    raw_input = getattr(step_input, "input", None)
    if isinstance(raw_input, dict):
        return raw_input
    if raw_input is None and isinstance(step_input, dict):
        raw_input = step_input.get("input", step_input)
    elif raw_input is None:
        raw_input = step_input

    if isinstance(raw_input, dict):
        return raw_input
    if isinstance(raw_input, WorkflowInput):
        return raw_input.model_dump()
    if hasattr(raw_input, "model_dump"):
        try:
            dumped = raw_input.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    return {}


def _build_theory_template_from_payload(payload: dict[str, Any]) -> str | None:
    material_type = str(payload.get("material_type") or "").strip()
    goal = str(payload.get("goal") or "").strip()
    composition = payload.get("composition") if isinstance(payload.get("composition"), dict) else {}
    processing = payload.get("processing") if isinstance(payload.get("processing"), dict) else {}
    features = payload.get("features") if isinstance(payload.get("features"), dict) else {}
    if not any([material_type, goal, composition, processing, features]):
        return None

    return (
        f"material type: {material_type or 'N/A'} | "
        f"goal: {goal or 'N/A'} | "
        f"composition constraints: {json.dumps(composition, ensure_ascii=False)} | "
        f"processing constraints: {json.dumps(processing, ensure_ascii=False)} | "
        f"features: {json.dumps(features, ensure_ascii=False)} | "
        "low lattice thermal conductivity mechanisms, phonon scattering, anharmonicity, mass disorder, lone pair, rattling"
    )


def _extract_runtime_overrides(step_input: Any, base_config: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
    payload = _extract_step_payload(step_input)
    if not payload:
        return {}, None

    try:
        wf_input = WorkflowInput.model_validate(payload)
        # Only use explicitly provided fields; avoid clobbering with schema defaults.
        normalized = wf_input.model_dump(exclude_unset=True, exclude_none=True)
    except ValidationError as exc:
        print(f"[agentos] workflow input validation failed, fallback to defaults: {exc}")
        normalized = payload

    overrides: dict[str, Any] = {}
    allowed_keys = {
        "samples",
        "n_structures",
        "top_k_bayes",
        "top_k_screen",
        "postprocess_workers",
        "novelty_workers",
        "websearch_enabled",
        "websearch_top_n",
        "phonon_imag_tol",
        "seed",
    }
    for key in allowed_keys:
        value = normalized.get(key)
        if value is not None and key in base_config:
            overrides[key] = value

    theory_template = _build_theory_template_from_payload(normalized)
    if theory_template:
        overrides["websearch_theory_template"] = theory_template

    requested_iterations = normalized.get("max_iterations")
    if requested_iterations is not None:
        try:
            requested_iterations = int(requested_iterations)
        except Exception:
            requested_iterations = None

    return overrides, requested_iterations


def _extract_formula(item: dict[str, Any]) -> str:
    return str(item.get("formula") or item.get("composition") or item.get("name") or "").strip()


def _extract_kappa(item: dict[str, Any]) -> float | None:
    for key in ("kappa", "k", "k_pred", "thermal_conductivity", "kappa_slack"):
        value = item.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _extract_result_path(item: dict[str, Any]) -> str:
    for key in ("Relative_CIF_Path", "CSV路径", "csv_path", "path", "result_path", "cif_path"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def _persist_runtime_memory(run_config: dict[str, Any], requested_iterations: int | None = None) -> None:
    csv_path = str(run_config.get("params_csv_path") or "").strip()
    if not csv_path:
        return
    payload = {
        "samples": run_config.get("samples"),
        "n_structures": run_config.get("n_structures"),
        "top_k_bayes": run_config.get("top_k_bayes"),
        "top_k_screen": run_config.get("top_k_screen"),
        "postprocess_workers": run_config.get("postprocess_workers"),
        "novelty_workers": run_config.get("novelty_workers"),
        "websearch_enabled": run_config.get("websearch_enabled"),
        "websearch_top_n": run_config.get("websearch_top_n"),
        "phonon_imag_tol": run_config.get("phonon_imag_tol"),
        "seed": run_config.get("seed"),
        "relax_timeout_sec": run_config.get("relax_timeout_sec"),
        "skip_doc_update": run_config.get("skip_doc_update"),
        "agentos_default_iterations": requested_iterations if requested_iterations is not None else run_config.get("agentos_default_iterations"),
        "agentos_ws_ping_interval": run_config.get("agentos_ws_ping_interval"),
        "agentos_ws_ping_timeout": run_config.get("agentos_ws_ping_timeout"),
    }
    updated, warnings = persist_param_values(csv_path, payload, keys=list(payload.keys()), enable_for_new_keys=False)
    if warnings:
        print(f"[agentos] param memory warnings: {warnings}")
    elif updated:
        print(f"[agentos] remembered runtime params to csv: {updated}")

def _compact_materials(materials: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for idx, item in enumerate(materials[:limit], start=1):
        compact.append(
            {
                "rank": idx,
                "name": _extract_formula(item),
                "path": _extract_result_path(item),
                "kappa_w_mk": _extract_kappa(item),
            }
        )
    return compact


def _first_existing_value(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _sort_summary_by_iteration(df: pd.DataFrame) -> pd.DataFrame:
    """Sort summary rows by numeric iteration, preserving order within an iteration."""
    if "iteration" not in df.columns or df.empty:
        return df

    return (
        df.assign(_summary_iteration_sort=pd.to_numeric(df["iteration"], errors="coerce"))
        .sort_values("_summary_iteration_sort", kind="mergesort", na_position="last")
        .drop(columns="_summary_iteration_sort")
        .reset_index(drop=True)
    )


def _aggregate_materials_from_results(results_root: str) -> dict[str, Any]:
    base = Path(results_root)
    summary_dir = base / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "success": ("success_materials.csv", "success_materials_deduped.csv"),
        "stable": ("stable_materials.csv", "stable_materials_deduped.csv"),
    }

    formula_keys = ["formula", "Formula", "composition", "name", "组分", "组成"]
    kappa_keys = ["thermal_conductivity_w_mk", "thermal_conductivity", "kappa", "k", "k_pred", "热导率(W/m·K)"]
    path_keys = ["relative_cif_path", "Relative_CIF_Path", "cif_file", "CIF文件", "csv_path", "CSV路径", "path"]
    sid_keys = ["structure_id", "结构ID", "Structure_ID", "id"]

    aggregate: dict[str, pd.DataFrame] = {}
    written_paths: dict[str, str] = {}

    for tag, filename_candidates in spec.items():
        frames: list[pd.DataFrame] = []
        for success_examples_dir in sorted(base.glob("iteration_*/success_examples")):
            csv_path = None
            for filename in filename_candidates:
                candidate = success_examples_dir / filename
                if candidate.exists():
                    csv_path = candidate
                    break
            if csv_path is None:
                continue

            iter_match = re.search(r"iteration_(\d+)", str(success_examples_dir))
            iteration = int(iter_match.group(1)) if iter_match else None
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
            except Exception:
                continue
            if df.empty:
                continue

            df = enrich_space_group_number(df, source_csv=csv_path)
            if "iteration" not in df.columns:
                df.insert(0, "iteration", iteration)
            else:
                df["iteration"] = df["iteration"].where(df["iteration"].notna(), iteration)
            df["source_type"] = tag
            df["source_file"] = str(csv_path)
            frames.append(df)

        if frames:
            out_df = pd.concat(frames, ignore_index=True)
            records = out_df.to_dict(orient="records")
            out_df["_summary_formula"] = [str(_first_existing_value(r, formula_keys) or "").strip() for r in records]
            out_df["_summary_rel_path"] = [str(_first_existing_value(r, path_keys) or "").strip() for r in records]
            out_df["_summary_sid"] = [str(_first_existing_value(r, sid_keys) or "").strip() for r in records]

            def _extract_kappa_value(record: dict[str, Any]) -> float | None:
                kappa = _first_existing_value(record, kappa_keys)
                if kappa is None:
                    for col_name, col_val in record.items():
                        key_text = str(col_name)
                        if "W/m" in key_text and ("K" in key_text or "k" in key_text):
                            kappa = col_val
                            break
                try:
                    return float(kappa) if kappa is not None else None
                except Exception:
                    return None

            out_df["_summary_kappa"] = [_extract_kappa_value(r) for r in records]
            out_df = out_df[out_df["_summary_formula"].astype(bool)]
            out_df = out_df[out_df["_summary_kappa"].notna()]
            out_df["dedup_key"] = out_df.apply(
                lambda r: f"{r['_summary_formula']}||{r['_summary_rel_path'] or r['_summary_sid'] or ''}", axis=1
            )
            out_df = out_df.sort_values(["_summary_kappa", "_summary_formula", "dedup_key"], ascending=True, kind="mergesort")
            out_df = out_df.drop_duplicates(subset=["dedup_key"], keep="first")
            out_df = out_df.drop(columns=["_summary_formula", "_summary_rel_path", "_summary_sid", "_summary_kappa", "dedup_key"])
            out_df = _sort_summary_by_iteration(out_df)
        else:
            out_df = pd.DataFrame(columns=["iteration", "source_type", "source_file"])

        aggregate[tag] = out_df
        output_path = summary_dir / f"{tag}_materials_summary.csv"
        out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        written_paths[f"{tag}_summary_csv"] = str(output_path)

    all_df = pd.concat([aggregate["success"], aggregate["stable"]], ignore_index=True)
    if not all_df.empty:
        records = all_df.to_dict(orient="records")
        all_df["_summary_formula"] = [str(_first_existing_value(r, formula_keys) or "").strip() for r in records]
        all_df["_summary_rel_path"] = [str(_first_existing_value(r, path_keys) or "").strip() for r in records]
        all_df["_summary_sid"] = [str(_first_existing_value(r, sid_keys) or "").strip() for r in records]

        def _extract_kappa_value_all(record: dict[str, Any]) -> float | None:
            kappa = _first_existing_value(record, kappa_keys)
            if kappa is None:
                for col_name, col_val in record.items():
                    key_text = str(col_name)
                    if "W/m" in key_text and ("K" in key_text or "k" in key_text):
                        kappa = col_val
                        break
            try:
                return float(kappa) if kappa is not None else None
            except Exception:
                return None

        all_df["_summary_kappa"] = [_extract_kappa_value_all(r) for r in records]
        all_df = all_df[all_df["_summary_formula"].astype(bool)]
        all_df = all_df[all_df["_summary_kappa"].notna()]
        all_df["dedup_key"] = all_df.apply(
            lambda r: f"{r['_summary_formula']}||{r['_summary_rel_path'] or r['_summary_sid'] or ''}", axis=1
        )
        all_df = all_df.sort_values(["_summary_kappa", "_summary_formula", "dedup_key"], ascending=True, kind="mergesort")
        all_df = all_df.drop_duplicates(subset=["dedup_key"], keep="first")
        all_df = all_df.drop(columns=["_summary_formula", "_summary_rel_path", "_summary_sid", "_summary_kappa", "dedup_key"])
        all_df = _sort_summary_by_iteration(all_df)
    output_path = summary_dir / "all_materials_summary.csv"
    all_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    written_paths["all_summary_csv"] = str(output_path)
    return {
        "success_summary_csv": written_paths.get("success_summary_csv"),
        "stable_summary_csv": written_paths.get("stable_summary_csv"),
        "all_summary_csv": written_paths.get("all_summary_csv"),
    }


def _compact_iteration_result(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    theory = result.get("theory", {}) if isinstance(result.get("theory"), dict) else {}
    top10 = result.get("top10", []) if isinstance(result.get("top10"), list) else []
    top20 = result.get("top20", []) if isinstance(result.get("top20"), list) else []
    chosen = top10 if top10 else top20
    return {
        "success": bool(result.get("success")),
        "iteration_num": result.get("iteration_num"),
        "failed_step": result.get("failed_step"),
        "materials": _compact_materials(chosen, limit=20),
        "updated_data_path": theory.get("updated_data_path"),
        "updated_doc_path": theory.get("updated_doc_path"),
    }


def _run_single_iteration(
    iteration_num: int,
    config: dict[str, Any],
    tracker=None,
    initial_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    train_result = run_train_step(iteration_num, config, tracker)
    if not train_result.get("success"):
        return {"success": False, "failed_step": "train_model", "train": train_result}

    bayes_result = run_bayesian_step(iteration_num, config, tracker, initial_samples)
    if not bayes_result.get("success"):
        return {"success": False, "failed_step": "bayesian_optimization", "bayes": bayes_result}

    all_candidates = bayes_result.get("all_materials") or bayes_result.get("top_materials", [])
    bayes_pool_size = int(config.get("samples", 100))
    bayes_pool = list(all_candidates)[:bayes_pool_size]

    proposal_artifact = Path(config["results_root"]) / f"iteration_{iteration_num}" / "selected_results" / "llm_formula_proposals.json"
    resume_ai_completed = bool(tracker and tracker.is_step_completed(iteration_num, "ai_evaluation"))
    saved_proposal_state = None
    if tracker:
        saved_proposal_state = _load_saved_llm_formula_artifact(proposal_artifact, iteration_num)

    # LLM formula generation starts from iteration 2 and is opt-in so legacy
    # runs remain Pure BO unless explicitly enabled in runtime configuration.
    if saved_proposal_state is not None:
        llm_proposals, parents, parent_stats = saved_proposal_state
        print(f"[resume] loaded saved LLM formula proposals: {len(llm_proposals)}")
    elif resume_ai_completed:
        # The completed screening artifact is authoritative.  Do not call the
        # proposal LLM again merely because its optional audit file is missing.
        llm_proposals, parents, parent_stats = [], [], {}
        print("[resume] LLM formula proposal artifact missing; continuing without regeneration")
    else:
        parents, parent_stats = _load_high_symmetry_parents(config, iteration_num)
        try:
            llm_proposals = _generate_diverse_llm_formula_proposals(parents, config, iteration_num, bayes_pool)
        except Exception as exc:
            print(f"[llm-proposal] generation failed; falling back to BO: {exc}")
            llm_proposals = []
        if any("bo_prediction_status" not in item for item in llm_proposals):
            llm_proposals = _enrich_llm_proposals_with_bo_features(
                llm_proposals,
                iteration_num=iteration_num,
                config=config,
            )
    try:
        proposal_artifact.parent.mkdir(parents=True, exist_ok=True)
        proposal_artifact.write_text(json.dumps({
            "iteration": iteration_num,
            "parent_count": len(parents),
            "parent_stats": parent_stats,
            "parents": parents,
            "proposals": llm_proposals,
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except OSError as exc:
        print(f"[llm-proposal] failed to save artifact: {exc}")
    if llm_proposals:
        bayes_pool.extend(llm_proposals)
        print(f"[llm-proposal] parents={len(parents)}, proposals={len(llm_proposals)}")

    novel_pool_ranked = rank_by_ei(bayes_pool)
    novel_top20 = novel_pool_ranked[: config["top_k_bayes"]]
    # Preserve generated formulas in the unified evaluator input even when
    # they do not yet have a BO EI score.
    selected_formulas = {_extract_formula(item) for item in novel_top20}
    for proposal in llm_proposals:
        if _extract_formula(proposal) not in selected_formulas:
            novel_top20.append(proposal)
            selected_formulas.add(_extract_formula(proposal))
    print(
        f"[screening] input_count={len(bayes_pool)}, "
        f"novel_count={len(novel_pool_ranked)}, "
        f"novel_top20_count={len(novel_top20)}"
    )

    cached_screen_result = None
    if tracker and tracker.is_step_completed(iteration_num, "ai_evaluation"):
        cached_screen_result = load_saved_ai_evaluation_result(
            config["results_root"],
            iteration_num,
        )
        if bool(config.get("chemistry_diversity_enabled", False)):
            cached_mode = str(cached_screen_result.get("screening_mode") or "") if isinstance(cached_screen_result, dict) else ""
            if cached_mode != CHEMISTRY_DIVERSITY_SCREENING_MODE:
                cached_screen_result = None

    if cached_screen_result is not None:
        # The final AI selection is already validated and persisted. Do not
        # repeat WebSearch or invoke the LLM merely to reach a later step.
        print("[resume] AI selection is complete; skipping WebSearch and LLM evaluation")
        websearch_enriched_candidates = []
        artifact_paths = {
            "selected_csv": cached_screen_result.get("csv_path"),
            "trace_json": cached_screen_result.get("trace_path"),
        }
        screen_result = run_ai_evaluation_step(
            iteration_num=iteration_num,
            config=config,
            candidate_materials=[],
            tracker=tracker,
        )
    else:
        websearch_theory_template = _resolve_websearch_theory_template(iteration_num, config)
        websearch_candidates = novel_top20
        if bool(config.get("chemistry_diversity_enabled", False)):
            # Keep supplementary search evidence independent of the BO/EI
            # ordering.  Candidate IDs/formulas provide a reproducible order.
            websearch_candidates = sorted(
                novel_top20,
                key=lambda item: canonical_formula(_extract_formula(item)) or _extract_formula(item),
            )
        websearch_enriched_candidates = enrich_topn_with_websearch(
            candidates=websearch_candidates,
            top_n=int(config.get("websearch_top_n", 5)),
            enabled=bool(config.get("websearch_enabled", True)),
            strategy=str(config.get("websearch_strategy", "hybrid")),
            queries_per_candidate=int(config.get("websearch_queries_per_candidate", 2)),
            theory_template=websearch_theory_template,
        )
        # Ensure proposals remain in the same evaluator pool even when the
        # web-search helper returns only its enriched top-N subset.
        enriched_formulas = {_extract_formula(item) for item in websearch_enriched_candidates}
        for proposal in llm_proposals:
            if _extract_formula(proposal) not in enriched_formulas:
                websearch_enriched_candidates.append(proposal)
                enriched_formulas.add(_extract_formula(proposal))
        websearch_attempted_candidates = min(len(novel_top20), int(config.get("websearch_top_n", 5)))
        total_queries = sum(
            len(item.get("websearch_queries", []))
            for item in websearch_enriched_candidates[:websearch_attempted_candidates]
        )
        success_queries = sum(
            int(item.get("websearch_success_count", 0))
            for item in websearch_enriched_candidates[:websearch_attempted_candidates]
        )
        failed_queries = max(total_queries - success_queries, 0)
        err_counter = Counter()
        for item in websearch_enriched_candidates[:websearch_attempted_candidates]:
            for err in item.get("websearch_errors", []):
                err_counter[str(err)] += 1
        top_errors = err_counter.most_common(3)
        print(
            f"[websearch] candidates={websearch_attempted_candidates}, queries={total_queries}, "
            f"success={success_queries}, failed={failed_queries}, top_errors={top_errors}"
        )

        artifact_paths = _save_screening_artifacts(
            iteration_num=iteration_num,
            results_root=config["results_root"],
            novel_pool=novel_pool_ranked,
            websearch_enriched_candidates=websearch_enriched_candidates,
        )

        screen_result = run_ai_evaluation_step(
            iteration_num=iteration_num,
            config=config,
            candidate_materials=websearch_enriched_candidates,
            tracker=tracker,
        )

    if not screen_result.get("success"):
        return {
            "success": False,
            "failed_step": "ai_evaluation",
            "screen": screen_result,
            "top20": novel_top20,
        }

    screened_top10 = screen_result.get("selected_materials", [])
    screening_mode = str(screen_result.get("screening_mode") or SCREENING_MODE)
    # Carry iteration-level provenance into the summary without affecting the
    # chemistry-only ranking decision.
    screen_result["high_symmetry_parent_count"] = len(parents)
    screen_result["high_symmetry_parent_stats"] = parent_stats
    screen_result["llm_proposal_count"] = len(llm_proposals)

    calculate_result = run_structure_step(
        iteration_num=iteration_num,
        config=config,
        materials=screened_top10,
        tracker=tracker,
    )
    if not calculate_result.get("success"):
        return {
            "success": False,
            "failed_step": "calculation",
            "calculate": calculate_result,
            "top20": novel_top20,
            "top10": screened_top10,
        }
    if not calculate_result.get("completed"):
        return {
            "success": False,
            "failed_step": "calculation_incomplete",
            "calculate": calculate_result,
            "top20": novel_top20,
            "top10": screened_top10,
        }

    merge_result = run_merge_step(iteration_num=iteration_num, config=config, tracker=tracker)
    if not merge_result.get("success"):
        return {
            "success": False,
            "failed_step": "merge",
            "merge": merge_result,
            "top20": novel_top20,
            "top10": screened_top10,
        }

    extract_result = run_extract_step(iteration_num, config, tracker)
    if not extract_result.get("success") and not extract_result.get("no_materials"):
        return {
            "success": False,
            "failed_step": "extract",
            "extract": extract_result,
            "top20": novel_top20,
            "top10": screened_top10,
        }

    screening_summary_path = _update_screening_summary(
        iteration_num=iteration_num,
        results_root=config["results_root"],
        screen_result=screen_result,
        extraction_result=extract_result,
    )

    materials_summary = _aggregate_materials_from_results(config["results_root"])
    print(f"[summary] summary files: {materials_summary}")

    theory_result = run_document_update_step(
        iteration_num=iteration_num,
        config=config,
        extraction_result=extract_result,
        tracker=tracker,
    )
    if not theory_result.get("success"):
        return {
            "success": False,
            "failed_step": "theory",
            "theory": theory_result,
            "top20": novel_top20,
            "top10": screened_top10,
            "extract": extract_result,
            "materials_summary": materials_summary,
            "screening_summary_path": screening_summary_path,
        }

    return {
        "success": True,
        "iteration_num": iteration_num,
        "top20": novel_top20,
        "top10": screened_top10,
        "screening_mode": screening_mode,
        "bayes_pool_size": len(bayes_pool),
        "high_symmetry_parent_count": len(parents),
        "high_symmetry_parent_stats": parent_stats,
        "llm_proposal_count": len(llm_proposals),
        "llm_proposal_artifact": str(proposal_artifact),
        "novel_pool": novel_pool_ranked,
        "novel_top20": novel_top20,
        "websearch_enriched_candidates": websearch_enriched_candidates,
        "train": train_result,
        "bayes": bayes_result,
        "screen": {
            "success": True,
            "selected_materials": screened_top10,
            "raw_ai_result": screen_result,
            "novel_pool": novel_pool_ranked,
            "novel_top20": novel_top20,
            "websearch_enriched_candidates": websearch_enriched_candidates,
            "artifact_paths": artifact_paths,
            "screening_mode": screening_mode,
        },
        "calculate": {"success": True, "structure": calculate_result, "merge": merge_result},
        "extract": extract_result,
        "theory": theory_result,
        "materials_summary": materials_summary,
        "screening_summary_path": screening_summary_path,
    }


def build_aslk_steps(
    config: dict[str, Any],
    tracker,
    start_iteration: int,
    max_iterations: int,
    initial_samples: list[dict[str, Any]] | None = None,
) -> list[Any]:
    from agno.workflow.step import Step, StepInput, StepOutput

    def orchestration_executor(step_input: StepInput, run_context=None) -> StepOutput:
        run_config = dict(config)
        verbose_output = bool(run_config.get("agentos_verbose_output", False))
        runtime_overrides, requested_iterations = _extract_runtime_overrides(step_input, run_config)
        print(f"[agentos] form payload keys={sorted(list(runtime_overrides.keys()))}, requested_iterations={requested_iterations}")
        if runtime_overrides:
            run_config.update(runtime_overrides)
            print(f"[agentos] runtime overrides from form: {runtime_overrides}")

        if bool(config.get("agentos_allow_text_iteration_override", True)):
            step_text = _step_input_to_text(step_input)
            text_requested_iterations = _extract_requested_iterations(step_text)
            if text_requested_iterations is not None:
                requested_iterations = text_requested_iterations
        loop_end = max_iterations
        if bool(run_config.get("max_iterations_locked", False)):
            requested_iterations = None
            loop_end = max_iterations
            print(
                f"[agentos] max_iterations locked by CLI/config, "
                f"start={start_iteration}, locked_end={loop_end}"
            )
        elif requested_iterations is not None:
            cap = int(run_config.get("agentos_max_iterations_cap", 20))
            loop_end = min(max(start_iteration, requested_iterations), cap)
            print(
                f"[agentos] requested_iterations={requested_iterations}, "
                f"start={start_iteration}, default_end={max_iterations}, resolved_end={loop_end}, cap={cap}"
            )
        _persist_runtime_memory(run_config, requested_iterations=requested_iterations)

        local_samples = initial_samples
        all_results: list[dict[str, Any]] = []
        compact_all_results: list[dict[str, Any]] = []

        session_state = getattr(run_context, "session_state", None)
        if isinstance(session_state, dict):
            for k, v in AGNO_SESSION_STATE_DEFAULT.items():
                session_state.setdefault(k, v if not isinstance(v, list) else list(v))

        for iteration in range(start_iteration, loop_end + 1):
            if iteration > int(max_iterations):
                print(
                    f"[agentos] stop guard reached: iteration={iteration}, "
                    f"max_iterations={max_iterations}"
                )
                break
            result = _run_single_iteration(iteration, run_config, tracker, local_samples)
            all_results.append(result)
            compact_result = _compact_iteration_result(result)
            compact_all_results.append(compact_result)
            try:
                next_samples, sample_source = extract_initial_samples_from_result(result.get("extract"))
                if next_samples:
                    local_samples = next_samples
                    print(
                        f"[warm-start] prepared {len(next_samples)} samples for next iteration "
                        f"(source: {sample_source})"
                    )
                elif local_samples and isinstance(result.get("extract"), dict):
                    local_samples = None
                    print("[warm-start] no reusable success/stable samples; next iteration falls back to config sampler")
            except Exception as exc:
                print(f"[warm-start] failed to prepare next-iteration samples: {exc}")

            if isinstance(session_state, dict):
                session_state[AGNO_STATE_KEYS["last_iteration"]] = iteration
                session_state[AGNO_STATE_KEYS["last_result"]] = result if verbose_output else compact_result
                session_state[AGNO_STATE_KEYS["all_results"]] = all_results if verbose_output else compact_all_results
                session_state[AGNO_STATE_KEYS["candidate_top20"]] = result.get("top20", []) if verbose_output else compact_result.get("materials", [])
                session_state[AGNO_STATE_KEYS["screened_top10"]] = result.get("top10", []) if verbose_output else compact_result.get("materials", [])
                session_state[AGNO_STATE_KEYS["calculation_results"]] = result.get("calculate", {}) if verbose_output else {}
                session_state[AGNO_STATE_KEYS["extraction_result"]] = result.get("extract", {}) if verbose_output else {}
                theory_result = result.get("theory", {})
                session_state[AGNO_STATE_KEYS["updated_data_path"]] = theory_result.get("updated_data_path")
                session_state[AGNO_STATE_KEYS["updated_doc_path"]] = theory_result.get("updated_doc_path")
                if not result.get("success"):
                    session_state[AGNO_STATE_KEYS["errors"]] = session_state.get(AGNO_STATE_KEYS["errors"], []) + [
                        result.get("failed_step", "unknown")
                    ]

            if result.get("success") and tracker and not tracker.is_round_completed(iteration):
                incomplete_step = tracker.get_next_incomplete_step(iteration) or "unknown"
                result["success"] = False
                result["failed_step"] = incomplete_step
                result["error"] = f"Iteration {iteration} returned before all required steps were completed"
                print(
                    f"[workflow] iteration {iteration} did not satisfy the completion barrier; "
                    f"first incomplete step: {incomplete_step}"
                )

            if not result.get("success"):
                break

        summary = {
            "runs": len(all_results),
            "all_success": all(r.get("success") for r in all_results),
            "last_result": (all_results[-1] if all_results else {}) if verbose_output else (compact_all_results[-1] if compact_all_results else {}),
            "all_results": all_results if verbose_output else compact_all_results,
            "materials_summary": (all_results[-1].get("materials_summary", {}) if all_results else {}),
            "requested_iterations": requested_iterations,
            "resolved_end_iteration": loop_end,
        }
        return StepOutput(content=summary)

    return [
        Step(
            name="aslk_orchestration",
            description="ASLK iterative orchestration with Agno Step executor",
            executor=orchestration_executor,
        )
    ]


__all__ = [
    "build_aslk_steps",
    "run_train_step",
    "run_bayesian_step",
    "run_extract_step",
]
