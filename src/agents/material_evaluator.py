# -*- coding: utf-8 -*-
"""Material evaluator used by step_ai_evaluation."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
SRC_DIR = Path(SCRIPT_DIR).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.ai_client import AIClient
from agents.document_reader import DocumentReader
from utils.theory_doc_context import build_websearch_theory_context
from utils.candidate_identity import canonical_formula


class InvalidLLMResponseError(RuntimeError):
    """Raised when an LLM response is truncated or violates its output contract."""


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _parse_strict_json_response(response: str) -> dict[str, Any]:
    """Parse raw JSON or one complete JSON markdown fence, never a partial prefix."""
    raw = str(response or "").strip()
    if not raw:
        raise InvalidLLMResponseError("LLM response is empty")

    candidates = [raw]
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    raise InvalidLLMResponseError(
        "LLM response is truncated or invalid JSON; workflow aborted"
    )


def validate_llm_evaluation_response(
    response: str,
    *,
    evaluation_mode: str,
    n_select: int,
    top_materials: List[Dict],
    candidate_identity_mode: bool = False,
) -> dict[str, Any]:
    """Validate and normalize a machine-readable response before persistence.

    New identity-mode calls use opaque IDs. Legacy formula-only responses keep
    their strict rank contract for backward compatibility.
    """
    payload = _parse_strict_json_response(response)

    key = "candidate_scores" if evaluation_mode == "candidate_scores" else "selected_materials"
    rows = payload.get(key)
    if not isinstance(rows, list):
        raise InvalidLLMResponseError(
            f"LLM response must contain a list named {key!r}; workflow aborted"
        )

    candidates = [item for item in top_materials if isinstance(item, dict) and str(item.get("formula", "")).strip()]
    candidate_formulas = [str(item.get("formula", "")).strip() for item in candidates]
    allowed_formulas = set(candidate_formulas)
    canonical_map: dict[str, str] = {}
    for formula in candidate_formulas:
        key = canonical_formula(formula)
        if not key or (key in canonical_map and canonical_map[key] != formula):
            raise InvalidLLMResponseError("candidate list contains duplicate or unparseable formulas")
        canonical_map[key] = formula
    candidate_ids = [str(item.get("candidate_id", "")).strip() for item in candidates]
    allowed_ids = {value for value in candidate_ids if value}
    if candidate_identity_mode and len(candidate_ids) != len(allowed_ids):
        raise InvalidLLMResponseError("candidate list contains missing or duplicate candidate_id")
    if len(candidate_formulas) != len(allowed_formulas):
        raise InvalidLLMResponseError("candidate list contains duplicate or missing formulas")

    def resolve(row: dict[str, Any]) -> tuple[str, str, int]:
        candidate_id = str(row.get("candidate_id", "")).strip()
        formula = str(row.get("formula", "")).strip()
        if candidate_identity_mode:
            if candidate_id not in allowed_ids:
                raise InvalidLLMResponseError(f"{key} contains an unknown candidate_id")
            index = candidate_ids.index(candidate_id)
            expected = candidate_formulas[index]
            if formula and canonical_formula(formula) != canonical_formula(expected):
                raise InvalidLLMResponseError(f"{key} formula conflicts with candidate_id")
            return candidate_id, expected, index + 1
        if formula in allowed_formulas:
            return candidate_id, formula, candidate_formulas.index(formula) + 1
        formula_key = canonical_formula(formula)
        if formula_key in canonical_map:
            expected = canonical_map[formula_key]
            return candidate_id, expected, candidate_formulas.index(expected) + 1
        raise InvalidLLMResponseError(f"{key} contains an unknown formula")

    if evaluation_mode == "candidate_scores":
        if len(rows) != len(candidate_formulas):
            raise InvalidLLMResponseError(
                f"LLM response is incomplete: expected scores for {len(candidate_formulas)} candidates, got {len(rows)}"
            )
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise InvalidLLMResponseError("candidate_scores contains a non-object row")
            candidate_id, formula, resolved_rank = resolve(row)
            if candidate_identity_mode:
                row["candidate_id"], row["formula"], row["original_rank"] = candidate_id, formula, resolved_rank
            rank = row.get("original_rank")
            if formula not in allowed_formulas or formula in seen:
                raise InvalidLLMResponseError("candidate_scores contains an unknown or duplicate formula")
            if type(rank) is not int or not 1 <= rank <= len(candidate_formulas):
                raise InvalidLLMResponseError("candidate_scores contains an invalid original_rank")
            if not candidate_identity_mode and candidate_formulas[rank - 1] != formula:
                raise InvalidLLMResponseError("candidate_scores original_rank does not match candidate order")
            score_names = (
                "chemical_plausibility_score",
                "low_kappa_mechanism_score",
                "stability_risk_score",
            ) if candidate_identity_mode else (
                "mechanism_fit_score",
                "stability_risk_score",
                "novelty_bonus_score",
                "bo_override_confidence",
            )
            for score_name in score_names:
                score = row.get(score_name)
                if type(score) is not int or not 0 <= score <= 10:
                    raise InvalidLLMResponseError(f"candidate_scores contains invalid {score_name}")
            seen.add(formula)
        return payload

    expected_count = min(max(int(n_select), 1), len(candidate_formulas))
    if expected_count <= 0 or len(rows) != expected_count:
        raise InvalidLLMResponseError(
            f"LLM response is incomplete: expected exactly {expected_count} selected materials, got {len(rows)}"
        )

    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise InvalidLLMResponseError("selected_materials contains a non-object row")
        candidate_id, formula, resolved_rank = resolve(row)
        if candidate_identity_mode:
            row["candidate_id"], row["formula"], row["original_rank"] = candidate_id, formula, resolved_rank
        final_rank = row.get("final_rank")
        original_rank = row.get("original_rank")
        if formula not in allowed_formulas or formula in seen:
            raise InvalidLLMResponseError("selected_materials contains an unknown or duplicate formula")
        if type(final_rank) is not int or final_rank != index:
            raise InvalidLLMResponseError("selected_materials final_rank must be continuous and ordered")
        if original_rank is not None and (
            type(original_rank) is not int or not 1 <= original_rank <= len(candidate_formulas)
        ):
            raise InvalidLLMResponseError("selected_materials contains an invalid original_rank")
        seen.add(formula)

    return payload


class MaterialEvaluator:
    def __init__(self, doc_path: str = "assets/theory.md", model_id: str | None = None):
        self.doc_path = doc_path
        self.doc_reader = DocumentReader(doc_path)
        self.doc_content = self.doc_reader.get_full_content()
        self.ai_client = AIClient()
        self.model_id = model_id or self.ai_client.get_default_model("workflow")
        model_info = self.ai_client.get_model_info(self.model_id) or {}
        self.model_name = str(model_info.get("model") or self.model_id)

        print(f"[evaluator] loaded theory doc: {doc_path}")
        print(f"[evaluator] theory length: {len(self.doc_content)}")

    def evaluate_materials(
        self,
        top_materials: List[Dict],
        n_select: int = 5,
        iteration_num: int = 1,
        results_root: str = "results",
        extra_instructions: str | None = None,
        evaluation_mode: Literal["selected_materials", "candidate_scores"] = "selected_materials",
        candidate_identity_mode: bool = False,
        evaluator_evidence: str = "full",
    ) -> Dict:
        print("\n" + "=" * 80)
        print("Material Evaluation")
        print("=" * 80)
        print(f"theory: {self.doc_path}")
        print(f"model: {self.model_name}")
        print(f"candidates: {len(top_materials)}")
        print(f"target select: {n_select}")
        print(f"iteration: {iteration_num}")
        print("=" * 80)

        if evaluation_mode == "selected_materials":
            if evaluator_evidence == "chemistry_only":
                # Stable, source-independent display order instead of BO rank
                # followed by appended proposals. Keep the records intact.
                top_materials = sorted(
                    top_materials,
                    key=lambda item: hashlib.sha256(str(item.get("formula", "")).strip().encode("utf-8")).hexdigest(),
                )
                materials_info = self._format_formula_info(top_materials)
            else:
                # Legacy llm_full_rerank input contract: BO metrics in the
                # original candidate order (BO rank + appended proposals).
                materials_info = self._format_materials_info(top_materials)
        else:
            # Preserve the metric-based input contract for candidate scoring.
            if evaluator_evidence == "chemistry_only":
                materials_info = self._format_formula_info_with_ids(top_materials)
            else:
                materials_info = self._format_materials_info(top_materials)
        websearch_info = self._format_websearch_summary(top_materials, iteration_num)
        if evaluation_mode == "candidate_scores":
            prompt = self._build_candidate_scoring_prompt(
                materials_info,
                websearch_info,
                extra_instructions=extra_instructions,
                chemistry_only=evaluator_evidence == "chemistry_only",
            )
            report_stem = "llm_candidate_scoring"
        else:
            prompt = self._build_evaluation_prompt(
                materials_info,
                websearch_info,
                n_select,
                extra_instructions=extra_instructions,
                evaluator_evidence=evaluator_evidence,
            )
            report_stem = "llm_evaluation"

        input_file = self._save_input(prompt, iteration_num, results_root, report_stem=report_stem)
        print(f"[evaluator] saved llm input: {input_file}")

        print("[evaluator] requesting LLM evaluation...")
        try:
            evaluation = self.ai_client.chat(
                prompt=prompt,
                model_id=self.model_id,
                temperature=self.ai_client.get_default_temperature("workflow"),
                max_tokens=8000,
                response_format={"type": "json_object"},
                auto_fallback=False,
            )
        except Exception as exc:
            invalid_file = self._save_invalid_output(
                "",
                iteration_num,
                results_root,
                report_stem=report_stem,
                error=f"LLM evaluation request failed; workflow aborted: {exc}",
            )
            raise RuntimeError(
                f"LLM evaluation request failed; workflow aborted: {exc}; diagnostic report: {invalid_file}"
            ) from exc

        try:
            validated = validate_llm_evaluation_response(
                evaluation,
                evaluation_mode=evaluation_mode,
                n_select=n_select,
                top_materials=top_materials,
                candidate_identity_mode=candidate_identity_mode,
            )
        except InvalidLLMResponseError as exc:
            invalid_file = self._save_invalid_output(
                evaluation,
                iteration_num,
                results_root,
                report_stem=report_stem,
                error=str(exc),
            )
            if candidate_identity_mode:
                allowed = [str(item.get("candidate_id")) for item in top_materials]
                correction = (
                    prompt + "\n\nCORRECTION: The previous response was invalid: " + str(exc)
                    + "\nReturn one valid score object for every allowed candidate_id exactly once. "
                    + "Do not output new formulas or substitute elements. Allowed IDs: " + json.dumps(allowed)
                )
            else:
                allowed = [str(item.get("formula", "")).strip() for item in top_materials]
                correction = (
                    prompt + "\n\nCORRECTION: The previous response was invalid: " + str(exc)
                    + "\nUse only the exact candidate formulas below. Do not invent, rewrite, or duplicate a formula. "
                    + "Allowed formulas: " + json.dumps(allowed, ensure_ascii=False)
                )
            try:
                evaluation = self.ai_client.chat(
                    prompt=correction, model_id=self.model_id,
                    temperature=self.ai_client.get_default_temperature("workflow"),
                    max_tokens=8000, response_format={"type": "json_object"}, auto_fallback=False,
                )
                validated = validate_llm_evaluation_response(
                    evaluation, evaluation_mode=evaluation_mode, n_select=n_select,
                    top_materials=top_materials, candidate_identity_mode=candidate_identity_mode,
                )
            except Exception as retry_exc:
                retry_file = self._save_invalid_output(
                    evaluation, iteration_num, results_root, report_stem=report_stem,
                    error=f"Retry failed: {retry_exc}",
                )
                raise InvalidLLMResponseError(f"LLM scoring failed after correction; diagnostic report: {retry_file}") from retry_exc

        if candidate_identity_mode:
            evaluation = json.dumps(validated, ensure_ascii=False)

        output_file = self._save_output(evaluation, iteration_num, results_root, report_stem=report_stem)
        print(f"[evaluator] saved llm output: {output_file}")

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "iteration_num": iteration_num,
            "top_materials": top_materials,
            "evaluation": evaluation,
            "n_candidates": len(top_materials),
            "n_selected": n_select,
            "evaluation_mode": evaluation_mode,
            "input_file": input_file,
            "output_file": output_file,
            "results_root": results_root,
        }

    def _format_formula_info(self, materials: List[Dict]) -> str:
        """Render the same composition-only evidence for every candidate source."""
        from pymatgen.core import Composition

        lines = ["Candidate Formulas (identifiers only, not a priority ranking):", ""]
        for index, material in enumerate(materials, 1):
            formula = str(material.get("formula", "")).strip()
            lines.append(f"[Material {index}] {formula}")
            try:
                amounts = Composition(formula, strict=True).get_el_amt_dict()
                elements = sorted(amounts)
                lines.append(f"  - Constituent elements: {', '.join(elements)}")
                counts = ", ".join(f"{element}: {amounts[element]:g}" for element in elements)
                lines.append(f"  - Stoichiometry: {counts}")
                lines.append(f"  - Number of elements: {len(elements)}")
                lines.append(f"  - Atoms in written formula: {sum(amounts.values()):g}")
            except (ValueError, TypeError) as exc:
                lines.append(f"  - Composition parsing: unavailable ({type(exc).__name__})")
            lines.append("")
        return "\n".join(lines)

    def _format_formula_info_with_ids(self, materials: List[Dict]) -> str:
        """Render chemistry-only evidence with stable IDs for model output."""
        from pymatgen.core import Composition

        lines = ["Candidate Formulas (identifiers are opaque and must be copied exactly):", ""]
        for index, material in enumerate(materials, 1):
            candidate_id = str(material.get("candidate_id") or f"cand_{index:04d}").strip()
            formula = str(material.get("formula", "")).strip()
            lines.append(f"[{candidate_id}] {formula}")
            try:
                amounts = Composition(formula, strict=True).get_el_amt_dict()
                elements = sorted(amounts)
                lines.append(f"  - Constituent elements: {', '.join(elements)}")
                lines.append("  - Stoichiometry: " + ", ".join(f"{element}: {amounts[element]:g}" for element in elements))
                lines.append(f"  - Number of elements: {len(elements)}")
                lines.append(f"  - Atoms in written formula: {sum(amounts.values()):g}")
            except (ValueError, TypeError):
                lines.append("  - Composition parsing: unavailable")
            lines.append("")
        return "\n".join(lines)

    def _format_materials_info(self, materials: List[Dict]) -> str:
        lines: List[str] = []
        lines.append("Candidate Materials List:")
        lines.append("")

        for i, mat in enumerate(materials, 1):
            lines.append(f"[Material {i}] {mat.get('formula', 'N/A')}")
            lines.append(f"  - Predicted thermal conductivity: {float(mat.get('k_pred', 0.0)):.4f} W/(m·K)")
            lines.append(f"  - Predicted mean (mu_log): {float(mat.get('mu_log', 0.0)):.4f}")
            lines.append(f"  - Predicted std dev (sigma_log): {float(mat.get('sigma_log', 0.0)):.4f}")
            lines.append(
                f"  - 95% confidence interval: [{float(mat.get('k_lower', 0.0)):.3f}, {float(mat.get('k_upper', 0.0)):.3f}] W/(m·K)"
            )
            lines.append(f"  - Acquisition function value (EI): {float(mat.get('ei', mat.get('score', 0.0))):.4f}")
            lines.append(f"  - Constituent elements: {mat.get('elements', 'N/A')}")
            lines.append(f"  - Number of elements: {mat.get('n_elements', 'N/A')}")
            lines.append(f"  - Total atoms: {mat.get('total_atoms', 'N/A')}")

            lines.append("")

        return "\n".join(lines)

    def _coerce_list_field(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, (tuple, set)):
            return list(value)
        if not isinstance(value, (str, bytes)):
            try:
                if bool(pd.isna(value)):
                    return []
            except (TypeError, ValueError):
                pass

        text = str(value).strip()
        if not text or text.lower() == "nan":
            return []
        if text[0] in "[{(":
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                return [text]
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, (tuple, set)):
                return list(parsed)
            return [parsed]
        return [text]

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(limit - 3, 0)].rstrip() + "..."

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any] | None:
        payload = str(text or "").strip()
        if not payload:
            return None
        try:
            data = json.loads(payload)
            return data if isinstance(data, dict) else None
        except Exception:
            pass

        match = re.search(r"\{.*\}", payload, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _format_websearch_summary(self, materials: List[Dict], iteration_num: int) -> str:
        queries: List[str] = []
        summaries: List[str] = []
        sources: List[str] = []
        errors: List[str] = []
        formulas: List[str] = []
        elements: set[str] = set()

        for mat in materials:
            formula = str(mat.get("formula", "") or "").strip()
            if formula and formula not in formulas:
                formulas.append(formula)
                elements.update(re.findall(r"[A-Z][a-z]?", formula))
            for q in self._coerce_list_field(mat.get("websearch_queries")):
                q_str = str(q).strip()
                if q_str and q_str not in queries:
                    queries.append(q_str)
            s = str(mat.get("websearch_summary", "") or "").strip()
            if s and s.lower() != "nan" and s not in summaries:
                summaries.append(s)
            for src in self._coerce_list_field(mat.get("websearch_sources")):
                src_str = str(src).strip()
                if src_str and src_str not in sources:
                    sources.append(src_str)
            for err in self._coerce_list_field(mat.get("websearch_errors")):
                err_str = str(err).strip()
                if err_str and err_str not in errors:
                    errors.append(err_str)

        return self._synthesize_websearch_summary(
            iteration_num=iteration_num,
            formulas=formulas,
            elements=sorted(elements),
            queries=queries,
            summaries=summaries,
            sources=sources,
            errors=errors,
        )

    def _build_websearch_synthesis_prompt(
        self,
        iteration_num: int,
        formulas: List[str],
        elements: List[str],
        queries: List[str],
        summaries: List[str],
        sources: List[str],
        errors: List[str],
    ) -> str:
        theory_context = build_websearch_theory_context(self.doc_content, max_chars=420) or "N/A"
        formula_text = ", ".join(formulas[:6]) if formulas else "N/A"
        query_line = self._truncate_text(queries[0], 240) if queries else "N/A"
        summary_block = (
            "\n".join(f"- {self._truncate_text(item, 280)}" for item in summaries[:3])
            if summaries
            else "- N/A"
        )
        source_block = "\n".join(f"- {item}" for item in sources[:6]) if sources else "- N/A"
        error_block = "\n".join(f"- {self._truncate_text(item, 180)}" for item in errors[:4]) if errors else "- None"

        return f"""# WebSearch Evidence Distillation Task

Return JSON only:
{{
  "distilled_evidence": [
    "evidence point 1",
    "evidence point 2",
    "evidence point 3"
  ]
}}

Constraints:
- Return 3 to 5 evidence points when evidence exists
- Each point must be a short sentence
- Use only claims supported by the raw search evidence
- Do not include URLs
- Do not repeat the query
- Do not use markdown headings
- Do not use mechanism templates

Iteration: {iteration_num}
Candidate formulas: {formula_text}

Current theory context:
{theory_context}

Executed unified query:
{query_line}

Raw search evidence:
{summary_block}

Sources:
{source_block}

Errors:
{error_block}
"""

    def _render_websearch_evidence(
        self,
        unified_query: str,
        distilled_evidence: list[str],
        sources: list[str],
    ) -> str:
        lines = [
            f"- Unified query: {str(unified_query or '').strip() or 'N/A'}",
            "- Distilled evidence:",
        ]
        if distilled_evidence:
            lines.extend(f"- {self._truncate_text(item, 280)}" for item in distilled_evidence[:5])
        else:
            lines.append("- N/A")

        lines.extend(["", "References"])
        if sources:
            lines.extend(sources[:10])
        else:
            lines.append("N/A")

        return "\n".join(lines)

    def _build_websearch_fallback(
        self,
        iteration_num: int,
        formulas: List[str],
        elements: List[str],
        queries: List[str],
        summaries: List[str],
        sources: List[str],
        errors: List[str],
    ) -> str:
        del iteration_num, formulas, elements, errors
        distilled = [self._truncate_text(snippet, 280) for snippet in summaries[:5]]
        return self._render_websearch_evidence(
            unified_query=queries[0] if queries else "",
            distilled_evidence=distilled,
            sources=sources,
        )

    def _synthesize_websearch_summary(
        self,
        iteration_num: int,
        formulas: List[str],
        elements: List[str],
        queries: List[str],
        summaries: List[str],
        sources: List[str],
        errors: List[str],
    ) -> str:
        del elements
        queries = queries[:1]
        summaries = summaries[:5]
        sources = sources[:10]
        errors = errors[:4]
        query_value = queries[0] if queries else ""

        if not queries and not summaries and not sources and not errors:
            return self._build_websearch_fallback(
                iteration_num=iteration_num,
                formulas=formulas,
                elements=[],
                queries=queries,
                summaries=summaries,
                sources=sources,
                errors=errors,
            )

        prompt = self._build_websearch_synthesis_prompt(
            iteration_num=iteration_num,
            formulas=formulas,
            elements=[],
            queries=queries,
            summaries=summaries,
            sources=sources,
            errors=errors,
        )
        try:
            synthesized = self.ai_client.chat(
                prompt=prompt,
                model_id=self.model_id,
                temperature=self.ai_client.get_default_temperature("workflow"),
                max_tokens=600,
                auto_fallback=False,
            )
            parsed = self._parse_json_object(synthesized)
            evidence = []
            if parsed:
                raw_items = parsed.get("distilled_evidence")
                if isinstance(raw_items, list):
                    evidence = [
                        self._truncate_text(str(item), 280)
                        for item in raw_items
                        if str(item).strip()
                    ][:5]
            if evidence:
                return self._render_websearch_evidence(
                    unified_query=query_value,
                    distilled_evidence=evidence,
                    sources=sources,
                )
        except Exception as exc:
            print(f"[evaluator] websearch synthesis failed, falling back to raw evidence: {exc}")

        return self._build_websearch_fallback(
            iteration_num=iteration_num,
            formulas=formulas,
            elements=[],
            queries=queries,
            summaries=summaries,
            sources=sources,
            errors=errors,
        )

    def _build_evaluation_prompt(
        self,
        materials_info: str,
        websearch_info: str,
        n_select: int,
        extra_instructions: str | None = None,
        evaluator_evidence: str = "full",
    ) -> str:
        extra_instruction_block = ""
        if extra_instructions:
            extra_instruction_block = f"""

4. **Additional Selection Constraints**:
{extra_instructions}
"""
        if evaluator_evidence == "chemistry_only":
            evaluation_task = f"""You are a materials-science expert. Analyze the chemical formulas themselves and select the top {n_select} candidates for downstream structure generation and phonon validation.

The goal is to identify formulas that are both chemically plausible for stable materials and promising for low lattice thermal conductivity.

Focus on the formula and its chemistry:

- Identify the A, B, and Ch elements in the formula.
- Assess whether the A-B-Ch combination is chemically plausible.
- Consider oxidation-state compatibility, bonding character, atomic size, and coordination tendencies.
- Consider heavy atoms, mass contrast, mass disorder, soft bonding, anharmonicity, lone-pair activity, and other plausible mechanisms for reducing lattice thermal conductivity.
- Identify the main chemical or structural risk of each candidate.
- Use the theoretical document and WebSearch Evidence to support the formula-level analysis.
- Use `WebSearch Evidence` as supplementary evidence only.

Allowed element groups:
- A = [Ag, Cu, In, Sn, Pb]
- B = [As, Sb, Ge, Bi, Ti, V]
- Ch = [S, Se, Te]

Evaluate every formula as an independent scientific hypothesis. The candidate's generation source and algorithmic metadata are not the scientific objective. Missing predicted values or scores do not by themselves make a formula unpromising. Do not invent numerical stability, phonon, formation-energy, or thermal-conductivity values.

**Selection Procedure**:

1. Analyze the formula and its chemical rationale for every candidate.
2. Rank candidates by the combined plausibility of chemical stability and low lattice thermal conductivity.
3. Keep only the top {n_select} candidates in final priority order.

**Reasoning Requirements**:

- `ranking_reason` must refer directly to the formula's chemistry and low-thermal-conductivity mechanism.
- `main_risk` must describe the most important uncertainty in the formula-level hypothesis.
- Keep both fields concise and specific.
{extra_instruction_block}"""
        else:
            # Legacy llm_full_rerank contract: rerank with the BO evidence that
            # `_format_materials_info` renders. Kept verbatim so the
            # pre-chemistry-diversity behaviour can be reproduced for ablation.
            evaluation_task = f"""You are a materials science expert. Based on the above theoretical principles document, rerank the candidate materials internally and keep only the top {n_select} materials that are most promising for stable low lattice thermal conductivity.

**Evaluation Requirements**:

1. **Ranking Objective**:
   - Prioritize candidates that are more likely to remain stable while achieving low lattice thermal conductivity.
   - Use the theory document as the primary ranking guide.
   - Use candidate parameters such as predicted thermal conductivity, uncertainty, EI/score, element set, and atom count as supporting evidence.
   - Use `WebSearch Evidence` as supplementary evidence only.

2. **Selection Procedure**:
   - Internally rerank the full candidate pool first.
   - Then keep only the top {n_select} materials.
   - The output must contain only the retained top {n_select} materials, in final priority order.
   - The output order itself is the final ranking order.

3. **Reasoning Requirements**:
   - `ranking_reason` should explain why this material belongs in the retained top-{n_select} list.
   - `main_risk` should describe the single most important uncertainty or failure risk.
   - Keep both reason fields concise and specific.
{extra_instruction_block}"""
        prompt = f"""# Low Thermal Conductivity Material Evaluation Task

## Theoretical Principles Document

{self.doc_content}

---

## Candidate Materials Information

{materials_info}

---

## WebSearch Evidence

{websearch_info}

---

## Evaluation Task

{evaluation_task}

4. **Output Format**: Return JSON strictly in this schema:

```json
{{
  "selected_materials": [
    {{
      "formula": "Bi2Te3",
      "final_rank": 1,
      "original_rank": 3,
      "ranking_reason": "brief reason for entering the retained top-n",
      "main_risk": "brief main uncertainty"
    }}
  ]
}}
```

**Strict Requirements**:
- Must select exactly {n_select} materials.
- `final_rank` must be continuous integers from 1 to {n_select}.
- `original_rank` is optional, but if present it must refer to the candidate list order shown above.
- JSON must be directly parsable by Python json.loads().
- Do not output any material outside the provided candidate list.
- Do not output any text outside JSON.
- Do not include candidates that are not in the final retained top-{n_select}.
- Do not output any text outside JSON.
"""
        return prompt

    def _build_candidate_scoring_prompt(
        self,
        materials_info: str,
        websearch_info: str,
        extra_instructions: str | None = None,
        chemistry_only: bool = False,
    ) -> str:
        extra_instruction_block = ""
        if extra_instructions:
            extra_instruction_block = f"""

4. **Additional Scoring Constraints**:
{extra_instructions}
"""
        if chemistry_only:
            return f"""# Low Thermal Conductivity Candidate Scoring Task

## Theoretical Principles Document

{self.doc_content}

---

## Candidate Materials Information

{materials_info}

---

## WebSearch Evidence

{websearch_info}

---

## Scoring Task

Score every candidate using only its formula-level chemistry and the theory document. Do not infer or use candidate source, parent, algorithmic rank, EI, GPR values, or generated explanations. Treat every candidate identically.
The theory document contains hypotheses and examples. Do not treat one prototype, element tuple, stoichiometric ratio, or historical family as mandatory or universally superior. Reward chemically plausible alternatives and keep scores independent across candidates so the final deterministic selector can preserve chemical diversity.

Required integer scores from 0 to 10:
- `chemical_plausibility_score`: oxidation-state, site-role, and composition plausibility.
- `low_kappa_mechanism_score`: mass contrast, soft bonding, anharmonicity, lone-pair, or disorder mechanisms.
- `stability_risk_score`: severity of unresolved chemical or structural risk (higher is worse).

Score every supplied candidate exactly once. Preserve each `candidate_id` exactly. Return JSON only:

```json
{{
  "candidate_scores": [
    {{
      "candidate_id": "cand_0001",
      "chemical_plausibility_score": 7,
      "low_kappa_mechanism_score": 8,
      "stability_risk_score": 3,
      "short_reason": "brief chemistry reason",
      "main_risk": "brief uncertainty"
    }}
  ]
}}
```

{extra_instruction_block}
Do not output formula substitutions, new formulas, duplicate IDs, unknown IDs, or any text outside JSON.
"""

        return f"""# Low Thermal Conductivity Candidate Scoring Task

## Theoretical Principles Document

{self.doc_content}

---

## Candidate Materials Information

{materials_info}

---

## WebSearch Evidence

{websearch_info}

---

## Scoring Task

You are a materials science expert. Score every candidate using the theory document as the mechanism-analysis guide while still accounting for the candidate parameters shown above.

**Scoring Requirements**:

1. **Decision Goal**:
   - Help a BO-dominant workflow decide which candidates deserve to remain in the final retained list.
   - Use the theory document to judge mechanism fit and stability risk.
   - Do not ignore BO-side evidence such as predicted thermal conductivity, uncertainty, EI/score, and original rank.
   - The theory document should guide mechanism interpretation, but it must not override strong BO evidence without strong risk evidence.

2. **Required Scores**:
   - `mechanism_fit_score`: integer 0-10, higher means stronger support for low lattice thermal conductivity mechanisms.
   - `stability_risk_score`: integer 0-10, higher means more severe stability or feasibility risk.
   - `novelty_bonus_score`: integer 0-10, higher means stronger novelty or under-explored upside.
   - `bo_override_confidence`: integer 0-10, higher means you are confident the theory evidence should materially affect BO ordering.

3. **Output Coverage**:
   - Score every candidate exactly once.
   - Preserve candidate identity using the provided formula and original rank.
   - Keep `short_reason` and `main_risk` concise and specific.
{extra_instruction_block}

4. **Output Format**: Return JSON strictly in this schema:

```json
{{
  "candidate_scores": [
    {{
      "formula": "Bi2Te3",
      "original_rank": 3,
      "mechanism_fit_score": 8,
      "stability_risk_score": 4,
      "novelty_bonus_score": 6,
      "bo_override_confidence": 7,
      "short_reason": "brief mechanism justification",
      "main_risk": "brief main uncertainty"
    }}
  ]
}}
```

**Strict Requirements**:
- Score all provided candidates.
- All four scores must be integers from 0 to 10.
- `original_rank` must match the shown candidate order.
- Do not output any material outside the provided candidate list.
- JSON must be directly parsable by Python json.loads().
- Do not output any text outside JSON.
"""

    def _save_input(
        self,
        prompt: str,
        iteration_num: int,
        results_root: str = "results",
        report_stem: str = "llm_evaluation",
    ) -> str:
        output_dir = Path(f"{results_root}/iteration_{iteration_num}/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        input_file = output_dir / f"{report_stem}_input.md"

        content = (
            "# Material Evaluation - LLM Input\n\n"
            f"**Iteration**: Iteration {iteration_num}\n\n"
            f"**Generation Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"**Model**: {self.model_name}\n\n"
            "---\n\n"
            f"{prompt}"
        )
        _atomic_write_text(input_file, content)
        return str(input_file)

    def _save_output(
        self,
        response: str,
        iteration_num: int,
        results_root: str = "results",
        report_stem: str = "llm_evaluation",
    ) -> str:
        output_dir = Path(f"{results_root}/iteration_{iteration_num}/reports")
        output_file = output_dir / f"{report_stem}_output.md"
        content = (
            "# Material Evaluation - LLM Output\n\n"
            f"**Iteration**: Iteration {iteration_num}\n\n"
            f"**Generation Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"**Model**: {self.model_name}\n\n"
            "---\n\n"
            f"{response.rstrip()}\n"
        )
        _atomic_write_text(output_file, content)
        return str(output_file)

    def _save_invalid_output(
        self,
        response: str,
        iteration_num: int,
        results_root: str,
        report_stem: str,
        error: str,
    ) -> str:
        output_dir = Path(f"{results_root}/iteration_{iteration_num}/reports")
        invalid_file = output_dir / f"{report_stem}_output.invalid.md"
        content = (
            f"# Invalid {report_stem.replace('_', ' ').title()} Response\n\n"
            f"**Error**: {error}\n\n"
            "## Raw model response\n\n"
            f"{str(response or '')}\n"
        )
        _atomic_write_text(invalid_file, content)
        return str(invalid_file)


def _extract_selected_materials(evaluation_text: str) -> list:
    selected: list = []
    data = MaterialEvaluator._parse_json_object(evaluation_text)
    if not data:
        return []

    raw_selected = data.get("selected_materials", [])
    if not isinstance(raw_selected, list):
        return []

    for i, mat in enumerate(raw_selected, 1):
        if not isinstance(mat, dict):
            continue
        formula = str(mat.get("formula", "")).strip()
        if not formula:
            continue
        try:
            final_rank = int(mat.get("final_rank", i))
        except Exception:
            final_rank = i
        try:
            original_rank = int(mat["original_rank"]) if mat.get("original_rank") is not None else None
        except Exception:
            original_rank = None
        selected.append(
            {
                "candidate_id": str(mat.get("candidate_id", "")).strip(),
                "final_rank": final_rank,
                "formula": formula,
                "ranking_reason": str(mat.get("ranking_reason", "")).strip(),
                "main_risk": str(mat.get("main_risk", "")).strip(),
                "original_rank": original_rank,
            }
        )

    selected = sorted(selected, key=lambda row: (int(row.get("final_rank", 10**9)), row.get("formula", "")))
    for i, row in enumerate(selected, 1):
        row["final_rank"] = i
        if row.get("original_rank") is None:
            row.pop("original_rank", None)
    return selected


def _extract_candidate_scores(evaluation_text: str) -> list[dict[str, Any]]:
    parsed = MaterialEvaluator._parse_json_object(evaluation_text)
    if not parsed:
        return []

    raw_scores = parsed.get("candidate_scores", [])
    if not isinstance(raw_scores, list):
        return []

    extracted: list[dict[str, Any]] = []
    for row in raw_scores:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id", "")).strip()
        formula = str(row.get("formula", "")).strip()
        if not formula and not candidate_id:
            continue
        try:
            original_rank = int(row.get("original_rank")) if row.get("original_rank") is not None else 0
        except Exception:
            original_rank = 0

        def _score(name: str) -> int:
            try:
                value = int(row.get(name))
            except Exception:
                value = 0
            return max(0, min(10, value))

        normalized = {
            "candidate_id": candidate_id,
            "formula": formula,
            "original_rank": original_rank,
            "chemical_plausibility_score": _score("chemical_plausibility_score"),
            "low_kappa_mechanism_score": _score("low_kappa_mechanism_score"),
            "mechanism_fit_score": _score("mechanism_fit_score"),
            "stability_risk_score": _score("stability_risk_score"),
            "novelty_bonus_score": _score("novelty_bonus_score"),
            "bo_override_confidence": _score("bo_override_confidence"),
            "short_reason": str(row.get("short_reason", "")).strip(),
            "main_risk": str(row.get("main_risk", "")).strip(),
        }
        if candidate_id:
            normalized["chemical_score"] = (
                normalized["chemical_plausibility_score"]
                + normalized["low_kappa_mechanism_score"]
                - normalized["stability_risk_score"]
            )
        extracted.append(normalized)

    extracted = sorted(extracted, key=lambda item: (item["original_rank"], item.get("candidate_id", ""), item["formula"]))
    deduped: list[dict[str, Any]] = []
    seen_ranks: set[int] = set()
    seen_formulas: set[str] = set()
    seen_ids: set[str] = set()
    for row in extracted:
        if row.get("candidate_id"):
            if row["candidate_id"] in seen_ids:
                continue
            seen_ids.add(row["candidate_id"])
            deduped.append(row)
            continue
        if row["original_rank"] in seen_ranks or row["formula"] in seen_formulas:
            continue
        seen_ranks.add(row["original_rank"])
        seen_formulas.add(row["formula"])
        deduped.append(row)
    return deduped


def _limit_selected_materials(selected_materials: list[dict[str, Any]], n_select: int) -> list[dict[str, Any]]:
    limited = list(selected_materials[: max(0, int(n_select))])
    for i, row in enumerate(limited, 1):
        row["final_rank"] = i
    return limited


def save_evaluation_results(result: Dict, output_dir: str | None = None):
    iteration_num = result.get("iteration_num", 1)
    results_root = result.get("results_root", "results")
    if output_dir is None:
        output_dir = f"{results_root}/iteration_{iteration_num}/selected_results"

    os.makedirs(output_dir, exist_ok=True)
    evaluation_mode = str(result.get("evaluation_mode") or "selected_materials")

    if evaluation_mode == "candidate_scores":
        candidate_scores = _extract_candidate_scores(result["evaluation"])
        if not candidate_scores:
            return None

        csv_file = os.path.join(output_dir, "ai_candidate_scores.csv")
        csv_data = []
        for score_row in candidate_scores:
            matched_material = None
            for candidate in result["top_materials"]:
                if candidate.get("formula") == score_row.get("formula"):
                    matched_material = candidate
                    break

            merged = dict(score_row)
            if matched_material:
                merged.update(
                    {
                        "k_pred": matched_material.get("k_pred", ""),
                        "mu_log": matched_material.get("mu_log", ""),
                        "sigma_log": matched_material.get("sigma_log", ""),
                        "ei": matched_material.get("ei", matched_material.get("score", "")),
                        "k_lower": matched_material.get("k_lower", ""),
                        "k_upper": matched_material.get("k_upper", ""),
                        "elements": matched_material.get("elements", ""),
                        "n_elements": matched_material.get("n_elements", ""),
                        "total_atoms": matched_material.get("total_atoms", ""),
                    }
                )
            csv_data.append(merged)

        pd.DataFrame(csv_data).to_csv(csv_file, index=False, encoding="utf-8-sig")
        print(f"[evaluator] saved candidate score csv: {csv_file}")
        return csv_file

    selected_materials = _limit_selected_materials(
        _extract_selected_materials(result["evaluation"]),
        int(result.get("n_selected", 0) or 0),
    )
    if not selected_materials:
        return None

    csv_file = os.path.join(output_dir, "ai_selected_materials.csv")
    csv_data = []
    for mat in selected_materials:
        matched_material = None
        for candidate in result["top_materials"]:
            if candidate.get("formula") == mat.get("formula"):
                matched_material = candidate
                break

        if matched_material:
            csv_data.append(
                {
                    "final_rank": mat["final_rank"],
                    "original_rank": mat.get("original_rank", ""),
                    "formula": mat["formula"],
                    "ranking_reason": mat.get("ranking_reason", ""),
                    "main_risk": mat.get("main_risk", ""),
                    "k_pred": matched_material.get("k_pred", ""),
                    "mu_log": matched_material.get("mu_log", ""),
                    "sigma_log": matched_material.get("sigma_log", ""),
                    "ei": matched_material.get("ei", matched_material.get("score", "")),
                    "k_lower": matched_material.get("k_lower", ""),
                    "k_upper": matched_material.get("k_upper", ""),
                    "elements": matched_material.get("elements", ""),
                    "n_elements": matched_material.get("n_elements", ""),
                    "total_atoms": matched_material.get("total_atoms", ""),
                }
            )
        else:
            csv_data.append(
                {
                    "final_rank": mat["final_rank"],
                    "original_rank": mat.get("original_rank", ""),
                    "formula": mat["formula"],
                    "ranking_reason": mat.get("ranking_reason", ""),
                    "main_risk": mat.get("main_risk", ""),
                }
            )

    pd.DataFrame(csv_data).to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"[evaluator] saved selected csv: {csv_file}")
    return csv_file


def save_evaluation_report(result: Dict, output_dir: str | None = None) -> str | None:
    return save_evaluation_results(result, output_dir)


if __name__ == "__main__":
    test_materials = [
        {
            "formula": "Bi2Te3",
            "k_pred": 0.868,
            "mu_log": -0.14,
            "sigma_log": 0.8,
            "score": 0.95,
            "ei": 0.002,
            "k_lower": 0.65,
            "k_upper": 1.16,
            "elements": "Bi, Te",
            "n_elements": 2,
            "total_atoms": 5,
        }
    ]
    evaluator = MaterialEvaluator()
    result = evaluator.evaluate_materials(test_materials, n_select=1, iteration_num=1)
    save_evaluation_results(result)
