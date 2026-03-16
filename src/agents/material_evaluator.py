# -*- coding: utf-8 -*-
"""Material evaluator used by step_ai_evaluation."""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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

        materials_info = self._format_materials_info(top_materials)
        websearch_info = self._format_websearch_summary(top_materials, iteration_num)
        prompt = self._build_evaluation_prompt(materials_info, websearch_info, n_select)

        input_file = self._save_input(prompt, iteration_num, results_root)
        print(f"[evaluator] saved llm input: {input_file}")

        print("[evaluator] requesting LLM evaluation...")
        try:
            evaluation = self.ai_client.chat(
                prompt=prompt,
                model_id=self.model_id,
                temperature=self.ai_client.get_default_temperature("workflow"),
                max_tokens=4000,
            )
        except Exception as e:
            print(f"[evaluator] llm call failed, fallback to empty selection: {e}")
            evaluation = json.dumps({"selected_materials": []}, ensure_ascii=False)

        output_file = self._save_output(evaluation, iteration_num, results_root)
        print(f"[evaluator] saved llm output: {output_file}")

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "iteration_num": iteration_num,
            "top_materials": top_materials,
            "evaluation": evaluation,
            "n_candidates": len(top_materials),
            "n_selected": n_select,
            "input_file": input_file,
            "output_file": output_file,
            "results_root": results_root,
        }

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
        if pd.isna(value):
            return []

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

    def _build_evaluation_prompt(self, materials_info: str, websearch_info: str, n_select: int) -> str:
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

You are a materials science expert. Based on the above theoretical principles document (especially **Section 2: Design Theory** and **Section 5: Evaluation Rules**), evaluate the candidate materials and select the {n_select} most promising low thermal conductivity materials.

**Evaluation Requirements**:

1. **Theoretical Consistency Analysis**:
   - Check if materials conform to the four major phonon scattering mechanisms.
   - Verify compliance with core evaluation parameters from the document.
   - Validate using successful-case theoretical summary if available.
   - Use `WebSearch Evidence` as supplementary evidence only.

2. **Comprehensive Judgment Principles**:
   - Theoretical principle compliance > acquisition function value (EI/Score).
   - Provide explicit physical-mechanism reasoning.

3. **Output Format**: Return JSON strictly in this schema:

```json
{{
  "selected_materials": [
    {{
      "formula": "Bi2Te3",
      "predicted_thermal_conductivity": 1.50,
      "reasoning": "brief physics-based reason",
      "confidence": "high/medium/low"
    }}
  ]
}}
```

**Strict Requirements**:
- Must select exactly {n_select} materials.
- JSON must be directly parsable by Python json.loads().
- Numerical fields must be numeric types.
- confidence must be one of high/medium/low.
- Do not output any text outside JSON.
"""
        return prompt

    def _save_input(self, prompt: str, iteration_num: int, results_root: str = "results") -> str:
        output_dir = Path(f"{results_root}/iteration_{iteration_num}/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        input_file = output_dir / "llm_evaluation_input.md"

        with open(input_file, "w", encoding="utf-8") as f:
            f.write("# Material Evaluation - LLM Input\n\n")
            f.write(f"**Iteration**: Iteration {iteration_num}\n\n")
            f.write(f"**Generation Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Model**: {self.model_name}\n\n")
            f.write("---\n\n")
            f.write(prompt)

        return str(input_file)

    def _save_output(self, response: str, iteration_num: int, results_root: str = "results") -> str:
        output_dir = Path(f"{results_root}/iteration_{iteration_num}/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "llm_evaluation_output.md"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Material Evaluation - LLM Output\n\n")
            f.write(f"**Iteration**: Iteration {iteration_num}\n\n")
            f.write(f"**Generation Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Model**: {self.model_name}\n\n")
            f.write("---\n\n")
            f.write(response)

        return str(output_file)


def _extract_selected_materials(evaluation_text: str) -> list:
    selected: list = []
    try:
        data = json.loads(evaluation_text.strip())
    except Exception:
        import re

        m = re.search(r"```json\s*(\{.*?\})\s*```", evaluation_text, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(1))
        except Exception:
            return []

    for i, mat in enumerate(data.get("selected_materials", []), 1):
        selected.append(
            {
                "rank": i,
                "formula": mat.get("formula", ""),
                "k_pred": mat.get("predicted_thermal_conductivity", 0.0),
            }
        )
    return selected


def save_evaluation_results(result: Dict, output_dir: str | None = None):
    iteration_num = result.get("iteration_num", 1)
    results_root = result.get("results_root", "results")
    if output_dir is None:
        output_dir = f"{results_root}/iteration_{iteration_num}/selected_results"

    os.makedirs(output_dir, exist_ok=True)

    selected_materials = _extract_selected_materials(result["evaluation"])
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
                    "rank": mat["rank"],
                    "formula": mat["formula"],
                    "k_pred": mat["k_pred"],
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
            csv_data.append({"rank": mat["rank"], "formula": mat["formula"], "k_pred": mat["k_pred"]})

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
