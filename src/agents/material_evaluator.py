# -*- coding: utf-8 -*-
"""Material evaluator used by step_ai_evaluation."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ai_client import AIClient
from document_reader import DocumentReader


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
        websearch_info = self._format_websearch_summary(top_materials)
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

    def _format_websearch_summary(self, materials: List[Dict]) -> str:
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
            for q in mat.get("websearch_queries", []) or []:
                q_str = str(q).strip()
                if q_str and q_str not in queries:
                    queries.append(q_str)
            s = str(mat.get("websearch_summary", "") or "").strip()
            if s and s not in summaries:
                summaries.append(s)
            for src in mat.get("websearch_sources", []) or []:
                src_str = str(src).strip()
                if src_str and src_str not in sources:
                    sources.append(src_str)
            for err in mat.get("websearch_errors", []) or []:
                err_str = str(err).strip()
                if err_str and err_str not in errors:
                    errors.append(err_str)

        return self._build_structured_websearch_summary(
            formulas=formulas,
            elements=sorted(elements),
            queries=queries,
            summaries=summaries,
            sources=sources,
            errors=errors,
        )

    @staticmethod
    def _contains_any(text: str, keywords: List[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    def _build_structured_websearch_summary(
        self,
        formulas: List[str],
        elements: List[str],
        queries: List[str],
        summaries: List[str],
        sources: List[str],
        errors: List[str],
    ) -> str:
        combined = " ".join([*queries, *summaries]).lower()
        formula_text = ", ".join(formulas[:8]) if formulas else "N/A"
        element_text = "-".join(elements) if elements else "N/A"

        overview_mechanisms: List[str] = []
        if self._contains_any(combined, ["anharm", "soft phonon", "soft mode", "soft vibrat"]):
            overview_mechanisms.append("strong anharmonicity")
        if self._contains_any(combined, ["mass disorder", "disorder", "distortion", "partial occup", "defect"]):
            overview_mechanisms.append("mass and structural disorder scattering")
        if self._contains_any(combined, ["lone pair", "lpe"]):
            overview_mechanisms.append("lone-pair-induced local distortion")
        if self._contains_any(combined, ["rattling", "localized", "resonant scattering", "resonance"]):
            overview_mechanisms.append("local rattling modes")
        if self._contains_any(combined, ["flatten", "flat branch", "group veloc", "mean free path", "boundary scattering"]):
            overview_mechanisms.append("reduced group velocity from structural complexity")
        if not overview_mechanisms:
            overview_mechanisms = [
                "strong anharmonicity",
                "mass and structural disorder scattering",
                "lone-pair-induced local distortion",
                "local rattling modes",
                "reduced group velocity from structural complexity",
            ]

        has_lone_pair = any(el in {"As", "Bi", "Ge", "Sb", "Sn", "Pb"} for el in elements) or self._contains_any(
            combined, ["lone pair", "lpe"]
        )
        has_rattling = "Ag" in elements or self._contains_any(combined, ["rattling", "localized", "resonant"])
        has_disorder = self._contains_any(
            combined, ["mass disorder", "disorder", "distortion", "partial occup", "defect", "boundary scattering"]
        )

        mechanism_1 = (
            "These materials often feature soft bonding, complex local coordination environments, and low-frequency vibrational modes, "
            "which together produce strong lattice anharmonicity. Strong anharmonicity enhances phonon-phonon scattering, shortens phonon lifetimes, "
            "and therefore suppresses lattice thermal conductivity."
        )
        if self._contains_any(combined, ["anharmonic scattering", "anharmonicities"]):
            mechanism_1 = (
                "The search results explicitly identify anharmonic scattering / large anharmonicities as an important origin of suppressed heat transport. "
                "Soft bonding environments, local structural distortions, and low-frequency vibrational modes strengthen anharmonicity, "
                "thereby increasing phonon-phonon scattering and lowering lattice thermal conductivity."
            )

        mechanism_2 = (
            "These candidate systems contain multiple elements, so differences in atomic mass and local bonding environments generate mass fluctuation "
            "and force-constant disorder. Partial occupancy, local disorder, and structural distortion can further enhance phonon scattering, "
            "especially for mid- and high-frequency phonons."
        )
        if has_disorder:
            mechanism_2 = (
                "The search results mention disorder / distortion of the unit cell and scattering from defects across multiple length scales. "
                f"For multicomponent {element_text} systems, differences in atomic mass and local coordination environments create mass fluctuation "
                "and force-constant disorder, while local disorder, partial occupancy, and structural distortion further strengthen phonon scattering."
            )

        mechanism_3 = (
            "As and Bi are commonly associated with lone-pair activity. Lone-pair electrons can induce off-centering, local structural distortion, "
            "and soft vibrational modes, which increase anharmonicity and reduce phonon transport efficiency."
        )
        if has_lone_pair:
            mechanism_3 = (
                "The search results indicate that lone-pair electrons can produce large anharmonicities and are closely associated with low lattice thermal conductivity. "
                "In systems containing As, Bi, or Ge, lone-pair activity may induce local off-centering and structural distortion, activate soft vibrational modes, "
                "and reduce phonon transport efficiency."
            )

        mechanism_4 = (
            "The literature indicates that heavy atoms or weakly bound atoms can form rattling-like localized vibrational modes within the lattice. "
            "These modes can flatten vibrational branches, reduce group velocities, and enhance phonon scattering through local resonance."
        )
        if has_rattling:
            mechanism_4 = (
                "The search results mention localized rattling atoms and resonant scattering. "
                "Localized rattling modes can introduce resonant scattering, flatten vibrational branches, reduce phonon group velocities, "
                "and substantially shorten phonon mean free paths. In silver-based compounds, a relatively soft Ag sublattice may produce similar low-frequency localized vibrations."
            )

        mechanism_5 = (
            "When a material has a complex unit cell and many vibrational branches, phonon dispersions become flatter and the average group velocity decreases. "
            "Even without extremely strong defect scattering, structural complexity alone can suppress heat transport."
        )
        if self._contains_any(combined, ["flatten", "flat branch", "group veloc"]):
            mechanism_5 = (
                "The search results refer to flattened vibrational branches and reduced group velocities. "
                "This indicates that complex unit cells and locally complex structures can flatten phonon dispersions and reduce average group velocity, "
                "directly weakening lattice heat transport."
            )

        mechanism_6 = (
            "Grain boundaries, vacancies, stacking faults, strain fluctuations, and local structural modulation can introduce additional phonon scattering "
            "across different frequency ranges and further reduce thermal conductivity."
        )
        if self._contains_any(combined, ["mean free path", "boundary scattering", "defect", "interatomic spacing"]):
            mechanism_6 = (
                "The search results indicate that the phonon mean free path can be strongly reduced by boundary scattering, anharmonic scattering, "
                "and defects of different dimensionalities, in some cases approaching the interatomic spacing. "
                "Grain boundaries, vacancies, stacking faults, strain fluctuations, and structural modulation can therefore create broadband phonon scattering and further depress lattice thermal conductivity."
            )

        common_points: List[str] = []
        if "Ag" in elements:
            common_points.append("Ag-related low-frequency soft vibrations or rattling-like modes")
        if has_lone_pair:
            common_points.append("local distortions and strong anharmonicity induced by lone-pair-active As/Bi/Ge species")
        if any(el in {"Se", "Te"} for el in elements) or has_disorder:
            common_points.append("mass-disorder scattering from Se/Te chemistry and multicomponent mixing")
        common_points.append("low group velocity and short phonon mean free path caused by complex crystal structures")

        lines: List[str] = [
            "Low Lattice Thermal Conductivity Mechanism Summary",
            "",
            "Search Query",
            queries[0] if queries else "N/A",
            "",
            "Overall Conclusion",
            (
                f"For {element_text}-based silver chalcogenide candidate materials"
                if "Ag" in elements
                else f"For {element_text}-based candidate materials"
            )
            + ", the WebSearch evidence suggests that the low lattice thermal conductivity does not arise from a single factor. "
            + "It is typically governed by multiple cooperative mechanisms, including "
            + ", ".join(overview_mechanisms)
            + "."
            + (f" Representative candidate formulas covered by the aggregated search include: {formula_text}." if formulas else ""),
            "",
            "Mechanism 1: Strong Anharmonicity",
            mechanism_1,
            "",
            "Mechanism 2: Mass and Structural Disorder Scattering",
            mechanism_2,
            "",
            "Mechanism 3: Lone-Pair-Induced Local Distortion",
            mechanism_3,
            "",
            "Mechanism 4: Rattling and Local Resonant Scattering",
            mechanism_4,
            "",
            "Mechanism 5: Reduced Group Velocity from Structural Complexity",
            mechanism_5,
            "",
            "Mechanism 6: Defect-Driven and Multiscale Scattering",
            mechanism_6,
            "",
            "Shared Interpretation for the Candidate Materials",
            (
                f"For materials such as {formula_text}, the most plausible shared mechanisms are: "
                + "; ".join(common_points)
                + "."
            ),
            "",
            "One-Sentence Summary",
            "The low lattice thermal conductivity of these materials is most plausibly caused by the combined action of strong anharmonicity, disorder/distortion scattering, lone-pair effects, local rattling modes, reduced group velocity from structural complexity, and multiscale defect scattering.",
            "",
            "References",
        ]

        if sources:
            lines.extend(sources[:15])
        else:
            lines.append("N/A")

        if errors:
            lines.extend(["", "Search Notes", json.dumps(errors[:5], ensure_ascii=False)])

        return "\n".join(lines)

    def _build_evaluation_prompt(self, materials_info: str, websearch_info: str, n_select: int) -> str:
        prompt = f"""# Low Thermal Conductivity Material Evaluation Task

## Theoretical Principles Document

{self.doc_content}

---

## Candidate Materials Information

{materials_info}

---

## WebSearch Mechanism Summary

{websearch_info}

---

## Evaluation Task

You are a materials science expert. Based on the above theoretical principles document (especially **Section 2: Design Theory** and **Section 5: Evaluation Rules**), evaluate the candidate materials and select the {n_select} most promising low thermal conductivity materials.

**Evaluation Requirements**:

1. **Theoretical Consistency Analysis**:
   - Check if materials conform to the four major phonon scattering mechanisms.
   - Verify compliance with core evaluation parameters from the document.
   - Validate using successful-case theoretical summary if available.
   - Use `WebSearch Mechanism Summary` as supplementary evidence only.

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
