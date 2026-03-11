from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
from pydantic import ValidationError

from agents.screening_agent import enrich_topn_with_websearch, rank_by_ei
from schemas import WorkflowInput
from utils.param_sheet import persist_param_values
from workflow.agno_state import AGNO_SESSION_STATE_DEFAULT, AGNO_STATE_KEYS
from workflow.step_ai_evaluation import step_ai_evaluation
from workflow.step_bayesian_optimization import step_bayesian_optimization
from workflow.step_extract_materials import step_extract_materials
from workflow.step_merge_results import step_merge_results
from workflow.step_structure_calculation import step_structure_calculation
from workflow.step_train_model import step_train_model
from workflow.step_update_data_doc import step_update_data_and_doc


def run_train_step(iteration_num: int, config: dict[str, Any], tracker=None) -> dict[str, Any]:
    step_key = "train_model"
    if tracker and tracker.is_step_completed(iteration_num, step_key):
        return {"success": True, "skipped": True}

    result = step_train_model(
        iteration_num=iteration_num,
        data_root=config["data_root"],
        models_root=config["models_root"],
        path_config=config.get("path_config"),
    )
    if result.get("success") and tracker:
        tracker.mark_step_completed(iteration_num, step_key)
    return result


def run_bayesian_step(
    iteration_num: int,
    config: dict[str, Any],
    tracker=None,
    initial_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    step_key = "bayesian_optimization"
    if tracker and tracker.is_step_completed(iteration_num, step_key):
        return {"success": True, "skipped": True, "top_materials": [], "all_materials": []}

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
    )
    if result.get("success") and tracker:
        tracker.mark_step_completed(iteration_num, step_key)
    return result


def run_extract_step(iteration_num: int, config: dict[str, Any], tracker=None) -> dict[str, Any]:
    step_key = "success_extraction"
    if tracker and tracker.is_step_completed(iteration_num, step_key):
        return {"success": True, "skipped": True}

    result = step_extract_materials(
        iteration_num=iteration_num,
        k_threshold=config["k_threshold"],
        imag_tol=config.get("phonon_imag_tol", -0.1),
        results_root=config["results_root"],
    )
    if (result.get("success") or result.get("no_materials")) and tracker:
        tracker.mark_step_completed(iteration_num, step_key)
    return result


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
    }


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

            df = df.copy()
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
            out_df = out_df.sort_values("_summary_kappa", ascending=True)
            out_df = out_df.drop_duplicates(subset=["dedup_key"], keep="first")
            out_df = out_df.drop(columns=["_summary_formula", "_summary_rel_path", "_summary_sid", "_summary_kappa", "dedup_key"])
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
        all_df = all_df.sort_values("_summary_kappa", ascending=True)
        all_df = all_df.drop_duplicates(subset=["dedup_key"], keep="first")
        all_df = all_df.drop(columns=["_summary_formula", "_summary_rel_path", "_summary_sid", "_summary_kappa", "dedup_key"])
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

    novel_pool_ranked = rank_by_ei(bayes_pool)
    novel_top20 = novel_pool_ranked[: config["top_k_bayes"]]
    print(
        f"[screening] input_count={len(bayes_pool)}, "
        f"novel_count={len(novel_pool_ranked)}, "
        f"novel_top20_count={len(novel_top20)}"
    )

    websearch_enriched_candidates = enrich_topn_with_websearch(
        candidates=novel_top20,
        top_n=int(config.get("websearch_top_n", 5)),
        enabled=bool(config.get("websearch_enabled", True)),
        strategy=str(config.get("websearch_strategy", "hybrid")),
        queries_per_candidate=int(config.get("websearch_queries_per_candidate", 2)),
        theory_template=config.get("websearch_theory_template"),
    )
    websearch_attempted_candidates = min(len(novel_top20), int(config.get("websearch_top_n", 5)))
    total_queries = sum(len(item.get("websearch_queries", [])) for item in websearch_enriched_candidates[:websearch_attempted_candidates])
    success_queries = sum(int(item.get("websearch_success_count", 0)) for item in websearch_enriched_candidates[:websearch_attempted_candidates])
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

    screen_result = step_ai_evaluation(
        iteration_num=iteration_num,
        candidate_materials=websearch_enriched_candidates,
        n_select=config["top_k_screen"],
        path_config=config.get("path_config"),
        results_root=config["results_root"],
    )

    if screen_result.get("success"):
        screened_top10 = screen_result.get("selected_materials", [])
        screening_mode = "ai_with_websearch"
        if tracker:
            tracker.mark_step_completed(
                iteration_num,
                "ai_evaluation",
                metadata={
                    "selected_materials": len(screened_top10),
                    "report_file": screen_result.get("report_file"),
                },
            )
    else:
        screened_top10 = novel_top20[: config["top_k_screen"]]
        screening_mode = "heuristic_fallback"

    calculate_result = step_structure_calculation(
        iteration_num=iteration_num,
        materials=screened_top10,
        n_structures=config["n_structures"],
        max_workers=config["max_workers"],
        relax_workers=config["relax_workers"],
        phonon_workers=config["phonon_workers"],
        pressure=config["pressure"],
        device=config["device"],
        gpus=config["gpus"],
        results_root=config["results_root"],
        tracker=tracker,
        allow_partial_completion=config.get("allow_partial_structure", False),
        path_config=config.get("path_config"),
        relax_timeout_sec=config.get("relax_timeout_sec", 120),
    )
    if not calculate_result.get("success"):
        return {
            "success": False,
            "failed_step": "calculation",
            "calculate": calculate_result,
            "top20": novel_top20,
            "top10": screened_top10,
        }

    merge_result = step_merge_results(iteration_num=iteration_num, results_root=config["results_root"], tracker=tracker)
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

    materials_summary = _aggregate_materials_from_results(config["results_root"])
    print(f"[summary] summary files: {materials_summary}")

    theory_result = step_update_data_and_doc(
        iteration_num=iteration_num,
        extraction_result=extract_result,
        version=config.get("version", 1),
        path_config=config.get("path_config"),
        data_root=config.get("data_root", "llm/data"),
        results_root=config.get("results_root", "llm/results"),
        doc_root=config.get("doc_root", "llm/doc"),
        skip_doc_update=config.get("skip_doc_update", False),
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
        }

    if tracker:
        tracker.mark_step_completed(
            iteration_num,
            "document_update",
            metadata={
                "updated_data_path": theory_result.get("updated_data_path"),
                "updated_doc_path": theory_result.get("updated_doc_path"),
            },
        )

    return {
        "success": True,
        "iteration_num": iteration_num,
        "top20": novel_top20,
        "top10": screened_top10,
        "screening_mode": screening_mode,
        "bayes_pool_size": len(bayes_pool),
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
    }


def build_aslk_steps(
    config: dict[str, Any],
    tracker,
    start_iteration: int,
    max_iterations: int,
    initial_samples: list[dict[str, Any]] | None = None,
) -> list[Step]:
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
        if requested_iterations is not None:
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
            result = _run_single_iteration(iteration, run_config, tracker, local_samples)
            all_results.append(result)
            compact_result = _compact_iteration_result(result)
            compact_all_results.append(compact_result)

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
