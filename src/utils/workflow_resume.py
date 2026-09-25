from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


THEORY_DOC_NAME = "Theoretical_principle_document.md"
THERMAL_COLUMN = "Kappa_Slack (W m-1 K-1)"


def file_sha256(path: str | Path) -> str:
    """Return a stable fingerprint for a persisted workflow input/output."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WorkflowProgressInconsistencyError(RuntimeError):
    """Raised when later iteration state exists behind an incomplete predecessor."""



def get_contiguous_completed_rounds(tracker) -> tuple[list[int], int]:
    """Return all completed rounds and the end of the contiguous prefix.

    A later round may exist in an old or manually interrupted progress file
    while an earlier round is incomplete. Such an orphan round must not move
    the resume cursor past the gap.
    """
    completed = sorted(set(tracker.get_completed_rounds()))
    completed_set = set(completed)
    contiguous_last = 0
    while contiguous_last + 1 in completed_set:
        contiguous_last += 1
    return completed, contiguous_last


def _load_llm_evaluation_payload(path: str | Path) -> dict[str, Any] | None:
    report_path = Path(path)
    if not report_path.exists():
        return None
    try:
        text = report_path.read_text(encoding="utf-8")
        _, separator, payload = text.partition("---")
        if not separator:
            return None
        payload = payload.strip()
        fenced = payload.startswith("```") and payload.endswith("```")
        if fenced:
            payload_lines = payload.splitlines()
            if len(payload_lines) < 3:
                return None
            payload = "\n".join(payload_lines[1:-1]).strip()
        data = json.loads(payload)
        return data if isinstance(data, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _is_valid_evaluation_payload(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False

    selected_rows = payload.get("selected_materials")
    score_rows = payload.get("candidate_scores")
    if selected_rows is not None:
        if not isinstance(selected_rows, list) or not selected_rows:
            return False
        seen_formulas: set[str] = set()
        for index, row in enumerate(selected_rows, start=1):
            if not isinstance(row, dict):
                return False
            formula = str(row.get("formula") or "").strip()
            final_rank = row.get("final_rank")
            if not formula or formula in seen_formulas:
                return False
            if type(final_rank) is not int or final_rank != index:
                return False
            seen_formulas.add(formula)
        return True

    if not isinstance(score_rows, list) or not score_rows:
        return False
    seen_formulas: set[str] = set()
    seen_ranks: set[int] = set()
    identity_score_mode = any(isinstance(row, dict) and row.get("candidate_id") for row in score_rows)
    score_names = (
        "chemical_plausibility_score",
        "low_kappa_mechanism_score",
        "stability_risk_score",
    ) if identity_score_mode else (
        "mechanism_fit_score",
        "stability_risk_score",
        "novelty_bonus_score",
        "bo_override_confidence",
    )
    seen_ids: set[str] = set()
    for row in score_rows:
        if not isinstance(row, dict):
            return False
        formula = str(row.get("formula") or "").strip()
        rank = row.get("original_rank")
        candidate_id = str(row.get("candidate_id") or "").strip()
        if identity_score_mode and not candidate_id:
            return False
        if candidate_id:
            if candidate_id in seen_ids:
                return False
            seen_ids.add(candidate_id)
        if not formula and not candidate_id:
            return False
        if formula and formula in seen_formulas:
            return False
        if (not candidate_id) and (type(rank) is not int or rank < 1 or rank in seen_ranks):
            return False
        if any(type(row.get(name)) is not int or not 0 <= row[name] <= 10 for name in score_names):
            return False
        seen_formulas.add(formula)
        seen_ranks.add(rank)
    return True


def is_valid_llm_evaluation_report(path: str | Path) -> bool:
    """只接受完整且符合字段合同的 evaluation 报告。"""
    return _is_valid_evaluation_payload(_load_llm_evaluation_payload(path))


def _is_valid_thermal_csv(path: Path, expected_cifs: set[str] | None = None) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError):
        return False
    if frame.empty or THERMAL_COLUMN not in frame.columns:
        return False
    values = frame[THERMAL_COLUMN].dropna().astype(str).str.strip()
    if values.empty or not values.ne("").any():
        return False
    if expected_cifs:
        if "CIF_File" not in frame.columns:
            return False
        written = set(frame["CIF_File"].dropna().astype(str).str.strip())
        return expected_cifs.issubset(written)
    return True


def _thermal_cif_names(path: Path) -> set[str]:
    """Return CIF names represented in a thermal CSV, if any."""
    if not path.exists():
        return set()
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError):
        return set()
    if "CIF_File" not in frame.columns:
        return set()
    return {
        str(value).strip()
        for value in frame["CIF_File"].dropna().tolist()
        if str(value).strip()
    }


def _load_dedup_status(path: Path) -> dict[str, Any] | None:
    """Load a completed dedup status used to explain removed relaxed CIFs."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or payload.get("status") != "completed":
        return None
    formulas = payload.get("formulas")
    if not isinstance(formulas, list):
        return None
    return payload


def _dedup_removed_cif_names(status: dict[str, Any] | None, formula: str) -> set[str]:
    if not isinstance(status, dict):
        return set()
    results = status.get("results")
    if not isinstance(results, dict) or not isinstance(results.get(formula), dict):
        return set()
    duplicate_files = results[formula].get("duplicate_files", [])
    if not isinstance(duplicate_files, list):
        return set()
    return {Path(str(value)).name for value in duplicate_files if str(value).strip()}


def _is_oom_text(*values: object) -> bool:
    text = " ".join(str(value or "") for value in values).lower()
    return "out of memory" in text or "cuda oom" in text


def is_valid_theory_document_artifact(
    doc_root: str | Path,
    iteration_num: int,
) -> bool:
    """Validate an updated document or an exact previous-document continuity copy."""
    current_path = Path(doc_root) / f"v0.0.{iteration_num}" / THEORY_DOC_NAME
    previous_path = Path(doc_root) / f"v0.0.{max(iteration_num - 1, 0)}" / THEORY_DOC_NAME
    if not current_path.exists() or not previous_path.exists():
        return False
    try:
        current_text = current_path.read_text(encoding="utf-8")
        previous_text = previous_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    if not current_text.strip() or not previous_text.strip():
        return False
    if current_text == previous_text:
        return True
    try:
        from agents.update_document import validate_updated_theory_document

        validate_updated_theory_document(current_text, previous_text)
        return True
    except Exception:
        return False


def _has_valid_generated_cif_set(
    results_root: str | Path,
    iteration_num: int,
    formula: str,
) -> bool:
    """Return whether every canonical CIF for a material is parseable."""
    processed_root = Path(results_root) / f"iteration_{iteration_num}" / "processed_structures"
    formula_root = processed_root / formula
    processed_dir = formula_root / "processed"
    search_dir = processed_dir if processed_dir.exists() else formula_root
    paths = sorted(search_dir.glob("*.cif")) if search_dir.exists() else []
    if not paths:
        return False
    try:
        from tools.structure_parallel import _filter_valid_generated_cif_outputs

        valid_paths, _ = _filter_valid_generated_cif_outputs(paths, formula)
    except Exception:
        return False
    return len(valid_paths) == len(paths)


def load_active_material_formulas(
    results_root: str | Path,
    iteration_num: int,
) -> set[str] | None:
    """Load the current iteration's selected formulas, if selection artifacts exist."""
    selected_dir = Path(results_root) / f"iteration_{iteration_num}" / "selected_results"
    for filename in ("ai_selected_materials.csv", "bo_selected_materials.csv"):
        path = selected_dir / filename
        frame = _read_nonempty_csv(path)
        if frame is None or "formula" not in frame.columns:
            continue
        formulas = {
            str(value).strip()
            for value in frame["formula"].tolist()
            if str(value).strip()
        }
        # Selection artifacts describe candidates, not necessarily materials
        # that produced valid structures. Generation failures/skips must not
        # enter relaxation, merge, extraction, or downstream data updates.
        manifest_path = selected_dir.parent / "processed_structures" / "generation_status.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entries = manifest.get("materials", {}) if isinstance(manifest, dict) else {}
            if isinstance(entries, dict):
                formulas = {
                    formula
                    for formula in formulas
                    if not (
                        isinstance(entries.get(formula), dict)
                        and (
                            entries[formula].get("status") == "fatal"
                            or (
                                entries[formula].get("status") in {"skipped", "failed"}
                                and not _has_valid_generated_cif_set(results_root, iteration_num, formula)
                            )
                        )
                    )
                }
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            pass

        # A generated material with no successful relaxation+phonon structure
        # has no valid downstream input and must not block merge/extraction.
        relax_root = selected_dir.parent / "MyRelaxStructure"
        for formula in list(formulas):
            log_path = relax_root / formula / "relax_phonon_results.csv"
            if not log_path.exists():
                continue
            try:
                rows = list(pd.read_csv(log_path, encoding="utf-8-sig").to_dict(orient="records"))
            except (OSError, UnicodeError, ValueError, pd.errors.ParserError):
                continue
            has_success = False
            successful_relaxed_cifs: set[str] = set()
            comp_dir = log_path.parent
            latest_rows: dict[str, dict] = {}
            for row in rows:
                source_name = str(row.get("CIF_File") or "").strip()
                if source_name:
                    latest_rows[source_name] = row
            for row in latest_rows.values():
                if str(row.get("Relax_Success") or "").strip().upper() != "Y":
                    continue
                if str(row.get("Phonon_Success") or "").strip().upper() != "Y":
                    continue
                relaxed_value = str(row.get("Relaxed_CIF") or "").strip()
                candidates = [Path(relaxed_value)] if relaxed_value else []
                cif_name = str(row.get("CIF_File") or "").strip()
                if cif_name:
                    candidates.append(comp_dir / cif_name)
                existing = next((candidate for candidate in candidates if candidate.exists()), None)
                if existing is not None:
                    has_success = True
                    successful_relaxed_cifs.add(existing.name)
            if not has_success or not _is_valid_thermal_csv(
                comp_dir / "thermal_conductivity.csv",
                successful_relaxed_cifs,
            ):
                # A material without a valid thermal artifact has no valid
                # downstream input and is terminally excluded from this round.
                formulas.discard(formula)
        return formulas
    return None


def is_valid_structure_artifacts(
    results_root: str | Path,
    iteration_num: int,
    expected_formulas: list[str] | None = None,
) -> bool:
    """Validate generated, relaxation/phonon and thermal artifacts without deleting them."""
    iteration_root = Path(results_root) / f"iteration_{iteration_num}"
    processed_root = iteration_root / "processed_structures"
    relax_root = iteration_root / "MyRelaxStructure"
    if not processed_root.exists() or not relax_root.exists():
        return False

    skipped_formulas: set[str] = set()
    terminal_thermal_formulas: set[str] = set()
    manifest: dict[str, Any] = {}
    manifest_path = processed_root / "generation_status.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                return False
            entries = manifest.get("materials", {})
            if isinstance(entries, dict):
                skipped_formulas = {
                    str(formula)
                    for formula, entry in entries.items()
                    if (
                        isinstance(entry, dict)
                        and (
                            entry.get("status") == "fatal"
                            or (
                                entry.get("status") in {"skipped", "failed"}
                                and not _has_valid_generated_cif_set(
                                    results_root,
                                    iteration_num,
                                    str(formula),
                                )
                            )
                        )
                    )
                }
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False

    thermal_status_path = relax_root / "thermal_status.json"
    if thermal_status_path.exists():
        try:
            thermal_status = json.loads(thermal_status_path.read_text(encoding="utf-8"))
            if isinstance(thermal_status, dict):
                failed_materials = thermal_status.get("failed_materials", [])
                if isinstance(failed_materials, list):
                    terminal_thermal_formulas = {
                        str(item.get("formula") or "").strip()
                        for item in failed_materials
                        if isinstance(item, dict) and str(item.get("formula") or "").strip()
                    }
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            return False

    # A valid partial generation can still have every relaxation/phonon task
    # fail for that one material.  That is a terminal material-level skip when
    # other selected materials have valid downstream artifacts; it must not be
    # treated as an unexplained missing formula during resume.  A batch with no
    # successful material is still rejected below because ``formula_dirs``
    # becomes empty.
    terminal_relaxation_formulas: set[str] = set()
    selected_formula_candidates = {
        str(formula).strip()
        for formula in (expected_formulas or [])
        if str(formula).strip()
    }
    if not selected_formula_candidates:
        for filename in ("ai_selected_materials.csv", "bo_selected_materials.csv"):
            frame = _read_nonempty_csv(iteration_root / "selected_results" / filename)
            if frame is not None and "formula" in frame.columns:
                selected_formula_candidates.update(
                    str(value).strip()
                    for value in frame["formula"].tolist()
                    if str(value).strip()
                )
    for formula in selected_formula_candidates:
        if not _has_valid_generated_cif_set(results_root, iteration_num, formula):
            continue
        log_path = relax_root / formula / "relax_phonon_results.csv"
        if not log_path.exists():
            continue
        try:
            rows = pd.read_csv(log_path, encoding="utf-8-sig").to_dict(orient="records")
        except (OSError, UnicodeError, ValueError, pd.errors.ParserError):
            continue
        generated_dir = processed_root / formula
        search_dir = generated_dir / "processed" if (generated_dir / "processed").exists() else generated_dir
        generated_names = {path.name for path in search_dir.glob("*.cif")}
        latest_rows = {
            str(row.get("CIF_File") or "").strip(): row
            for row in rows
            if str(row.get("CIF_File") or "").strip()
        }
        if not generated_names or set(latest_rows) != generated_names:
            continue
        has_success = any(
            str(row.get("Relax_Success") or "").strip().upper() == "Y"
            and str(row.get("Phonon_Success") or "").strip().upper() == "Y"
            for row in latest_rows.values()
        )
        if not has_success:
            terminal_relaxation_formulas.add(formula)

    excluded_formulas = (
        skipped_formulas
        | terminal_thermal_formulas
        | terminal_relaxation_formulas
    )
    if expected_formulas:
        expected_set = {str(formula) for formula in expected_formulas}
        # A selected batch with no valid relaxed/phonon artifact is not a
        # completed structure step.  Explicit generation skips are handled at
        # the material level only when another selected material has a valid
        # artifact; an all-excluded batch must be rebuilt or stopped.
        if expected_set and expected_set.issubset(excluded_formulas):
            # Generation-level skips may still have valid partial CIFs and
            # complete downstream artifacts; validate those directories below.
            # A batch excluded by thermal/relaxation failure has no valid
            # downstream input and must remain incomplete.
            if expected_set.issubset(skipped_formulas) and not (
                expected_set & terminal_thermal_formulas
            ):
                # A generation skip can be a partial-generation status rather
                # than an empty material.  Accept it only when its persisted
                # relax/phonon log and thermal CSV prove usable output.
                for formula in expected_formulas:
                    relax_dir = relax_root / str(formula)
                    log_path = relax_dir / "relax_phonon_results.csv"
                    if not relax_dir.exists() or not _is_valid_thermal_csv(
                        relax_dir / "thermal_conductivity.csv"
                    ):
                        return False
                    try:
                        rows = pd.read_csv(log_path, encoding="utf-8-sig").to_dict(orient="records")
                    except (OSError, UnicodeError, ValueError, pd.errors.ParserError):
                        return False
                    if not rows or not any(
                        str(row.get("Relax_Success") or "").strip().upper() == "Y"
                        and str(row.get("Phonon_Success") or "").strip().upper() == "Y"
                        for row in rows
                    ):
                        return False
                return True
            return False
        else:
            formula_dirs = [
                processed_root / formula
                for formula in expected_formulas
                if str(formula) not in excluded_formulas
            ]
    else:
        active_formulas = load_active_material_formulas(results_root, iteration_num)
        if active_formulas is not None:
            selected_formulas = set(active_formulas)
            for filename in ("ai_selected_materials.csv", "bo_selected_materials.csv"):
                frame = _read_nonempty_csv(
                    iteration_root / "selected_results" / filename
                )
                if frame is not None and "formula" in frame.columns:
                    selected_formulas.update(
                        str(value).strip()
                        for value in frame["formula"].tolist()
                        if str(value).strip()
                    )
            # A selected material can be absent from active_formulas only when
            # it was explicitly skipped/failed during generation or thermal
            # calculation.  Without such a terminal record, a missing thermal
            # artifact must not be silently treated as a valid completed step.
            unaccounted_formulas = selected_formulas - active_formulas - excluded_formulas
            if unaccounted_formulas:
                return False
            formula_dirs = [
                processed_root / formula
                for formula in sorted(active_formulas)
                if formula not in excluded_formulas
            ]
        else:
            formula_dirs = [
                path for path in processed_root.iterdir()
                if path.is_dir()
                and not path.name.startswith(".")
                and path.name not in excluded_formulas
            ]
    if not formula_dirs:
        return False

    for generated_dir in formula_dirs:
        if not generated_dir.exists():
            return False
        generated_cifs = list((generated_dir / "processed").glob("*.cif")) if (generated_dir / "processed").exists() else list(generated_dir.glob("*.cif"))
        if not generated_cifs:
            return False
        formula = generated_dir.name
        try:
            from tools.structure_parallel import _filter_valid_generated_cif_outputs

            valid_cifs, _ = _filter_valid_generated_cif_outputs(generated_cifs, formula)
        except (ImportError, TypeError, ValueError):
            return False
        if not valid_cifs or len(valid_cifs) != len(generated_cifs):
            return False
        relax_dir = relax_root / formula
        log_path = relax_dir / "relax_phonon_results.csv"
        if not relax_dir.exists() or not log_path.exists():
            return False
        try:
            rows = list(pd.read_csv(log_path, encoding="utf-8-sig").to_dict(orient="records"))
        except (OSError, UnicodeError, ValueError, pd.errors.ParserError):
            return False
        if not rows:
            return False

        generated_names = {path.name for path in generated_cifs}
        latest_rows: dict[str, dict] = {}
        for row in rows:
            source_name = str(row.get("CIF_File") or "").strip()
            if source_name in generated_names:
                latest_rows[source_name] = row
        processed_names = set(latest_rows)
        completed_relaxed: set[str] = set()
        missing_after_dedup: set[str] = set()
        dedup_status = _load_dedup_status(relax_root / "deduplication_status.json")
        dedup_formulas = {
            str(value).strip()
            for value in (dedup_status or {}).get("formulas", [])
            if str(value).strip()
        }
        dedup_completed = formula in dedup_formulas
        removed_names = _dedup_removed_cif_names(dedup_status, formula)
        thermal_path = relax_dir / "thermal_conductivity.csv"

        for source_name, row in latest_rows.items():
            relax_ok = str(row.get("Relax_Success") or "").strip().upper() == "Y"
            phonon_ok = str(row.get("Phonon_Success") or "").strip().upper() == "Y"
            if not (phonon_ok and relax_ok):
                # OOM, worker crash, timeout, and other per-structure failures
                # are explicit terminal skips once their latest row is persisted.
                continue
            relaxed_value = str(row.get("Relaxed_CIF") or "").strip()
            candidates = [Path(relaxed_value)] if relaxed_value else []
            candidates.append(relax_dir / source_name)
            resolved_candidates = [
                path if path.is_absolute() else relax_dir / path
                for path in candidates
            ]
            relaxed_path = next(
                (path for path in resolved_candidates if path.exists()),
                None,
            )
            if relaxed_path is not None:
                completed_relaxed.add(relaxed_path.name)
            elif dedup_completed and source_name in removed_names:
                # Deduplication intentionally removed this successful source
                # CIF; its representative remains in the thermal CSV.
                missing_after_dedup.add(source_name)
            elif dedup_completed:
                # Legacy status files did not persist duplicate mappings.  A
                # successful row absent from disk is accepted only when the
                # thermal CSV is valid and does not claim that same CIF.
                missing_after_dedup.add(source_name)
            else:
                # A claimed success without a persisted relaxed CIF and without
                # completed deduplication is not a resumable final artifact.
                return False

        if processed_names != generated_names:
            return False
        if not completed_relaxed:
            # Every generated CIF may have a persisted failure row, but that
            # is not a completed structure contract without at least one
            # valid relaxation+phonon artifact.
            return False
        if not _is_valid_thermal_csv(thermal_path, completed_relaxed):
            return False
        if missing_after_dedup:
            thermal_names = _thermal_cif_names(thermal_path)
            if not completed_relaxed or not _is_valid_thermal_csv(thermal_path, completed_relaxed):
                return False
            if missing_after_dedup & thermal_names:
                return False
            if removed_names and not missing_after_dedup.issubset(removed_names):
                return False

    return True


def _read_nonempty_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    return frame if not frame.empty else None


def _is_valid_dataset_artifact(path: Path) -> bool:
    frame = _read_nonempty_csv(path)
    return frame is not None


def _is_valid_model_artifacts(models_root: Path, model_iteration: int) -> bool:
    model_dir = models_root / f"iteration_{model_iteration}"
    return all(
        path.exists() and path.stat().st_size > 0
        for path in (
            model_dir / "gpr_thermal_conductivity.joblib",
            model_dir / "gpr_scaler.joblib",
        )
    )


def _is_valid_merge_artifacts(results_root: Path, iteration_num: int) -> bool:
    active_formulas = load_active_material_formulas(results_root, iteration_num)
    if active_formulas is None or not active_formulas:
        return False
    relax_root = results_root / f"iteration_{iteration_num}" / "MyRelaxStructure"
    for formula in active_formulas:
        comp_dir = relax_root / formula
        thermal_csv = comp_dir / "thermal_conductivity.csv"
        if not comp_dir.exists() or not _is_valid_thermal_csv(thermal_csv):
            return False
        try:
            frame = pd.read_csv(thermal_csv, encoding="utf-8-sig")
        except (OSError, UnicodeError, ValueError, pd.errors.ParserError):
            return False
        if "Phonon_Success" not in frame.columns:
            return False
    return True


def load_saved_bayesian_result(results_root: str | Path, iteration_num: int, n_top: int) -> dict[str, Any] | None:
    base_dir = Path(results_root) / f"iteration_{iteration_num}" / "selected_results"
    all_samples_path = base_dir / "all_samples.csv"
    df = _read_nonempty_csv(all_samples_path)
    if df is None or "formula" not in df.columns:
        return None

    rows = df.to_dict(orient="records")
    metadata: dict[str, Any] = {}
    top_path = base_dir / f"top{n_top}_materials.json"
    if top_path.exists():
        try:
            payload = json.loads(top_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                metadata = {
                    key: payload[key]
                    for key in (
                        "sampling_status",
                        "sampler_generated_count",
                        "candidate_pool_count",
                        "candidate_dedup_removed",
                        "raw_sample_count",
                        "selected_count",
                        "raw_samples_file",
                    )
                    if key in payload
                }
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            metadata = {}

    return {
        "success": True,
        "skipped": True,
        "n_materials": len(rows),
        "n_top": n_top,
        "top_materials": rows[:n_top],
        "top10_materials": rows[:10],
        "all_materials": rows,
        "artifact_path": str(all_samples_path),
        **metadata,
    }


def load_saved_extract_result(results_root: str | Path, iteration_num: int) -> dict[str, Any] | None:
    success_dir = Path(results_root) / f"iteration_{iteration_num}" / "success_examples"
    if not success_dir.exists():
        return None

    success_file = success_dir / "success_materials.csv"
    stable_file = success_dir / "stable_materials.csv"
    success_deduped_file = success_dir / "success_materials_deduped.csv"
    stable_deduped_file = success_dir / "stable_materials_deduped.csv"
    novelty_csv = success_dir / "final_materials_db_novelty.csv"
    novelty_json = success_dir / "final_materials_db_novelty.json"
    novelty_summary = success_dir / "final_materials_db_novelty_summary.md"
    status_path = success_dir / "extraction_status.json"

    if status_path.exists():
        try:
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            status_payload = None
        if isinstance(status_payload, dict) and status_payload.get("status") == "no_materials":
            return {
                "success": True,
                "skipped": True,
                "no_materials": True,
                "has_success": False,
                "has_stable": False,
                "extraction_status": str(status_path),
            }

    valid_success_file = next(
        (path for path in (success_deduped_file, success_file) if _read_nonempty_csv(path) is not None),
        None,
    )
    valid_stable_file = next(
        (path for path in (stable_deduped_file, stable_file) if _read_nonempty_csv(path) is not None),
        None,
    )
    has_success = valid_success_file is not None
    has_stable = valid_stable_file is not None
    if not has_success and not has_stable:
        return None

    return {
        "success": True,
        "skipped": True,
        "has_success": has_success,
        "has_stable": has_stable,
        "success_file": str(success_file) if valid_success_file is not None and success_file.exists() else None,
        "stable_file": str(stable_file) if valid_stable_file is not None and stable_file.exists() else None,
        "success_deduped_file": str(success_deduped_file) if valid_success_file == success_deduped_file else None,
        "stable_deduped_file": str(stable_deduped_file) if valid_stable_file == stable_deduped_file else None,
        "final_db_novelty_file": str(novelty_csv) if novelty_csv.exists() else None,
        "final_db_novelty_json": str(novelty_json) if novelty_json.exists() else None,
        "final_db_novelty_summary_file": str(novelty_summary) if novelty_summary.exists() else None,
        "extraction_status": str(status_path) if status_path.exists() else None,
        "novelty_summary": {},
    }


def load_saved_ai_evaluation_result(results_root: str | Path, iteration_num: int) -> dict[str, Any] | None:
    base_dir = Path(results_root) / f"iteration_{iteration_num}" / "selected_results"
    selected_path = base_dir / "ai_selected_materials.csv"
    trace_path = base_dir / "selection_trace.json"
    reports_dir = Path(results_root) / f"iteration_{iteration_num}" / "reports"
    evaluation_report = reports_dir / "llm_evaluation_output.md"
    candidate_report = reports_dir / "llm_candidate_scoring_output.md"
    # Prefer the identity-based candidate-score artifact when both reports are
    # present (for example after migrating an iteration from the legacy
    # formula/rank evaluator).  Falling back to the legacy report preserves old
    # completed runs.
    report_candidates = [path for path in (candidate_report, evaluation_report) if path.exists()]
    report_path = None
    payload = None
    for candidate_path in report_candidates:
        candidate_payload = _load_llm_evaluation_payload(candidate_path)
        if _is_valid_evaluation_payload(candidate_payload):
            report_path = candidate_path
            payload = candidate_payload
            break
    if report_path is None or not _is_valid_evaluation_payload(payload):
        return None

    df = _read_nonempty_csv(selected_path)
    if df is None or "formula" not in df.columns or "final_rank" not in df.columns:
        return None
    csv_formulas = [str(value).strip() for value in df["formula"].tolist()]
    if not csv_formulas or len(csv_formulas) != len(set(csv_formulas)):
        return None
    if list(df["final_rank"]) != list(range(1, len(df) + 1)):
        return None

    selected_rows = payload.get("selected_materials") if isinstance(payload, dict) else None
    if isinstance(selected_rows, list):
        report_formulas = [str(row.get("formula") or "").strip() for row in selected_rows]
        report_ranks = [row.get("final_rank") for row in selected_rows]
        if report_formulas != csv_formulas or report_ranks != list(range(1, len(df) + 1)):
            return None

    if not trace_path.exists():
        return None
    try:
        trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(trace_payload, dict):
        return None
    trace_rows = trace_payload.get("rows")
    if not isinstance(trace_rows, list) or not all(isinstance(row, dict) for row in trace_rows):
        return None
    trace_selected = trace_payload.get("selected_formulas")
    if trace_selected is not None and list(trace_selected) != csv_formulas:
        return None
    selected_trace_formulas = [
        str(row.get("formula") or "").strip()
        for row in trace_rows
        if row.get("selected_for_calc") is True
    ]
    # The trace preserves the original BO order, while the selected-materials
    # CSV is written in final-rank order. Validate membership, not ordering.
    if selected_trace_formulas and (
        len(selected_trace_formulas) != len(csv_formulas)
        or set(selected_trace_formulas) != set(csv_formulas)
    ):
        return None

    return {
        "success": True,
        "skipped": True,
        "n_selected": len(df),
        "selected_materials": df.to_dict(orient="records"),
        "csv_path": str(selected_path),
        "trace_path": str(trace_path),
        "selection_trace": trace_rows,
        "diversity_audit": trace_payload.get("selection_audit", {}),
        "screening_mode": trace_payload.get("screening_mode"),
        "report_path": str(report_path),
    }


def load_saved_document_update_result(
    results_root: str | Path,
    data_root: str | Path,
    doc_root: str | Path,
    iteration_num: int,
) -> dict[str, Any] | None:
    data_path = Path(data_root) / f"iteration_{iteration_num}" / "data.csv"
    doc_path = Path(doc_root) / f"v0.0.{iteration_num}" / THEORY_DOC_NAME
    if _read_nonempty_csv(data_path) is None:
        return None
    if not is_valid_theory_document_artifact(doc_root, iteration_num):
        return None

    return {
        "success": True,
        "skipped": True,
        "has_success_materials": True,
        "updated_data_path": str(data_path),
        "updated_doc_path": str(doc_path),
    }


def reset_steps_from(tracker, iteration_num: int, start_step: str) -> list[str]:
    reset_steps: list[str] = []
    should_reset = False
    round_data = tracker.progress.get(f"iteration_{iteration_num}", {})
    if not isinstance(round_data, dict):
        return reset_steps
    for step in tracker.steps:
        if step == start_step:
            should_reset = True
        if not should_reset:
            continue
        if tracker.progress.get(f"iteration_{iteration_num}", {}).get(step) is None:
            continue
        tracker.reset_step(iteration_num, step)
        reset_steps.append(step)
    return reset_steps


def reconcile_progress_with_filesystem(
    tracker,
    path_config,
    *,
    allow_inconsistent_history: bool = False,
) -> list[str]:
    """Reconcile progress without rerunning valid completed work.

    A valid final artifact can backfill a missing completion marker. An invalid
    completed marker is reset from that step onward. In normal recovery, later
    iteration state behind an incomplete predecessor is a hard progress
    inconsistency: it is diagnosed and execution stops without modifying that
    later state. ``allow_inconsistent_history`` is reserved for an explicit
    user-requested ``--reset``.
    """
    messages: list[str] = []
    results_root = Path(path_config.results_root)
    models_root = Path(path_config.models_root)
    doc_root = Path(path_config.doc_root) if path_config.doc_root else None

    iteration_numbers: set[int] = set()
    for key in tracker.progress.keys():
        if not key.startswith("iteration_"):
            continue
        try:
            iteration_numbers.add(int(key.split("_", 1)[1]))
        except Exception:
            continue
    # A process can stop after writing final artifacts but before persisting
    # the completion marker. Include artifact directories so they can be
    # validated and backfilled instead of rerunning the step.
    for path in results_root.glob("iteration_*") if results_root.exists() else []:
        if not path.is_dir():
            continue
        try:
            iteration_numbers.add(int(path.name.split("_", 1)[1]))
        except (ValueError, IndexError):
            continue
    # Model iteration M is produced by workflow iteration M+1.
    for path in models_root.glob("iteration_*") if models_root.exists() else []:
        if not path.is_dir():
            continue
        try:
            iteration_numbers.add(int(path.name.split("_", 1)[1]) + 1)
        except (ValueError, IndexError):
            continue

    # Do not check later-round consistency before validating earlier rounds.
    # A predecessor may have a complete final artifact but a missing progress
    # marker; the sequential pass below must be allowed to backfill it first.
    # Once each earlier round has been reconciled, the same pass raises at the
    # first genuinely orphaned later-round state without deleting it.

    for iteration_num in sorted(iteration_numbers):
        if iteration_num < 1:
            continue
        key = f"iteration_{iteration_num}"
        predecessor_key = f"iteration_{iteration_num - 1}"
        if iteration_num > 1 and not tracker.is_round_completed(iteration_num - 1):
            round_data = tracker.progress.get(key, {})
            has_state = isinstance(round_data, dict) and any(
                isinstance(round_data.get(step), dict) for step in tracker.steps
            )
            has_artifacts = has_state
            if not allow_inconsistent_history and has_state:
                raise WorkflowProgressInconsistencyError(
                    f"Inconsistent iteration history: {key} has later state/artifacts while "
                    f"predecessor iteration_{iteration_num - 1} is incomplete"
                )
            if has_state and tracker.steps:
                reset_steps = reset_steps_from(tracker, iteration_num, tracker.steps[0])
                messages.append(
                    f"iteration_{iteration_num}: reset {reset_steps} because predecessor "
                    f"iteration_{iteration_num - 1} is incomplete; later iteration is not a valid resume point"
                )
            elif (
                (results_root / key).exists()
                or (Path(path_config.data_root) / key).exists()
                or (doc_root is not None and (doc_root / f"v0.0.{iteration_num}").exists())
            ):
                messages.append(
                    f"iteration_{iteration_num}: ignored historical filesystem artifacts because "
                    f"predecessor iteration_{iteration_num - 1} is incomplete"
                )
            continue

        model_iteration = max(iteration_num - 1, 0)
        model_candidates = [
            models_root / f"iteration_{model_iteration}" / "gpr_thermal_conductivity.joblib",
            models_root / f"iteration_{iteration_num}" / "gpr_thermal_conductivity.joblib",
        ]
        all_samples_path = results_root / f"iteration_{iteration_num}" / "selected_results" / "all_samples.csv"
        training_data_path = Path(path_config.data_root) / f"iteration_{model_iteration}" / "data.csv"
        data_path = Path(path_config.data_root) / f"iteration_{iteration_num}" / "data.csv"
        versioned_doc = doc_root / f"v0.0.{iteration_num}" / THEORY_DOC_NAME if doc_root else None

        def _current_round() -> dict[str, Any]:
            value = tracker.progress.get(key, {})
            return value if isinstance(value, dict) else {}

        def step_completed(step_name: str) -> bool:
            step_data = _current_round().get(step_name)
            return isinstance(step_data, dict) and bool(step_data.get("completed"))

        def step_in_tracker(step_name: str) -> bool:
            return step_name in tracker.steps

        def previous_steps_completed(step_name: str) -> bool:
            try:
                index = tracker.steps.index(step_name)
            except ValueError:
                return False
            return all(step_completed(previous) for previous in tracker.steps[:index])

        # Train: preserve legacy completed markers with the old model layout,
        # but require the immediate input dataset. New markers require both
        # model and scaler artifacts before they are backfilled.
        model_valid = any(path.exists() for path in model_candidates)
        strict_model_valid = _is_valid_model_artifacts(models_root, model_iteration)
        train_data_metadata = _current_round().get("train_model", {})
        train_data_metadata = (
            train_data_metadata.get("metadata", {})
            if isinstance(train_data_metadata, dict)
            else {}
        )
        recorded_training_hash = (
            train_data_metadata.get("training_data_sha256")
            if isinstance(train_data_metadata, dict)
            else None
        )
        training_hash_changed = False
        if recorded_training_hash and training_data_path.exists():
            try:
                training_hash_changed = file_sha256(training_data_path) != recorded_training_hash
            except OSError:
                training_hash_changed = True
        if step_completed("train_model"):
            if not training_data_path.exists() or not model_valid or training_hash_changed:
                reset_steps = reset_steps_from(tracker, iteration_num, "train_model")
                if not training_data_path.exists():
                    reason = "required training dataset is missing"
                elif training_hash_changed:
                    reason = "training dataset fingerprint changed"
                else:
                    reason = "model artifact is missing"
                messages.append(f"iteration_{iteration_num}: reset {reset_steps} because {reason}")
                continue
        elif step_in_tracker("train_model") and training_data_path.exists() and strict_model_valid:
            tracker.mark_step_completed(
                iteration_num,
                "train_model",
                metadata={
                    "reconciled": True,
                    "training_data": str(training_data_path),
                    "training_data_sha256": file_sha256(training_data_path),
                },
            )
            messages.append(f"iteration_{iteration_num}: marked train_model completed from valid artifacts")

        bayes_frame = _read_nonempty_csv(all_samples_path)
        bayes_valid = bayes_frame is not None and "formula" in bayes_frame.columns
        if step_completed("bayesian_optimization"):
            if not bayes_valid:
                reset_steps = reset_steps_from(tracker, iteration_num, "bayesian_optimization")
                messages.append(f"iteration_{iteration_num}: reset {reset_steps} because BO artifacts are missing or invalid")
                continue
        elif (
            step_in_tracker("bayesian_optimization")
            and previous_steps_completed("bayesian_optimization")
            and bayes_valid
        ):
            tracker.mark_step_completed(
                iteration_num,
                "bayesian_optimization",
                metadata={"reconciled": True, "all_samples": str(all_samples_path)},
            )
            messages.append(f"iteration_{iteration_num}: marked bayesian_optimization completed from valid artifacts")

        if step_in_tracker("ai_evaluation"):
            ai_valid = load_saved_ai_evaluation_result(results_root, iteration_num) is not None
            if step_completed("ai_evaluation"):
                if not ai_valid:
                    reset_steps = reset_steps_from(tracker, iteration_num, "ai_evaluation")
                    messages.append(f"iteration_{iteration_num}: reset {reset_steps} because AI screening artifacts are missing or invalid")
                    continue
            elif previous_steps_completed("ai_evaluation") and ai_valid:
                tracker.mark_step_completed(
                    iteration_num,
                    "ai_evaluation",
                    metadata={"reconciled": True},
                )
                messages.append(f"iteration_{iteration_num}: marked ai_evaluation completed from valid artifacts")

        if step_in_tracker("structure_calculation"):
            structure_valid = is_valid_structure_artifacts(results_root, iteration_num)
            if step_completed("structure_calculation"):
                if not structure_valid:
                    reset_steps = reset_steps_from(tracker, iteration_num, "structure_calculation")
                    messages.append(f"iteration_{iteration_num}: reset {reset_steps} because structure artifacts are missing or invalid")
                    continue
            elif previous_steps_completed("structure_calculation") and structure_valid:
                tracker.mark_step_completed(
                    iteration_num,
                    "structure_calculation",
                    metadata={"reconciled": True},
                )
                messages.append(f"iteration_{iteration_num}: marked structure_calculation completed from valid artifacts")

        if step_in_tracker("merge_results"):
            merge_valid = _is_valid_merge_artifacts(results_root, iteration_num)
            if step_completed("merge_results"):
                if not merge_valid:
                    reset_steps = reset_steps_from(tracker, iteration_num, "merge_results")
                    messages.append(f"iteration_{iteration_num}: reset {reset_steps} because merge artifacts are missing or invalid")
                    continue
            elif previous_steps_completed("merge_results") and merge_valid:
                tracker.mark_step_completed(
                    iteration_num,
                    "merge_results",
                    metadata={"reconciled": True},
                )
                messages.append(f"iteration_{iteration_num}: marked merge_results completed from valid artifacts")

        if step_in_tracker("success_extraction"):
            extraction_data = _current_round().get("success_extraction", {})
            extraction_metadata = extraction_data.get("metadata", {}) if isinstance(extraction_data, dict) else {}
            no_materials_marker = (
                isinstance(extraction_metadata, dict)
                and extraction_metadata.get("no_materials") is True
            )
            extraction_valid = load_saved_extract_result(results_root, iteration_num) is not None
            if step_completed("success_extraction"):
                if not extraction_valid and not no_materials_marker:
                    reset_steps = reset_steps_from(tracker, iteration_num, "success_extraction")
                    messages.append(f"iteration_{iteration_num}: reset {reset_steps} because extraction artifacts are missing or invalid")
                    continue
            elif previous_steps_completed("success_extraction") and extraction_valid:
                tracker.mark_step_completed(
                    iteration_num,
                    "success_extraction",
                    metadata={"reconciled": True},
                )
                messages.append(f"iteration_{iteration_num}: marked success_extraction completed from valid artifacts")

        if step_in_tracker("document_update"):
            document_metadata = _current_round().get("document_update", {})
            document_metadata = (
                document_metadata.get("metadata", {})
                if isinstance(document_metadata, dict)
                else {}
            )
            document_data_hash_changed = False
            recorded_document_data_hash = (
                document_metadata.get("data_sha256")
                if isinstance(document_metadata, dict)
                else None
            )
            if recorded_document_data_hash and data_path.exists():
                try:
                    document_data_hash_changed = file_sha256(data_path) != recorded_document_data_hash
                except OSError:
                    document_data_hash_changed = True
            document_valid = (
                versioned_doc is not None
                and not document_data_hash_changed
                and load_saved_document_update_result(
                    results_root,
                    path_config.data_root,
                    doc_root,
                    iteration_num,
                )
                is not None
            )
            if step_completed("document_update"):
                if not document_valid:
                    reset_steps = reset_steps_from(tracker, iteration_num, "document_update")
                    messages.append(f"iteration_{iteration_num}: reset {reset_steps} because versioned theory document or dataset is missing or invalid")
                    continue
            elif previous_steps_completed("document_update") and document_valid:
                tracker.mark_step_completed(
                    iteration_num,
                    "document_update",
                    metadata={"reconciled": True},
                )
                messages.append(f"iteration_{iteration_num}: marked document_update completed from valid artifacts")

        if step_in_tracker("data_update"):
            dataset_valid = _is_valid_dataset_artifact(data_path)
            data_metadata = _current_round().get("data_update", {})
            data_metadata = (
                data_metadata.get("metadata", {})
                if isinstance(data_metadata, dict)
                else {}
            )
            recorded_data_hash = (
                data_metadata.get("data_sha256")
                if isinstance(data_metadata, dict)
                else None
            )
            data_hash_changed = False
            if recorded_data_hash and data_path.exists():
                try:
                    data_hash_changed = file_sha256(data_path) != recorded_data_hash
                except OSError:
                    data_hash_changed = True
            if step_completed("data_update"):
                if not dataset_valid or data_hash_changed:
                    reset_steps = reset_steps_from(tracker, iteration_num, "data_update")
                    reason = "updated dataset fingerprint changed" if data_hash_changed else "updated dataset is missing or invalid"
                    messages.append(f"iteration_{iteration_num}: reset {reset_steps} because {reason}")
                    continue
            elif previous_steps_completed("data_update") and training_data_path.exists() and dataset_valid:
                tracker.mark_step_completed(
                    iteration_num,
                    "data_update",
                    metadata={
                        "reconciled": True,
                        "data_path": str(data_path),
                        "data_sha256": file_sha256(data_path),
                    },
                )
                messages.append(f"iteration_{iteration_num}: marked data_update completed from valid artifacts")

        missing_step: str | None = None
        seen_completed_after_gap = False
        for step in tracker.steps:
            if not step_completed(step):
                if missing_step is None:
                    missing_step = step
                continue
            if missing_step is not None:
                seen_completed_after_gap = True
                break
        if missing_step is not None and seen_completed_after_gap:
            reset_steps = reset_steps_from(tracker, iteration_num, missing_step)
            messages.append(f"iteration_{iteration_num}: reset {reset_steps} because downstream steps existed after missing {missing_step}")

    return messages
