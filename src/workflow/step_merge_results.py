# -*- coding: utf-8 -*-
"""
Workflow Step: Merge phonon results into thermal conductivity CSVs.
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

from utils.workflow_resume import load_active_material_formulas


def _merge_phonon_into_kappa(comp_dir: Path) -> bool:
    kappa_file = comp_dir / "thermal_conductivity.csv"
    phonon_file = comp_dir / "relax_phonon_results.csv"
    if not kappa_file.exists() or not phonon_file.exists():
        return False

    try:
        kappa_df = pd.read_csv(kappa_file, encoding="utf-8-sig")
        phonon_df = pd.read_csv(phonon_file, encoding="utf-8-sig")
    except Exception:
        return False

    if "CIF_File" not in kappa_df.columns or "CIF_File" not in phonon_df.columns:
        return False

    phonon_cols = [
        "Has_Imaginary_Freq",
        "Min_Frequency",
        "Gamma_Min_Optical",
        "Gamma_Max_Acoustic",
        "Phonon_Success",
    ]
    lineage_cols = [
        "Formula",
        "Composition",
        "Material_Dir",
        "Composition_Mismatch",
    ]
    merge_cols = phonon_cols + [
        col for col in lineage_cols if col in phonon_df.columns
    ]
    for col in merge_cols:
        if col not in phonon_df.columns:
            phonon_df[col] = None

    phonon_view = phonon_df[["CIF_File"] + merge_cols].copy()
    merged = kappa_df.merge(phonon_view, on="CIF_File", how="left", suffixes=("", "_ph"))
    for col in merge_cols:
        ph_col = f"{col}_ph"
        if col in kappa_df.columns:
            if ph_col in merged.columns:
                merged[col] = merged[col].fillna(merged[ph_col])
                merged = merged.drop(columns=[ph_col])
        else:
            if ph_col in merged.columns:
                merged[col] = merged[ph_col]
                merged = merged.drop(columns=[ph_col])

    temp_file = kappa_file.with_name(f".{kappa_file.name}.{os.getpid()}.tmp")
    try:
        merged.to_csv(temp_file, index=False, encoding="utf-8-sig")
        os.replace(temp_file, kappa_file)
    finally:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass
    return True


def _merge_material_result(comp_dir: Path) -> tuple[str, bool, str]:
    """Merge one material directory and contain failures to that material."""
    try:
        return comp_dir.name, _merge_phonon_into_kappa(comp_dir), ""
    except Exception as exc:
        return comp_dir.name, False, str(exc)


def step_merge_results(
    iteration_num: int,
    results_root: str = "results",
    tracker=None,
    path_config=None,
    max_workers: int = 1,
) -> dict:
    """
    Merge phonon results into thermal conductivity CSVs for an iteration.
    """
    print("=" * 80)
    print(f"Step: Merge Results (Iteration {iteration_num})")
    print("=" * 80)

    project_root = Path(__file__).resolve().parents[2]
    if path_config is not None:
        relax_dir = path_config.get_iteration_results_path(iteration_num) / "MyRelaxStructure"
    else:
        relax_dir = project_root / results_root / f"iteration_{iteration_num}" / "MyRelaxStructure"
    if not relax_dir.exists():
        print(f"[WARN] Relaxation directory not found: {relax_dir}")
        return {"success": False, "error": "Relaxation directory not found"}

    merged_count = 0
    merged_formulas: set[str] = set()
    active_formulas = load_active_material_formulas(
        path_config.results_root if path_config is not None else project_root / results_root,
        iteration_num,
    )
    if active_formulas is not None and not active_formulas:
        error = "No active materials with valid relaxation/phonon/thermal artifacts"
        print(f"[ERROR] {error}; merge cannot complete")
        return {
            "success": False,
            "completed": False,
            "no_active_materials": True,
            "error": error,
        }

    material_dirs = []
    for comp_dir in sorted(relax_dir.iterdir(), key=lambda p: p.name):
        if not comp_dir.is_dir():
            continue
        if active_formulas is not None and comp_dir.name not in active_formulas:
            print(f"[SKIP] Ignoring stale relaxation directory: {comp_dir.name}")
            continue
        material_dirs.append(comp_dir)

    worker_count = max(1, min(int(max_workers), len(material_dirs))) if material_dirs else 1
    merge_results: dict[str, tuple[bool, str]] = {}
    if worker_count == 1:
        for comp_dir in material_dirs:
            formula, merged, error = _merge_material_result(comp_dir)
            merge_results[formula] = (merged, error)
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="merge-material") as executor:
            futures = {
                executor.submit(_merge_material_result, comp_dir): comp_dir.name
                for comp_dir in material_dirs
            }
            for future in as_completed(futures):
                formula, merged, error = future.result()
                merge_results[formula] = (merged, error)

    for formula in sorted(merge_results):
        merged, error = merge_results[formula]
        if error:
            print(f"[WARN] Merge failed for {formula}: {error}")
        if merged:
            merged_count += 1
            merged_formulas.add(formula)

    if active_formulas is not None:
        missing_formulas = sorted(active_formulas - merged_formulas)
        if missing_formulas:
            error = f"Merge artifacts missing for active materials: {missing_formulas}"
            print(f"[ERROR] {error}")
            return {"success": False, "error": error, "materials_processed": merged_count}

    if tracker:
        tracker.mark_step_completed(
            iteration_num,
            "merge_results",
            {"materials_processed": merged_count},
        )

    print(f"[OK] Merged {merged_count} materials")
    return {"success": True, "materials_processed": merged_count}
