# -*- coding: utf-8 -*-
"""
Workflow Step: Merge phonon results into thermal conductivity CSVs.
"""
from pathlib import Path
import pandas as pd


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
    for col in phonon_cols:
        if col not in phonon_df.columns:
            phonon_df[col] = None

    phonon_view = phonon_df[["CIF_File"] + phonon_cols].copy()
    merged = kappa_df.merge(phonon_view, on="CIF_File", how="left", suffixes=("", "_ph"))
    for col in phonon_cols:
        ph_col = f"{col}_ph"
        if col in kappa_df.columns:
            if ph_col in merged.columns:
                merged[col] = merged[col].fillna(merged[ph_col])
                merged = merged.drop(columns=[ph_col])
        else:
            if ph_col in merged.columns:
                merged[col] = merged[ph_col]
                merged = merged.drop(columns=[ph_col])

    merged.to_csv(kappa_file, index=False, encoding="utf-8-sig")
    return True


def step_merge_results(iteration_num: int, results_root: str = "results", tracker=None) -> dict:
    """
    Merge phonon results into thermal conductivity CSVs for an iteration.
    """
    print("=" * 80)
    print(f"Step: Merge Results (Iteration {iteration_num})")
    print("=" * 80)

    project_root = Path(__file__).parent.parent.parent
    relax_dir = project_root / results_root / f"iteration_{iteration_num}" / "MyRelaxStructure"
    if not relax_dir.exists():
        print(f"[WARN] Relaxation directory not found: {relax_dir}")
        return {"success": False, "error": "Relaxation directory not found"}

    merged_count = 0
    for comp_dir in relax_dir.iterdir():
        if not comp_dir.is_dir():
            continue
        if _merge_phonon_into_kappa(comp_dir):
            merged_count += 1

    if tracker:
        tracker.mark_step_completed(
            iteration_num,
            "merge_results",
            {"materials_processed": merged_count},
        )

    print(f"[OK] Merged {merged_count} materials")
    return {"success": True, "materials_processed": merged_count}
