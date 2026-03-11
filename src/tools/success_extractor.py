"""Success materials extractor.

Extracts two sets from each thermal_conductivity.csv:
- success_materials.csv: dynamically stable and k < k_threshold
- stable_materials.csv: dynamically stable and k < 5.0

Dynamic stability rule (configurable):
- Min_Frequency >= imag_tol (default -0.1 THz) => stable
- if Min_Frequency missing, fallback to Has_Imaginary_Freq == '否'/no/false
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def extract_formula_from_structure(structure_str: str) -> str:
    if not structure_str or structure_str == "N/A":
        return ""
    match = re.search(r"Full Formula \(([^)]+)\)", structure_str)
    return match.group(1).strip() if match else ""


class SuccessMaterialsExtractor:
    def __init__(
        self,
        myrelax_dir: str,
        output_dir: str,
        k_threshold: float = 1.0,
        imag_tol: float = -0.1,
    ):
        if not myrelax_dir:
            raise ValueError("myrelax_dir must be provided")
        if not output_dir:
            raise ValueError("output_dir must be provided")

        self.myrelax_dir = Path(myrelax_dir)
        self.output_dir = Path(output_dir)
        self.k_threshold = float(k_threshold)
        self.imag_tol = float(imag_tol)

    @staticmethod
    def _is_no_imag_flag(value) -> bool:
        v = str(value).strip().lower()
        if v in {"否", "no", "n", "false", "0"}:
            return True
        if v in {"是", "yes", "y", "true", "1"}:
            return False
        return False

    def _classify_stability(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        imag_col = "Has_Imaginary_Freq"
        has_imag_col = imag_col in df.columns
        no_imag = df[imag_col].apply(self._is_no_imag_flag) if has_imag_col else pd.Series(False, index=df.index)

        has_min_freq = "Min_Frequency" in df.columns
        min_freq = pd.to_numeric(df.get("Min_Frequency"), errors="coerce") if has_min_freq else pd.Series(float("nan"), index=df.index)

        if has_min_freq:
            dyn_stable = (min_freq >= self.imag_tol) | (min_freq.isna() & no_imag)
            cls = pd.Series("unstable", index=df.index, dtype=object)
            cls[min_freq >= 0] = "strict_stable"
            cls[(min_freq < 0) & (min_freq >= self.imag_tol)] = "quasi_stable"
            cls[min_freq < self.imag_tol] = "unstable"
            cls[min_freq.isna() & no_imag] = "strict_stable_by_flag"
            cls[min_freq.isna() & (~no_imag)] = "unstable_by_flag"
            return dyn_stable, cls

        dyn_stable = no_imag
        cls = pd.Series("unstable_by_flag", index=df.index, dtype=object)
        cls[no_imag] = "strict_stable_by_flag"
        return dyn_stable, cls

    def extract(self) -> Optional[str]:
        logger.info("=" * 80)
        logger.info("Start extracting success/stable materials")
        logger.info("source=%s", self.myrelax_dir)
        logger.info("output=%s", self.output_dir)
        logger.info("criteria: Min_Frequency >= %.3f THz and k < %.3f", self.imag_tol, self.k_threshold)
        logger.info("=" * 80)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        cif_output_dir_success = self.output_dir / "cif_files_success"
        cif_output_dir_stable = self.output_dir / "cif_files_stable"
        cif_output_dir_success.mkdir(parents=True, exist_ok=True)
        cif_output_dir_stable.mkdir(parents=True, exist_ok=True)

        if not self.myrelax_dir.exists():
            logger.warning("Relax dir not found: %s", self.myrelax_dir)
            return None

        success_rows: list[dict] = []
        stable_rows: list[dict] = []
        success_idx = 1
        stable_idx = 1

        csv_files = list(self.myrelax_dir.rglob("thermal_conductivity.csv"))
        logger.info("Found %d thermal_conductivity.csv files", len(csv_files))

        for csv_file in csv_files:
            rel = csv_file.relative_to(self.myrelax_dir)
            material_name = rel.parts[0] if len(rel.parts) >= 1 else "Unknown"
            parent_dir_name = csv_file.parent.name
            if "original" in parent_dir_name.lower():
                struct_type = "original"
            elif "primitive" in parent_dir_name.lower():
                struct_type = "primitive"
            elif "conventional" in parent_dir_name.lower():
                struct_type = "conventional"
            else:
                struct_type = parent_dir_name

            try:
                df = pd.read_csv(csv_file, encoding="utf-8-sig")
            except Exception as e:
                logger.warning("Failed reading %s: %s", rel, e)
                continue

            kappa_col = "Kappa_Slack (W m-1 K-1)"
            if kappa_col not in df.columns:
                logger.warning("Missing required column in %s: %s", rel, kappa_col)
                continue

            df = df.copy()
            df["_dyn_stable"], df["_stability_class"] = self._classify_stability(df)

            mask_success = df["_dyn_stable"] & (pd.to_numeric(df[kappa_col], errors="coerce") < self.k_threshold)
            mask_stable = df["_dyn_stable"] & (pd.to_numeric(df[kappa_col], errors="coerce") < 5.0)

            success_df = df[mask_success]
            stable_df = df[mask_stable]
            quasi_count = int((df["_stability_class"] == "quasi_stable").sum())
            logger.info(
                "%s: total=%d, dyn_stable=%d, quasi_stable=%d, success=%d, stable=%d",
                rel,
                len(df),
                int(df["_dyn_stable"].sum()),
                quasi_count,
                len(success_df),
                len(stable_df),
            )

            for _, row in success_df.iterrows():
                item = self._process_row(row, material_name, csv_file, rel, struct_type, success_idx, cif_output_dir_success)
                if item:
                    success_rows.append(item)
                    success_idx += 1

            for _, row in stable_df.iterrows():
                item = self._process_row(row, material_name, csv_file, rel, struct_type, stable_idx, cif_output_dir_stable)
                if item:
                    stable_rows.append(item)
                    stable_idx += 1

        stable_csv = None
        if stable_rows:
            stable_csv = self.output_dir / "stable_materials.csv"
            pd.DataFrame(stable_rows).to_csv(stable_csv, index=False, encoding="utf-8-sig")
            logger.info("Saved stable materials: %s (%d)", stable_csv, len(stable_rows))

        if success_rows:
            success_csv = self.output_dir / "success_materials.csv"
            pd.DataFrame(success_rows).to_csv(success_csv, index=False, encoding="utf-8-sig")
            logger.info("Saved success materials: %s (%d)", success_csv, len(success_rows))
            return str(success_csv)

        if stable_csv is not None:
            logger.warning("No strict success materials, return stable file for fallback: %s", stable_csv)
            return str(stable_csv)

        logger.warning("No materials passed stability criteria")
        return None

    def _process_row(
        self,
        row: pd.Series,
        material_name: str,
        csv_file: Path,
        relative_path: Path,
        struct_type: str,
        index: int,
        cif_output_dir: Path | None = None,
    ) -> Optional[dict]:
        composition = row.get("Composition", material_name)
        structure_id = row.get("Structure_ID", f"{material_name}_{index}")
        kappa = row.get("Kappa_Slack (W m-1 K-1)")
        cif_filename = row.get("CIF_File", "")

        structure_str = "N/A"
        volume = row.get("Volume (Å³)", row.get("Volume", "N/A"))
        density = row.get("Density (g/cm³)", row.get("Density", "N/A"))
        n_atoms = row.get("N_Atoms", row.get("Number of Atoms", "N/A"))
        space_group = row.get("Space_Group", row.get("Space Group Symbol", "N/A"))
        debye_temp = row.get("Debye_Temperature (K)", row.get("Debye Temperature", ""))
        gruneisen = row.get("Gruneisen_Parameter", row.get("Grüneisen_Parameter", ""))
        min_frequency = row.get("Min_Frequency")
        gamma_min_optical = row.get("Gamma_Min_Optical")
        gamma_max_acoustic = row.get("Gamma_Max_Acoustic")

        if cif_filename:
            cif_source = csv_file.parent / str(cif_filename)
            if cif_source.exists():
                try:
                    from pymatgen.core import Structure

                    structure = Structure.from_file(str(cif_source))
                    primitive_structure = structure.get_primitive_structure()
                    structure_str = str(primitive_structure)
                    volume = primitive_structure.volume
                    density = primitive_structure.density
                    n_atoms = primitive_structure.num_sites
                    try:
                        space_group = primitive_structure.get_space_group_info()[0]
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning("Failed reading/converting structure %s: %s", cif_source, e)
                    structure_str = f"Error: {e}"

        formula = extract_formula_from_structure(structure_str)

        item = {
            "index": index,
            "composition": composition,
            "formula": formula,
            "material_dir": material_name,
            "structure_type": struct_type,
            "thermal_conductivity_w_mk": kappa,
            "structure_id": structure_id,
            "cif_file": cif_filename,
            "structure": structure_str,
            "space_group": space_group,
            "volume_a3": volume,
            "density_g_cm3": density,
            "n_atoms": n_atoms,
            "debye_temperature_k": debye_temp,
            "gruneisen_parameter": gruneisen,
            "min_frequency": min_frequency,
            "gamma_min_optical": gamma_min_optical,
            "gamma_max_acoustic": gamma_max_acoustic,
            "csv_path": str(relative_path),
            "relative_cif_path": str(self.myrelax_dir / relative_path.parent / str(cif_filename)) if cif_filename else "",
            "stability_class": row.get("_stability_class", "unknown"),
            "dynamic_stable": bool(row.get("_dyn_stable", False)),
            "imag_tol_thz": self.imag_tol,
        }

        if cif_output_dir and cif_filename:
            cif_source = csv_file.parent / str(cif_filename)
            if cif_source.exists():
                dest_dir = cif_output_dir / material_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                cif_dest = dest_dir / str(cif_filename)
                shutil.copy2(cif_source, cif_dest)

                safe_formula = str((composition if pd.notna(composition) else material_name)).replace(" ", "")
                stem = Path(str(cif_filename)).stem
                phonon_dir = csv_file.parent / f"{stem}_phonon"
                if phonon_dir.exists():
                    band_candidates = list(phonon_dir.glob("*_phonon_band.png"))
                    dos_candidates = list(phonon_dir.glob("*_phonon_dos.png"))
                    band_src = band_candidates[0] if band_candidates else (phonon_dir / "phonon_band.png")
                    dos_src = dos_candidates[0] if dos_candidates else (phonon_dir / "phonon_dos.png")
                    spectrum_src = phonon_dir / "phonon_spectrum.png"

                    if band_src.exists():
                        shutil.copy2(band_src, dest_dir / f"{stem}_{safe_formula}_phonon_band.png")
                    if dos_src.exists():
                        shutil.copy2(dos_src, dest_dir / f"{stem}_{safe_formula}_phonon_dos.png")
                    if (not band_src.exists() and not dos_src.exists()) and spectrum_src.exists():
                        shutil.copy2(spectrum_src, dest_dir / f"{stem}_{safe_formula}_phonon_spectrum.png")
        return item


def extract_success_materials(
    myrelax_dir: str,
    output_dir: str,
    k_threshold: float = 1.0,
    imag_tol: float = -0.1,
) -> Optional[str]:
    extractor = SuccessMaterialsExtractor(
        myrelax_dir=myrelax_dir,
        output_dir=output_dir,
        k_threshold=k_threshold,
        imag_tol=imag_tol,
    )
    return extractor.extract()
