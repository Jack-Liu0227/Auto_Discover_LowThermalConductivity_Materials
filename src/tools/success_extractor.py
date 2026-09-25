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
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temp_path, index=False, encoding="utf-8-sig")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def extract_formula_from_structure(structure_str: str) -> str:
    if not structure_str or structure_str == "N/A":
        return ""
    match = re.search(r"Full Formula \(([^)]+)\)", structure_str)
    return match.group(1).strip() if match else ""


def enrich_space_group_number(df: pd.DataFrame, source_csv: str | Path | None = None) -> pd.DataFrame:
    """Ensure a canonical ``space_group_number`` column is present.

    Existing values are preserved when they are valid. Missing values are
    calculated from each row's CIF, using the primitive structure and the same
    pymatgen settings as the success/stable extractor. Unresolvable rows are
    left as ``NA`` rather than assigned a guessed number.
    """
    result = df.copy()
    source_path = Path(source_csv) if source_csv else None

    def _first_value(row: pd.Series, keys: tuple[str, ...]):
        for key in keys:
            value = row.get(key)
            if value is not None and pd.notna(value) and str(value).strip() not in {"", "N/A", "nan"}:
                return value
        return None

    def _valid_number(value) -> int | None:
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return None
        return number if 1 <= number <= 230 else None

    def _candidate_paths(row: pd.Series) -> list[Path]:
        candidates: list[Path] = []
        raw_relative = _first_value(row, ("relative_cif_path", "Relative_CIF_Path"))
        if raw_relative:
            raw_text = str(raw_relative).strip()
            candidates.extend([Path(raw_text), Path(raw_text.replace("\\", "/"))])
            if source_path and not Path(raw_text).is_absolute():
                candidates.append(source_path.parent / raw_text)

        cif_name = _first_value(row, ("cif_file", "CIF_File", "CIF文件"))
        material = _first_value(row, ("material_dir", "material", "formula", "composition"))
        if source_path and cif_name:
            cif_text = str(cif_name).strip()
            if material:
                candidates.extend(
                    [
                        source_path.parent / "cif_files_success" / str(material) / cif_text,
                        source_path.parent / "cif_files_stable" / str(material) / cif_text,
                    ]
                )
            candidates.extend(
                [
                    source_path.parent / "cif_files_success" / cif_text,
                    source_path.parent / "cif_files_stable" / cif_text,
                    source_path.parent / cif_text,
                ]
            )
        return list(dict.fromkeys(path for path in candidates if path.exists()))

    numbers: list[int | object] = []
    try:
        from pymatgen.core import Structure
    except ImportError:
        Structure = None

    for _, row in result.iterrows():
        number = _valid_number(_first_value(row, ("space_group_number", "Space_Group_Number", "Space Group Number")))
        if number is None and Structure is not None:
            for cif_path in _candidate_paths(row):
                try:
                    structure = Structure.from_file(cif_path).get_primitive_structure()
                    number = _valid_number(structure.get_space_group_info()[1])
                    if number is not None:
                        break
                except Exception:
                    continue
        numbers.append(number if number is not None else pd.NA)

    result["space_group_number"] = numbers
    columns = [column for column in result.columns if column != "space_group_number"]
    if "space_group" in columns:
        columns.insert(columns.index("space_group") + 1, "space_group_number")
        result = result[columns]
    return result


class SuccessMaterialsExtractor:
    def __init__(
        self,
        myrelax_dir: str,
        output_dir: str,
        k_threshold: float = 1.0,
        imag_tol: float = -0.1,
        allowed_formulas: set[str] | None = None,
        max_workers: int = 1,
    ):
        if not myrelax_dir:
            raise ValueError("myrelax_dir must be provided")
        if not output_dir:
            raise ValueError("output_dir must be provided")

        self.myrelax_dir = Path(myrelax_dir)
        self.output_dir = Path(output_dir)
        self.k_threshold = float(k_threshold)
        self.imag_tol = float(imag_tol)
        self.allowed_formulas = set(allowed_formulas) if allowed_formulas is not None else None
        self.max_workers = max(1, int(max_workers))

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

    def _merge_phonon_results_if_needed(self, df: pd.DataFrame, csv_file: Path, rel: Path) -> pd.DataFrame:
        needed_cols = [
            "Has_Imaginary_Freq",
            "Min_Frequency",
            "Gamma_Min_Optical",
            "Gamma_Max_Acoustic",
        ]
        if all(col in df.columns for col in needed_cols):
            return df

        phonon_csv = csv_file.parent / "relax_phonon_results.csv"
        if not phonon_csv.exists():
            phonon_csv = csv_file.parent / "phonon_results.csv"
        if not phonon_csv.exists():
            logger.warning("%s: missing phonon metadata CSV, cannot backfill stability columns", rel)
            return df

        try:
            phonon_df = pd.read_csv(phonon_csv, encoding="utf-8-sig")
        except Exception as exc:
            logger.warning("%s: failed reading %s: %s", rel, phonon_csv.name, exc)
            return df

        join_keys = [key for key in ("CIF_File", "Structure_ID") if key in df.columns and key in phonon_df.columns]
        if not join_keys:
            logger.warning("%s: no common key to join %s", rel, phonon_csv.name)
            return df

        available_cols = [col for col in needed_cols if col in phonon_df.columns]
        if not available_cols:
            logger.warning("%s: %s has no stability columns to backfill", rel, phonon_csv.name)
            return df

        right_cols = join_keys + available_cols
        left_existing = set(df.columns)
        merged = df.merge(phonon_df[right_cols], on=join_keys, how="left", suffixes=("", "__phonon"))
        for col in available_cols:
            phonon_col = f"{col}__phonon" if col in left_existing else col
            if col not in left_existing:
                continue
            if phonon_col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[phonon_col])
                merged.drop(columns=[phonon_col], inplace=True, errors="ignore")

        logger.info("%s: backfilled stability columns from %s using %s", rel, phonon_csv.name, ",".join(join_keys))
        return merged

    def _prepare_csv_file(self, csv_file: Path) -> dict | None:
        """Read and classify one material CSV without writing shared outputs."""
        rel = csv_file.relative_to(self.myrelax_dir)
        material_name = rel.parts[0] if rel.parts else "Unknown"
        if self.allowed_formulas is not None and material_name not in self.allowed_formulas:
            logger.info("Skipping stale material directory: %s", material_name)
            return None

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
        except Exception as exc:
            logger.warning("Failed reading %s: %s", rel, exc)
            return None

        kappa_col = "Kappa_Slack (W m-1 K-1)"
        if kappa_col not in df.columns:
            logger.warning("Missing required column in %s: %s", rel, kappa_col)
            return None

        df = self._merge_phonon_results_if_needed(df.copy(), csv_file, rel)
        df["_dyn_stable"], df["_stability_class"] = self._classify_stability(df)

        mask_success = df["_dyn_stable"] & (
            pd.to_numeric(df[kappa_col], errors="coerce") < self.k_threshold
        )
        mask_stable = df["_dyn_stable"] & (
            pd.to_numeric(df[kappa_col], errors="coerce") < 5.0
        )
        quasi_count = int((df["_stability_class"] == "quasi_stable").sum())
        logger.info(
            "%s: total=%d, dyn_stable=%d, quasi_stable=%d, success=%d, stable=%d",
            rel,
            len(df),
            int(df["_dyn_stable"].sum()),
            quasi_count,
            int(mask_success.sum()),
            int(mask_stable.sum()),
        )
        return {
            "csv_file": csv_file,
            "relative_path": rel,
            "material_name": material_name,
            "struct_type": struct_type,
            "success_records": df.loc[mask_success].to_dict(orient="records"),
            "stable_records": df.loc[mask_stable].to_dict(orient="records"),
        }

    def _copy_row_assets(
        self,
        item: dict,
        csv_file: Path,
        cif_output_dir: Path,
    ) -> None:
        """Copy one row's CIF and phonon images after parent-side indexing."""
        cif_filename = item.get("cif_file")
        if not cif_filename:
            return
        cif_source = csv_file.parent / str(cif_filename)
        if not cif_source.exists():
            return

        material_name = str(item.get("material_dir") or "Unknown")
        dest_dir = cif_output_dir / material_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        cif_dest = dest_dir / str(cif_filename)
        shutil.copy2(cif_source, cif_dest)

        composition = item.get("composition")
        safe_formula = str(
            composition if pd.notna(composition) else material_name
        ).replace(" ", "")
        stem = Path(str(cif_filename)).stem
        phonon_dir = csv_file.parent / f"{stem}_phonon"
        if not phonon_dir.exists():
            return

        band_candidates = sorted(phonon_dir.glob("*_phonon_band.png"))
        dos_candidates = sorted(phonon_dir.glob("*_phonon_dos.png"))
        band_src = band_candidates[0] if band_candidates else (phonon_dir / "phonon_band.png")
        dos_src = dos_candidates[0] if dos_candidates else (phonon_dir / "phonon_dos.png")
        spectrum_src = phonon_dir / "phonon_spectrum.png"

        if band_src.exists():
            shutil.copy2(band_src, dest_dir / f"{stem}_{safe_formula}_phonon_band.png")
        if dos_src.exists():
            shutil.copy2(dos_src, dest_dir / f"{stem}_{safe_formula}_phonon_dos.png")
        if (not band_src.exists() and not dos_src.exists()) and spectrum_src.exists():
            shutil.copy2(spectrum_src, dest_dir / f"{stem}_{safe_formula}_phonon_spectrum.png")

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

        csv_files = sorted(self.myrelax_dir.rglob("thermal_conductivity.csv"))
        logger.info("Found %d thermal_conductivity.csv files", len(csv_files))
        if not csv_files:
            logger.error(
                "No thermal artifacts are available; this is an upstream "
                "relaxation/thermal failure, not a stability-threshold result."
            )

        prepared: list[dict | None] = [None] * len(csv_files)
        worker_count = max(1, min(self.max_workers, len(csv_files))) if csv_files else 1
        if worker_count == 1:
            for index, csv_file in enumerate(csv_files):
                prepared[index] = self._prepare_csv_file(csv_file)
        else:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="extract-material",
            ) as executor:
                futures = {
                    executor.submit(self._prepare_csv_file, csv_file): index
                    for index, csv_file in enumerate(csv_files)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        prepared[index] = future.result()
                    except Exception as exc:
                        logger.warning("Failed preparing %s: %s", csv_files[index], exc)

        row_tasks: list[tuple[str, pd.Series, str, Path, Path, str, int]] = []
        for payload in prepared:
            if payload is None:
                continue
            csv_file = Path(payload["csv_file"])
            rel = Path(payload["relative_path"])
            material_name = str(payload["material_name"])
            struct_type = str(payload["struct_type"])
            for record in payload["success_records"]:
                row_tasks.append(
                    (
                        "success",
                        pd.Series(record),
                        material_name,
                        csv_file,
                        rel,
                        struct_type,
                        success_idx,
                    )
                )
                success_idx += 1
            for record in payload["stable_records"]:
                row_tasks.append(
                    (
                        "stable",
                        pd.Series(record),
                        material_name,
                        csv_file,
                        rel,
                        struct_type,
                        stable_idx,
                    )
                )
                stable_idx += 1

        def _process_row_task(
            task: tuple[str, pd.Series, str, Path, Path, str, int],
        ) -> tuple[str, Path, dict | None]:
            kind, row, material_name, csv_file, rel, struct_type, index = task
            try:
                item = self._process_row(
                    row,
                    material_name,
                    csv_file,
                    rel,
                    struct_type,
                    index,
                )
            except Exception as exc:
                logger.warning("Failed processing %s row %d: %s", rel, index, exc)
                item = None
            return kind, csv_file, item

        processed_rows: list[tuple[str, Path, dict | None] | None] = [None] * len(row_tasks)
        worker_count = max(1, min(self.max_workers, len(row_tasks))) if row_tasks else 1
        if worker_count == 1:
            for index, task in enumerate(row_tasks):
                processed_rows[index] = _process_row_task(task)
        else:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="extract-structure-row",
            ) as executor:
                futures = {
                    executor.submit(_process_row_task, task): index
                    for index, task in enumerate(row_tasks)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        processed_rows[index] = future.result()
                    except Exception as exc:
                        logger.warning("Failed processing extraction row %d: %s", index, exc)

        for processed in processed_rows:
            if processed is None:
                continue
            kind, csv_file, item = processed
            if item is None:
                continue
            if kind == "success":
                self._copy_row_assets(item, csv_file, cif_output_dir_success)
                success_rows.append(item)
            else:
                self._copy_row_assets(item, csv_file, cif_output_dir_stable)
                stable_rows.append(item)

        stable_csv = None
        if stable_rows:
            stable_csv = self.output_dir / "stable_materials.csv"
            _atomic_to_csv(pd.DataFrame(stable_rows), stable_csv)
            logger.info("Saved stable materials: %s (%d)", stable_csv, len(stable_rows))

        if success_rows:
            success_csv = self.output_dir / "success_materials.csv"
            _atomic_to_csv(pd.DataFrame(success_rows), success_csv)
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
        composition = row.get("Composition")
        if composition is None or pd.isna(composition) or not str(composition).strip():
            composition = row.get("composition")
        if composition is None or pd.isna(composition) or not str(composition).strip():
            composition = row.get("Material_Dir")
        if composition is None or pd.isna(composition) or not str(composition).strip():
            composition = material_name
        composition = str(composition).strip()

        structure_id = row.get("Structure_ID", f"{material_name}_{index}")
        kappa = row.get("Kappa_Slack (W m-1 K-1)")
        cif_filename = row.get("CIF_File", "")

        structure_str = "N/A"
        actual_formula = ""
        volume = row.get("Volume (Å³)", row.get("Volume", "N/A"))
        density = row.get("Density (g/cm³)", row.get("Density", "N/A"))
        n_atoms = row.get("N_Atoms", row.get("Number of Atoms", "N/A"))
        space_group = row.get("Space_Group", row.get("Space Group Symbol", "N/A"))
        space_group_number = row.get(
            "space_group_number",
            row.get("Space_Group_Number", row.get("Space Group Number", "N/A")),
        )
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
                    actual_formula = primitive_structure.composition.reduced_formula
                    structure_str = str(primitive_structure)
                    volume = primitive_structure.volume
                    density = primitive_structure.density
                    n_atoms = primitive_structure.num_sites
                    try:
                        space_group, space_group_number = primitive_structure.get_space_group_info()
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning("Failed reading/converting structure %s: %s", cif_source, e)
                    structure_str = f"Error: {e}"

        # The actual formula must come from the CIF, never from the material
        # directory. Keep the directory/request formula separately as lineage.
        formula = actual_formula

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
            "space_group_number": space_group_number,
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

        if cif_output_dir:
            self._copy_row_assets(item, csv_file, cif_output_dir)
        return item


def extract_success_materials(
    myrelax_dir: str,
    output_dir: str,
    k_threshold: float = 1.0,
    imag_tol: float = -0.1,
    allowed_formulas: set[str] | None = None,
    max_workers: int = 1,
) -> Optional[str]:
    extractor = SuccessMaterialsExtractor(
        myrelax_dir=myrelax_dir,
        output_dir=output_dir,
        k_threshold=k_threshold,
        imag_tol=imag_tol,
        allowed_formulas=allowed_formulas,
        max_workers=max_workers,
    )
    return extractor.extract()
