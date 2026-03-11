# -*- coding: utf-8 -*-
"""Structure deduplication utilities for relaxed CIF files."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

logger = logging.getLogger(__name__)


class StructureDeduplicator:
    """Deduplicate crystal structures in a directory."""

    def __init__(
        self,
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5.0,
        attempt_supercell: bool = True,
    ) -> None:
        self.ltol = ltol
        self.stol = stol
        self.angle_tol = angle_tol
        self.attempt_supercell = attempt_supercell
        self.matcher = StructureMatcher(
            ltol=self.ltol,
            stol=self.stol,
            angle_tol=self.angle_tol,
            primitive_cell=True,
            scale=True,
            attempt_supercell=self.attempt_supercell,
        )

    def deduplicate_structures(
        self,
        cif_dir: Path,
        keep_duplicates: bool = False,
        update_csv: bool = True,
    ) -> Dict[str, Any]:
        """Deduplicate CIF files in one directory and optionally sync CSV."""
        cif_dir = Path(cif_dir)
        if not cif_dir.exists():
            logger.warning("Directory not found: %s", cif_dir)
            return {
                "total": 0,
                "unique": 0,
                "removed": 0,
                "unique_files": [],
                "duplicate_files": [],
                "csv_updated": False,
                "matcher_config": self._matcher_config(),
            }

        cif_files = sorted(cif_dir.glob("*.cif"))
        if not cif_files:
            logger.info("No CIF files in %s", cif_dir)
            return {
                "total": 0,
                "unique": 0,
                "removed": 0,
                "unique_files": [],
                "duplicate_files": [],
                "csv_updated": False,
                "matcher_config": self._matcher_config(),
            }

        logger.info(
            "Start dedup in %s, total=%d (ltol=%s, stol=%s, angle_tol=%s, attempt_supercell=%s)",
            cif_dir,
            len(cif_files),
            self.ltol,
            self.stol,
            self.angle_tol,
            self.attempt_supercell,
        )

        structures: List[Tuple[Path, Structure]] = []
        for cif_file in cif_files:
            try:
                struct = Structure.from_file(str(cif_file))
                structures.append((cif_file, struct))
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", cif_file.name, exc)

        if not structures:
            logger.warning("No valid structures in %s", cif_dir)
            return {
                "total": len(cif_files),
                "unique": 0,
                "removed": 0,
                "unique_files": [],
                "duplicate_files": [str(f) for f in cif_files],
                "csv_updated": False,
                "matcher_config": self._matcher_config(),
            }

        # Group by reduced formula and number of sites to reduce false comparisons.
        grouped: Dict[Tuple[str, int], List[Tuple[Path, Structure]]] = {}
        for cif_file, struct in structures:
            key = (struct.composition.reduced_formula, struct.num_sites)
            grouped.setdefault(key, []).append((cif_file, struct))

        logger.info("Detected %d groups", len(grouped))

        unique_files: List[Path] = []
        duplicate_files: List[Path] = []

        for (formula, num_sites), group_structs in grouped.items():
            unique_in_group: List[Tuple[Path, Structure]] = []
            group_removed = 0

            for cif_file, struct in group_structs:
                is_duplicate = False
                for _, kept_struct in unique_in_group:
                    try:
                        if self.matcher.fit(struct, kept_struct):
                            is_duplicate = True
                            duplicate_files.append(cif_file)
                            group_removed += 1
                            break
                    except Exception as exc:
                        logger.debug("Matcher failed for %s: %s", cif_file.name, exc)

                if not is_duplicate:
                    unique_in_group.append((cif_file, struct))
                    unique_files.append(cif_file)

            logger.info(
                "  Group %s (num_sites=%d): %d -> %d (removed %d)",
                formula,
                num_sites,
                len(group_structs),
                len(unique_in_group),
                group_removed,
            )

        csv_updated = False
        if update_csv and duplicate_files:
            csv_updated = self._sync_csv_with_files(cif_dir, unique_files, duplicate_files)

        if duplicate_files:
            if keep_duplicates:
                dup_dir = cif_dir / "duplicates"
                dup_dir.mkdir(exist_ok=True)
                for dup_file in duplicate_files:
                    try:
                        shutil.move(str(dup_file), str(dup_dir / dup_file.name))
                    except Exception as exc:
                        logger.warning("Failed to move duplicate %s: %s", dup_file.name, exc)
            else:
                for dup_file in duplicate_files:
                    try:
                        dup_file.unlink()
                    except Exception as exc:
                        logger.warning("Failed to remove duplicate %s: %s", dup_file.name, exc)

        result = {
            "total": len(cif_files),
            "unique": len(unique_files),
            "removed": len(duplicate_files),
            "unique_files": [str(f) for f in unique_files],
            "duplicate_files": [str(f) for f in duplicate_files],
            "csv_updated": csv_updated,
            "matcher_config": self._matcher_config(),
        }
        logger.info(
            "Dedup finished: %d -> %d (removed %d)",
            result["total"],
            result["unique"],
            result["removed"],
        )
        return result

    def _matcher_config(self) -> Dict[str, Any]:
        return {
            "ltol": self.ltol,
            "stol": self.stol,
            "angle_tol": self.angle_tol,
            "attempt_supercell": self.attempt_supercell,
        }

    def _sync_csv_with_files(
        self,
        cif_dir: Path,
        unique_files: List[Path],
        duplicate_files: List[Path],
    ) -> bool:
        """Sync thermal_conductivity.csv by removing rows for duplicate CIFs."""
        csv_file = cif_dir / "thermal_conductivity.csv"
        if not csv_file.exists():
            logger.debug("CSV not found, skip sync: %s", csv_file)
            return False

        try:
            import pandas as pd

            df = pd.read_csv(csv_file)
            if df.empty:
                logger.debug("CSV empty, skip sync: %s", csv_file)
                return False

            duplicate_stems = {f.stem for f in duplicate_files}

            id_column: Optional[str] = None
            for col in ["Structure_ID", "CIF_File", "structure_id"]:
                if col in df.columns:
                    id_column = col
                    break

            if id_column is None:
                logger.warning("CSV has no structure id column, skip sync: %s", csv_file)
                return False

            def _stem(value: Any) -> Optional[str]:
                if value is None:
                    return None
                text = str(value)
                if text.lower().endswith(".cif"):
                    return text[:-4]
                return text

            original_len = len(df)
            df["_temp_stem"] = df[id_column].apply(_stem)
            df_filtered = df[~df["_temp_stem"].isin(duplicate_stems)].drop(columns=["_temp_stem"])
            new_len = len(df_filtered)

            if new_len < original_len:
                df_filtered.to_csv(csv_file, index=False, encoding="utf-8-sig")
                logger.info(
                    "CSV synced: %d -> %d (removed %d)",
                    original_len,
                    new_len,
                    original_len - new_len,
                )
                return True

            return False
        except Exception as exc:
            logger.warning("Failed to sync CSV %s: %s", csv_file, exc)
            return False

    def deduplicate_multi_dirs(
        self,
        base_dir: Path,
        formula_dirs: Optional[List[str]] = None,
        keep_duplicates: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Deduplicate multiple formula subdirectories under one base dir."""
        base_dir = Path(base_dir)
        if not base_dir.exists():
            logger.warning("Base directory not found: %s", base_dir)
            return {}

        if formula_dirs is None:
            formula_dirs = [d.name for d in base_dir.iterdir() if d.is_dir()]

        results: Dict[str, Dict[str, Any]] = {}
        for formula in formula_dirs:
            comp_dir = base_dir / formula
            if not comp_dir.exists():
                logger.warning("Formula directory not found: %s", comp_dir)
                continue
            results[formula] = self.deduplicate_structures(comp_dir, keep_duplicates)
        return results


def deduplicate_relaxed_structures(
    relax_dir: Path,
    formulas: Optional[List[str]] = None,
    keep_duplicates: bool = False,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
    attempt_supercell: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Convenience wrapper for relaxed-structure deduplication."""
    deduplicator = StructureDeduplicator(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        attempt_supercell=attempt_supercell,
    )
    return deduplicator.deduplicate_multi_dirs(relax_dir, formulas, keep_duplicates)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Structure deduplication tool")
    parser.add_argument("directory", type=str, help="Directory to deduplicate")
    parser.add_argument("--keep-duplicates", action="store_true", help="Move duplicates to duplicates/")
    parser.add_argument("--ltol", type=float, default=0.2, help="Lattice tolerance")
    parser.add_argument("--stol", type=float, default=0.3, help="Site tolerance")
    parser.add_argument("--angle-tol", type=float, default=5.0, help="Angle tolerance in degrees")
    parser.add_argument(
        "--attempt-supercell",
        action="store_true",
        default=True,
        help="Enable supercell matching (default: True)",
    )
    parser.add_argument(
        "--no-attempt-supercell",
        action="store_false",
        dest="attempt_supercell",
        help="Disable supercell matching",
    )

    args = parser.parse_args()
    deduplicator = StructureDeduplicator(
        ltol=args.ltol,
        stol=args.stol,
        angle_tol=args.angle_tol,
        attempt_supercell=args.attempt_supercell,
    )
    result = deduplicator.deduplicate_structures(
        Path(args.directory),
        keep_duplicates=args.keep_duplicates,
    )

    print("\nDedup result:")
    print(f"  total: {result['total']}")
    print(f"  unique: {result['unique']}")
    print(f"  removed: {result['removed']}")
