#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Operations for AI4Kappa Application

This module provides functions to process CIF files, extract crystallographic
data, and manage file operations for the AI4Kappa application.

Author: [Your Name]
Date: [Current Date]
"""

from __future__ import annotations

import os
import glob
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_and_save_files(
    file_paths: List[str | Path],
    root_dir_path: str | Path
) -> Dict[str, Structure]:
    """
    Process CIF files and convert structures to primitive format.

    Args:
        file_paths (List[str | Path]): List of paths to CIF files.
        root_dir_path (str | Path): Directory path for saving processed files.

    Returns:
        Dict[str, Structure]: Dictionary mapping filenames to primitive pymatgen Structure objects.
    """
    root_dir_path = Path(root_dir_path)
    root_dir_path.mkdir(parents=True, exist_ok=True)

    primitive_structures = {}

    for file_path in file_paths:
        try:
            file_path = Path(file_path)
            
            # Load the CIF file
            structure = Structure.from_file(file_path)
            primitive_structure = structure.get_primitive_structure()

            # Extract filename without extension
            file_name = file_path.stem
            primitive_structures[file_name] = primitive_structure

            # Save the primitive structure as a CIF file
            save_path = root_dir_path / file_path.name
            CifWriter(primitive_structure, symprec=0.1).write_file(str(save_path))

            logger.info(f"Successfully processed and saved {file_name}")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            try:
                # Fallback: copy the original file if conversion fails
                save_path = root_dir_path / Path(file_path).name
                shutil.copy2(file_path, save_path)
                logger.info(f"Saved original file {file_path.name} due to processing error")
            except Exception as save_error:
                logger.error(f"Error saving original file {file_path}: {str(save_error)}")

    logger.info("Processed all files successfully")
    return primitive_structures


def get_crystalline_data(structure: Structure) -> Dict[str, float | str | int]:
    """
    Extract crystallographic data from a pymatgen Structure object.

    Args:
        structure (Structure): Pymatgen Structure object.

    Returns:
        Dict[str, float | str | int]: Dictionary containing crystallographic properties.

    Raises:
        ValueError: If required data extraction fails.
    """
    try:
        # Basic structure data
        data = {
            "Formula": structure.composition.reduced_formula,
            "Number of Atoms": structure.composition.num_atoms,
            "Density (g cm-3)": structure.density,
            "Volume (Å3)": structure.volume,
            "Total Atomic Mass (amu)": sum(site.specie.atomic_mass for site in structure.sites)
        }

        # Lattice parameters
        lattice = structure.lattice
        lattice_data = {
            "a (Å)": lattice.a,
            "b (Å)": lattice.b,
            "c (Å)": lattice.c,
            "alpha (°)": lattice.alpha,
            "beta (°)": lattice.beta,
            "gamma (°)": lattice.gamma
        }
        data.update(lattice_data)

        # Symmetry analysis using SpacegroupAnalyzer
        analyzer = SpacegroupAnalyzer(structure)
        symmetry_data = {
            "Space Group Symbol": analyzer.get_space_group_symbol(),
            "Space Group Number": analyzer.get_space_group_number(),
            "Crystal System": analyzer.get_crystal_system(),
            "Point Group": analyzer.get_point_group_symbol(),
            
        }
        data.update(symmetry_data)

        # Validate data completeness
        null_fields = [key for key, value in data.items() if value is None]
        if null_fields:
            raise ValueError(f"Failed to extract fields: {', '.join(null_fields)}")

        logger.info("Successfully extracted crystallographic data for structure")
        return data

    except Exception as e:
        error_msg = f"Error extracting crystalline data: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def get_dir_crystalline_data(root_dir_path: str | Path) -> pd.DataFrame:
    """
    Extract crystal data for all CIF files in a directory.

    Args:
        root_dir_path (str | Path): Directory containing CIF files.

    Returns:
        pd.DataFrame: DataFrame containing crystallographic data for all structures.
    """
    try:
        root_dir_path = Path(root_dir_path)
        cif_paths = list(root_dir_path.glob('*.cif'))

        if not cif_paths:
            logger.warning(f"No CIF files found in {root_dir_path}")
            return pd.DataFrame()

        logger.info(f"Found {len(cif_paths)} CIF files in {root_dir_path}")

        data_list = []
        file_names = []

        for cif_path in cif_paths:
            try:
                file_name = cif_path.stem

                # Load structure and convert to primitive
                structure = Structure.from_file(cif_path)
                primitive_structure = SpacegroupAnalyzer(structure).get_primitive_standard_structure()

                # Extract crystallographic data
                data = get_crystalline_data(primitive_structure)
                data_list.append(data)
                file_names.append(file_name)

                logger.info(f"Extracted crystallographic data for {file_name}")

            except Exception as e:
                logger.error(f"Error processing {cif_path.name}: {e}")
                continue

        if not data_list:
            logger.warning("No valid crystal data found")
            return pd.DataFrame()

        # Create DataFrame with extracted data
        df = pd.DataFrame(data_list, index=file_names)
        logger.info(f"Created DataFrame with {len(df)} entries")
        return df

    except Exception as e:
        logger.error(f"Error processing directory data: {str(e)}")
        return pd.DataFrame()


def is_valid_cif(file_path: str | Path) -> bool:
    """
    Check if a CIF file is valid.

    Args:
        file_path (str | Path): Path to the CIF file.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        Structure.from_file(file_path)
        logger.info(f"Valid CIF file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Invalid CIF file {file_path}: {str(e)}")
        return False


def create_id_prop(root_dir_path: str | Path) -> Tuple[List[str], List[str]]:
    """
    Create an 'id_prop.csv' file and return lists of CIF file paths and their base names.

    Args:
        root_dir_path (str | Path): Directory containing CIF files.

    Returns:
        Tuple[List[str], List[str]]: List of CIF file paths and corresponding base filenames.
    """
    try:
        # Get all CIF files in the directory
        cif_path_list = glob.glob(os.path.join(root_dir_path, '*.[cC][iI][fF]'))
        if not cif_path_list:
            logger.warning(f"No CIF files found in {root_dir_path}")
            return [], []

        # Extract base filenames without extensions
        cif_name_list = [
            Path(path).stem for path in cif_path_list if os.path.exists(path)
        ]

        if not cif_name_list:
            logger.warning("No valid CIF files found")
            return [], []

        # Create a DataFrame without headers and save as 'id_prop.csv'
        df = pd.DataFrame({'name': cif_name_list, 'target': 0})
        csv_path = os.path.join(root_dir_path, 'id_prop.csv')
        df.to_csv(csv_path, index=False, header=False)
        logger.info(f"Created 'id_prop.csv' with {len(cif_name_list)} entries at {csv_path}")

        return cif_path_list, cif_name_list

    except Exception as e:
        logger.error(f"Error creating 'id_prop.csv': {str(e)}")
        return [], []


def del_cif_file(path: str | Path) -> None:
    """
    Delete CIF and related temporary files in a specified directory.

    Args:
        path (str | Path): Directory path.
    """
    try:
        for file in os.listdir(path):
            file_path = Path(path) / file
            if file.lower().endswith('.cif') or file in {"id_prop.csv", "pre-trained.pth.tar", "test_results.csv"}:
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting CIF files in {path}: {str(e)}")


def del_temp_file(path: str | Path) -> None:
    """
    Delete temporary files in a specified directory.

    Args:
        path (str | Path): Directory path.
    """
    try:
        test_results = Path(path) / "test_results.csv"
        if test_results.exists():
            test_results.unlink()
            logger.info(f"Deleted temporary file: {test_results}")
    except Exception as e:
        logger.error(f"Error deleting temporary file in {path}: {str(e)}")


def clean_root_dir(root_dir_path: str | Path) -> None:
    """
    Clean the root directory by removing all files except 'atom_init.json'.

    Args:
        root_dir_path (str | Path): Path to the root directory.
    """
    try:
        for file_path in Path(root_dir_path).glob('*'):
            if file_path.name != 'atom_init.json':
                file_path.unlink()
                logger.info(f"Removed: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning root_dir {root_dir_path}: {str(e)}")


def get_N_cif(
    start_idx: int,
    end_idx: int,
    source_dir: str,
    target_dir: str
) -> str:
    """
    Copy CIF files from source directory to target directory within a specified index range,
    converting structures to primitive cells.

    Args:
        start_idx (int): Starting index (1-based) of files to process.
        end_idx (int): Ending index (inclusive) of files to process.
        source_dir (str): Path to source directory containing CIF files.
        target_dir (str): Path to target directory where files will be copied.

    Returns:
        str: Name of the last processed file.

    Raises:
        ValueError: If no files are copied or invalid index range.
    """
    try:
        # Ensure target directory exists
        target_dir_path = Path(target_dir).resolve()
        target_dir_path.mkdir(parents=True, exist_ok=True)

        # Get list of CIF files from source directory
        cif_files = sorted([
            file for file in os.listdir(source_dir)
            if file.lower().endswith('.cif')
        ])

        if not cif_files:
            raise ValueError(f"No CIF files found in source directory: {source_dir}")

        # Adjust indices to 0-based
        start_idx = max(1, start_idx) - 1
        end_idx = min(end_idx, len(cif_files)) if end_idx is not None else len(cif_files)

        if start_idx >= end_idx:
            raise ValueError("Invalid index range provided for CIF file processing")

        last_file = None

        # Process and copy files within the specified range
        for idx in range(start_idx, end_idx):
            try:
                source_file = Path(source_dir) / cif_files[idx]
                filename = source_file.stem
                target_file = target_dir_path / (filename + '.cif')

                # Load and convert structure to primitive
                structure = Structure.from_file(source_file)
                primitive_structure = structure.get_primitive_structure()

                # Save the primitive structure as a CIF file
                CifWriter(primitive_structure, symprec=0.1).write_file(target_file)

                last_file = cif_files[idx]
                logger.info(f"Processed and copied {cif_files[idx]} ({idx + 1}/{end_idx})")

            except Exception as e:
                logger.error(f"Error processing {cif_files[idx]}: {str(e)}")
                # Fallback: copy the original file if conversion fails
                try:
                    shutil.copy2(source_file, target_file)
                    last_file = cif_files[idx]
                    logger.info(f"Copied original file {cif_files[idx]} due to processing error")
                except Exception as copy_error:
                    logger.error(f"Error copying original file {cif_files[idx]}: {str(copy_error)}")
                    continue

        if last_file is None:
            raise ValueError("No files were successfully copied")

        logger.info(f"Last processed file: {last_file}")
        return last_file

    except Exception as e:
        logger.error(f"Error in get_N_cif: {str(e)}")
        raise

