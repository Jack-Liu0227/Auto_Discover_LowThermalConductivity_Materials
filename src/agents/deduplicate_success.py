
import os
import logging
import pandas as pd
from typing import Optional, List
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

logger = logging.getLogger(__name__)


def _pick(row, keys):
    for key in keys:
        if key in row and pd.notna(row[key]):
            return row[key]
    return None

def deduplicate_success_materials(
    input_csv: str, 
    output_csv: str,
    matcher: Optional[StructureMatcher] = None
) -> Optional[str]:
    """
    Read success materials CSV, deduplicate based on structure similarity, 
    and save to output CSV.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output deduplicated CSV file
        matcher: Pymatgen StructureMatcher instance. If None, default will be used.
        
    Returns:
        Path to output_csv if successful, None otherwise.
    """
    if not os.path.exists(input_csv):
        logger.error(f"Input file not found: {input_csv}")
        return None

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return None
    
    initial_count = len(df)
    if initial_count == 0:
        logger.warning("Input CSV is empty.")
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        return output_csv

    # If no structure column, fall back to exact row duplication
    if 'Structure' not in df.columns and 'structure' not in df.columns:
        logger.warning(f"'Structure/structure' column not found in {input_csv}. Deduplicating by exact row match.")
        df_dedup = df.drop_duplicates()
        df_dedup.to_csv(output_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Deduplicated (Exact Match): {initial_count} -> {len(df_dedup)}")
        return output_csv

    # Use StructureMatcher
    if matcher is None:
        # Default matcher with generous tolerance for "same structure"
        # ltol=0.2, stol=0.3, angle_tol=5 are pymatgen defaults
        matcher = StructureMatcher()

    # We will group by composition first to speed up comparison
    # Assuming '组分' is the composition/formula column based on previous files.
    # If not present, we will try to parse structure to get composition.
    
    # Add a temporary index column to keep track of rows
    df['original_index'] = df.index
    
    unique_indices = []
    

    # Helper to parse structure
    def normalize_path(path_str):
        if pd.isna(path_str): return None
        # Handle windows/unix path separators
        return str(path_str).replace('\\', '/')

    def load_structure_from_row(row, csv_dir):
        # 1. Try Loading from Relative_CIF_Path
        rel_path = _pick(row, ['relative_cif_path', 'Relative_CIF_Path'])
        if rel_path is not None:
            # Try as absolute path first
            if os.path.exists(rel_path):
                try:
                    return Structure.from_file(rel_path)
                except:
                    pass
            
            # Try relative to project root (csv_dir's parent's parent's parent...)
            # Assuming csv is in results/iteration_X/success_examples/
            # and rel_path is results/iteration_X/...
            # We need to find the project root.
            # Let's try relative to CWD (d:\XJTU\ImportantFile\aslk)
            if os.path.exists(rel_path): # Check again in CWD
                 try:
                    return Structure.from_file(rel_path)
                 except:
                    pass
            
        # 2. Try looking in cif_files subdirectory
        cif_files_dir = os.path.join(csv_dir, 'cif_files')
        if os.path.exists(cif_files_dir):
            # Try to match by filename
            cif_name = _pick(row, ['cif_file', 'CIF文件'])
            if cif_name is not None:
                 # exact match
                 p = os.path.join(cif_files_dir, cif_name)
                 if os.path.exists(p):
                     try: return Structure.from_file(p)
                     except: pass
                 
                 # Fuzzy match: the files in cif_files might have prefixes like 001_...
                 # We look for a file that ENDS with cif_name or contains it
                 for f in os.listdir(cif_files_dir):
                     if f.endswith(cif_name) or (cif_name in f):
                         try: return Structure.from_file(os.path.join(cif_files_dir, f))
                         except: pass

            # Try to match by structure ID
            sid = _pick(row, ['structure_id', '结构ID'])
            if sid is not None:
                for f in os.listdir(cif_files_dir):
                     if sid in f and f.endswith('.cif'):
                         try: return Structure.from_file(os.path.join(cif_files_dir, f))
                         except: pass
        
        # 3. Last resort: Try parsing 'Structure' string (unreliable for Pymatgen string output)
        # But maybe it is a POSCAR string?
        s_str = row.get('Structure', row.get('structure', ''))
        if s_str and isinstance(s_str, str):
            try: return Structure.from_str(s_str, fmt="poscar")
            except: pass
            try: return Structure.from_str(s_str, fmt="cif")
            except: pass

        return None

    # Parse all structures
    structures = []
    valid_indices = []
    csv_dir = os.path.dirname(os.path.abspath(input_csv))
    
    logger.info("Parsing structures for deduplication...")
    for idx, row in df.iterrows():
        s = load_structure_from_row(row, csv_dir)
        if s:
            structures.append((idx, s))
            valid_indices.append(idx)
        else:
            logger.warning(f"Could not load structure at index {idx} (ID: {_pick(row, ['structure_id', '结构ID']) or 'Unknown'})")
    
    if not structures:
        # If no structures were parsed, check if we should just warn and continue with raw duplicates?
        # User wants to deduplicate "same structure", if we can't parse, we can't deduplicate by structure.
        # Fallback to exact row duplicate.
        logger.warning("No valid structures found. Falling back to simple row deduplication.")
        df.drop_duplicates().to_csv(output_csv, index=False, encoding="utf-8-sig")
        return output_csv

    # Group by composition formula
    grouped_structures = {}
    for idx, s in structures:
        formula = s.composition.reduced_formula
        if formula not in grouped_structures:
            grouped_structures[formula] = []
        grouped_structures[formula].append((idx, s))
        
    logger.info(f"Found {len(grouped_structures)} unique compositions.")

    unique_indices = []
    
    for formula, struct_list in grouped_structures.items():
        # For each group, pick unique structures
        unique_in_group = [] # list of (idx, Structure)
        
        for idx, s in struct_list:
            is_duplicate = False
            for _, u_s in unique_in_group:
                if matcher.fit(s, u_s):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_in_group.append((idx, s))
        
        unique_indices.extend([x[0] for x in unique_in_group])

    # Create final dataframe
    # We only keep the valid unique structures? 
    # Or should we keep rows that failed parsing?
    # Usually better to keep failing rows if we aren't sure, but here the task is "success examples",
    # so maybe only valid ones are useful. 
    # Let's keep valid ones.
    
    df_dedup = df.loc[sorted(unique_indices)].copy()
    
    # Remove our temp column if we added one
    if 'original_index' in df_dedup.columns:
        df_dedup = df_dedup.drop(columns=['original_index'])
        
    df_dedup.to_csv(output_csv, index=False, encoding="utf-8-sig")
    
    logger.info(f"Deduplication complete: {initial_count} -> {len(df_dedup)} structures.")
    
    return output_csv

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 2:
        inp = sys.argv[1]
        outp = sys.argv[2]
        deduplicate_success_materials(inp, outp)
    else:
        print("Usage: python deduplicate_success.py input.csv output.csv")
