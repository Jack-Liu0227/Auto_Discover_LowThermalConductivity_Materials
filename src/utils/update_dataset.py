
import os
import sys
import pandas as pd
import logging
from pymatgen.core import Composition

logger = logging.getLogger(__name__)


def _pick(row, keys):
    for key in keys:
        value = row.get(key)
        if pd.isna(value) or value == "":
            continue
        return value
    return None

def update_dataset(
    success_csv: str = "results/iteration_1/success_examples/success_materials_deduped.csv",
    origin_csv: str = "data/iteration_0/data.csv",
    output_dir: str = "data/iteration_1"
):
    """
    Update dataset with success examples.
    
    1. Reads origin dataset (features + target) - 应使用上一轮的data.csv作为基础
    2. Reads success examples
    3. Featurizes success examples to match origin schema (Elemental fractions)
    4. Merges and saves to output_dir
    
    注意: origin_csv 应该是上一轮iteration的data.csv (如 iteration_0/data.csv)
    这样可以确保数据是在上一轮基础上累积的
    """
    if not os.path.exists(success_csv):
        logger.warning(f"Success CSV not found: {success_csv}. No update performed.")
        return None

    if not os.path.exists(origin_csv):
        logger.error(f"Origin CSV not found: {origin_csv}")
        return None

    try:
        df_origin = pd.read_csv(origin_csv)
        df_success = pd.read_csv(success_csv)
    except Exception as e:
        logger.error(f"Error reading CSVs: {e}")
        return None

    if len(df_success) == 0:
        logger.warning("Success CSV is empty.")
        return None

    # Columns in origin: ['Formula', 'Ag', 'As', ..., 'k(W/Km)']
    # We need to transform df_success to this format.
    
    # Identify feature columns (elements)
    # Exclude 'Formula' and target 'k(W/Km)'
    feature_cols = [c for c in df_origin.columns if c not in ['Formula', 'k(W/Km)', 'Unnamed: 0']]
    
    new_rows = []
    
    logger.info(f"Processing {len(df_success)} new materials...")
    
    for idx, row in df_success.iterrows():
        # 优先读取英文列名，同时兼容旧中文列名
        formula_str = _pick(row, ['formula', 'composition', '组分'])
        kappa = _pick(row, ['thermal_conductivity_w_mk', '热导率(W/m·K)', '热导率 (W/m·K)'])
        
        if pd.isna(formula_str) or pd.isna(kappa):
            continue
            
        try:
            # 使用 formula 列来解析元素组成
            # formula 格式如 "Ag5 Sb6 As4"，需要去除空格才能被 pymatgen 解析
            formula_for_parse = formula_str.replace(' ', '')
            comp = Composition(formula_for_parse)

            el_dict = comp.get_el_amt_dict()
            # Do NOT normalize to sum=1. The original dataset seems to use raw atomic counts (subscripts).
            # total_atoms = sum(el_dict.values())
            
            # Formula 列使用解析后的化学式（去除空格后的格式，如 "Ag5Sb6As4"）
            row_data = {'Formula': formula_for_parse, 'k(W/Km)': kappa}
            
            for el in feature_cols:
                # Use raw amount
                row_data[el] = el_dict.get(el, 0.0)
                
            new_rows.append(row_data)
            logger.debug(f"  Added: {formula_for_parse} (k={kappa})")
            
        except Exception as e:
            logger.warning(f"Failed to process {formula_str}: {e}")
            
    if not new_rows:
        logger.warning("No valid new rows generated.")
        return None
        
    df_new = pd.DataFrame(new_rows)
    
    # Concatenate
    df_combined = pd.concat([df_origin, df_new], ignore_index=True)
    
    # -------------------------------------------------------------------------
    # Deduplicate: Keep lowest thermal conductivity for same Formula
    # -------------------------------------------------------------------------
    if 'k(W/Km)' in df_combined.columns:
        logger.info("Deduplicating entries based on Formula (keeping lowest k)...")
        
        # 1. Normalize Formula: remove spaces to treat "Ag5 Sn5" and "Ag5Sn5" as same
        if 'Formula' in df_combined.columns:
            df_combined['Formula'] = df_combined['Formula'].astype(str).str.replace(' ', '')
            subset_cols = ['Formula']
        else:
            logger.warning("'Formula' column missing, falling back to composition features for deduplication.")
            subset_cols = feature_cols

        # 2. Sort by k(W/Km) ascending so that 'first' keeps the lowest value
        df_combined.sort_values(by='k(W/Km)', ascending=True, inplace=True)
        
        # 3. Deduplicate 
        df_combined.drop_duplicates(subset=subset_cols, keep='first', inplace=True)
    else:
        logger.warning("'k(W/Km)' column not found in dataset, skipping deduplication.")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "data.csv")
    df_combined.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"Dataset updated. Original: {len(df_origin)}, New: {len(df_new)}, Total: {len(df_combined)}")
    logger.info(f"Saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    update_dataset()
