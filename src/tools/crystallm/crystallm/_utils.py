"""
A collection of utility functions for parsing, manipulating, and generating
crystallographic information file (CIF) data, primarily using the pymatgen library.
"""

import math
import re
from typing import Dict, List, Any, Union

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.io.cif import CifBlock
# 正确的导入路径：SpaceGroup 位于 pymatgen.symmetry.groups
from pymatgen.symmetry.groups import SpaceGroup

# --- 模块级常量 ---
# 将常量移至模块顶部，并使用大写命名，以提高性能和可读性
NOBLE_GAS_VDW_RADII: Dict[str, float] = {
    "He": 1.40, "Ne": 1.54, "Ar": 1.88, "Kr": 2.02, "Xe": 2.16, "Rn": 2.20,
}
ALLEN_ELECTRONEGATIVITY: Dict[str, float] = {
    "He": 4.16, "Ne": 4.79, "Ar": 3.24,
}


def get_unit_cell_volume(
    a: float, b: float, c: float,
    alpha_deg: float, beta_deg: float, gamma_deg: float
) -> float:
    """
    Calculates the volume of a unit cell given its lattice parameters.

    Args:
        a, b, c: The three lattice vector lengths.
        alpha_deg, beta_deg, gamma_deg: The three lattice angles in degrees.

    Returns:
        The volume of the unit cell in the same cubic units as the lengths.
    """
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)

    cos_a, cos_b, cos_g = math.cos(alpha), math.cos(beta), math.cos(gamma)

    volume_squared = (a**2 * b**2 * c**2 *
                      (1 - cos_a**2 - cos_b**2 - cos_g**2 + 2 * cos_a * cos_b * cos_g))
    
    # Handle potential floating point inaccuracies leading to a negative radicand
    if volume_squared < 0:
        return 0.0
    return math.sqrt(volume_squared)


def get_atomic_props_block_for_formula(formula: str, oxi: bool = False) -> str:
    """
    Creates a CIF atomic properties block for a given chemical formula string.

    Args:
        formula: The chemical formula string (e.g., "Ag4Se4").
        oxi: Whether to include oxidation state information.

    Returns:
        A string representing the CIF atomic properties loop.
    """
    comp = Composition(formula)
    return get_atomic_props_block(comp, oxi)


def get_atomic_props_block(composition: Composition, oxi: bool = False) -> str:
    """
    Generates a CIF loop block with atomic properties for a composition.

    Args:
        composition: A pymatgen Composition object.
        oxi: Whether to include oxidation state information.

    Returns:
        A string representing the CIF atomic properties loop.
    """
    data: Dict[str, List[Any]] = {
        "_atom_type_symbol": [],
        "_atom_type_electronegativity": [],
        "_atom_type_radius": [],
        "_atom_type_ionic_radius": [],
    }

    elements = sorted(composition.elements, key=lambda el: el.X if (hasattr(el, 'X') and el.X is not None and not math.isnan(el.X)) else 999)

    for el in elements:
        symbol = str(el)
        data["_atom_type_symbol"].append(symbol)

        # Safely get Pauling electronegativity
        pauling_X = None
        if hasattr(el, 'X') and el.X is not None and not math.isnan(el.X):
            pauling_X = el.X
        electronegativity = pauling_X if pauling_X is not None else ALLEN_ELECTRONEGATIVITY.get(symbol, 0.0)
        data["_atom_type_electronegativity"].append(f"{electronegativity:.4f}")

        # Safely get atomic radius
        atomic_radius = 0.0
        if hasattr(el, 'atomic_radius') and el.atomic_radius is not None:
            atomic_radius = el.atomic_radius
        else:
            atomic_radius = NOBLE_GAS_VDW_RADII.get(symbol, 0.0)
        data["_atom_type_radius"].append(f"{atomic_radius:.4f}")
        
        # Safely get average ionic radius
        avg_ionic_radius = 0.0
        if hasattr(el, 'average_ionic_radius') and el.average_ionic_radius is not None:
            avg_ionic_radius = el.average_ionic_radius
        data["_atom_type_ionic_radius"].append(f"{avg_ionic_radius:.4f}")

    loop_vals = list(data.keys())

    if oxi:
        oxi_numbers = []
        ionic_radii = []
        for el in elements:
            # Safely get oxidation state
            oxi_state = 0.0
            if hasattr(el, 'oxi_state') and el.oxi_state is not None:
                oxi_state = float(el.oxi_state)
            oxi_numbers.append(oxi_state)
            
            # Safely get ionic radius
            ionic_radius = 0.0
            if hasattr(el, 'ionic_radius') and el.ionic_radius is not None:
                ionic_radius = el.ionic_radius
            ionic_radii.append(f"{ionic_radius:.4f}")
        
        data["_atom_type_oxidation_number"] = oxi_numbers
        data["_atom_type_ionic_radius"] = ionic_radii
        loop_vals.append("_atom_type_oxidation_number")

    cif_block = CifBlock(data, [loop_vals], "atomic_props")
    return str(cif_block).replace("data_atomic_props\n", "")


def _matrix_to_cif_expression(row, translation):
    """
    Convert a row of the rotation matrix and translation to CIF expression format.
    
    Args:
        row: A row of the rotation matrix [a, b, c]
        translation: The translation component
        
    Returns:
        String representation in CIF format (e.g., 'x', '-y', 'x+1/2')
    """
    terms = []
    
    # Handle x component
    if abs(row[0]) > 1e-6:
        if abs(row[0] - 1) < 1e-6:
            terms.append('x')
        elif abs(row[0] + 1) < 1e-6:
            terms.append('-x')
        else:
            terms.append(f'{row[0]:.0f}*x')
    
    # Handle y component
    if abs(row[1]) > 1e-6:
        if abs(row[1] - 1) < 1e-6:
            terms.append('y')
        elif abs(row[1] + 1) < 1e-6:
            terms.append('-y')
        else:
            terms.append(f'{row[1]:.0f}*y')
    
    # Handle z component
    if abs(row[2]) > 1e-6:
        if abs(row[2] - 1) < 1e-6:
            terms.append('z')
        elif abs(row[2] + 1) < 1e-6:
            terms.append('-z')
        else:
            terms.append(f'{row[2]:.0f}*z')
    
    # Handle translation
    if abs(translation) > 1e-6:
        if abs(translation - 0.5) < 1e-6:
            terms.append('1/2')
        elif abs(translation + 0.5) < 1e-6:
            terms.append('-1/2')
        elif abs(translation - 0.25) < 1e-6:
            terms.append('1/4')
        elif abs(translation + 0.25) < 1e-6:
            terms.append('-1/4')
        elif abs(translation - 0.75) < 1e-6:
            terms.append('3/4')
        elif abs(translation + 0.75) < 1e-6:
            terms.append('-3/4')
        else:
            terms.append(f'{translation:.3f}')
    
    if not terms:
        return '0'
    
    return '+'.join(terms).replace('+-', '-').replace('++', '+').replace('--', '+').replace('+0', '').replace('0+', '')


def replace_symmetry_operators(cif_str: str, space_group_symbol: str) -> str:
    """
    Replaces the symmetry operators in a CIF string using the modern pymatgen API.

    Args:
        cif_str: The string content of the CIF file.
        space_group_symbol: The Hermann-Mauguin symbol (e.g., 'P m -3 m').

    Returns:
        The updated CIF string with a complete list of symmetry operators.
    """
    try:
        space_group = SpaceGroup(space_group_symbol)
        symmetry_ops = list(space_group.symmetry_ops)  # Convert set to list
    except ValueError as e:
        raise ValueError(f"Invalid space group symbol '{space_group_symbol}': {e}")

    # Convert symmetry operations to CIF format
    ops = []
    for op in symmetry_ops:
        # Get the transformation matrix and translation vector
        rotation_matrix = op.rotation_matrix
        translation_vector = op.translation_vector
        
        # Convert to CIF format: 'x, y, z' or 'x, -y, z' etc.
        x_expr = _matrix_to_cif_expression(rotation_matrix[0], translation_vector[0])
        y_expr = _matrix_to_cif_expression(rotation_matrix[1], translation_vector[1])
        z_expr = _matrix_to_cif_expression(rotation_matrix[2], translation_vector[2])
        
        ops.append(f"'{x_expr}, {y_expr}, {z_expr}'")
    data = {
        "_symmetry_equiv_pos_site_id": [str(i) for i in range(1, len(ops) + 1)],
        "_symmetry_equiv_pos_as_xyz": ops,
    }
    loops = [["_symmetry_equiv_pos_site_id", "_symmetry_equiv_pos_as_xyz"]]
    
    symm_block_str = str(CifBlock(data, loops, "symmetry"))
    symm_block_str = symm_block_str.replace("data_symmetry\n", "")

    pattern = re.compile(
        r"loop_\s+_symmetry_equiv_pos_site_id\s+_symmetry_equiv_pos_as_xyz.*?'x,\s*y,\s*z'.*?(?=\n\S|\Z)",
        re.DOTALL | re.IGNORECASE
    )

    if pattern.search(cif_str):
        return pattern.sub(symm_block_str, cif_str, count=1)

    print(f"Warning: Symmetry block not found in CIF for space group '{space_group_symbol}'. Returning original CIF.")
    return cif_str


def extract_cif_property(cif_str: str, property_tag: str) -> str:
    """
    Extracts a string value for a given tag from a CIF file.
    Handles values that are single-quoted, double-quoted, or unquoted.

    Args:
        cif_str: The string content of the CIF file.
        property_tag: The CIF tag to extract.

    Returns:
        The extracted property value as a string.
    """
    pattern = re.compile(rf"{property_tag}\s+(?:'([^']*)'|\"([^\"]*)\"|(\S+))", re.IGNORECASE)
    match = pattern.search(cif_str)
    
    if match:
        return next(g for g in match.groups() if g is not None)
        
    raise ValueError(f"Could not extract '{property_tag}' from CIF string.")


def extract_numeric_property(cif_str: str, prop: str, numeric_type: type = float) -> Union[float, int]:
    """Extracts a numeric property, removing uncertainty in parentheses."""
    value_str = extract_cif_property(cif_str, prop)
    value_str = re.sub(r'\(.*\)', '', value_str)
    return numeric_type(value_str)


# --- Specific Extraction Functions (using the generic helpers) ---

def extract_space_group_symbol(cif_str: str) -> str:
    return extract_cif_property(cif_str, "_symmetry_space_group_name_H-M")

def extract_volume(cif_str: str) -> float:
    return extract_numeric_property(cif_str, "_cell_volume")

def extract_formula_units(cif_str: str) -> int:
    return extract_numeric_property(cif_str, "_cell_formula_units_Z", numeric_type=int)

def extract_data_formula(cif_str: str) -> str:
    match = re.search(r"data_([A-Za-z0-9]+)", cif_str)
    if match:
        return match.group(1)
    raise ValueError("Could not find 'data_' line in CIF string.")

def extract_formula_nonreduced(cif_str: str) -> str:
    return extract_cif_property(cif_str, "_chemical_formula_sum")


# --- CIF Manipulation Functions ---

def semisymmetrize_cif(cif_str: str) -> str:
    """
    Reduces a CIF's symmetry operations to only the identity 'x, y, z'.

    Args:
        cif_str: The string content of the CIF file.

    Returns:
        The updated CIF string with a minimal symmetry loop.
    """
    return re.sub(
        r"(_symmetry_equiv_pos_as_xyz\s*\n)(.*?)",
        r"\1  1  'x, y, z'\n",
        cif_str,
        flags=re.DOTALL | re.IGNORECASE
    )


def replace_data_formula_with_nonreduced_formula(cif_str: str) -> str:
    """
    Replaces the `data_...` line with the value from `_chemical_formula_sum`.

    Args:
        cif_str: The string content of the CIF file.

    Returns:
        The updated CIF string.
    """
    try:
        chemical_formula = extract_formula_nonreduced(cif_str)
        # Clean the formula: remove quotes and spaces
        cleaned_formula = re.sub(r"['\s]", "", chemical_formula)
        # Replace the `data_...` line
        return re.sub(r"^(data_)([^\n]*)", r"\1" + cleaned_formula, cif_str, count=1, flags=re.MULTILINE)
    except ValueError as e:
        raise ValueError(f"Cannot replace data formula: Could not find _chemical_formula_sum. Original error: {e}")


def add_atomic_props_block(cif_str: str, oxi: bool = False) -> str:
    """
    Adds a CIF block with atomic properties to a CIF string using a safe regex substitution.
    The block is inserted just before the symmetry space group definition.
    """
    try:
        formula = extract_formula_nonreduced(cif_str)
        comp = Composition(formula)
    except ValueError as e:
        raise ValueError(f"Cannot add atomic props block: Failed to extract formula. Original error: {e}")

    block = get_atomic_props_block(composition=comp, oxi=oxi)
    
    # 使用 re.sub 进行安全的替换，而不是字符串切片
    # 这确保了只在找到匹配模式时才进行操作，且操作是原子的
    insertion_point_pattern = r"(_symmetry_space_group_name_H-M)"
    
    # 将原子属性块和两个换行符插入到找到的 insertion_point 之前
    replacement_str = block + "\n\n" + r"\1"
    
    modified_cif, num_subs = re.subn(insertion_point_pattern, replacement_str, cif_str, count=1)

    if num_subs == 0:
        raise ValueError(f"Cannot add atomic props block: Insertion point '{insertion_point_pattern}' not found in CIF.")
    
    return modified_cif


def remove_atom_props_block(cif_str: str) -> str:
    """
    Removes the atomic properties loop block from a CIF string safely.
    """
    # 这个模式精确匹配从loop_开始，包含_atom_type_symbol，直到_symmetry...之前的所有内容
    # 它捕获_symmetry...部分，以便在替换时将其保留下来
    pattern = re.compile(
        r"loop_\s+_atom_type_symbol.*?(_symmetry_space_group_name_H-M)",
        re.DOTALL | re.IGNORECASE
    )
    
    # 使用 re.sub 将匹配到的整个块替换为被捕获的组(\1)，有效地删除了前面的原子属性块。
    modified_cif, num_subs = pattern.subn(r"\1", cif_str, count=1)
    
    if num_subs == 0:
        print("Warning: Atomic props block to remove was not found.")
        
    return modified_cif


def round_numbers(cif_str: str, decimal_places: int = 4) -> str:
    """
    Rounds all floating-point numbers in a CIF string to a set number of decimal places.

    Args:
        cif_str: The string content of the CIF file.
        decimal_places: The number of decimal places to round to.

    Returns:
        The updated CIF string with rounded numbers.
    """
    # Regex to find floating point numbers, avoiding integers
    pattern = re.compile(r"([-+]?\d+\.\d+)")

    def round_match(match: re.Match) -> str:
        number_str = match.group(1)
        # Check if rounding is necessary
        if len(number_str.split('.')[-1]) <= decimal_places:
            return number_str
        return f"{float(number_str):.{decimal_places}f}"

    return pattern.sub(round_match, cif_str)


# --- Utility Functions ---

def array_split(arr: List[Any], num_splits: int) -> List[np.ndarray]:
    """
    Splits a list into a specified number of sub-lists using numpy.
    """
    return np.array_split(np.array(arr), num_splits)


def embeddings_from_csv(embedding_csv: str) -> Dict[str, np.ndarray]:
    """
    Reads element embedding vectors from a CSV file.

    Args:
        embedding_csv: Path to the CSV file with an 'element' index column.

    Returns:
        A dictionary mapping element symbols to their numpy embedding vectors.
    """
    df = pd.read_csv(embedding_csv, index_col="element")
    return {element: df.loc[element].to_numpy() for element in df.index}