"""
Thermal Conductivity Calculator
Simplified wrapper for ai4kappa library adapted for MCP tools
"""
import os
import glob
import shutil
import tempfile
import pandas as pd
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from streamlit_scripts.file_op import get_dir_crystalline_data, create_id_prop, clean_root_dir
    from streamlit_scripts import chang_model as cm
    from streamlit_scripts import calculate_K as calk
    import predict
    KAPPA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import kappa modules: {e}")
    KAPPA_AVAILABLE = False


class ThermalConductivityCalculator:
    """
    Thermal conductivity calculator for materials using CIF files.
    
    This calculator supports two methods:
    1. Kappa-P: Physics-based Slack model
    2. Kappa-MTP: Machine learning prediction
    """
    
    def __init__(self, cif_dir_path: str, model_dir: str = "model") -> None:
        """
        Initialize the calculator.
        
        Args:
            cif_dir_path: Path to directory containing CIF files
            model_dir: Path to model directory (default: "model")
        """
        import uuid
        self.cif_dir_path = os.path.abspath(cif_dir_path)
        self.model_dir = model_dir
        self.lib_path = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.lib_path, model_dir)
        
        # Create unique working directory to avoid conflicts in multi-process environment
        self.unique_id = str(uuid.uuid4())
        self.root_dir = os.path.join(self.lib_path, f"root_dir_{self.unique_id}")
        
        # Create working directory
        os.makedirs(self.root_dir, exist_ok=True)
        
        # Validate and copy CIF files
        self._validate_and_copy_cifs()
    
    def _validate_and_copy_cifs(self) -> None:
        """Validate CIF directory and copy files to working directory."""
        if not os.path.exists(self.cif_dir_path):
            # Try to resolve relative to current directory if absolute path fails
            if os.path.exists(os.path.abspath(self.cif_dir_path)):
                self.cif_dir_path = os.path.abspath(self.cif_dir_path)
            else:
                raise FileNotFoundError(f"CIF directory not found: {self.cif_dir_path}")
        
        cif_files = glob.glob(os.path.join(self.cif_dir_path, '*.cif'))
        if not cif_files:
            raise ValueError(f"No CIF files found in: {self.cif_dir_path}")
        
        print(f"[{self.unique_id}] Found {len(cif_files)} CIF files")
        
        # Copy CIF files to working directory
        for cif_file in cif_files:
            dest_file = os.path.join(self.root_dir, os.path.basename(cif_file))
            shutil.copy2(cif_file, dest_file)
    
    def _prepare_data(self) -> pd.DataFrame:
        """Prepare data for thermal conductivity calculation."""
        try:
            # Create property files and copy atom initialization data
            cif_path_list, cif_name_list = create_id_prop(self.root_dir)
            
            # Copy atom_init.json to the unique root dir
            atom_init_src = os.path.join(self.lib_path, "atom_init.json")
            if os.path.exists(atom_init_src):
                shutil.copy2(atom_init_src, self.root_dir)
            else:
                print(f"Warning: atom_init.json not found at {atom_init_src}")

            results_csv_path = "test_results.csv"
            
            # Get model paths
            model_path_list, model_name_list = cm.get_model_path(self.model_path)
            
            # Run first model prediction
            # COPY MODEL TO UNIQUE ROOT DIR instead of LIB PATH
            target_model_path = os.path.join(self.root_dir, "pre-trained.pth.tar")
            shutil.copy2(model_path_list[0], target_model_path)
            
            # Predict using the model in the unique dir
            predict.main(target_model_path, self.root_dir)
            pre_df = cm.get_pre_dataframe(results_csv_path, model_name_list[0])
            
            # Run additional model predictions
            for model_path, model_name in zip(model_path_list[1:], model_name_list[1:]):
                # Copy next model to unique dir
                shutil.copy2(model_path, target_model_path)
                
                predict.main(target_model_path, self.root_dir)
                pre_df1 = cm.get_pre_dataframe(results_csv_path, model_name)
                pre_df = pd.merge(pre_df, pre_df1, left_index=True, right_index=True)
            
            # Get crystal structure data
            all_cry_df = get_dir_crystalline_data(self.root_dir)
            if all_cry_df.empty:
                raise ValueError("Failed to extract crystal data")
            
            # Merge crystal data with predictions
            whole_info_df = pd.merge(all_cry_df, pre_df, left_index=True, right_index=True)
            
            if whole_info_df.empty:
                raise ValueError("Failed to merge crystal data with predictions")
            
            return whole_info_df
            
        except Exception as e:
            print(f"[{self.unique_id}] Error in _prepare_data: {str(e)}")
            raise
    
    def calculate_kappa_p(self) -> pd.DataFrame:
        """
        Calculate thermal conductivity using Kappa-P method (Slack model).
        
        Returns:
            pd.DataFrame: Results with thermal conductivity values
        """
        try:
            print("\n=== Kappa-P Method (Slack Model) ===")
            df = self._prepare_data()
            
            # Calculate Debye temperature
            Debye_df = calk.cal_Debye_T(df)
            
            # Calculate Grüneisen parameter
            gamma_df = calk.cal_gamma(Debye_df)
            
            # Calculate A parameter
            A_df = calk.cal_A(gamma_df, 1)
            
            # Calculate thermal conductivity using Slack model
            result_df = calk.cal_K_Slack(A_df)
            
            return result_df
            
        finally:
            self._cleanup()
    
    def calculate_kappa_mtp(self) -> pd.DataFrame:
        """
        Calculate thermal conductivity using Kappa-MTP method (ML prediction).
        
        Returns:
            pd.DataFrame: Results with predicted thermal conductivity values
        """
        try:
            print("\n=== Kappa-MTP Method (Machine Learning) ===")
            df = self._prepare_data()
            
            # Calculate Debye temperature
            Debye_df = calk.cal_Debye_T(df)
            
            # Calculate Grüneisen parameter
            gamma_df = calk.cal_gamma(Debye_df)
            
            # Calculate thermal conductivity using MTP model
            result_df = calk.by_MTP(gamma_df)
            
            return result_df
            
        finally:
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Clean up temporary files."""
        print(f"[{self.unique_id}] Cleaning up temporary files...")
        try:
            if os.path.exists(self.root_dir):
                shutil.rmtree(self.root_dir)
                print(f"[{self.unique_id}] Removed temporary directory: {self.root_dir}")
        except Exception as e:
            print(f"[{self.unique_id}] Error cleaning up: {e}")
        
        print(f"[{self.unique_id}] Cleanup completed")


def calculate_kappa_p(cif_dir: str) -> pd.DataFrame:
    """
    Calculate thermal conductivity using Kappa-P method.
    
    Args:
        cif_dir: Path to directory containing CIF files
        
    Returns:
        pd.DataFrame: Calculation results
    """
    calculator = ThermalConductivityCalculator(cif_dir)
    return calculator.calculate_kappa_p()


def calculate_kappa_mtp(cif_dir: str) -> pd.DataFrame:
    """
    Calculate thermal conductivity using Kappa-MTP method.
    
    Args:
        cif_dir: Path to directory containing CIF files
        
    Returns:
        pd.DataFrame: Prediction results
    """
    calculator = ThermalConductivityCalculator(cif_dir)
    return calculator.calculate_kappa_mtp()


# Helper function to convert material data dict to CIF file
def create_cif_from_material_data(material_data: Dict[str, Any], output_dir: str) -> str:
    """
    Create a CIF file from material data dictionary.
    
    Args:
        material_data: Dictionary with material properties
        output_dir: Directory to save CIF file
        
    Returns:
        str: Path to created CIF file
    """
    # This is a placeholder - in real implementation, you would need
    # to convert material_data to proper CIF format
    # For now, we'll create a minimal CIF file
    
    composition = material_data.get("composition", "Unknown")
    cif_content = f"""data_{composition}
_chemical_name '{composition}'
_cell_length_a {material_data.get('volume', 10.0)**(1/3)}
_cell_length_b {material_data.get('volume', 10.0)**(1/3)}
_cell_length_c {material_data.get('volume', 10.0)**(1/3)}
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_name_H-M_alt 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
{composition}1 0.0 0.0 0.0
"""
    
    os.makedirs(output_dir, exist_ok=True)
    cif_path = os.path.join(output_dir, f"{composition}.cif")
    
    with open(cif_path, 'w') as f:
        f.write(cif_content)
    
    return cif_path


def is_kappa_available() -> bool:
    """Check if kappa library is available."""
    return KAPPA_AVAILABLE

