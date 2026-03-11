# -*- coding: utf-8 -*-
"""
GPR Model Training Script
Trains a Gaussian Process Regressor on composition-thermal conductivity data.
Uses 10-fold cross-validation to select the best model.
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import json
import copy
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))

# Import utility for elements list
sys.path.append(os.path.join(PROJECT_ROOT, 'data', 'algorithms'))
try:
    from utils import ALL_ELEMENTS
except ImportError:
    ALL_ELEMENTS = ['Ag', 'As', 'Bi', 'Cu', 'Ge', 'In', 'Pb', 'S', 'Sb', 'Se', 'Sn', 'Te', 'Ti', 'V']

def load_processed_data(data_path):
    """Load and prepare data for training."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    df = pd.read_csv(data_path)
    X = df[ALL_ELEMENTS].values
    y = df['k(W/Km)'].values
    return X, y, ALL_ELEMENTS

def plot_cross_validation(cv_scores, save_path):
    """Plot 10-fold Cross-Validation scores."""
    plt.figure(figsize=(10, 6))
    folds = range(1, len(cv_scores) + 1)
    
    bars = plt.bar(folds, cv_scores, color='skyblue', alpha=0.8, edgecolor='black', width=0.6)
    
    mean_score = np.mean(cv_scores)
    plt.axhline(y=mean_score, color='crimson', linestyle='--', linewidth=2, 
                label=f'Mean R² = {mean_score:.4f}')
    
    plt.xlabel('Fold Number', fontsize=12, fontweight='bold')
    plt.ylabel('R² Score', fontsize=12, fontweight='bold')
    plt.title('10-Fold Cross-Validation Performance', fontsize=14, fontweight='bold')
    plt.xticks(folds)
    plt.legend(loc='lower right')
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Adjust Y-axis
    max_val = max(cv_scores)
    min_val = min(cv_scores)
    padding = (max_val - min_val) * 0.2 if max_val != min_val else 0.1
    # Ensure top isn't cut off if R2 is close to 1
    top_limit = max(1.05, max_val + padding)
    bottom_limit = min(0, min_val - padding)
    plt.ylim(bottom_limit, top_limit)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_prediction(y_true, y_pred, save_path, title=None):
    """Plot Predicted vs True values."""
    if title is None:
        title = 'Model Prediction Performance'
        
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, c='royalblue', edgecolors='black', linewidth=0.5, s=60)
    
    # 45-degree line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    margin = (max_val - min_val) * 0.05
    limit_min = min_val - margin
    limit_max = max_val + margin
    
    plt.plot([limit_min, limit_max], [limit_min, limit_max], 'r--', linewidth=2, label='Ideal (y=x)')
    
    plt.xlabel('True k (W/Km)', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted k (W/Km)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(limit_min, limit_max)
    plt.ylim(limit_min, limit_max)
    
    # Add Metrics Text
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    textstr = '\n'.join((
        f'$R^2 = {r2:.4f}$',
        f'RMSE = {rmse:.4f}',
        f'MAE = {mae:.4f}'
    ))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def train_gpr_model(data_path, output_dir):
    print("=" * 60)
    print("Starting GPR Model Training Pipeline")
    print("=" * 60)
    
    # Ensure paths are absolute or relative to PROJECT_ROOT
    if not os.path.isabs(data_path):
        data_path = os.path.join(PROJECT_ROOT, data_path)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)

    # 1. Setup Output Directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 2. Load Data
    print(f"Loading data from: {data_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        X, y, feature_names = load_processed_data(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. Preprocessing (Log Transform for Target)
    y_log = np.log(y)
    
    # 4. Split Data (10% Test, 90% Train+Val)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_log, test_size=0.1, random_state=42
    )
    
    print(f"Total samples: {len(X)}")
    print(f"Training+Validation samples: {len(X_trainval)}")
    print(f"Test samples: {len(X_test)}")

    # 5. Feature Scaling
    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval)
    # Important: Transform test set with the SAME scaler fitted on training data
    X_test_scaled = scaler.transform(X_test)

    # 6. Kernel Definition
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) * \
             RationalQuadratic(alpha=1.0, length_scale=1.0) + \
             WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))

    # 7. 10-Fold Cross-Validation
    print("\nRunning 10-Fold Cross-Validation on TrainVal set...")
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    
    cv_scores = []
    
    # We will track the actual best model instance
    best_fold_score = -np.inf
    best_model = None
    best_fold_idx = -1

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_trainval_scaled), 1):
        X_fold_train, X_fold_val = X_trainval_scaled[train_idx], X_trainval_scaled[val_idx]
        y_fold_train, y_fold_val = y_trainval[train_idx], y_trainval[val_idx]
        
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=42,
            alpha=1e-6
        )
        
        gpr.fit(X_fold_train, y_fold_train)
        y_pred_val = gpr.predict(X_fold_val)
        score = r2_score(y_fold_val, y_pred_val)
        cv_scores.append(score)
        
        # Save the best model
        if score > best_fold_score:
            best_fold_score = score
            # Deep copy to ensure we keep this specific state
            best_model = copy.deepcopy(gpr)
            best_fold_idx = fold

        print(f"  Fold {fold}: R² = {score:.4f}")

    print(f"\nAverage CV R²: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
    print(f"Best Fold: {best_fold_idx} (R² = {best_fold_score:.4f})")
    print("NOTE: Saving the specific model trained on Fold {}, NOT retraining on full data.".format(best_fold_idx))

    # 8. Save Final Model Comparison (CV Plot)
    plot_cross_validation(cv_scores, os.path.join(output_dir, 'Final_Model_Comparison.png'))

    # 9. Evaluate Best Model on Test Set
    # This evaluates the partial model (trained on 90% of TrainVal) on the unseen Test Set
    if best_model is None:
        print("Error: No model trained.")
        return

    y_pred_log, y_std = best_model.predict(X_test_scaled, return_std=True)
    y_pred = np.exp(y_pred_log)
    y_test_original = np.exp(y_test)
    
    test_r2 = r2_score(y_test_original, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
    
    print(f"\nTest Set Evaluation (using Best Fold Model):")
    print(f"  R²: {test_r2:.4f}")
    print(f"  RMSE: {test_rmse:.4f} W/Km")

    # 10. Save Best Model Prediction Plot
    plot_prediction(y_test_original, y_pred, 
                   os.path.join(output_dir, 'Best_Model_Prediction.png'),
                   title=f'Best Fold Model (Fold {best_fold_idx}) on Test Set')

    # 11. Save Models
    print("\nSaving Artifacts...")
    joblib.dump(best_model, os.path.join(output_dir, 'gpr_thermal_conductivity.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'gpr_scaler.joblib'))
    
    # Save metadata
    meta_info = {
        'training_timestamp': datetime.now().isoformat(),
        'model_source': f'Best Fold ({best_fold_idx}) from 10-Fold CV',
        'n_samples_trained_on': len(X_trainval) * 0.9, # Approx (9/10ths)
        'cv_scores': [float(x) for x in cv_scores],
        'cv_mean_r2': float(np.mean(cv_scores)),
        'best_fold_val_r2': float(best_fold_score),
        'test_r2': float(test_r2),
        'test_rmse': float(test_rmse),
        'final_kernel': str(best_model.kernel_)
    }
    
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(meta_info, f, indent=4)

    print("\nAll tasks completed successfully.")
    print(f"Outputs located in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GPR Model for Thermal Conductivity')
    
    # Default paths
    default_input = os.path.join('data', 'iteration_0', 'data.csv')
    default_output = os.path.join('models', 'GPR', 'iteration_0')
    
    parser.add_argument('--input', type=str, default=default_input,
                      help=f'Path to input CSV file (default: {default_input})')
    parser.add_argument('--output', type=str, default=default_output,
                      help=f'Path to output directory (default: {default_output})')
    
    args = parser.parse_args()
    
    train_gpr_model(args.input, args.output)
