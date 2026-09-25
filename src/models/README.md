# GPR thermal-conductivity model

This directory contains the Gaussian-process regression (GPR) training script
used by the ADLM workflow. The model predicts the logarithm of the thermal
conductivity from the composition features in the input CSV.

## Input data contract

The default feature order is defined by `ALL_ELEMENTS` in
`src/models/train_gpr_model.py`:

```text
Ag, As, Bi, Cu, Ge, In, Pb, S, Sb, Se, Sn, Te, Ti, V
```

The input CSV must contain all 14 element columns and the target column:

```text
k(W/Km)
```

The repository-root initial dataset is:

```text
data/processed_data.csv
```

The training script applies `log(y)` to the target and stores a scaler for the
14 composition features.

## Train from the repository root

Use the same environment as the main workflow:

```powershell
conda activate kappap
python -m pip install -r requirements.txt
python .\src\models\train_gpr_model.py `
  --input data\processed_data.csv `
  --output llm\models\GPR\iteration_0
```

Both `--input` and `--output` accept relative or absolute paths. Relative
paths are resolved from the repository root by the script. The workflow passes
its own iteration-specific input and output paths when training models for
later iterations.

## Generated artifacts

The output directory contains:

```text
llm/models/GPR/iteration_0/
|- gpr_thermal_conductivity.joblib
|- gpr_scaler.joblib
|- model_metadata.json
|- Final_Model_Comparison.png
|- Best_Model_Prediction.png
```

`model_metadata.json` records the training timestamp, cross-validation scores,
best fold, test metrics, and fitted kernel. The two joblib files are the model
and feature scaler consumed by the BO workflow.

## Load and predict

Run the following from the repository root after training:

```python
import joblib
import numpy as np

model = joblib.load("llm/models/GPR/iteration_0/gpr_thermal_conductivity.joblib")
scaler = joblib.load("llm/models/GPR/iteration_0/gpr_scaler.joblib")

# Ag2Se; columns follow the ALL_ELEMENTS order above.
x_new = np.array([[2.0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0]])
x_scaled = scaler.transform(x_new)
y_pred_log, y_std = model.predict(x_scaled, return_std=True)
k_pred = np.exp(y_pred_log)

print(f"predicted k: {k_pred[0]:.4f} W/Km")
print(f"log-space standard deviation: {y_std[0]:.4f}")
```

The model prediction is converted back from log space with `exp`. The
uncertainty returned by scikit-learn is in log space.

## Verification

From the repository root:

```powershell
conda activate kappap
python -m compileall -q src\models\train_gpr_model.py
python .\src\models\train_gpr_model.py --help
```

Training is deterministic for the configured train/test split, cross-validation
split, and model random state. It still requires the input data and the normal
Python scientific dependencies. Training creates or overwrites only the
specified output directory artifacts.
