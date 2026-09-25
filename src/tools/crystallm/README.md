# CrystaLLM structure generation

This directory contains the local CrystaLLM implementation used by ADLM. The
workflow entry point is `src/tools/structure_parallel.py`, which calls
`src/tools/crystallm_wrapper.py`. The wrapper uses this repository's CrystaLLM
source and does not fall back to pymatgen, mock structures, or another
structure generator.

## Requirements

Run commands from the repository root in a Python environment that has the
packages in `requirements.txt` installed:

```powershell
conda activate kappap
python -m pip install -r requirements.txt
```

The local pretrained model required by the current adapter is:

```text
src/tools/crystallm/pre-trained-model/crystallm_v1_small/ckpt.pt
```

A CUDA-capable GPU is recommended for the full workflow. Use `device="cuda"`
for GPU generation or explicitly use `device="cpu"` for CPU generation. The
workflow does not silently change an explicit CPU request to CUDA.

## Direct Python API

The package is located below `src/tools`, so set `PYTHONPATH` when using it
directly from the repository root:

```powershell
conda activate kappap
$env:PYTHONPATH = (Resolve-Path .\src\tools).Path
python -c "from crystallm import generate_crystal_from_composition; print('crystallm_import=ok')"
```

Example generation:

```python
from crystallm import generate_crystal_from_composition

result = generate_crystal_from_composition(
    composition="GaN",
    device="cuda",
    num_samples=1,
    seed=42,
    output_dir="tmp/crystallm_generation",
)

if not result["success"]:
    raise RuntimeError(result.get("error", "CrystaLLM generation failed"))

print(result["cif_file_paths"])
```

The direct API returns a dictionary containing these fields on success:

```text
success
cif_file_paths
cif_filenames
cif_directory
composition
generation_id
num_generated
cif_source
model_used
device
frontend_structures
num_frontend_structures
```

On failure, inspect `success` and `error`. A successful result always points to
CIF files written under the requested output directory. The generated
structure directory contains:

```text
<output_dir>/<composition>/
|- prompts/
|- generated/
|- processed/
```

The `processed/` directory is the preferred source of CIF files.

## ADLM workflow usage

For the complete BO-LLM workflow, run the command in the repository root:

```powershell
conda activate kappap
python .\main.py `
  --runtime workflow `
  --max-iterations 3 `
  --n-structures 1 `
  --num-gpus 1 `
  --device cuda `
  --seed 42 `
  --init-data data/processed_data.csv `
  --init-doc doc/Theoretical_principle_document.md `
  --params-csv config/agentos_params.csv
```

For the BO-only baseline:

```powershell
conda activate kappap
python .\main_bo_only.py `
  --start-iteration 1 `
  --max-iterations 3 `
  --n-structures 1 `
  --num-gpus 1 `
  --device cuda `
  --seed 42 `
  --init-data data/processed_data.csv
```

Use `--reset` only when starting a new experiment. Omit it when recovering an
interrupted run. The main repository README documents the complete parameter
set, checkpoints, model assets, API configuration, and failure behavior.

## Structure-level retry behavior

When the workflow receives a local structure-quality failure, it retries the
same composition once with a deterministic new seed. If the retry succeeds,
the valid CIF is promoted to the standard processed-structure directory. If
both attempts are structurally invalid, the composition is recorded as
`skipped` and is excluded from relaxation, phonon, and thermal-conductivity
steps. Attempt outputs are preserved under:

```text
processed_structures/.crystallm_attempts/<formula>/attempt_1/
processed_structures/.crystallm_attempts/<formula>/attempt_2/
```

Backend, dependency, model-loading, and system-level errors remain fatal.

## Verification

From the repository root:

```powershell
conda activate kappap
python -m compileall -q src\tools\crystallm src\tools\crystallm_wrapper.py src\tools\structure_parallel.py
python -c "import sys; sys.path.insert(0, r'src/tools'); import crystallm; print('crystallm_import=ok')"
```

The direct API performs model inference and may require substantial GPU memory.
For a full reproducibility check, also verify the model file and run the
workflow-level tests described in `README.md`.
