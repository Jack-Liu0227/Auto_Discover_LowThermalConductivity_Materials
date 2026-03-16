# ADLM

Auto_Discover_LowThermalConductivity_Materials (ADLM) is an automated low-thermal-conductivity materials discovery workflow. It combines Bayesian Optimization, LLM-based screening, structure generation, thermal conductivity prediction, phonon or stability validation, and iterative dataset updates.

[阅读中文文档](./README.zh-CN.md)

## What This Repository Provides

The repository currently exposes two runnable entry points:

- `main.py`: LLM-assisted workflow built on Agno
- `main_bo_only.py`: BO-only workflow without LLM screening or theory-document update

Both workflows share the same general iteration loop:

1. Train or refresh the surrogate model from accumulated data.
2. Sample candidate compositions with Bayesian Optimization.
3. Optionally screen candidates with an LLM.
4. Generate structures and run downstream relaxation, phonon, and thermal-conductivity calculations.
5. Extract successful or stable materials.
6. Update the dataset for the next iteration.
7. In LLM mode, update the theory document version after each round.

## Repository Layout

```text
aslk/
|- main.py
|- main_bo_only.py
|- README.md
|- README.zh-CN.md
|- .env.example
|- requirements.txt
|- uv.lock
|- config/
|  |- config.yaml
|  |- agentos_params.csv
|- src/
|  |- agents/
|  |- database/
|  |- generators/
|  |- models/
|  |  |- README.md
|  |- tools/
|  |- utils/
|  |- workflow/
|- data/
|  |- processed_data.csv
|- doc/
|  |- Theoretical_principle_document.md
```

> [!NOTE]
> This README is calibrated to the files currently tracked by git. Runtime-generated folders such as `llm/` and `bo/` are outputs, not tracked source files.

## Requirements

- Python `3.10+`
- `uv` recommended for dependency management
- CUDA-capable GPU recommended for full structure and property calculations

Key dependencies declared in `requirements.txt` and locked in `uv.lock` include:

- `agno`
- `google-adk`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `ase`, `pymatgen`
- `scikit-learn`, `xgboost`, `joblib`
- `torch`, `torchvision`, `torchaudio`
- `mattersim`

> [!IMPORTANT]
> `requirements.txt` includes CUDA 12.4 PyTorch packages. If your machine uses a different CUDA stack, install a matching PyTorch build manually.

## Installation

### `pip`

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you use `uv`, the lockfile tracked in this repository is `uv.lock`.

Manual CUDA PyTorch example:

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

## Environment Variables

Create `.env` from the template:

```bash
copy .env.example .env
```

The current template defines:

- `WORKFLOW_MODEL`
- `WORKFLOW_API_KEY`
- `WORKFLOW_BASE_URL`
- `THEORY_UPDATE_MODEL`
- `THEORY_UPDATE_API_KEY`
- `THEORY_UPDATE_BASE_URL`
- `TEMPERATURE`
- `MP_API_KEY`
- `AFLOW_BASE_URL` (optional)

Minimal example:

```dotenv
WORKFLOW_MODEL=deepseek-chat
WORKFLOW_API_KEY=your_api_key
WORKFLOW_BASE_URL=https://api.deepseek.com/v1

THEORY_UPDATE_MODEL=deepseek-chat
THEORY_UPDATE_API_KEY=your_api_key
THEORY_UPDATE_BASE_URL=https://api.deepseek.com/v1

TEMPERATURE=0.3
MP_API_KEY=your_materials_project_key
```

Notes:

- `WORKFLOW_*` is used for candidate screening in `main.py`.
- `THEORY_UPDATE_*` is used for theory-document evolution in `main.py`.
- `MP_API_KEY` is required for Materials Project queries.

## Quick Start

Install dependencies:

```bash
uv sync
```

Create `.env`:

```bash
copy .env.example .env
```

Run the LLM workflow:

```bash
python main.py
```

Run the BO-only workflow:

```bash
python main_bo_only.py
```

## LLM Workflow

Entry point:

```bash
python main.py
```

Default roots used by the code:

- `llm/results`
- `llm/models/GPR`
- `llm/data`
- `llm/doc`

Default initial inputs:

- dataset: `data/processed_data.csv`
- theory document: `doc/Theoretical_principle_document.md`

### Common commands

Run a fixed number of iterations:

```bash
python main.py --max-iterations 3
```

Continue from existing progress:

```bash
python main.py --add-iterations 2
```

Override BO or screening parameters:

```bash
python main.py --samples 150 --top-k-bayes 30 --top-k-screen 10 --n-structures 5
```

Set GPU count:

```bash
python main.py --num-gpus 2
```

Disable web search:

```bash
python main.py --no-websearch-enabled
```

Allow partial structure completion:

```bash
python main.py --allow-partial-structure
```

Reset recorded progress:

```bash
python main.py --reset
```

Override initial dataset and theory document:

```bash
python main.py --init-data data/processed_data.csv --init-doc doc/Theoretical_principle_document.md
```

### AgentOS runtime

```bash
python main.py --runtime agentos --agentos-host 127.0.0.1 --agentos-port 7777
```

`main.py` also reads and persists values through `config/agentos_params.csv`. In practice this file acts as:

- editable defaults for the runtime UI
- runtime override sheet
- lightweight parameter memory between runs

Common keys include:

- `websearch_enabled`
- `websearch_top_n`
- `top_k_bayes`
- `top_k_screen`
- `samples`
- `n_structures`
- `relax_timeout_sec`
- `skip_doc_update`
- `agentos_default_iterations`

## BO-Only Workflow

Entry point:

```bash
python main_bo_only.py
```

Current output roots in code:

- `bo/results`
- `bo/models/GPR`
- `bo/data`

The BO-only workflow performs:

1. model training
2. BO candidate generation
3. top-`top_k_screen` candidate selection for downstream calculation
4. structure generation and calculation
5. result merging and success extraction
6. dataset update

### Common commands

Run 10 iterations:

```bash
python main_bo_only.py --max-iterations 10
```

Start from a specific iteration:

```bash
python main_bo_only.py --start-iteration 3 --max-iterations 10
```

Override sampling or structure parameters:

```bash
python main_bo_only.py --samples 150 --top-k-bayes 30 --top-k-screen 10 --n-structures 5
```

Use a custom initial dataset:

```bash
python main_bo_only.py --init-data data/processed_data.csv
```

Allow partial structure completion:

```bash
python main_bo_only.py --allow-partial-structure
```

Reset and rerun:

```bash
python main_bo_only.py --reset
```

## Configuration Files

### `config/config.yaml`

Main algorithm and tool configuration, including:

- loop control
- BO acquisition and sampling settings
- model defaults
- generator constraints
- external tool parameters
- thresholds and logging/output examples

Fields that are especially relevant to current runs:

- `loop.max_iterations`
- `bayesian_optimization.acquisition.xi`
- `bayesian_optimization.sampling.n_samples`
- `bayesian_optimization.sampling.hard_constraints`
- `tools.crystallm.model_path`
- `tools.ai4kappa.k_threshold`
- `tools.mattersim.imaginary_freq_threshold`

## Output Layout

### LLM mode

```text
llm/
|- data/
|  |- iteration_0/data.csv
|  |- iteration_1/data.csv
|- models/
|  |- GPR/iteration_1/...
|- results/
|  |- progress.json
|  |- run_YYYYMMDD_HHMMSS.log
|  |- iteration_1/
|     |- reports/
|     |- selected_results/
|     |- success_examples/
|- doc/
|  |- v0.0.0/Theoretical_principle_document.md
|  |- v0.0.1/Theoretical_principle_document.md
```

### BO-only mode

```text
bo/
|- data/
|  |- iteration_0/data.csv
|- models/
|  |- GPR/iteration_1/...
|- results/
|  |- progress.json
|  |- iteration_1/
|     |- selected_results/
|     |- success_examples/
```

### `progress.json`

Both workflows use `progress.json` to skip completed work when resuming. Depending on mode, tracked steps may include:

- `train_model`
- `bayesian_optimization`
- `ai_evaluation`
- `structure_calculation`
- `merge_results`
- `success_extraction`
- `document_update`
- `data_update`

## Troubleshooting

### No usable LLM configured

Check:

- `.env` exists
- `WORKFLOW_API_KEY` is set
- `THEORY_UPDATE_API_KEY` is set
- the configured model names are valid for the selected endpoints

### Materials Project queries fail

Check that `MP_API_KEY` is set in `.env`.

### CUDA or PyTorch mismatch

Reinstall PyTorch with a build that matches your local CUDA or CPU environment.

### Windows encoding issues

If terminal output is garbled:

```bash
set PYTHONIOENCODING=utf-8
python main.py
```

## Reference Files

If you want to extend the documentation further, the most relevant files are:

- `main.py`
- `main_bo_only.py`
- `src/workflow/agno_pipeline.py`
- `src/workflow/agno_steps.py`
- `src/agents/llm_models.py`
- `config/config.yaml`
- `src/utils/path_config.py`
- `src/utils/progress_tracker.py`
- `data/processed_data.csv`
- `doc/Theoretical_principle_document.md`
