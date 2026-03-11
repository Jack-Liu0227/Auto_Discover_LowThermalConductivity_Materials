# ASLK

Active Search for Low Kappa materials.

[Read in Chinese / 中文文档](./README.zh-CN.md)

ASLK is an automated low-thermal-conductivity materials discovery workflow. It combines Bayesian Optimization, LLM-based screening, structure generation, thermal conductivity prediction, phonon or stability validation, and iterative dataset updates.

The repository currently exposes two main entry points:

- `main.py`: LLM + Agno workflow
- `main_bo_only.py`: Bayesian-optimization-only workflow

## Overview

The project is designed to iteratively discover low-kappa candidate materials in a constrained composition space. A typical loop includes:

1. Train or update the surrogate model from prior data.
2. Generate new candidates with Bayesian Optimization.
3. Screen candidates with LLM reasoning in `main.py`.
4. Generate structures and run downstream calculations.
5. Extract successful or stable materials.
6. Update the dataset for the next iteration.
7. In LLM mode, update the theory document across iterations.

## Run Modes

### LLM Workflow

Entry point:

```bash
python main.py
```

Characteristics:

- Uses Agno workflow orchestration
- Supports `workflow` and `agentos` runtime modes
- Uses DeepSeek as the default model provider
- Supports fallback providers configured from `.env`
- Maintains `llm/data`, `llm/results`, `llm/models`, and `llm/doc`
- Supports parameter overrides through `config/agentos_params.csv`

### BO-only Workflow

Entry point:

```bash
python main_bo_only.py
```

Characteristics:

- Runs BO, structure calculation, extraction, and dataset update only
- Does not require LLM evaluation or theory-document updates
- Maintains `bo_new/data`, `bo_new/results`, and `bo_new/models`

## Repository Layout

```text
aslk/
|- main.py
|- main_bo_only.py
|- README.md
|- README.zh-CN.md
|- .env.example
|- requirements.txt
|- pyproject.toml
|- config/
|  |- config.yaml
|  |- llm_config.yaml
|  |- agentos_params.csv
|- src/
|  |- agents/
|  |- workflow/
|  |- database/
|  |- tools/
|  |- utils/
|- scripts/
|  |- summarize_results.py
|  |- check_screening_tools.py
|- analysis_scripts/
|- data/
|- doc/
|- llm/
|- bo_new/
```

## Requirements

Recommended environment:

- Python `3.10+`
- `uv` for dependency management
- A CUDA-capable GPU if you want full structure or property workflow performance

Common scientific or runtime dependencies used by the project:

- `torch`
- `pymatgen`
- `ase`
- `mattersim`
- `phonopy`

> [!IMPORTANT]
> `pyproject.toml` currently pins CUDA-enabled PyTorch packages for `cu124`. If your machine uses a different CUDA version, install a matching PyTorch build manually.

## Installation

### Option 1: uv

```bash
pip install uv
uv sync
```

For development extras:

```bash
uv sync --extra dev
```

### Option 2: pip

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If needed, install CUDA PyTorch manually:

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

## Environment Variables

The project loads environment variables from the repository-root `.env`. Start from:

```bash
copy .env.example .env
```

### Minimal model configuration

The project now uses two explicit model slots:

- `WORKFLOW_MODEL`
- `WORKFLOW_API_KEY`
- `WORKFLOW_BASE_URL`
- `THEORY_UPDATE_MODEL`
- `THEORY_UPDATE_API_KEY`
- `THEORY_UPDATE_BASE_URL`
- `TEMPERATURE`

Notes:

- `WORKFLOW_MODEL` is used by the main workflow screening step.
- `THEORY_UPDATE_MODEL` is used by the theory-document update step.
- Both default to `deepseek-chat`.
- Both default base URLs point to `https://api.deepseek.com/v1`.
- The variable name is `WORKFLOW_BASE_URL`, not `WORKFLOW_BASW_URL`.

### Database or query variables

- `MP_API_KEY`: required for Materials Project queries
- `AFLOW_BASE_URL`: optional, defaults to `https://aflowlib.org/API/aflux/`

### Minimal example

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

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Create and edit `.env`

```bash
copy .env.example .env
```

At minimum, configure:

- `WORKFLOW_API_KEY`
- `THEORY_UPDATE_API_KEY`
- `MP_API_KEY` if you rely on Materials Project queries

### 3. Run the LLM workflow

```bash
python main.py
```

### 4. Or run the BO-only workflow

```bash
python main_bo_only.py
```

### 5. Summarize results

```bash
python scripts/summarize_results.py --results-dir llm/results
python scripts/summarize_results.py --results-dir bo_new/results
```

## LLM Workflow Usage

### Basic run

```bash
python main.py
```

By default it writes outputs to:

- `llm/results`
- `llm/data`
- `llm/models/GPR`
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

Override sampling or screening parameters:

```bash
python main.py --samples 150 --n-top-candidates 30 --n-select 10 --n-structures 5
```

Specify GPU count:

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

Reset progress:

```bash
python main.py --reset
```

Set initial dataset and theory document:

```bash
python main.py --init-data data/processed_data.csv --init-doc doc/Theoretical_principle_document.md
```

### AgentOS mode

```bash
python main.py --runtime agentos --agentos-host 0.0.0.0 --agentos-port 8000
```

Use this when you want to expose the workflow through AgentOS.

### `config/agentos_params.csv`

`main.py` reads and persists values from `config/agentos_params.csv`. It acts as:

- a UI or default parameter sheet
- a runtime override sheet
- a lightweight memory store for selected parameters

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

## BO-only Workflow Usage

### Basic run

```bash
python main_bo_only.py
```

### Common commands

Run 10 iterations:

```bash
python main_bo_only.py --max-iterations 10
```

Start from a given iteration:

```bash
python main_bo_only.py --start-iteration 3 --max-iterations 10
```

Override BO parameters:

```bash
python main_bo_only.py --samples 150 --n-top-candidates 30 --n-select 10 --n-structures 5
```

Use a custom initial dataset:

```bash
python main_bo_only.py --init-data data/processed_data.csv
```

Reset and rerun:

```bash
python main_bo_only.py --reset
```

## Configuration Files

### `config/config.yaml`

This is the main algorithm or tool configuration file. It contains settings for:

- loop control
- Bayesian Optimization
- model parameters
- generator or tool parameters
- thresholds and timeouts

Notable fields:

- `loop.max_iterations`
- `bayesian_optimization.acquisition.xi`
- `bayesian_optimization.sampling.n_samples`
- `tools.crystallm.model_path`
- `tools.ai4kappa.k_threshold`
- `tools.mattersim.imaginary_freq_threshold`

### `config/llm_config.yaml`

This file documents LLM input or output layout and sample model config. In practice, the active model chain is primarily driven by `.env` and `src/agents/llm_models.py`.

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
bo_new/
|- data/
|- models/
|- results/
```

### `progress.json`

The workflow uses `progress.json` to avoid re-running completed steps. Common tracked steps include:

- `train_model`
- `bayesian_optimization`
- `ai_evaluation`
- `structure_calculation`
- `merge_results`
- `success_extraction`
- `document_update`

## Utility Scripts

### Result summarization

```bash
python scripts/summarize_results.py --results-dir llm/results
```

This consolidates per-iteration success or stable CSVs into summary CSV files.

### Environment or tool probing

`scripts/check_screening_tools.py` is kept for troubleshooting, not as a default quick-start step.

### Analysis scripts

`analysis_scripts/` contains post-processing and plotting helpers, including:

- `analysis_scripts/run_and_filter_iterations.py`
- `analysis_scripts/compare_and_plot.py`

These are intended for experiment analysis after runs complete.

## Troubleshooting

### No usable LLM configured

Check:

- `.env` exists
- `WORKFLOW_API_KEY` and `THEORY_UPDATE_API_KEY` are set
- `WORKFLOW_MODEL` and `THEORY_UPDATE_MODEL` are valid model names for the configured endpoint

### Materials Project query fails

Check that `MP_API_KEY` is set in `.env`.

### OQMD or AFLOW query issues

The code already contains Python-API to HTTP fallbacks. If both fail, it is often an upstream service or environment problem rather than a core workflow bug.

### CUDA or PyTorch mismatch

If CUDA packages do not match your machine, reinstall PyTorch with the correct build for your environment.

### Windows encoding issues

If your terminal shows garbled output:

```bash
set PYTHONIOENCODING=utf-8
python main.py
```

## Reference Files

If you need to extend the documentation further, start with:

- `main.py`
- `main_bo_only.py`
- `src/workflow/agno_pipeline.py`
- `src/agents/llm_models.py`
- `src/utils/param_sheet.py`
- `src/utils/progress_tracker.py`
- `scripts/summarize_results.py`
