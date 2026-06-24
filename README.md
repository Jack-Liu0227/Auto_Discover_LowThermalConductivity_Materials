# ADLM: Auto Discovery of Low-Thermal-Conductivity Materials

[中文文档](./README.zh-CN.md)

ADLM is a research workflow for discovering low-thermal-conductivity materials. It combines Bayesian optimization, optional LLM-based candidate screening, crystal structure generation, thermal-conductivity prediction, stability or phonon validation, and iterative dataset updates.

The repository is organized as a runnable Python project with two main workflows:

- `main.py`: LLM-assisted discovery workflow powered by Agno-style workflow steps.
- `main_bo_only.py`: Bayesian-optimization-only workflow for experiments without LLM screening or theory-document updates.

> [!NOTE]
> Runtime outputs such as `llm/`, `bo/`, `featureEngeering/`, logs, caches, and temporary calculator folders are intentionally ignored. The tracked files are the source code, configuration, seed data, reference document, and local tool integrations required to reproduce new runs.

## Table of Contents

- [Features](#features)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Environment Configuration](#environment-configuration)
- [Quick Start](#quick-start)
- [LLM-Assisted Workflow](#llm-assisted-workflow)
- [BO-Only Workflow](#bo-only-workflow)
- [Configuration](#configuration)
- [Data and Output Layout](#data-and-output-layout)
- [Local Tool Integrations](#local-tool-integrations)
- [Development Notes](#development-notes)
- [License](#license)
- [Troubleshooting](#troubleshooting)

## Features

- Iterative materials discovery loop for low lattice thermal conductivity.
- Gaussian-process surrogate model training and refresh.
- Expected-improvement Bayesian optimization over constrained composition spaces.
- Optional LLM screening of candidate materials.
- Crystal structure generation through the bundled CrystaLLM integration.
- Thermal-conductivity estimation through the bundled kappa/CGCNN integration.
- MatterSim-based relaxation and phonon-oriented validation hooks.
- Materials Project, AFLOW, and OQMD helper modules for external materials data.
- Resume support through per-run progress files.
- Separate runtime roots for LLM-assisted and BO-only experiments.

## Repository Layout

```text
aslk/
|- main.py                         # LLM-assisted workflow entry point
|- main_bo_only.py                 # BO-only workflow entry point
|- requirements.txt                # Python dependency list
|- uv.lock                         # uv lockfile
|- .env.example                    # environment variable template
|- config/
|  |- config.yaml                  # algorithm and tool configuration
|  |- agentos_params.csv           # editable runtime parameter sheet
|- data/
|  |- processed_data.csv           # seed dataset
|- doc/
|  |- Theoretical_principle_document.md
|- src/
|  |- agents/                      # LLM clients, screening, document updates
|  |- analysis/                    # offline feature analysis utilities
|  |- database/                    # MP/AFLOW/OQMD access helpers
|  |- generators/                  # BO samplers and acquisition functions
|  |- models/                      # GPR training code
|  |- schemas/                     # typed workflow inputs
|  |- tools/                       # CrystaLLM, kappa, MatterSim, structure tools
|  |- utils/                       # configuration, paths, progress, reproducibility
|  |- workflow/                    # step-level workflow implementation
|- README.md
|- README.zh-CN.md
```

## Requirements

- Python 3.10 or newer.
- Windows PowerShell, Linux shell, or a comparable terminal.
- CUDA-capable GPU recommended for full structure generation and MatterSim/torch workloads.
- API keys for the LLM provider and Materials Project when using those features.

Core dependency groups:

- Workflow and LLM runtime: `agno`, `google-adk`, `litellm`
- Data and ML: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`
- Materials science: `ase`, `pymatgen`, `phonopy`, `spglib`
- Deep learning and calculators: `torch`, `torchvision`, `torchaudio`, `mattersim`
- Development and analysis: `pytest`, `black`, `isort`, `jupyter`, `shap`, `plotly`

> [!IMPORTANT]
> `requirements.txt` declares PyTorch package names, but GPU builds are environment-specific. Install the PyTorch wheel that matches your CUDA or CPU environment before running heavy workflows.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you use `uv`, install from the lockfile:

```bash
uv sync
```

CUDA 12.4 PyTorch example:

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

## Environment Configuration

Create a local `.env` file:

```bash
copy .env.example .env
```

Fill in the required values:

```dotenv
WORKFLOW_MODEL=deepseek-chat
WORKFLOW_API_KEY=your_workflow_llm_key
WORKFLOW_BASE_URL=https://api.deepseek.com/v1

THEORY_UPDATE_MODEL=your_theory_update_model
THEORY_UPDATE_API_KEY=your_theory_update_key
THEORY_UPDATE_BASE_URL=https://your-provider.example/v1

TEMPERATURE=0.3
MP_API_KEY=your_materials_project_key
```

Variable usage:

- `WORKFLOW_*`: LLM candidate screening in `main.py`.
- `THEORY_UPDATE_*`: theory-document update step in `main.py`.
- `TEMPERATURE`: default LLM sampling temperature.
- `MP_API_KEY`: Materials Project queries.
- `AFLOW_BASE_URL`: optional AFLOW endpoint override.

Never commit `.env`; it is ignored by `.gitignore`.

## Quick Start

Run the LLM-assisted workflow:

```bash
python main.py --max-iterations 1
```

Run the BO-only workflow:

```bash
python main_bo_only.py --max-iterations 1
```

Run a lightweight syntax check:

```bash
python -m compileall main.py main_bo_only.py src
```

## LLM-Assisted Workflow

Entry point:

```bash
python main.py
```

Default seed inputs:

- Dataset: `data/processed_data.csv`
- Theory document: `doc/Theoretical_principle_document.md`

Default runtime roots:

- `llm/data`
- `llm/models/GPR`
- `llm/results`
- `llm/doc`

Common commands:

```bash
python main.py --max-iterations 3
python main.py --add-iterations 2
python main.py --samples 150 --top-k-bayes 30 --top-k-screen 10 --n-structures 5
python main.py --num-gpus 2
python main.py --no-websearch-enabled
python main.py --skip-doc-update
python main.py --allow-partial-structure
python main.py --reset
python main.py --init-data data/processed_data.csv --init-doc doc/Theoretical_principle_document.md
```

AgentOS runtime:

```bash
python main.py --runtime agentos --agentos-host 127.0.0.1 --agentos-port 7777
```

`config/agentos_params.csv` is used as an editable parameter sheet and lightweight runtime memory. It includes values such as `websearch_enabled`, `websearch_top_n`, `samples`, `top_k_bayes`, `top_k_screen`, `n_structures`, `relax_timeout_sec`, and `skip_doc_update`.

## BO-Only Workflow

Entry point:

```bash
python main_bo_only.py
```

Default runtime roots:

- `bo/data`
- `bo/models/GPR`
- `bo/results`

Common commands:

```bash
python main_bo_only.py --max-iterations 10
python main_bo_only.py --start-iteration 3 --max-iterations 10
python main_bo_only.py --samples 150 --top-k-bayes 30 --top-k-screen 10 --n-structures 5
python main_bo_only.py --init-data data/processed_data.csv
python main_bo_only.py --allow-partial-structure
python main_bo_only.py --reset
```

The BO-only pipeline trains the surrogate model, samples candidate compositions, selects top candidates, generates structures, runs downstream calculations, merges results, extracts successful materials, and updates the dataset for the next iteration.

## Configuration

`config/config.yaml` contains the main algorithm and tool settings:

- Loop metadata and default iteration hints.
- Bayesian optimization acquisition and sampling settings.
- Allowed element set and hard composition constraints.
- Thermal-conductivity and stability thresholds.
- CrystaLLM, kappa, and MatterSim tool settings.
- Runtime layout documentation for `llm/` and `bo/` outputs.

Key fields:

- `bayesian_optimization.acquisition.xi`
- `bayesian_optimization.sampling.n_samples`
- `bayesian_optimization.sampling.allowed_elements`
- `bayesian_optimization.sampling.hard_constraints`
- `thresholds.thermal_conductivity`
- `tools.crystallm.model_path`
- `tools.ai4kappa.k_threshold`
- `tools.mattersim.imaginary_freq_threshold`

## Data and Output Layout

Seed data and documents:

```text
data/processed_data.csv
doc/Theoretical_principle_document.md
```

LLM workflow outputs:

```text
llm/
|- data/
|  |- iteration_0/data.csv
|  |- iteration_1/data.csv
|- models/
|  |- GPR/iteration_1/
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

BO-only outputs:

```text
bo/
|- data/
|  |- iteration_0/data.csv
|- models/
|  |- GPR/iteration_1/
|- results/
|  |- progress.json
|  |- iteration_1/
|     |- selected_results/
|     |- success_examples/
```

Progress tracking:

- `progress.json` allows a run to skip completed steps during resume.
- Typical step keys include `train_model`, `bayesian_optimization`, `ai_evaluation`, `structure_calculation`, `merge_results`, `success_extraction`, `document_update`, and `data_update`.

## Local Tool Integrations

### CrystaLLM

The local CrystaLLM integration is under `src/tools/crystallm/`. It provides structure-generation utilities, model configuration files, tokenizer/model code, and helper scripts.

The configured model path is:

```text
src/tools/crystallm/pre-trained-model/crystallm_v1_small
```

Large pretrained model files are intentionally not tracked. Place the required model files in that directory before running CrystaLLM-backed generation.

### kappa / CGCNN

The kappa integration is under `src/tools/kappa_lib/`. It includes CGCNN code and tracked pretrained thermal-property models under `src/tools/kappa_lib/model/`.

Temporary prediction folders matching `src/tools/kappa_lib/root_dir*/` are ignored and should not be committed.

### MatterSim

MatterSim-related wrapper code lives in `src/tools/mattersim_wrapper.py`. Full MatterSim runs may require a GPU-enabled environment and compatible PyTorch installation.

## Development Notes

Useful checks:

```bash
python -m compileall main.py main_bo_only.py src
python -m pytest
```

The repository may include optional test and development dependencies in `requirements.txt`. Some tests or workflows can require external credentials, model weights, GPU support, or generated runtime data.

Ignored by design:

- `.env`
- `.venv/`
- `__pycache__/`, `.pytest_cache/`, `htmlcov/`
- `llm/`, `llm_*/`
- `bo/`, `bo_*/`
- `Paper/`, `archive/`, `figures/`
- `featureEngeering/`
- CrystaLLM generated structures
- kappa temporary prediction directories

## License

The original source code in this repository is licensed under the MIT License. See [LICENSE](./LICENSE).

Bundled or integrated third-party tools, pretrained models, datasets, manuscripts, and generated outputs are not relicensed by this repository-level license. They remain subject to their original licenses, terms, or data-use restrictions. See [NOTICE.md](./NOTICE.md).

When using MatterSim or CrystaLLM features, comply with their upstream licenses and cite the original works. See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

## Troubleshooting

### LLM calls fail

Check that `.env` exists and that `WORKFLOW_API_KEY`, `WORKFLOW_BASE_URL`, and `WORKFLOW_MODEL` match your provider.

### Theory-document updates fail

Check `THEORY_UPDATE_API_KEY`, `THEORY_UPDATE_BASE_URL`, and `THEORY_UPDATE_MODEL`. You can bypass the document update step with:

```bash
python main.py --skip-doc-update
```

### Materials Project queries fail

Check that `MP_API_KEY` is set and valid.

### CUDA or PyTorch errors

Install a PyTorch build that matches your CUDA driver, or use a CPU-compatible setup for lightweight code paths.

### CrystaLLM model not found

Place the pretrained CrystaLLM model files under:

```text
src/tools/crystallm/pre-trained-model/crystallm_v1_small
```

### Windows terminal encoding is garbled

Use UTF-8 mode:

```bash
set PYTHONIOENCODING=utf-8
python main.py
```
