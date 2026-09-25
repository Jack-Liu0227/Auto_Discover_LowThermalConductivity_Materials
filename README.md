# ADLM: Automated Discovery of Low-Thermal-Conductivity Materials

English | [简体中文](README.zh-CN.md)

ADLM searches the A-B-Ch ternary composition space for low lattice thermal conductivity materials
with an iterative **Bayesian optimization (BO) + LLM** loop: BO samples candidate formulas in the
constrained space, the LLM proposes and screens candidates and writes the lessons learned back into
the theory document, CrystaLLM generates crystal structures, MatterSim performs relaxation and
phonon stability analysis, and AI4Kappa (CGCNN) predicts elastic moduli and derives `κ_L`.

The repository has two entry points:

| Entry point | Purpose | Result directories |
|---|---|---|
| `main.py` | BO + LLM screening workflow (with LLM candidate generation switches) | `llm_gen_diverse/`, `llm_gen_legacy/`, `llm_nogen_diverse/`, `llm_nogen_legacy/` |
| `main_bo_only.py` | BO-only baseline (no LLM calls) | `bo/` |

Run every command from the repository root. Examples use PowerShell on Windows (line continuation
is a backtick); on Linux/macOS use bash with `\`.

---

## 1. Environment preparation

### 1.1 Hardware and system requirements

- **GPU**: at least one NVIDIA CUDA GPU. The reference machine uses a single RTX 4090 D (24 GB).
  16 GB or more VRAM is recommended.
- **Multiple GPUs**: optional. `--num-gpus N` is validated against the devices PyTorch can actually
  see and is mapped to `cuda:0..cuda:N-1`.
- **Network**: the first run needs access to Zenodo (CrystaLLM checkpoint), GitHub raw (MatterSim
  checkpoint), your own LLM endpoint, and the Materials Project / AFLOW databases.

### 1.2 Create the Python environment

Python 3.10 with CUDA 12.1 is recommended:

```powershell
conda create -n kappap python=3.10.18 -y
conda activate kappap
$PY = 'python'
& $PY -m pip install --upgrade pip
& $PY -m pip install -r requirements.txt
& $PY -m pip check
```

Notes:

- `requirements.txt` is a full `pip freeze` snapshot. It already embeds
  `--extra-index-url https://download.pytorch.org/whl/cu121` and installs `torch==2.5.1+cu121`,
  `mattersim==1.2.0`, `phonopy`, `pymatgen`, `agno==2.5.6` and everything else.
- The vendored CrystaLLM (`src/tools/crystallm`) is plain source code. It needs **no** extra
  `pip install`; `src/tools` is put on `sys.path` at runtime.
- Reference environment: Python 3.10.18 / PyTorch 2.5.1+cu121 / CUDA runtime 12.1.

### 1.3 Download and place the pretrained models (required)

The repository **already ships the AI4Kappa elastic model weights** (see 1.3.3; they are tiny),
but it does not contain the CrystaLLM and MatterSim weights (`*.pt` / `*.pth` are covered by
`.gitignore`). Those two must be obtained separately.

#### 1.3.1 CrystaLLM (structure generation) — required

Upstream project: [lantunes/CrystaLLM](https://github.com/lantunes/CrystaLLM); weights are hosted on
Zenodo record [10642388](https://zenodo.org/records/10642388).

The repository ships the official download script (pinned to the `files` directory of that Zenodo
record). Just run it:

```powershell
conda activate kappap
cd <repo root>
python .\src\tools\crystallm\bin\download.py crystallm_v1_small.tar.gz -o .\src\tools\crystallm\pre-trained-model
cd .\src\tools\crystallm\pre-trained-model
tar -xvf crystallm_v1_small.tar.gz
```

After extraction you must end up with:

```text
src/tools/crystallm/pre-trained-model/crystallm_v1_small/ckpt.pt     ~311 MB (296 MiB)
```

The same folder can also hold `crystallm_v1_small.tar.gz`, `crystallm_v1_large.tar.gz`
(~2.2 GB) and `crystallm_v1_minus_mpts_52_small.tar.gz`. The current code **only reads
`crystallm_v1_small`** (see `src/tools/crystallm/generator.py` and
`tools.crystallm.model_path` in `config/config.yaml`), so a `crystallm_v1_large/` folder would be
ignored.

#### 1.3.2 MatterSim (relaxation + phonons) — required

`mattersim==1.2.0` **downloads the checkpoint automatically** the first time a
`MatterSimCalculator` is instantiated, into the default cache directory
`~/.local/mattersim/pretrained_models/` (on Windows:
`C:\Users\<you>\.local\mattersim\pretrained_models\`).

- Default checkpoint: `MatterSim-v1.0.0-1M.pth` (smaller and faster).
- The current code does not expose MatterSim's `load_path`, so the default 1M checkpoint is what is
  actually used.

For offline or pre-warmed machines, download it into the cache directory ahead of time:

```powershell
conda activate kappap
& $PY -c "from mattersim.utils.download_utils import download_checkpoint; download_checkpoint('MatterSim-v1.0.0-1M.pth')"
```

If that function path differs in your `mattersim` version, fetch the file directly and place it in
the cache directory:

```text
https://raw.githubusercontent.com/microsoft/mattersim/main/pretrained_models/MatterSim-v1.0.0-1M.pth
  -> ~/.local/mattersim/pretrained_models/MatterSim-v1.0.0-1M.pth
```

#### 1.3.3 AI4Kappa CGCNN elastic models (`κ_L` estimation) — required

`src/tools/kappa_lib/` is a local module adapted from
[Jack-Liu0227/AI4Kappa](https://github.com/Jack-Liu0227/AI4Kappa). It predicts the **bulk modulus**
and **shear modulus** from a CIF and then derives `κ_L` through the Slack/PINK relations. It needs
three resource files:

```text
src/tools/kappa_lib/model/Bulk modulus (GPa)-pre-trained.pth.tar     ~660 KB
src/tools/kappa_lib/model/Shear modulus (GPa)-pre-trained.pth.tar    ~660 KB
src/tools/kappa_lib/atom_init.json                                   ~28 KB
```

**All three files are committed to this repository** (see `git ls-files src/tools/kappa_lib`), so
if they are present you need to download nothing. Only when they are missing (for instance after a
manual cleanup) restore them from upstream:

```powershell
conda activate kappap
cd <repo root>
git clone --depth=1 https://github.com/Jack-Liu0227/AI4Kappa.git tmp/AI4Kappa
Copy-Item "tmp\AI4Kappa\model\Bulk modulus (GPa)-pre-trained.pth.tar"  src\tools\kappa_lib\model\
Copy-Item "tmp\AI4Kappa\model\Shear modulus (GPa)-pre-trained.pth.tar" src\tools\kappa_lib\model\
Copy-Item "tmp\AI4Kappa\root_dir\atom_init.json"                       src\tools\kappa_lib\
```

**Keep the file names exactly as they are.** The code globs `model/*-pre-trained.pth.tar` and uses
the file name minus `-pre-trained.pth.tar` as the result column name
(`Bulk modulus (GPa)` / `Shear modulus (GPa)`), and the downstream formulas index those columns.
Renaming a file breaks the `κ_L` computation.

#### 1.3.4 Model self-check

```powershell
conda activate kappap
$PY = 'python'
& $PY -c "
from pathlib import Path
checks = {
 'crystallm': 'src/tools/crystallm/pre-trained-model/crystallm_v1_small/ckpt.pt',
 'kappa_bulk': 'src/tools/kappa_lib/model/Bulk modulus (GPa)-pre-trained.pth.tar',
 'kappa_shear': 'src/tools/kappa_lib/model/Shear modulus (GPa)-pre-trained.pth.tar',
 'atom_init': 'src/tools/kappa_lib/atom_init.json',
}
for k, v in checks.items():
    p = Path(v)
    size = f'{p.stat().st_size/1e6:.1f} MB' if p.exists() else ''
    print(f'{k:12s}', 'OK' if p.exists() else 'MISSING', size, v)
"
```

The MatterSim checkpoint only lands in its cache directory during the first GPU computation, so it
is not checked here.

### 1.4 Prepare the initial dataset and theory document

```text
data/processed_data.csv                        # initial dataset (iteration 0)
doc/Theoretical_principle_document.md          # initial theory document
```

- `data/processed_data.csv` must contain a `Formula` column (e.g.
  `(Ag0.05Sb0.05Sn0.9)(S0.05Se0.05Te0.9)`), one atomic-fraction column per element
  (`Ag, As, Bi, Cu, Ge, In, Pb, S, Sb, Se, Sn, Te, Ti, V`) and the target column `k(W/Km)`
  (lattice thermal conductivity in W/(m·K)). On the first run it is bootstrapped into
  `<run_mode>/data/iteration_0/data.csv`, and later iterations append new data.
- `doc/Theoretical_principle_document.md` is bootstrapped into `<run_mode>/doc/v0.0.0/` and is then
  updated by the LLM every iteration into a new version (`v0.0.1`, `v0.0.2`, ..., i.e.
  `v0.0.<iteration>`).
- If the document carries machine-readable markers such as `**Composition prior**` or
  `**Success threshold**`, the program validates the document against `config/config.yaml` and fails
  on a mismatch. The document shipped in this repository does not contain those markers.

### 1.5 Configure the APIs and secrets (`.env`)

```powershell
cd <repo root>
Copy-Item .env.example .env
notepad .env
```

`main.py` / `main_bo_only.py` load the `.env` in the repository root automatically. `.env` is
ignored by `.gitignore` — **never commit secrets**.

| Variable | Default | Purpose |
|---|---|---|
| `WORKFLOW_MODEL` | `deepseek-chat` | Main workflow model name (candidate generation, screening, material evaluation) |
| `WORKFLOW_API_KEY` | empty | API key for the main workflow model |
| `WORKFLOW_BASE_URL` | empty (SDK default) | OpenAI-compatible endpoint for the main workflow model |
| `THEORY_UPDATE_MODEL` | falls back to `WORKFLOW_MODEL` | Model used for theory document updates (`update_document`) |
| `THEORY_UPDATE_API_KEY` | falls back to `WORKFLOW_API_KEY` | Dedicated key for theory document updates |
| `THEORY_UPDATE_BASE_URL` | falls back to `WORKFLOW_BASE_URL` | Dedicated endpoint for theory document updates |
| `THEORY_UPDATE_MAX_TOKENS` | `32000` | Max output tokens for a theory document update |
| `THEORY_UPDATE_PROXY` | empty | Proxy applied only to theory document update requests |
| `TEMPERATURE` | `0.3` | Sampling temperature for all LLM calls |
| `LLM_NUM_RETRIES` | `2` | Number of retries for failed LLM requests |
| `LLM_REQUEST_TIMEOUT_SEC` | `120` | Timeout of a single LLM request (seconds) |
| `MP_API_KEY` | empty | Materials Project API key (novelty / deduplication checks) |
| `AFLOW_BASE_URL` | official AFLOW endpoint | AFLOW endpoint (second source for novelty checks) |
| `PHONON_PARALLEL_WORKERS` | `4` | Upper bound of phonon worker processes (falls back to the config value) |
| `PHONON_GPUS` | empty | Overrides the GPU list used for phonon calculations, e.g. `0,1` |
| `PYTORCH_CUDA_ALLOC_CONF` | on non-Windows automatically `expandable_segments:True,max_split_size_mb:128` | CUDA allocator tuning; it is removed automatically on Windows (unsupported) |

Minimal working example (your own OpenAI-compatible service):

```dotenv
WORKFLOW_MODEL=gpt-5
WORKFLOW_API_KEY=sk-xxxxxxxx
WORKFLOW_BASE_URL=https://your-endpoint/v1
THEORY_UPDATE_MODEL=gpt-5
THEORY_UPDATE_API_KEY=sk-xxxxxxxx
THEORY_UPDATE_BASE_URL=https://your-endpoint/v1
MP_API_KEY=your-mp-key
TEMPERATURE=0.3
```

### 1.6 Installation self-check

```powershell
conda activate kappap
$PY = 'python'
& $PY -c "import agno, litellm, ase, phonopy, pymatgen, mattersim, torch; print('imports ok', torch.cuda.is_available(), torch.cuda.device_count())"
& $PY main.py --help
& $PY main_bo_only.py --help
```

`torch.cuda.is_available()` must be `True`. If it is not, check that your driver matches the CUDA
build of `torch`.

### 1.7 Minimal smoke run (optional but strongly recommended)

First run the cheapest possible combination for a single iteration to confirm the whole chain
(BO -> structure generation -> relaxation/phonons -> elasticity -> `κ_L` -> success extraction ->
result merging) works end to end, then launch the real experiment. Turning both switches off writes
into `llm_nogen_legacy/`, so it cannot pollute the production directories:

```powershell
conda activate kappap
$PY = 'python'
& $PY main.py --runtime workflow --max-iterations 1 --samples 20 `
  --n-structures 1 --top-k-bayes 3 --top-k-screen 1 `
  --no-llm-formula-generation --no-chemistry-diversity
```

Then check:

```text
llm_nogen_legacy/data/iteration_1/            # dataset after iteration 1
llm_nogen_legacy/results/iteration_1/         # structure / relaxation / phonon artifacts
llm_nogen_legacy/models/GPR/iteration_0/      # GPR trained during iteration 1
llm_nogen_legacy/results/progress.json        # iteration progress
llm_nogen_legacy/results/run_<timestamp>.log  # run log
```

> **Note**: `--max-iterations` is the **total iteration target** (not "run N more iterations").
> If you pass no iteration argument, `main.py` takes its default target from
> `agentos_default_iterations` in the parameter sheet `config/agentos_params.csv` (currently **50**),
> while `main_bo_only.py` defaults to **20**. Always pass `--max-iterations` explicitly for real
> experiments.

---

## 2. Run modes: two switches, four result directories

`main.py` exposes two **independent** tri-state switches controlling "does the LLM generate
candidates" and "which screening strategy is used". Tri-state means: omitted = keep the configured
default (`config/config.yaml` + parameter sheet); passed explicitly = override it.

| Switch | Effect |
|---|---|
| `--llm-formula-generation` / `--no-llm-formula-generation` | Whether the LLM proposes new candidate formulas from iteration N onward (enabled by default) |
| `--chemistry-diversity` / `--no-chemistry-diversity` | Screening strategy: on = chemistry-diverse reranking; off = the original "rerank by BO prediction and uncertainty" path |

The combination of the two switches derives a `run_mode` and writes into a **fully isolated result
root**, so all four modes can run before/after or next to each other without reading or writing each
other's state:

| Command | LLM proposes candidates | chemistry diversity | `run_mode` / directory | Meaning |
|---|---|---|---|---|
| default (no switch) | on | on | `llm_gen_diverse/` | current full pipeline |
| `--no-chemistry-diversity` | on | off | `llm_gen_legacy/` | LLM candidates + original metric screening |
| `--no-llm-formula-generation` | off | on | `llm_nogen_diverse/` | BO-only candidates + chemistry-diverse screening |
| both off | off | off | `llm_nogen_legacy/` | the pre-extension pipeline, reproduced |

Key points:

- `run_mode` is derived from the **effective** switch values (including parameter-sheet overrides),
  so two runs with different behavior can never share a result directory.
- Each of the four directories bootstraps independently: the first run creates
  `<run_mode>/data/iteration_0/` and `<run_mode>/doc/v0.0.0/` from `data/processed_data.csv` and
  `doc/Theoretical_principle_document.md`. They share **no** state.
- `--reset` and `--rebuild-from` only affect the directory of the current switch combination (see
  3.5 and 5.3).
- The startup summary prints the active `run mode:`, `llm formula generation:` and
  `screening strategy:` lines; trust the log.

**How to choose**

- Reproducibility / ablation studies: run `llm_gen_diverse`, `llm_gen_legacy`, `llm_nogen_diverse`
  and `llm_nogen_legacy` and compare `selection_trace.csv` and the final material sets.
- BO baseline only (no LLM service at all): use `main_bo_only.py`, results in `bo/`.

---

## 3. Running the full pipeline

All commands are run from the repository root with the `kappap` environment active. The examples
assume `$PY = 'python'`.

### 3.1 Production run (BO + LLM, 20 iterations)

```powershell
conda activate kappap
cd <repo root>
$PY = 'python'
& $PY main.py --runtime workflow `
  --max-iterations 20 `
  --samples 100 `
  --n-structures 5 `
  --top-k-bayes 20 `
  --top-k-screen 10 `
  --num-gpus 1 `
  --postprocess-workers 2 `
  --novelty-workers 4 `
  --seed 42
```

Before starting, a summary is printed (log file, applied parameter-sheet overrides,
`effective_target_iterations`, `execution_range=[1 .. 20]`, `run mode`, `screening strategy`, ...).
Each iteration then performs, in order:

1. **BO sampling and ranking**: sample `samples` formulas in the constrained A-B-Ch space, predict
   `κ_L` and its uncertainty with the GPR, rank by EI and keep the top `top_k_bayes` (historical
   data from previous rounds is merged in).
2. **Candidate pool construction** (from iteration 2 on): with `llm_formula_generation` enabled the
   LLM proposes new formulas based on the screening feedback and parent candidates, scores them and
   merges them with the BO candidates into one reranked pool; when disabled this step is skipped and
   only an empty `llm_formula_proposals.json` is written to preserve the resume contract.
3. **LLM screening**: pick `top_k_screen` materials from the merged pool. With chemistry diversity
   enabled this uses the chemistry-diverse reranker (writing `candidate_scores`); disabled uses the
   original "rerank by predicted value and uncertainty" path (writing `selected_materials` and
   restoring BO metric evidence in the evaluation prompt).
4. **Structure generation**: CrystaLLM generates `n_structures` structures per material (one lane
   per GPU when several are used).
5. **Relaxation + phonons**: MatterSim relaxes the structures and applies the `--phonon-imag-tol`
   dynamical stability criterion.
6. **`κ_L` calculation**: AI4Kappa CGCNN predicts the elastic moduli, then Slack/PINK gives `κ_L`.
7. **Success extraction**: materials that are both stable and below the `κ_L` threshold are selected
   and written back into the dataset.
8. **Theory document update**: the LLM writes the iteration's lessons into the document and produces
   a new `doc/v0.0.<iteration>/` version.

### 3.2 Resume after an interruption / add iterations

```powershell
# Case A: the previous run stopped at iteration 7; continue up to iteration 20
& $PY main.py --runtime workflow --max-iterations 20

# Case B: 20 iterations are done; add 10 more (target becomes 30)
& $PY main.py --runtime workflow --add-iterations 10
```

- Resuming validates `progress.json` against the artifacts on disk. If predecessor artifacts are
  missing the run is refused with a hint to use `--reset` or `--rebuild-from`.
- **Use exactly the same switch combination as before**, otherwise the run lands in a different
  directory and looks like it restarted from iteration 1.

### 3.3 BO-only baseline

```powershell
& $PY main_bo_only.py --max-iterations 20 --samples 100 --n-structures 5 `
  --top-k-bayes 20 --top-k-screen 10 --num-gpus 1 --seed 42
```

It needs no LLM endpoint or API key and writes into `bo/`. Running, resuming, `--reset` and
`--rebuild-from` behave exactly as in `main.py`.

### 3.4 Ablation comparison (the four run modes)

```powershell
# 1) LLM generation + chemistry-diverse screening (default)
& $PY main.py --runtime workflow --max-iterations 20

# 2) LLM generation + original metric screening
& $PY main.py --runtime workflow --max-iterations 20 --no-chemistry-diversity

# 3) BO-only candidates + chemistry-diverse screening
& $PY main.py --runtime workflow --max-iterations 20 --no-llm-formula-generation

# 4) the pre-extension pipeline (BO-only candidates + original metric screening)
& $PY main.py --runtime workflow --max-iterations 20 `
  --no-llm-formula-generation --no-chemistry-diversity
```

These produce `llm_gen_diverse/`, `llm_gen_legacy/`, `llm_nogen_diverse/` and
`llm_nogen_legacy/`. When comparing, look at `screening_mode` in
`results/iteration_N/selected_results/selection_trace.csv`
(`chemistry_diverse_rerank` or `llm_full_rerank`) and at the number of entries in
`llm_formula_proposals.json`.

### 3.5 Rebuild a failed iteration suffix

When structure generation / relaxation / phonons fail at scale in some iteration and you want to
rerun it while keeping the earlier iterations:

```powershell
# Keep iterations 1..2, archive the artifacts from iteration 3 on and restart there
& $PY main.py --runtime workflow --rebuild-from 3 --max-iterations 20
```

- Archive directory: `<run_mode>_rebuild_<timestamp>/`, containing a `rebuild_manifest.json` that
  records what was archived.
- Only the directory of the current `run_mode` is touched; the other combinations are unaffected.
- Iterations `1..N-1` must all be complete, otherwise the command fails immediately (so an
  incomplete prefix is never mistaken for a trustworthy baseline).

To tolerate partial structure failures (by default only a fully unusable batch stops the run):

```powershell
& $PY main.py --runtime workflow --max-iterations 20 --allow-partial-structure
```

### 3.6 Parallelism and multiple GPUs

```powershell
# Use 2 GPUs; structure generation is split into one lane per GPU
& $PY main.py --runtime workflow --max-iterations 20 --num-gpus 2

# Run phonons only on GPUs 0 and 1 (environment variables win)
$env:PHONON_GPUS = '0,1'
$env:PHONON_PARALLEL_WORKERS = '4'
& $PY main.py --runtime workflow --max-iterations 20 --num-gpus 2
```

- `--num-gpus N` is validated against `torch.cuda.device_count()`; an `N` larger than the visible
  device count aborts immediately.
- With several GPUs each GPU runs `max_workers` structure-generation lanes (config value, default
  4); with a single GPU that value is 1.
- To run several run modes in parallel on one machine, use separate processes, give each a bounded
  `--num-gpus`, or pin them with `CUDA_VISIBLE_DEVICES`. Because the run-mode directories are
  isolated, they never overwrite each other's files.

What controls the concurrency of each stage:

| Stage | Concurrency control |
|---|---|
| Structure generation | Lanes per GPU = `max_workers` (config value, default 4; 1 on a single GPU) |
| Relaxation + phonons | Lanes per GPU = `relax_workers` (default 1, to limit MatterSim memory pressure) |
| `κ_L` calculation | One task per GPU at a time, assigned round-robin across the configured GPUs |
| Post-processing (dedup / extraction / merge) | `--postprocess-workers` (default 2), parallel per formula/material directory |
| Final database novelty queries | `--novelty-workers` (default 4), parallel per material; the database calls inside one material are serialized to avoid nested API overload |

For debugging or strict external rate limits, set both `postprocess_workers` and `novelty_workers`
to 1 for fully serial execution.

### 3.7 AgentOS mode (optional)

```powershell
& $PY main.py --runtime agentos --agentos-host 127.0.0.1 --agentos-port 7777
```

It prints `agentos endpoint: http://127.0.0.1:7777` and `workflow id: aslk-agentos-workflow`. In
this mode the form can override `max_iterations`, `samples`, `n_structures`, `top_k_bayes`,
`top_k_screen`, `websearch_enabled` and `websearch_top_n` for a run; everything else (including the
two switches from section 2 and the resulting `run_mode`) keeps the CLI/config values from startup.

### 3.8 Start over

```powershell
# Archive the whole current run_mode directory as <run_mode>_old_<timestamp>/ and rebuild from iteration 0
& $PY main.py --runtime workflow --reset --max-iterations 20
```

`--reset` only archives the directory of the current switch combination; for example
`python main.py --reset --no-llm-formula-generation` only affects `llm_nogen_diverse/` (plus any
legacy archives with the same prefix).

---

## 4. Parameter reference (what every parameter does)

Parameter precedence (highest first):

1. **Explicit CLI arguments** (tables below)
2. **Parameter sheet** rows with `enabled=1` in `config/agentos_params.csv` (see 4.4)
3. **Code defaults** (`DEFAULT_CONFIG` in `main.py` / `main_bo_only.py`)
4. **`config/config.yaml`** (BO / sampling / thresholds / tools, see 4.5)

> Note: arguments such as `--top-k-bayes` (help shows 20) and `--seed` (42) in `main.py` only
> override the parameter sheet when they are **explicitly present on the command line**; otherwise
> the sheet value wins.

### 4.1 Arguments shared by both entry points (a few are `main.py`-only and marked as such)

| Argument | Default | Purpose |
|---|---|---|
| `--max-iterations N` | `main.py`: see below; `main_bo_only.py`: **20** | **Total iteration target** (not "N more iterations"). With `k` completed iterations it runs from `k+1` to `N`. When `main.py` gets no iteration argument it uses the sheet key `agentos_default_iterations` (currently **50**) |
| `--add-iterations N` (`main.py` only) | none | Add N iterations on top of the **currently completed** rounds (equivalent to `--max-iterations = completed + N`). Mutually exclusive with `--max-iterations` |
| `--samples N` | 100 | Number of candidate formulas sampled by BO in the constrained A-B-Ch space per iteration. Larger = broader exploration, slower GPR training and EI ranking |
| `--n-structures N` | 5 | Number of CrystaLLM structures per selected material. Drives the cost of structure generation / relaxation / phonons and is the dominant runtime factor |
| `--top-k-bayes N` | 20 | Candidates kept in the pool after EI ranking; also the size of the merged BO + LLM pool (`top_k_bayes`) |
| `--top-k-screen N` | 10 | Number of **materials** finally sent to structure calculation (the LLM screening output, `top_k_screen`) |
| `--phonon-imag-tol V` | -0.1 | Minimum phonon frequency threshold (THz). A minimum frequency below it marks the structure dynamically unstable (imaginary modes). Keep it in sync with `thresholds.dynamic_min_frequency` in `config/config.yaml` |
| `--num-gpus N` | none (all visible) | Number of GPUs to use, mapped to `cuda:0..cuda:N-1`; validated against `torch.cuda.device_count()` and aborts when exceeded |
| `--device {cuda,cpu}` | cuda | Device for MatterSim structure calculations. `cpu` also sets `gpus` to `["cpu"]`; debugging only, extremely slow |
| `--postprocess-workers N` | 2 | CPU workers for structure deduplication, success extraction and result merging |
| `--novelty-workers N` | 4 | Workers for final database novelty queries (MP / AFLOW) |
| `--seed N` | 42 | Base random seed. The seed of iteration `i` is `seed + i * seed_stride` (`seed_stride` defaults to 1000), so every iteration is reproducible while the iterations differ from each other |
| `--non-deterministic-torch` | off | Disable deterministic PyTorch kernels (faster, loses strict reproducibility) |
| `--reset` | off | Archive the current `run_mode` directory as `<run_mode>_old_<timestamp>/` and rebuild from iteration 0 |
| `--rebuild-from N` | none | Keep iterations `1..N-1` and archive `data/`, `models/`, `results/` and `doc/` from iteration `N` on into `<run_mode>_rebuild_<timestamp>/` (with `rebuild_manifest.json`), then continue from iteration `N` |
| `--allow-partial-structure` | off | Allow the run to continue when some structure tasks fail (by default a fully unusable structure batch stops the iteration) |
| `--init-data PATH` | `data/processed_data.csv` | Initial dataset path; bootstrapped into `<run_mode>/data/iteration_0/` on the first run |
| `--init-doc PATH` (`main.py` only) | `doc/Theoretical_principle_document.md` | Initial theory document path; bootstrapped into `<run_mode>/doc/v0.0.0/` on the first run |
| `--skip-doc-update` | off | Skip the theory document update step for this run (faster debugging, no document versions produced) |

Iteration-window semantics (`resolve_iteration_window` in `main.py`/`main_bo_only.py`), which are
the most common source of confusion:

```text
completed      = number of completed iterations read from progress.json
--add-iterations N  -> target = completed + N
--max-iterations N  -> target = N
neither             -> main.py: agentos_default_iterations (sheet, currently 50)
                       main_bo_only.py: 20
execution range     = [completed + 1 .. target]      # nothing is rerun
```

### 4.2 `main.py`-only arguments

| Argument | Default | Purpose |
|---|---|---|
| `--runtime {workflow,agentos}` | workflow | `workflow` = run the full loop locally; `agentos` = start an HTTP service that is triggered by a form, which may override some parameters |
| `--params-csv PATH` | `config/agentos_params.csv` | Editable parameter sheet (CSV). When present, rows with `enabled=1` override the code defaults; explicit CLI arguments still win |
| `--llm-formula-generation` / `--no-llm-formula-generation` | config default (on) | Whether the LLM may propose new candidate formulas from `llm_formula_generation.start_iteration` (iteration 2 by default) on. When disabled no proposer call is made at all, but an empty `llm_formula_proposals.json` (with `parents`/`parent_stats`) is still written to keep the resume contract. **Disabling it switches to a `llm_nogen_*` directory** |
| `--chemistry-diversity` / `--no-chemistry-diversity` | config default (on) | Screening strategy. On = chemistry-diverse reranking (candidates carry `candidate_id`, evaluation uses `candidate_scores` and only formula-level evidence); off = the original `llm_full_rerank` (evaluation uses `selected_materials` and restores BO metric evidence: `k_pred`/`mu_log`/`sigma_log`/95% CI/EI/rank). **Disabling it switches to a `llm_*_legacy` directory** |
| `--websearch-enabled` / `--no-websearch-enabled` | on | Whether WebSearch evidence is used to enrich material assessment during evaluation. Disabling saves time and outbound network access |
| `--websearch-top-n N` | 5 (the sheet currently says 10) | The first N candidates entering evaluation get a WebSearch lookup (the rest use local evidence only) |
| `--skip-doc-update` / `--no-skip-doc-update` | off | Skip this iteration's theory document update, producing no new `doc/v0.0.<iteration>/` |
| `--agentos-host HOST` | 127.0.0.1 | Bind address for `--runtime agentos` |
| `--agentos-port PORT` | 7777 | Bind port for `--runtime agentos` |

`main_bo_only.py` offers two extra arguments:

| Argument | Default | Purpose |
|---|---|---|
| `--relax-timeout-sec N` | 900 | Timeout of a single "relaxation + phonon" task in seconds. A timeout counts as failure; long-range or large-cell systems need a larger value |
| `--start-iteration N` | none | Explicitly start at iteration N (by default resume from the first incomplete iteration) |

Hidden compatibility arguments (`argparse.SUPPRESS`, absent from `--help`, kept for older scripts):

| Argument | Equivalent to |
|---|---|
| `--n-top-candidates N` | `--top-k-bayes N` |
| `--n-select N` | `--top-k-screen N` |

### 4.3 Tuning knobs that only exist in the sheet / config

These have no CLI switch. Edit `config/agentos_params.csv` (or `config/config.yaml`):

| Key | Default | Purpose |
|---|---|---|
| `relax_timeout_sec` | 900 | Timeout of a single relaxation + phonon task (seconds). `main.py` can only set it through the parameter sheet |
| `xi` | 0.01 | Exploration coefficient of the EI acquisition function; larger values favor high uncertainty (exploration) |
| `k_threshold` | 1.0 | `κ_L` success threshold in W/(m·K). Only materials below it count as low-thermal-conductivity hits |
| `pressure` | 0.0 | Pressure applied during relaxation (GPa) |
| `seed_stride` | 1000 | Per-iteration seed stride (see `--seed`) |
| `max_workers` | 4 | Structure-generation concurrency per GPU (1 on a single GPU) |
| `relax_workers` | 1 | Relaxation concurrency per GPU |
| `phonon_workers` | 1 | Phonon concurrency (currently relaxed and phonons run together; the key is kept) |
| `prefer_isolated_relax_process` | true | Prefer running relaxation in a dedicated subprocess (isolates VRAM and crashes) |
| `allow_in_process_relax_fallback` | true | Allow falling back to in-process execution when a dedicated subprocess is unavailable |
| `websearch_strategy` | hybrid | WebSearch retrieval strategy |
| `websearch_queries_per_candidate` | 2 | Number of search queries per candidate |
| `websearch_theory_template` | none | Theory template used for retrieval (may be left empty) |
| `llm_formula_generation_start_iteration` | 2 | First iteration at which the LLM may propose candidates (never in iteration 1) |
| `llm_formula_proposal_count` | 20 | Number of candidate formulas the LLM proposes per iteration |
| `llm_formula_parent_count` | 10 | Number of parent candidates shown to the LLM |
| `llm_formula_generation_max_tokens` | 12000 | Max output tokens of a candidate-generation request |
| `llm_formula_generation_max_retries` | 3 | Retries for a failed candidate-generation request |
| `chemistry_evaluator_evidence` | chemistry_only | On the chemistry-diverse path the evaluator only gets formula-level evidence (it automatically becomes `full` when that path is disabled) |
| `max_same_element_tuple` | 2 | Same element set may appear at most twice (diversity constraint) |
| `max_same_stoichiometry_pattern` | 4 | Same stoichiometry pattern may appear at most four times (diversity constraint) |
| `high_symmetry_parent_enabled` | true | Whether high-symmetry historical structures are used as parent seeds |
| `high_symmetry_min_crystal_system` | orthorhombic | Lowest crystal system accepted as a high-symmetry parent |
| `high_symmetry_parent_scope` | historical_seed_only | Scope in which high-symmetry parents are used |
| `agentos_default_iterations` | 50 | **Default total iteration target of `main.py` when no iteration argument is passed** (the sheet currently says 50) |
| `agentos_max_iterations_cap` | 20 | Upper bound for the iteration count accepted from the AgentOS form |
| `agentos_allow_text_iteration_override` | true | Allow overriding the iteration count from AgentOS conversation text |
| `agentos_ws_ping_interval` / `agentos_ws_ping_timeout` | none | AgentOS WebSocket keep-alive interval / timeout |

### 4.4 Parameter sheet `config/agentos_params.csv`

Format: `key,value,enabled,notes`. Only rows with `enabled=1` (or `true/yes/on`) take effect; rows
with `enabled=0` are UI prefill only and never override the configuration.

```csv
key,value,enabled,notes
websearch_enabled,true,1,Enable/disable web search
websearch_top_n,10,1,How many top candidates to enrich
top_k_bayes,20,1,Bayes top-k candidates
top_k_screen,10,1,AI screening top-k
samples,100,1,Bayesian sample size
n_structures,5,1,Structures generated per material
relax_timeout_sec,900,1,Relaxation timeout per task
postprocess_workers,2,1,Bounded CPU workers for deduplication extraction and merge
novelty_workers,4,1,Bounded workers for final database novelty queries
skip_doc_update,false,1,Skip theory update step
agentos_default_iterations,50,1,Default iterations for AgentOS
agentos_ws_ping_interval,none,1,Auto-added by runtime memory
agentos_ws_ping_timeout,none,1,Auto-added by runtime memory
phonon_imag_tol,-0.1,1,Auto-added by runtime memory
seed,42,0,Auto-added by runtime memory
```

> That is the current content of the repository file (it drifts slowly as runs persist values).
> Note that `websearch_top_n=10` with `enabled=1` overrides the code default of 5, while `seed=42`
> has `enabled=0` and therefore does not apply. `agentos_default_iterations=50` is the default total
> iteration target of `main.py` when no iteration argument is given.

Notes:

- Only keys **already present in the configuration** are accepted; unknown keys trigger an
  `Unknown config key skipped` warning and are ignored.
- At the end of every `main.py` run the **effective values** of `PARAM_MEMORY_KEYS` (`samples`,
  `n_structures`, `top_k_bayes`, `top_k_screen`, `postprocess_workers`, `novelty_workers`,
  `websearch_enabled`, `websearch_top_n`, `phonon_imag_tol`, `seed`, `relax_timeout_sec`,
  `skip_doc_update`, `agentos_default_iterations`, `agentos_ws_ping_interval`,
  `agentos_ws_ping_timeout`) are written back into that CSV (new rows default to `enabled=0`).
- `llm_formula_generation_enabled` and `chemistry_diversity_enabled` can also be added by hand
  (both already exist in the configuration). In that case the sheet decides `run_mode` and the
  result directory even **without** CLI switches.

### 4.5 `config/config.yaml`

Read by `src/utils/config_loader.py`; it mainly covers the BO search space, thresholds and tool
behaviour:

| Location | Key values | Purpose |
|---|---|---|
| `bayesian_optimization.acquisition` | `function: EI`, `xi: 0.01` | Acquisition function and exploration coefficient |
| `bayesian_optimization.sampling` | `method: mcmc`, `n_samples: 100`, `max_atoms: 20` | Sampling method, sample count, maximum atom count |
| `sampling.allowed_elements` | 14 elements | The allowed element set |
| `sampling.hard_constraints.schema` | A/B/Ch grouping | The A-B-Ch constraint (A: Ag/Cu/In/Sn/Pb; B: As/Sb/Ge/Bi/Ti/V; Ch: S/Se/Te) |
| `sampling.hard_constraints.stoichiometry` | 1-10 per group (2-10 for A) | Atom-count ranges per group |
| `sampling.mcmc` | `w_ei/w_k/w_sigma/temperature/burn_in/thin` | MCMC sampling weights and iteration parameters |
| `model.retrain_from_round` | 2 | Iteration from which the experiment's own GPR replaces the initial model |
| `thresholds.thermal_conductivity` | 1.0 | `κ_L` success threshold (linked to `k_threshold`) |
| `thresholds.dynamic_min_frequency` | -0.1 | Imaginary phonon frequency threshold (linked to `--phonon-imag-tol`) |
| `tools.crystallm.model_path` | `src/tools/crystallm/pre-trained-model/crystallm_v1_small` | CrystaLLM weights directory |
| `tools.mattersim.phonon_supercell` | `[2, 2, 2]` | Phonon supercell size |
| `tools.mattersim.displacement` | 0.01 | Finite-displacement amplitude (Å) |
| `tools.mattersim.timeout` | 600 | MatterSim per-call timeout (seconds) |
| `tools.ai4kappa.temperature` | 300 | Temperature used for the `κ_L` calculation (K) |
| `loop.*` | `max_iterations: 20` etc. | **Informational only**; the real iteration count comes from the CLI / parameter sheet |

Also note: `tools.crystallm.num_samples/top_k/max_new_tokens` are currently hard-coded in the code
(`num_samples = --n-structures`, `top_k = 10`, `max_new_tokens = 2000`), so editing them has no
effect.

### 4.6 Environment variables

See the table in 1.5. The ones that change runtime behaviour rather than LLM access are:

| Variable | Effect |
|---|---|
| `PHONON_GPUS` | Overrides the GPU list used for phonon calculations (`0,1`); when unset the same GPUs as the workflow are used |
| `PHONON_PARALLEL_WORKERS` | Upper bound for phonon worker processes (falls back to the configured value) |
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA allocator tuning; set automatically off Windows and removed on Windows |

---

## 5. Output directories and artifacts

### 5.1 Directory layout

Taking `llm_gen_diverse` (the default combination) as the example, where `<run_mode>` can be
`llm_gen_diverse` / `llm_gen_legacy` / `llm_nogen_diverse` / `llm_nogen_legacy`, and the BO-only
baseline is `bo`:

```text
<run_mode>/
├── data/
│   └── iteration_<N>/data.csv              # dataset at the end of iteration N (iteration_0 = bootstrap data)
├── doc/
│   └── v0.0.<N>/Theoretical_principle_document.md   # theory document per iteration (v0.0.0 = initial document)
├── models/
│   └── GPR/iteration_<N>/
│       ├── gpr_thermal_conductivity.joblib  # GPR trained during iteration N+1 (iteration_0 = model trained by the first iteration)
│       ├── gpr_scaler.joblib                # feature scaler
│       ├── model_metadata.json              # training-data hash, sample count, metadata
│       ├── Best_Model_Prediction.png
│       └── Final_Model_Comparison.png
└── results/
    ├── progress.json                        # per-iteration, per-step completion state (resume source of truth)
    ├── workflow.db                          # runtime workflow database
    ├── screening_summary.csv                # per-iteration screening metrics (see 5.2)
    ├── summary/                             # cross-iteration material summaries
    │   ├── all_materials_summary.csv
    │   ├── stable_materials_summary.csv
    │   ├── success_materials_summary.csv
    │   └── final_materials_db_novelty.csv
    ├── run_<timestamp>.log                   # full log of each run
    └── iteration_<N>/
        ├── logs/ei_acquisition.log          # BO/EI sampling and ranking log
        ├── reports/                         # archived LLM interactions (input/output pairs)
        │   ├── llm_candidate_scoring_input.md / _output.md
        │   ├── llm_formula_generation_input.md / _output.md
        │   └── llm_theory_update_input.md / _output.md
        ├── selected_results/                # all intermediate/final screening results
        ├── processed_structures/<formula>/  # generated structures, prompts, postprocessing per material
        ├── MyRelaxStructure/<formula>/      # relaxation + phonon artifacts
        │   ├── sample_<i>.cif
        │   ├── sample_<i>_phonon/
        │   ├── relax_phonon_results.csv
        │   └── thermal_conductivity.csv      # κ_L of this material
        └── success_examples/                 # successful materials
            ├── cif_files_stable/             # dynamically stable structures
            ├── cif_files_success/            # structures below the κ_L threshold
            ├── extraction_status.json
            └── final_materials_db_novelty.csv / .json   # MP/AFLOW novelty check
```

### 5.2 Key files per iteration (`results/iteration_<N>/selected_results/`)

| File | Contents |
|---|---|
| `raw_samples.csv` | Every candidate and prediction produced by the BO sampler (`formula`, `k_pred`, `mu_log`, `sigma_log`, `ei`, 95% CI), unsorted |
| `all_samples.csv` | The same, sorted by EI with a `rank` column: the full EI-ranked view |
| `top20_materials.json` | BO acquisition output: the `top20` list (`rank`/`formula`/`k_pred`/`mu_log`/`sigma_log`/`ei`/`k_lower`/`k_upper`) plus metadata of that sampling round (`xi`, `f_min`, `n_samples`, ...) |
| `llm_formula_proposals.json` | Candidate formulas proposed by the LLM and their scores. **Written even when generation is disabled**, with an empty `proposals` array and `parents`/`parent_stats` retained |
| `novel_candidates.csv` | The complete candidate pool: BO candidates merged with LLM candidates and sorted by EI (includes `candidate_source`, `original_bo_rank`, `bo_prediction_status`) |
| `websearch_enriched_candidates.csv` | The candidate table after WebSearch enrichment (includes `websearch_queries`/`websearch_summary`/`websearch_sources`) |
| `merged_screening_candidates.csv` | The final candidate table sent to LLM evaluation: the previous file plus `candidate_id`; this is the evaluation input on the chemistry-diverse path |
| `selection_trace.csv` / `.json` | Per-candidate screening trace with `screening_mode` (`chemistry_diverse_rerank` or `llm_full_rerank`) and rejection reasons |
| `ai_candidate_scores.csv` | Per-candidate scores on the chemistry-diverse path (`candidate_scores` mode) |
| `ai_selected_materials.csv` | The finally selected materials (`top_k_screen` of them); written on both paths |

Key columns of `results/screening_summary.csv` (one appended row per run, comparable across
iterations):

| Column | Meaning |
|---|---|
| `iteration` / `screening_mode` | Iteration / screening mode used (`chemistry_diverse_rerank` or `llm_full_rerank`) |
| `selected_count` | Number of materials actually sent to structure calculation |
| `success_count` / `stable_count` | Materials below the `κ_L` threshold / dynamically stable materials |
| `success_rate_at_k` / `stable_rate_at_k` | The two counts above divided by `selected_count` (the core ablation metric) |
| `best_kappa_in_selected_success` | Lowest `κ_L` among this iteration's successes |
| `selected_from_bo_top3_count` / `selected_from_bo_13_20_count` | How many selected materials came from BO ranks 1-3 / 13-20 (did the run exploit long-tail candidates) |
| `tail_promotions_count` / `tail_promotions_success_count` | Long-tail candidates promoted into the selection / how many of them succeeded |
| `protected_top3_dropped_count` | How many of the BO top 3 were dropped |
| `candidate_count` / `bo_candidate_count` / `llm_candidate_count` | Candidate-pool size / BO candidates / LLM-proposed candidates |
| `llm_selected_count` / `llm_stable_count` / `llm_success_count` / `llm_proposal_count` | Selected materials that the LLM proposed / how many are stable / how many succeeded / total LLM proposals this iteration (all 0 when generation is off) |
| `high_symmetry_parent_count` / `high_symmetry_parent_stats` | Number and distribution of high-symmetry parent seeds |
| `final_p1_count` / `final_low_symmetry_count` / `final_high_symmetry_count` / `unresolved_symmetry_count` | Final structures by symmetry class (P1 / low / high / unresolved) |

### 5.3 Archive directories

| Trigger | Archive directory | Contents |
|---|---|---|
| `--reset` | `<run_mode>_old_<timestamp>/` | A copy of the whole `run_mode` tree |
| `--rebuild-from N` | `<run_mode>_rebuild_<timestamp>/` | Invalidated suffix: `results/iteration_{N..}`, `data/iteration_{N..}`, `models/GPR/iteration_{N-1..}`, `doc/v0.0.{N..}` and the previous `progress.json`, plus a `rebuild_manifest.json`. Requires iterations `1..N-1` to be complete |

Archive directories are also covered by the `.gitignore` rules `llm_*/` / `bo_*/`.
Archived artifacts are **audit-only** and are never used as resume inputs; the program only reads
the active state under `<run_mode>/`.

---

## 6. Core rules

1. **The two entry points never interfere**: `main.py` only touches `llm_*` directories,
   `main_bo_only.py` only touches `bo/`.
2. **The switches decide the directory**: the four combinations bootstrap and resume
   independently; a resume must use the same combination.
3. **Resume validates first**: `progress.json` must agree with the artifacts on disk; missing
   predecessor artifacts make the run refuse to resume (fix it with `--reset` or `--rebuild-from`)
   so an untrustworthy history is never produced. A resume continues from the **first incomplete
   step**, and the next iteration starts only once the current one is fully complete.
4. **Every iteration is reproducible**: a fixed `seed` plus an increasing `seed_stride` means rerunning
   the same iteration gives the same result; `--non-deterministic-torch` gives that up.
5. **Failure boundaries per stage**: failed structure generation / relaxation / phonon tasks are
   recorded in `relax_phonon_results.csv` and filtered out during extraction; only a fully unusable
   structure batch stops the iteration (`--allow-partial-structure` relaxes this).
6. **The LLM only advises, computation decides**: the LLM generates candidates, screens them and
   updates the document; dynamical stability comes from phonons, `κ_L` from elastic moduli plus
   Slack/PINK, and the final success call is based on `k_threshold`.
7. **Secrets live only in `.env`**: never write API keys into code, configuration or logs.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'agno'` | Wrong environment or dependencies not installed: `conda activate kappap && pip install -r requirements.txt` |
| `CrystaLLM model not found` | `src/tools/crystallm/pre-trained-model/crystallm_v1_small/ckpt.pt` is missing, see 1.3.1 |
| All `κ_L` are 0 / elastic moduli missing | `src/tools/kappa_lib/model/*-pre-trained.pth.tar` missing or renamed, see 1.3.3 |
| MatterSim keeps failing to download | Restricted network; pre-place `MatterSim-v1.0.0-1M.pth` in `~/.local/mattersim/pretrained_models/` as in 1.3.2 |
| `Requested N GPU(s) but only M visible` | `--num-gpus` exceeds the visible device count; lower it or set `CUDA_VISIBLE_DEVICES` |
| Resume complains about missing predecessor artifacts | A different switch combination was used (so it landed in another directory), or history was cleaned; add `--reset` / `--rebuild-from` |
| "How do I really restart from iteration 1?" | Add `--reset` (it archives first; nothing is deleted outright) |
| Want to see what the LLM actually said in an iteration | Read the `*_input.md` / `*_output.md` pairs under `results/iteration_<N>/reports/` |
| CrystaLLM subprocesses crash on VRAM allocation on Windows | `PYTORCH_CUDA_ALLOC_CONF` contains `expandable_segments` (unsupported on Windows); the program strips it, so do not add it back in your own environment |
