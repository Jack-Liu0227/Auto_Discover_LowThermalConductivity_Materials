# ASLK

低热导率材料自动搜索工作流。

[Read in English / 英文文档](./README.md)

ASLK 是一个面向低热导率材料发现的自动化工作流系统，结合了 Bayesian Optimization、LLM 筛选、结构生成、热导率预测、声子或稳定性验证，以及迭代式数据更新。

当前仓库主要提供两个入口：

- `main.py`：LLM + Agno 工作流
- `main_bo_only.py`：纯 Bayesian Optimization 工作流

## 项目概览

典型迭代流程包括：

1. 基于已有数据训练或更新代理模型。
2. 使用 Bayesian Optimization 生成新候选材料。
3. 在 `main.py` 中使用 LLM 做进一步筛选。
4. 生成结构并执行下游计算。
5. 提取成功材料或稳定材料。
6. 更新下一轮数据集。
7. 在 LLM 模式下持续更新理论文档。

## 运行模式

### LLM 工作流

入口：

```bash
python main.py
```

特点：

- 使用 Agno workflow 编排流程
- 支持 `workflow` 和 `agentos`
- 默认模型提供商为 DeepSeek
- 支持通过 `.env` 配置回退提供商
- 维护 `llm/data`、`llm/results`、`llm/models`、`llm/doc`
- 支持通过 `config/agentos_params.csv` 覆盖参数

### BO-only 工作流

入口：

```bash
python main_bo_only.py
```

特点：

- 只执行 BO、结构计算、结果提取和数据更新
- 不依赖 LLM 评估和理论文档更新
- 维护 `bo_new/data`、`bo_new/results`、`bo_new/models`

## 仓库结构

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

## 环境要求

推荐环境：

- Python `3.10+`
- 使用 `uv` 管理依赖
- 如果要完整运行结构和物性流程，建议使用支持 CUDA 的 GPU

常见依赖包括：

- `torch`
- `pymatgen`
- `ase`
- `mattersim`
- `phonopy`

> [!IMPORTANT]
> `pyproject.toml` 当前默认锁定 `cu124` 版本的 PyTorch。如果你的 CUDA 版本不同，需要手动安装匹配版本。

## 安装

### 方式一：uv

```bash
pip install uv
uv sync
```

安装开发依赖：

```bash
uv sync --extra dev
```

### 方式二：pip

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果需要手动安装 CUDA 版 PyTorch：

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

## 环境变量

项目会从仓库根目录 `.env` 中读取环境变量。先复制模板：

```bash
copy .env.example .env
```

### 最小模型配置

项目现在只保留两个明确的模型槽位：

- `WORKFLOW_MODEL`
- `WORKFLOW_API_KEY`
- `WORKFLOW_BASE_URL`
- `THEORY_UPDATE_MODEL`
- `THEORY_UPDATE_API_KEY`
- `THEORY_UPDATE_BASE_URL`
- `TEMPERATURE`

说明：

- `WORKFLOW_MODEL` 用于主 workflow 的筛选步骤
- `THEORY_UPDATE_MODEL` 用于文档更新步骤
- 两者默认都是 `deepseek-chat`
- 两者默认 base URL 都指向 `https://api.deepseek.com/v1`
- 变量名应为 `WORKFLOW_BASE_URL`，不是 `WORKFLOW_BASW_URL`

### 数据库或查询相关变量

- `MP_API_KEY`：Materials Project 查询必需
- `AFLOW_BASE_URL`：可选，默认 `https://aflowlib.org/API/aflux/`

### 最小示例

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

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 创建并编辑 `.env`

```bash
copy .env.example .env
```

至少配置：

- `WORKFLOW_API_KEY`
- `THEORY_UPDATE_API_KEY`
- 如果依赖 Materials Project，则配置 `MP_API_KEY`

### 3. 运行 LLM 工作流

```bash
python main.py
```

### 4. 或运行 BO-only 工作流

```bash
python main_bo_only.py
```

### 5. 汇总结果

```bash
python scripts/summarize_results.py --results-dir llm/results
python scripts/summarize_results.py --results-dir bo_new/results
```

## LLM 工作流用法

### 基础运行

```bash
python main.py
```

默认输出路径：

- `llm/results`
- `llm/data`
- `llm/models/GPR`
- `llm/doc`

默认初始输入：

- 数据集：`data/processed_data.csv`
- 理论文档：`doc/Theoretical_principle_document.md`

### 常用命令

固定轮数：

```bash
python main.py --max-iterations 3
```

基于现有进度继续：

```bash
python main.py --add-iterations 2
```

覆盖采样或筛选参数：

```bash
python main.py --samples 150 --n-top-candidates 30 --n-select 10 --n-structures 5
```

指定 GPU 数量：

```bash
python main.py --num-gpus 2
```

关闭 web search：

```bash
python main.py --no-websearch-enabled
```

允许结构计算部分失败：

```bash
python main.py --allow-partial-structure
```

重置进度：

```bash
python main.py --reset
```

指定初始数据与理论文档：

```bash
python main.py --init-data data/processed_data.csv --init-doc doc/Theoretical_principle_document.md
```

### AgentOS 模式

```bash
python main.py --runtime agentos --agentos-host 0.0.0.0 --agentos-port 8000
```

### `config/agentos_params.csv`

`main.py` 会读取并回写 `config/agentos_params.csv`。它可以用作：

- UI 或默认参数表
- 运行时覆盖参数表
- 轻量级参数记忆表

常见字段：

- `websearch_enabled`
- `websearch_top_n`
- `top_k_bayes`
- `top_k_screen`
- `samples`
- `n_structures`
- `relax_timeout_sec`
- `skip_doc_update`
- `agentos_default_iterations`

## BO-only 工作流用法

### 基础运行

```bash
python main_bo_only.py
```

### 常用命令

运行 10 轮：

```bash
python main_bo_only.py --max-iterations 10
```

从指定轮次开始：

```bash
python main_bo_only.py --start-iteration 3 --max-iterations 10
```

覆盖 BO 参数：

```bash
python main_bo_only.py --samples 150 --n-top-candidates 30 --n-select 10 --n-structures 5
```

指定初始数据：

```bash
python main_bo_only.py --init-data data/processed_data.csv
```

重置并重跑：

```bash
python main_bo_only.py --reset
```

## 配置文件

### `config/config.yaml`

主算法或工具配置文件，包含：

- 循环控制
- Bayesian Optimization 参数
- 模型参数
- 生成器或工具参数
- 阈值与超时

重点字段：

- `loop.max_iterations`
- `bayesian_optimization.acquisition.xi`
- `bayesian_optimization.sampling.n_samples`
- `tools.crystallm.model_path`
- `tools.ai4kappa.k_threshold`
- `tools.mattersim.imaginary_freq_threshold`

### `config/llm_config.yaml`

这个文件主要描述 LLM 输入输出布局和示例模型配置。实际运行时的模型链路主要由 `.env` 和 `src/agents/llm_models.py` 决定。

## 输出结构

### LLM 模式

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

### BO-only 模式

```text
bo_new/
|- data/
|- models/
|- results/
```

### `progress.json`

工作流使用 `progress.json` 避免重复执行已完成步骤。常见跟踪步骤包括：

- `train_model`
- `bayesian_optimization`
- `ai_evaluation`
- `structure_calculation`
- `merge_results`
- `success_extraction`
- `document_update`

## 辅助脚本

### 结果汇总

```bash
python scripts/summarize_results.py --results-dir llm/results
```

该脚本会把每轮的 success 或 stable CSV 汇总成总表。

### 环境或工具检查

`scripts/check_screening_tools.py` 主要用于排错，不作为默认 quick start 步骤。

### 分析脚本

`analysis_scripts/` 中包含实验后处理和画图脚本，例如：

- `analysis_scripts/run_and_filter_iterations.py`
- `analysis_scripts/compare_and_plot.py`

这些脚本更适合在实验完成后使用。

## 排错

### 没有可用的 LLM

检查：

- `.env` 是否存在
- `WORKFLOW_API_KEY` 和 `THEORY_UPDATE_API_KEY` 是否已设置
- `WORKFLOW_MODEL` 和 `THEORY_UPDATE_MODEL` 是否是当前 endpoint 支持的模型名

### Materials Project 查询失败

检查 `.env` 中是否已配置 `MP_API_KEY`。

### OQMD 或 AFLOW 查询异常

代码已经实现 Python API 到 HTTP 的 fallback。如果两者都失败，通常是上游服务或环境问题，而不是主流程逻辑问题。

### CUDA 或 PyTorch 不匹配

请按本机环境重新安装对应版本的 PyTorch。

### Windows 编码问题

如果终端输出乱码：

```bash
set PYTHONIOENCODING=utf-8
python main.py
```

## 参考文件

如果你后续还要继续扩展文档，可以优先查看：

- `main.py`
- `main_bo_only.py`
- `src/workflow/agno_pipeline.py`
- `src/agents/llm_models.py`
- `src/utils/param_sheet.py`
- `src/utils/progress_tracker.py`
- `scripts/summarize_results.py`
