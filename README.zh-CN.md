# ADLM

Auto_Discover_LowThermalConductivity_Materials (ADLM) 是一个面向低热导率材料发现的自动化工作流。它将 Bayesian Optimization、基于 LLM 的筛选、结构生成、热导率预测、声子或稳定性验证，以及迭代式数据集更新整合到同一套流程中。

[Read the English version](./README.md)

## 项目提供的能力

当前仓库有两个主要入口：

- `main.py`：基于 Agno 的 LLM 增强工作流
- `main_bo_only.py`：不包含 LLM 筛选与理论文档更新的纯 BO 工作流

两种模式共享同一类迭代逻辑：

1. 基于已有数据训练或刷新代理模型。
2. 使用 Bayesian Optimization 采样候选组分。
3. 在 LLM 模式下对候选材料进行额外筛选。
4. 生成结构并执行弛豫、声子和热导率相关计算。
5. 提取成功材料或稳定材料。
6. 更新下一轮迭代数据集。
7. 在 LLM 模式下按轮次更新理论文档版本。

## 仓库结构

```text
aslk/
|- main.py
|- main_bo_only.py
|- README.md
|- README.zh-CN.md
|- .env.example
|- pyproject.toml
|- requirements.txt
|- config/
|  |- config.yaml
|  |- llm_config.yaml
|  |- agentos_params.csv
|- src/
|  |- agents/
|  |- database/
|  |- generators/
|  |- models/
|  |- tools/
|  |- utils/
|  |- workflow/
|- scripts/
|- analysis_scripts/
|- data/
|- doc/
|- llm/
|- bo_first_iteration/
```

> [!NOTE]
> `main_bo_only.py` 当前已经切换为写入 `bo/...`。仓库中仍然保留了一些历史脚本、旧实验目录或旧命名，例如 `bo_new/...`。本文档以下内容以当前入口脚本的实际行为为准。

## 环境要求

- Python `3.10+`
- 推荐使用 `uv` 管理依赖
- 如需完整运行结构和物性计算流程，建议使用支持 CUDA 的 GPU

`pyproject.toml` 中的主要依赖包括：

- `agno`
- `google-adk`
- `pandas`、`numpy`、`matplotlib`、`seaborn`
- `ase`、`pymatgen`
- `scikit-learn`、`xgboost`、`joblib`
- `torch`、`torchvision`、`torchaudio`
- `mattersim`

> [!IMPORTANT]
> 当前 `pyproject.toml` 默认固定了 CUDA 12.4 对应的 PyTorch 轮子（`cu124`）。如果你的环境不是这一版本，需要手动安装匹配的 PyTorch。

## 安装

### 方式一：使用 `uv`

```bash
pip install uv
uv sync
```

安装开发依赖：

```bash
uv sync --extra dev
```

### 方式二：使用 `pip`

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

手动安装 CUDA 12.4 版 PyTorch 示例：

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

## 环境变量

先从模板生成 `.env`：

```bash
copy .env.example .env
```

当前模板中包含：

- `WORKFLOW_MODEL`
- `WORKFLOW_API_KEY`
- `WORKFLOW_BASE_URL`
- `THEORY_UPDATE_MODEL`
- `THEORY_UPDATE_API_KEY`
- `THEORY_UPDATE_BASE_URL`
- `TEMPERATURE`
- `MP_API_KEY`
- `AFLOW_BASE_URL`（可选）

最小示例：

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

说明：

- `WORKFLOW_*` 主要用于 `main.py` 中的候选材料筛选。
- `THEORY_UPDATE_*` 主要用于 `main.py` 中的理论文档更新。
- `MP_API_KEY` 用于 Materials Project 查询。

## 快速开始

安装依赖：

```bash
uv sync
```

创建 `.env`：

```bash
copy .env.example .env
```

运行 LLM 工作流：

```bash
python main.py
```

运行纯 BO 工作流：

```bash
python main_bo_only.py
```

汇总结果：

```bash
python scripts/summarize_results.py --results-dir llm/results
python scripts/summarize_results.py --results-dir bo/results
```

## LLM 工作流

入口：

```bash
python main.py
```

当前代码默认使用的根目录：

- `llm/results`
- `llm/models/GPR`
- `llm/data`
- `llm/doc`

默认初始输入：

- 数据集：`data/processed_data.csv`
- 理论文档：`doc/Theoretical_principle_document.md`

### 常用命令

固定运行轮数：

```bash
python main.py --max-iterations 3
```

基于已有进度继续运行：

```bash
python main.py --add-iterations 2
```

覆盖采样和筛选参数：

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

允许部分结构计算失败后继续：

```bash
python main.py --allow-partial-structure
```

重置进度：

```bash
python main.py --reset
```

指定初始数据和理论文档：

```bash
python main.py --init-data data/processed_data.csv --init-doc doc/Theoretical_principle_document.md
```

### AgentOS 运行模式

```bash
python main.py --runtime agentos --agentos-host 127.0.0.1 --agentos-port 7777
```

`main.py` 还会读写 `config/agentos_params.csv`。这个文件在当前项目中的作用基本是：

- 运行界面的可编辑默认参数表
- 运行时覆盖参数表
- 跨运行的轻量参数记忆表

常见参数键包括：

- `websearch_enabled`
- `websearch_top_n`
- `top_k_bayes`
- `top_k_screen`
- `samples`
- `n_structures`
- `relax_timeout_sec`
- `skip_doc_update`
- `agentos_default_iterations`

## 纯 BO 工作流

入口：

```bash
python main_bo_only.py
```

当前代码中的输出根目录：

- `bo/results`
- `bo/models/GPR`
- `bo/data`

纯 BO 工作流执行的核心步骤包括：

1. 模型训练
2. BO 候选生成
3. 从候选中选取前 `n_select` 个进入后续计算
4. 结构生成与下游计算
5. 合并结果并提取成功材料
6. 更新下一轮数据集

### 常用命令

运行 10 轮：

```bash
python main_bo_only.py --max-iterations 10
```

从指定轮次开始：

```bash
python main_bo_only.py --start-iteration 3 --max-iterations 10
```

覆盖采样或结构参数：

```bash
python main_bo_only.py --samples 150 --n-top-candidates 30 --n-select 10 --n-structures 5
```

指定初始数据：

```bash
python main_bo_only.py --init-data data/processed_data.csv
```

允许部分结构计算失败后继续：

```bash
python main_bo_only.py --allow-partial-structure
```

重置并重新运行：

```bash
python main_bo_only.py --reset
```

## 配置文件

### `config/config.yaml`

这是主算法和工具配置文件，包含：

- 循环控制
- BO 采样和采集函数参数
- 模型默认参数
- 组分生成约束
- 外部工具参数
- 阈值、日志和输出示例配置

当前项目里比较关键的字段包括：

- `loop.max_iterations`
- `bayesian_optimization.acquisition.xi`
- `bayesian_optimization.sampling.n_samples`
- `bayesian_optimization.sampling.hard_constraints`
- `tools.crystallm.model_path`
- `tools.ai4kappa.k_threshold`
- `tools.mattersim.imaginary_freq_threshold`

### `config/llm_config.yaml`

该文件主要描述 LLM 工作流中的理论文档版本命名方式，例如：

- `llm/doc/v0.0.{version}/Theoretical_principle_document.md`

## 输出目录说明

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

### BO 模式

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

两种工作流都使用 `progress.json` 来避免重复执行已完成步骤。按模式不同，常见跟踪步骤包括：

- `train_model`
- `bayesian_optimization`
- `ai_evaluation`
- `structure_calculation`
- `merge_results`
- `success_extraction`
- `document_update`
- `data_update`

## 辅助脚本

汇总各轮成功材料或稳定材料：

```bash
python scripts/summarize_results.py --results-dir llm/results
python scripts/summarize_results.py --results-dir bo/results
```

排查环境或筛选工具：

- `scripts/check_screening_tools.py`

实验分析辅助脚本：

- `analysis_scripts/run_and_filter_iterations.py`
- `analysis_scripts/compare_and_plot.py`
- `scripts/compare_llm_bo_formula_overlap.py`

> [!NOTE]
> 某些分析脚本内部仍然保留了 `bo_new` 这样的历史命名标签，但 `main_bo_only.py` 当前真实输出目录已经是 `bo`。

## 排错

### 没有可用的 LLM 配置

检查：

- `.env` 是否存在
- `WORKFLOW_API_KEY` 是否已设置
- `THEORY_UPDATE_API_KEY` 是否已设置
- 所填模型名是否能被对应接口正常支持

### Materials Project 查询失败

检查 `.env` 中是否已配置 `MP_API_KEY`。

### CUDA 或 PyTorch 不匹配

请按本机 CUDA 或 CPU 环境重新安装匹配版本的 PyTorch。

### Windows 编码问题

如果终端输出乱码：

```bash
set PYTHONIOENCODING=utf-8
python main.py
```

## 参考文件

如果后续还要继续扩展文档，建议优先查看：

- `main.py`
- `main_bo_only.py`
- `src/workflow/agno_pipeline.py`
- `src/workflow/agno_steps.py`
- `src/agents/llm_models.py`
- `src/utils/path_config.py`
- `src/utils/progress_tracker.py`
- `scripts/summarize_results.py`
