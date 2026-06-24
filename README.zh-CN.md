# ADLM：低热导率材料自动发现系统

[English README](./README.md)

ADLM 是一个面向低热导率材料发现的研究型自动化工作流。项目将贝叶斯优化、可选的 LLM 候选材料筛选、晶体结构生成、热导率预测、稳定性或声子验证，以及迭代式数据集更新整合到同一套 Python 流程中。

当前仓库提供两个主要运行入口：

- `main.py`：带 LLM 筛选与理论文档更新能力的工作流。
- `main_bo_only.py`：不依赖 LLM 筛选的纯贝叶斯优化工作流。

> [!NOTE]
> `llm/`、`bo/`、`featureEngeering/`、日志、缓存和计算器临时目录都属于运行产物，默认不进入版本控制。仓库只保留复现实验所需的源码、配置、初始数据、参考文档和本地工具集成代码。

## 目录

- [核心能力](#核心能力)
- [仓库结构](#仓库结构)
- [环境要求](#环境要求)
- [安装](#安装)
- [环境变量配置](#环境变量配置)
- [快速开始](#快速开始)
- [LLM 增强工作流](#llm-增强工作流)
- [纯 BO 工作流](#纯-bo-工作流)
- [配置说明](#配置说明)
- [数据与输出目录](#数据与输出目录)
- [本地工具集成](#本地工具集成)
- [开发与验证](#开发与验证)
- [许可](#许可)
- [常见问题](#常见问题)

## 核心能力

- 面向低晶格热导率材料的迭代式发现流程。
- 高斯过程代理模型训练与按轮次刷新。
- 基于 Expected Improvement 的贝叶斯优化候选生成。
- 可选 LLM 候选材料筛选。
- 通过内置 CrystaLLM 集成生成候选晶体结构。
- 通过内置 kappa/CGCNN 集成估计热导率。
- MatterSim 结构弛豫与声子验证相关接口。
- Materials Project、AFLOW、OQMD 数据库访问辅助模块。
- 通过 `progress.json` 支持断点续跑。
- LLM 模式与纯 BO 模式使用互相隔离的输出根目录。

## 仓库结构

```text
aslk/
|- main.py                         # LLM 增强工作流入口
|- main_bo_only.py                 # 纯 BO 工作流入口
|- requirements.txt                # Python 依赖
|- uv.lock                         # uv 锁文件
|- .env.example                    # 环境变量模板
|- config/
|  |- config.yaml                  # 算法与工具配置
|  |- agentos_params.csv           # 可编辑运行参数表
|- data/
|  |- processed_data.csv           # 初始数据集
|- doc/
|  |- Theoretical_principle_document.md
|- src/
|  |- agents/                      # LLM 客户端、筛选、文档更新
|  |- analysis/                    # 离线特征分析工具
|  |- database/                    # MP/AFLOW/OQMD 查询辅助
|  |- generators/                  # BO 采样器与采集函数
|  |- models/                      # GPR 训练代码
|  |- schemas/                     # 工作流输入结构
|  |- tools/                       # CrystaLLM、kappa、MatterSim、结构工具
|  |- utils/                       # 配置、路径、进度、可复现性工具
|  |- workflow/                    # 分步骤工作流实现
|- README.md
|- README.zh-CN.md
```

## 环境要求

- Python 3.10 或更高版本。
- Windows PowerShell、Linux shell 或等价终端。
- 如需完整运行结构生成、MatterSim 或 torch 相关计算，建议使用支持 CUDA 的 GPU。
- 使用 LLM 筛选、理论文档更新、Materials Project 查询时，需要准备相应 API key。

主要依赖类别：

- 工作流与 LLM：`agno`、`google-adk`、`litellm`
- 数据与机器学习：`pandas`、`numpy`、`scikit-learn`、`xgboost`、`joblib`
- 材料科学：`ase`、`pymatgen`、`phonopy`、`spglib`
- 深度学习与计算器：`torch`、`torchvision`、`torchaudio`、`mattersim`
- 开发与分析：`pytest`、`black`、`isort`、`jupyter`、`shap`、`plotly`

> [!IMPORTANT]
> `requirements.txt` 中列出了 PyTorch 包名，但 GPU 版本需要与本机 CUDA 环境匹配。建议先按本机 CUDA 或 CPU 环境安装合适的 PyTorch，再运行完整工作流。

## 安装

创建并激活虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果使用 `uv`：

```bash
uv sync
```

CUDA 12.4 版 PyTorch 安装示例：

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

## 环境变量配置

复制模板：

```bash
copy .env.example .env
```

填写必要字段：

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

变量用途：

- `WORKFLOW_*`：`main.py` 中的 LLM 候选材料筛选。
- `THEORY_UPDATE_*`：`main.py` 中的理论文档更新。
- `TEMPERATURE`：默认 LLM 采样温度。
- `MP_API_KEY`：Materials Project 查询。
- `AFLOW_BASE_URL`：可选的 AFLOW 接口地址覆盖。

不要提交 `.env`，它已经被 `.gitignore` 忽略。

## 快速开始

运行 LLM 增强工作流：

```bash
python main.py --max-iterations 1
```

运行纯 BO 工作流：

```bash
python main_bo_only.py --max-iterations 1
```

做一次轻量语法检查：

```bash
python -m compileall main.py main_bo_only.py src
```

## LLM 增强工作流

入口命令：

```bash
python main.py
```

默认初始输入：

- 数据集：`data/processed_data.csv`
- 理论文档：`doc/Theoretical_principle_document.md`

默认运行输出根目录：

- `llm/data`
- `llm/models/GPR`
- `llm/results`
- `llm/doc`

常用命令：

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

AgentOS 运行模式：

```bash
python main.py --runtime agentos --agentos-host 127.0.0.1 --agentos-port 7777
```

`config/agentos_params.csv` 是可编辑参数表，也承担轻量运行参数记忆功能。常见字段包括 `websearch_enabled`、`websearch_top_n`、`samples`、`top_k_bayes`、`top_k_screen`、`n_structures`、`relax_timeout_sec` 和 `skip_doc_update`。

## 纯 BO 工作流

入口命令：

```bash
python main_bo_only.py
```

默认输出根目录：

- `bo/data`
- `bo/models/GPR`
- `bo/results`

常用命令：

```bash
python main_bo_only.py --max-iterations 10
python main_bo_only.py --start-iteration 3 --max-iterations 10
python main_bo_only.py --samples 150 --top-k-bayes 30 --top-k-screen 10 --n-structures 5
python main_bo_only.py --init-data data/processed_data.csv
python main_bo_only.py --allow-partial-structure
python main_bo_only.py --reset
```

纯 BO 流程会依次执行代理模型训练、BO 候选生成、候选筛选、结构生成、下游计算、结果合并、成功材料提取，以及下一轮数据集更新。

## 配置说明

`config/config.yaml` 包含主要算法和工具配置：

- 循环控制和默认迭代提示。
- 贝叶斯优化采集函数与采样配置。
- 允许元素集合和硬性组成约束。
- 热导率与稳定性阈值。
- CrystaLLM、kappa、MatterSim 工具参数。
- `llm/` 与 `bo/` 输出目录布局说明。

重点字段：

- `bayesian_optimization.acquisition.xi`
- `bayesian_optimization.sampling.n_samples`
- `bayesian_optimization.sampling.allowed_elements`
- `bayesian_optimization.sampling.hard_constraints`
- `thresholds.thermal_conductivity`
- `tools.crystallm.model_path`
- `tools.ai4kappa.k_threshold`
- `tools.mattersim.imaginary_freq_threshold`

## 数据与输出目录

初始数据和文档：

```text
data/processed_data.csv
doc/Theoretical_principle_document.md
```

LLM 工作流输出：

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

纯 BO 工作流输出：

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

进度记录：

- `progress.json` 用于断点续跑，避免重复执行已完成步骤。
- 常见步骤键包括 `train_model`、`bayesian_optimization`、`ai_evaluation`、`structure_calculation`、`merge_results`、`success_extraction`、`document_update` 和 `data_update`。

## 本地工具集成

### CrystaLLM

CrystaLLM 本地集成位于 `src/tools/crystallm/`，包括结构生成工具、模型配置、tokenizer/model 代码和辅助脚本。

配置中的模型路径为：

```text
src/tools/crystallm/pre-trained-model/crystallm_v1_small
```

大型预训练模型文件不进入版本控制。运行 CrystaLLM 结构生成前，需要把所需模型文件放到该目录。

### kappa / CGCNN

kappa 集成位于 `src/tools/kappa_lib/`，包括 CGCNN 代码和 `src/tools/kappa_lib/model/` 下的预训练热学性质模型。

`src/tools/kappa_lib/root_dir*/` 是预测临时目录，已被忽略，不应提交。

### MatterSim

MatterSim 包装代码位于 `src/tools/mattersim_wrapper.py`。完整 MatterSim 流程通常需要 GPU 环境和匹配的 PyTorch 安装。

## 开发与验证

常用检查命令：

```bash
python -m compileall main.py main_bo_only.py src
python -m pytest
```

`requirements.txt` 包含部分测试、开发和分析依赖。某些测试或完整工作流可能需要外部 API key、预训练模型权重、GPU 支持或已生成的运行数据。

默认忽略的内容：

- `.env`
- `.venv/`
- `__pycache__/`、`.pytest_cache/`、`htmlcov/`
- `llm/`、`llm_*/`
- `bo/`、`bo_*/`
- `Paper/`、`archive/`、`figures/`
- `featureEngeering/`
- CrystaLLM 生成结构
- kappa 临时预测目录

## 许可

本仓库中的原创源代码采用 MIT License 授权，详见 [LICENSE](./LICENSE)。

本仓库层面的许可不会重新授权已打包或集成的第三方工具、预训练模型、数据集、论文稿件和运行生成结果。它们仍受各自原始许可、使用条款或数据使用限制约束，详见 [NOTICE.md](./NOTICE.md)。

使用 MatterSim 或 CrystaLLM 相关功能时，请遵守其上游许可并引用原始论文，详见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。

## 常见问题

### LLM 调用失败

检查 `.env` 是否存在，以及 `WORKFLOW_API_KEY`、`WORKFLOW_BASE_URL`、`WORKFLOW_MODEL` 是否与服务商匹配。

### 理论文档更新失败

检查 `THEORY_UPDATE_API_KEY`、`THEORY_UPDATE_BASE_URL` 和 `THEORY_UPDATE_MODEL`。也可以跳过文档更新：

```bash
python main.py --skip-doc-update
```

### Materials Project 查询失败

检查 `MP_API_KEY` 是否已设置且有效。

### CUDA 或 PyTorch 报错

请安装与本机 CUDA 驱动匹配的 PyTorch，或者在轻量流程中使用 CPU 环境。

### 找不到 CrystaLLM 模型

将 CrystaLLM 预训练模型文件放到：

```text
src/tools/crystallm/pre-trained-model/crystallm_v1_small
```

### Windows 终端输出乱码

使用 UTF-8 输出：

```bash
set PYTHONIOENCODING=utf-8
python main.py
```
