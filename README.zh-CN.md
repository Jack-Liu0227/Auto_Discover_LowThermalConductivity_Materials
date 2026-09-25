# ADLM：低热导率材料自动发现

[English](README.md) | 简体中文

ADLM 用“贝叶斯优化（BO）+ 大模型（LLM）”在 A-B-Ch 三元组分空间中迭代搜索低晶格热导率
材料：BO 在约束空间采样候选化学式，LLM 负责候选生成与筛选并把经验写回理论文档，CrystaLLM
生成晶体结构，MatterSim 做弛豫与声子稳定性判定，AI4Kappa（CGCNN）估计弹性模量并给出 κ_L。

仓库提供两个入口：

| 入口 | 用途 | 结果目录 |
|---|---|---|
| `main.py` | BO + LLM 筛选工作流（含 LLM 候选生成开关） | `llm_gen_diverse/`、`llm_gen_legacy/`、`llm_nogen_diverse/`、`llm_nogen_legacy/` |
| `main_bo_only.py` | 纯 BO 基线（不调用 LLM） | `bo/` |

所有命令都从仓库根目录执行。Windows 示例用 PowerShell（续行符为反引号），
Linux/macOS 用 bash（续行符为 `\`）。

---

## 1. 环境准备

### 1.1 硬件与系统要求

- **GPU**：至少 1 张 NVIDIA CUDA GPU。参考机器为单张 RTX 4090 D（24 GB）。显存建议 ≥ 16 GB。
- **多卡**：可选。`--num-gpus N` 会与 PyTorch 实际可见设备数校验，并映射为 `cuda:0..cuda:N-1`。
- **网络**：首次运行需要访问 Zenodo（下载 CrystaLLM 权重）、GitHub raw（下载 MatterSim 权重）、
  以及你自己配置的 LLM 服务端和 Materials Project / AFLOW 数据库。

### 1.2 创建 Python 环境

推荐 Python 3.10 + CUDA 12.1：

```powershell
conda create -n kappap python=3.10.18 -y
conda activate kappap
$PY = 'python'
& $PY -m pip install --upgrade pip
& $PY -m pip install -r requirements.txt
& $PY -m pip check
```

说明：

- `requirements.txt` 是完整 `pip freeze` 冻结清单，已内置
  `--extra-index-url https://download.pytorch.org/whl/cu121`，会安装
  `torch==2.5.1+cu121`、`mattersim==1.2.0`、`phonopy`、`pymatgen`、`agno==2.5.6` 等全部依赖。
- 仓库内置的 CrystaLLM（`src/tools/crystallm`）是源码形式，**不需要**额外 `pip install`，
  运行时通过 `sys.path` 注入 `src/tools`。
- 参考环境：Python 3.10.18 / PyTorch 2.5.1+cu121 / CUDA runtime 12.1。

### 1.3 下载并放置预训练模型（必需）

仓库**已随源码提供 AI4Kappa 的弹性模型权重**（见 1.3.3，体积很小），但不包含 CrystaLLM 与
MatterSim 的权重（`*.pt` / `*.pth` 已被 `.gitignore` 忽略），这两组需要自行获取。

#### 1.3.1 CrystaLLM（结构生成）— 必需

上游项目：[lantunes/CrystaLLM](https://github.com/lantunes/CrystaLLM)；权重托管在 Zenodo 记录
[10642388](https://zenodo.org/records/10642388)。

仓库已内置官方下载脚本（固定指向该 Zenodo 记录的 `files` 目录），直接执行：

```powershell
conda activate kappap
cd <仓库根目录>
python .\src\tools\crystallm\bin\download.py crystallm_v1_small.tar.gz -o .\src\tools\crystallm\pre-trained-model
cd .\src\tools\crystallm\pre-trained-model
tar -xvf crystallm_v1_small.tar.gz
```

解包后必须形成：

```text
src/tools/crystallm/pre-trained-model/crystallm_v1_small/ckpt.pt     ≈ 311 MB (296 MiB)
```

该目录下还可放置 `crystallm_v1_small.tar.gz`、`crystallm_v1_large.tar.gz`（约 2.2 GB）、
`crystallm_v1_minus_mpts_52_small.tar.gz` 等。当前代码**只读取 `crystallm_v1_small`**
（见 `src/tools/crystallm/generator.py` 与 `config/config.yaml` 的 `tools.crystallm.model_path`），
`crystallm_v1_large/` 放在那里也不会被使用。

#### 1.3.2 MatterSim（弛豫 + 声子）— 必需

`mattersim==1.2.0` 会在第一次实例化 `MatterSimCalculator` 时**自动下载**权重，默认缓存目录为
`~/.local/mattersim/pretrained_models/`（Windows 即
`C:\Users\<你>\.local\mattersim\pretrained_models\`）。

- 默认 checkpoint：`MatterSim-v1.0.0-1M.pth`（较小、较快）。
- 当前代码没有暴露 MatterSim 的 `load_path`，因此**实际使用默认的 1M 版本**。

无外网或需要预热的机器，可提前手动下载到该缓存目录：

```powershell
conda activate kappap
& $PY -c "from mattersim.utils.download_utils import download_checkpoint; download_checkpoint('MatterSim-v1.0.0-1M.pth')"
```

若该函数路径与你的 `mattersim` 版本不一致，可直接从 GitHub 下载并放入缓存目录：

```text
https://raw.githubusercontent.com/microsoft/mattersim/main/pretrained_models/MatterSim-v1.0.0-1M.pth
  -> ~/.local/mattersim/pretrained_models/MatterSim-v1.0.0-1M.pth
```

#### 1.3.3 AI4Kappa CGCNN 弹性模型（κ_L 估计）— 必需

`src/tools/kappa_lib/` 是从 [Jack-Liu0227/AI4Kappa](https://github.com/Jack-Liu0227/AI4Kappa)
适配而来的本地模块，用于从 CIF 预测**体模量**与**剪切模量**，再由 Slack/PINK 关系给出 κ_L。
它需要 3 个资源文件：

```text
src/tools/kappa_lib/model/Bulk modulus (GPa)-pre-trained.pth.tar     ≈ 660 KB
src/tools/kappa_lib/model/Shear modulus (GPa)-pre-trained.pth.tar    ≈ 660 KB
src/tools/kappa_lib/atom_init.json                                   ≈ 28 KB
```

**这 3 个文件已经提交在仓库里**（`git ls-files src/tools/kappa_lib` 可见），如果它们存在就不需要任何下载；
只有在缺失（例如被手工清理）时，才从上游恢复：

```powershell
conda activate kappap
cd <仓库根目录>
git clone --depth=1 https://github.com/Jack-Liu0227/AI4Kappa.git tmp/AI4Kappa
Copy-Item "tmp\AI4Kappa\model\Bulk modulus (GPa)-pre-trained.pth.tar"  src\tools\kappa_lib\model\
Copy-Item "tmp\AI4Kappa\model\Shear modulus (GPa)-pre-trained.pth.tar" src\tools\kappa_lib\model\
Copy-Item "tmp\AI4Kappa\root_dir\atom_init.json"                       src\tools\kappa_lib\
```

**文件名必须保持原样。** 代码按 `model/*-pre-trained.pth.tar` 通配匹配，并把文件名去掉
`-pre-trained.pth.tar` 后的前缀当作结果列名（`Bulk modulus (GPa)` / `Shear modulus (GPa)`），
下游公式按列名取值，改名会导致 κ_L 计算出错。

#### 1.3.4 模型自检

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

MatterSim 权重在第一次 GPU 计算时才落到缓存目录，这里不检查。

### 1.4 准备初始数据与理论文档

```text
data/processed_data.csv                        # 初始数据集（迭代 0）
doc/Theoretical_principle_document.md          # 初始理论原理文档
```

- `data/processed_data.csv`：必须含 `Formula` 列（如 `(Ag0.05Sb0.05Sn0.9)(S0.05Se0.05Te0.9)`）、
  每个元素的原子分数列（`Ag, As, Bi, Cu, Ge, In, Pb, S, Sb, Se, Sn, Te, Ti, V`），
  以及目标列 `k(W/Km)`（晶格热导率，单位 W/(m·K)）。首次运行会把它 bootstrap 成
  `<run_mode>/data/iteration_0/data.csv`，后续轮次在此基础上追加新数据。
- `doc/Theoretical_principle_document.md`：初始理论文档，会被 bootstrap 成
  `<run_mode>/doc/v0.0.0/`，之后每轮由 LLM 更新为一个新版本（`v0.0.1`、`v0.0.2` …，即 `v0.0.<迭代号>`）。
- 若文档中包含 `**Composition prior**` / `**Success threshold**` 之类的机器可读标记，
  程序会校验文档与 `config/config.yaml` 的一致性并在不一致时报错；当前仓库文档未包含这些标记。

### 1.5 配置 API 与密钥（`.env`）

```powershell
cd <仓库根目录>
Copy-Item .env.example .env
notepad .env
```

`main.py` / `main_bo_only.py` 启动时会自动加载仓库根目录的 `.env`。`.env` 已被 `.gitignore`
忽略，**不要提交任何密钥**。

| 变量 | 默认 | 作用 |
|---|---|---|
| `WORKFLOW_MODEL` | `deepseek-chat` | 主工作流 LLM 模型名（候选生成、候选筛选、材料评测） |
| `WORKFLOW_API_KEY` | 空 | 主工作流 LLM 的 API Key |
| `WORKFLOW_BASE_URL` | 空（用 SDK 默认） | 主工作流 LLM 的 OpenAI 兼容 endpoint |
| `THEORY_UPDATE_MODEL` | 回退到 `WORKFLOW_MODEL` | 理论文档更新（`update_document`）用的模型 |
| `THEORY_UPDATE_API_KEY` | 回退到 `WORKFLOW_API_KEY` | 理论文档更新专用 Key |
| `THEORY_UPDATE_BASE_URL` | 回退到 `WORKFLOW_BASE_URL` | 理论文档更新专用 endpoint |
| `THEORY_UPDATE_MAX_TOKENS` | `32000` | 理论文档更新的最大输出 token |
| `THEORY_UPDATE_PROXY` | 空 | 仅给理论文档更新请求使用的代理 |
| `TEMPERATURE` | `0.3` | 所有 LLM 调用的采样温度 |
| `LLM_NUM_RETRIES` | `2` | LLM 请求失败重试次数 |
| `LLM_REQUEST_TIMEOUT_SEC` | `120` | 单次 LLM 请求超时（秒） |
| `MP_API_KEY` | 空 | Materials Project API Key（最终材料的新颖性/去重查询） |
| `AFLOW_BASE_URL` | AFLOW 官方地址 | AFLOW 查询 endpoint（新颖性检查的第二数据源） |
| `PHONON_PARALLEL_WORKERS` | `4` | 声子并行进程上限（缺省回退到配置值） |
| `PHONON_GPUS` | 空 | 覆盖声子计算使用的 GPU 列表，如 `0,1` |
| `PYTORCH_CUDA_ALLOC_CONF` | 非 Windows 自动设为 `expandable_segments:True,max_split_size_mb:128` | CUDA 显存分配器调优；Windows 上会被自动移除（不支持） |

最小可用配置示例（自有 OpenAI 兼容服务）：

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

### 1.6 安装自检

```powershell
conda activate kappap
$PY = 'python'
& $PY -c "import agno, litellm, ase, phonopy, pymatgen, mattersim, torch; print('imports ok', torch.cuda.is_available(), torch.cuda.device_count())"
& $PY main.py --help
& $PY main_bo_only.py --help
```

`torch.cuda.is_available()` 必须为 `True`，否则请检查驱动与 `torch` 的 CUDA 版本匹配。

### 1.7 最小冒烟运行（可选，强烈建议）

先用“最便宜”的组合跑 1 轮，确认整条链路（BO → 结构生成 → 弛豫/声子 → 弹性 → κ_L →
成功提取 → 结果合并）通畅，再启动正式实验。两个开关都关会把结果写入 `llm_nogen_legacy/`，
不会污染正式实验目录：

```powershell
conda activate kappap
$PY = 'python'
& $PY main.py --runtime workflow --max-iterations 1 --samples 20 `
  --n-structures 1 --top-k-bayes 3 --top-k-screen 1 `
  --no-llm-formula-generation --no-chemistry-diversity
```

跑完后检查：

```text
llm_nogen_legacy/data/iteration_1/            # 数据已更新
llm_nogen_legacy/results/iteration_1/         # 结构/弛豫/声子产物
llm_nogen_legacy/models/GPR/iteration_0/      # 迭代 1 训练出的 GPR 模型
llm_nogen_legacy/results/progress.json        # 迭代进度
llm_nogen_legacy/results/run_<时间戳>.log      # 运行日志
```

> ⚠️ **注意**：`--max-iterations` 是**总迭代目标**（不是“再跑 N 轮”）。不传任何迭代参数时，
> `main.py` 的默认目标来自参数表 `config/agentos_params.csv` 的
> `agentos_default_iterations`（当前为 **50**）；`main_bo_only.py` 的默认是 **20**。
> 因此正式实验请显式传 `--max-iterations`。

---

## 2. 运行模式：两个开关与四个结果目录

`main.py` 提供两个**互相独立**的三态开关，用来控制“LLM 是否生成候选”与“筛选策略”。
三态含义：不传 = 沿用配置层默认值（`config/config.yaml` + 参数表）；显式传参则覆盖。

| 开关 | 作用 |
|---|---|
| `--llm-formula-generation` / `--no-llm-formula-generation` | 是否让 LLM 从第 N 轮起提出新的候选化学式（默认开启） |
| `--chemistry-diversity` / `--no-chemistry-diversity` | 筛选策略：开启 = chemistry-diverse 重排；关闭 = 原始“按 BO 预测值与不确定性重排” |

两个开关的组合会推导出不同的 `run_mode`，并写入**完全独立的结果根目录**，因此四种模式
可以并行/先后运行、互不读写：

| 命令 | LLM 生成候选 | chemistry diversity | `run_mode` / 结果目录 | 含义 |
|---|---|---|---|---|
| 默认（不带开关） | 开 | 开 | `llm_gen_diverse/` | 当前完整流程 |
| `--no-chemistry-diversity` | 开 | 关 | `llm_gen_legacy/` | LLM 候选 + 原始指标筛选 |
| `--no-llm-formula-generation` | 关 | 开 | `llm_nogen_diverse/` | 纯 BO 候选 + chemistry-diverse 筛选 |
| 两个都关 | 关 | 关 | `llm_nogen_legacy/` | 完全复现 LLM 扩展改造前的原始流程 |

要点：

- `run_mode` 由**有效开关值**推导（含参数表中的覆盖值），所以“行为不同的运行”永远不会
  共用同一个结果目录。
- 四个目录各自独立 bootstrap：首次运行会从 `data/processed_data.csv` 与
  `doc/Theoretical_principle_document.md` 生成 `<run_mode>/data/iteration_0/` 与
  `<run_mode>/doc/v0.0.0/`。它们之间**不共享**任何状态。
- `--reset` 与 `--rebuild-from` 只影响本次开关组合对应的那个目录（见 3.5、5.3）。
- 直接运行 `python main.py` 时使用的 `run_mode` 会打印在启动摘要的 `run mode:` 行，
  同时打印 `llm formula generation:` 与 `screening strategy:`，请以日志为准。

**选择建议**

- 复现/对照实验：同时跑 `llm_gen_diverse`、`llm_gen_legacy`、`llm_nogen_diverse`、
  `llm_nogen_legacy` 做消融，比较 `selection_trace.csv` 与最终材料集合的差异。
- 只用 BO 基线（完全不需要 LLM 服务）：用 `main_bo_only.py`，结果在 `bo/`。

---

## 3. 运行完整流程

所有命令都在仓库根目录、`kappap` 环境下执行。以下假设 `$PY = 'python'`。

### 3.1 正式实验（BO + LLM，20 轮）

```powershell
conda activate kappap
cd <仓库根目录>
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

启动前会打印一段摘要（日志文件、参数表生效项、`effective_target_iterations`、
`execution_range=[1 .. 20]`、`run mode`、`screening strategy` 等）。执行过程中每个迭代会依次：

1. **BO 采样与排序**：在 A-B-Ch 约束空间采样 `samples` 个化学式，GPR 预测 κ_L 与不确定性，
   按 EI 排序取前 `top_k_bayes` 个（多参考轮次会合并历史数据）。
2. **候选池构建**（第 2 轮起）：`llm_formula_generation` 开启时，LLM 基于筛选反馈与父代候选
   提出新化学式并打分，与 BO 候选统一重排；关闭时跳过，只写一个空的
   `llm_formula_proposals.json` 以保持续跑契约。
3. **LLM 筛选**：从合并池中挑出 `top_k_screen` 个材料；开启 chemistry diversity 时使用
   chemistry-diverse 重排（并写入 `candidate_scores`），关闭时使用原始“按预测值与不确定性
   重排”（写入 `selected_materials`，评测 prompt 恢复 BO 指标证据）。
4. **结构生成**：CrystaLLM 为每个材料生成 `n_structures` 个结构（多卡按 GPU 分 lane）。
5. **弛豫 + 声子**：MatterSim 弛豫并按 `--phonon-imag-tol` 判定动力学稳定性。
6. **κ_L 计算**：AI4Kappa CGCNN 预测弹性模量 → Slack/PINK 公式给出 κ_L。
7. **成功提取**：筛出“稳定 + κ_L 达标”的材料，写回数据集。
8. **理论文档更新**：LLM 把本轮经验写回文档，生成新版本 `doc/v0.0.<迭代号>/`。

### 3.2 中断后继续 / 追加轮数

```powershell
# 场景 A：上次跑到第 7 轮中断，现在继续跑到 20 轮
& $PY main.py --runtime workflow --max-iterations 20

# 场景 B：已完成 20 轮，在此基础上再补 10 轮（目标变成 30）
& $PY main.py --runtime workflow --add-iterations 10
```

- 续跑会读取 `progress.json` 与磁盘产物做一致性校验；缺失前序产物时会拒绝续跑并提示
  使用 `--reset` 或 `--rebuild-from`。
- **必须使用与上次完全相同的开关组合**，否则会落到另一个目录，看起来像“从第 1 轮重新开始”。

### 3.3 纯 BO 基线

```powershell
& $PY main_bo_only.py --max-iterations 20 --samples 100 --n-structures 5 `
  --top-k-bayes 20 --top-k-screen 10 --num-gpus 1 --seed 42
```

不依赖任何 LLM/API Key，结果写入 `bo/`。跑法、续跑、`--reset`、`--rebuild-from` 语义与
`main.py` 一致。

### 3.4 消融对照（四个 run_mode）

```powershell
# 1) LLM 生成 + chemistry-diverse 筛选（默认）
& $PY main.py --runtime workflow --max-iterations 20

# 2) LLM 生成 + 原始指标筛选
& $PY main.py --runtime workflow --max-iterations 20 --no-chemistry-diversity

# 3) 纯 BO 候选 + chemistry-diverse 筛选
& $PY main.py --runtime workflow --max-iterations 20 --no-llm-formula-generation

# 4) 完全复现改造前流程（纯 BO 候选 + 原始指标筛选）
& $PY main.py --runtime workflow --max-iterations 20 `
  --no-llm-formula-generation --no-chemistry-diversity
```

四组命令产出四个目录：`llm_gen_diverse/`、`llm_gen_legacy/`、`llm_nogen_diverse/`、
`llm_nogen_legacy/`。对照时关注每轮 `results/iteration_N/selected_results/selection_trace.csv`
的 `screening_mode`（`chemistry_diverse_rerank` 或 `llm_full_rerank`）与
`llm_formula_proposals.json` 中 `proposals` 的数量。

### 3.5 重建失败的迭代后缀

当某轮的结构生成/弛豫/声子大规模失败，希望在保留前面轮次的前提下重跑：

```powershell
# 保留第 1..2 轮，归档第 3 轮起的产物，并从第 3 轮重新开始
& $PY main.py --runtime workflow --rebuild-from 3 --max-iterations 20
```

- 归档目录：`<run_mode>_rebuild_<时间戳>/`，内部有 `rebuild_manifest.json` 记录被归档的内容。
- 只影响当前 `run_mode` 的目录，其他组合目录不受影响。
- 要求第 `1..N-1` 轮已全部完成，否则会直接报错（避免把不完整的前缀当成可信基线）。
- 允许部分结构失败并继续（默认整批失败才中止）：

```powershell
& $PY main.py --runtime workflow --max-iterations 20 --allow-partial-structure
```

### 3.6 并行与多 GPU

```powershell
# 使用 2 张 GPU；结构生成按 GPU 分 lane
& $PY main.py --runtime workflow --max-iterations 20 --num-gpus 2

# 仅用第 0、1 号卡跑声子（环境变量优先）
$env:PHONON_GPUS = '0,1'
$env:PHONON_PARALLEL_WORKERS = '4'
& $PY main.py --runtime workflow --max-iterations 20 --num-gpus 2
```

- `--num-gpus N` 会与 `torch.cuda.device_count()` 校验；`N` 大于可见设备数会直接退出。
- 多卡时每张卡的结构生成并发为 `max_workers`（配置项，默认 4）；单卡时该值为 1。
- 想跑多个 run_mode 组合并行的机器，建议分开进程、各自限制 `--num-gpus`，
  或把 `CUDA_VISIBLE_DEVICES` 分别指到不同卡；因为各 run_mode 目录独立，不会互相覆盖文件。

各阶段的并行度含义：

| 阶段 | 并发由谁控制 |
|---|---|
| 结构生成 | 每张 GPU 的 lane 数 = `max_workers`（配置项，默认 4；单卡时为 1） |
| 弛豫 + 声子 | 每张 GPU 的 lane 数 = `relax_workers`（默认 1，限制 MatterSim 显存压力） |
| κ_L 计算 | 每张 GPU 同一时刻 1 个任务，任务按轮转分配到各卡 |
| 后处理（去重/提取/合并） | `--postprocess-workers`（默认 2），按 formula/材料目录并行 |
| 最终数据库新颖性查询 | `--novelty-workers`（默认 4），按材料并行；同一材料内部的多个数据库调用在本模式下串行，避免嵌套 API 过载 |

调试或外部服务限流严格时，把 `postprocess_workers` / `novelty_workers` 都设为 1 即为全串行。

### 3.7 AgentOS 模式（可选）

```powershell
& $PY main.py --runtime agentos --agentos-host 127.0.0.1 --agentos-port 7777
```

启动后打印 `agentos endpoint: http://127.0.0.1:7777` 与 `workflow id: aslk-agentos-workflow`。
该模式下可以在表单里传入 `max_iterations`、`samples`、`n_structures`、`top_k_bayes`、
`top_k_screen`、`websearch_enabled`、`websearch_top_n` 覆盖本次运行，其余参数沿用启动命令的
CLI/配置值（包括第 2 节的两个开关与由此决定的 `run_mode`）。

### 3.8 重新开始

```powershell
# 归档整个当前 run_mode 目录为 <run_mode>_old_<时间戳>/，然后从第 0 轮重建
& $PY main.py --runtime workflow --reset --max-iterations 20
```

`--reset` 只归档“本次开关组合”对应的目录；例如 `python main.py --reset
--no-llm-formula-generation` 只影响 `llm_nogen_diverse/`（以及可能存在的同名历史归档）。

---

## 4. 参数手册（每个参数的作用）

参数优先级（从高到低）：

1. **显式传入的 CLI 参数**（本节表格）
2. **参数表** `config/agentos_params.csv` 中 `enabled=1` 的行（见 4.5）
3. **代码默认值**（`main.py` / `main_bo_only.py` 的 `DEFAULT_CONFIG`）
4. **`config/config.yaml`**（BO/采样/阈值/工具配置，见 4.6）

> 注意：`main.py` 里像 `--top-k-bayes`（help 显示默认 20）、`--seed`（42）这类参数，
> 只有在命令行**显式出现**时才会覆盖参数表；否则以参数表的值为准。

### 4.1 两个入口通用参数（个别仅 `main.py` 提供，已标注）

| 参数 | 默认 | 作用 |
|---|---|---|
| `--max-iterations N` | `main.py`: 见下文；`main_bo_only.py`: **20** | **总迭代目标数**（不是“再跑 N 轮”）。已完成 `k` 轮时从第 `k+1` 轮跑到第 `N` 轮。`main.py` 未显式传参时取参数表 `agentos_default_iterations`（当前 **50**） |
| `--add-iterations N`（仅 `main.py`） | 无 | 在**当前已完成轮数**基础上再追加 N 轮（等效于 `--max-iterations = 已完成 + N`）。与 `--max-iterations` 二选一 |
| `--samples N` | 100 | 每轮 BO 在 A-B-Ch 约束空间中采样的候选化学式数量。越大探索越广、GPR 训练与 EI 排序越慢 |
| `--n-structures N` | 5 | 每个入选材料用 CrystaLLM 生成的结构个数。直接决定结构生成/弛豫/声子的工作量，是最主要的耗时来源 |
| `--top-k-bayes N` | 20 | BO（EI）排序后进入候选池的候选数量；也是 LLM 候选合并池的大小（`top_k_bayes`） |
| `--top-k-screen N` | 10 | 最终送入结构计算的**材料数量**（LLM 筛选输出，`top_k_screen`） |
| `--phonon-imag-tol V` | -0.1 | 声子最小频率阈值（THz）。最低频率低于该值判为动力学不稳定（虚频）。改这里时注意与 `config/config.yaml` 的 `thresholds.dynamic_min_frequency` 保持一致 |
| `--num-gpus N` | 无（按可见设备） | 使用 GPU 数量，映射为 `cuda:0..cuda:N-1`；与 `torch.cuda.device_count()` 校验，超出则退出 |
| `--device {cuda,cpu}` | cuda | MatterSim 结构计算设备。`cpu` 会同时把 `gpus` 置为 `["cpu"]`，仅用于调试，速度极慢 |
| `--postprocess-workers N` | 2 | 结构去重、成功提取、结果合并等 CPU 后处理的并发进程数 |
| `--novelty-workers N` | 4 | 最终材料数据库新颖性查询（MP / AFLOW）的并发数 |
| `--seed N` | 42 | 随机种子基准值。第 `i` 轮实际种子 = `seed + i * seed_stride`（`seed_stride` 默认 1000），因此每轮可复现但轮间不重复 |
| `--non-deterministic-torch` | 关闭 | 关闭 PyTorch 确定性内核（提速，牺牲严格可复现） |
| `--reset` | 关闭 | 归档当前 `run_mode` 目录为 `<run_mode>_old_<时间戳>/`，并从第 0 轮全新开始 |
| `--rebuild-from N` | 无 | 保留第 `1..N-1` 轮，把第 `N` 轮起的 `data/`、`models/`、`results/`、`doc/` 归档到 `<run_mode>_rebuild_<时间戳>/`（附 `rebuild_manifest.json`），并从第 `N` 轮继续 |
| `--allow-partial-structure` | 关闭 | 允许部分结构任务失败仍继续流程（默认要求整批结构产物可用，否则停在本轮） |
| `--init-data PATH` | `data/processed_data.csv` | 初始数据集路径，首次运行 bootstrap 到 `<run_mode>/data/iteration_0/` |
| `--init-doc PATH`（仅 `main.py`） | `doc/Theoretical_principle_document.md` | 初始理论文档路径，首次运行 bootstrap 到 `<run_mode>/doc/v0.0.0/` |

`main_bo_only.py` 额外提供：

| 参数 | 默认 | 作用 |
|---|---|---|
| `--relax-timeout-sec N` | 900 | 单个“弛豫 + 声子”任务的超时时间（秒）。超时判为任务失败；长程/大胞体系需要调大 |
| `--start-iteration N` | 无 | 显式指定从第 N 轮开始（默认自动从第一个未完成轮次续跑） |

### 4.2 `main.py` 专属参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `--runtime {workflow,agentos}` | workflow | `workflow` = 本地直接跑完整循环；`agentos` = 启动 HTTP 服务，由表单触发并可在表单里覆盖部分参数 |
| `--params-csv PATH` | `config/agentos_params.csv` | 可编辑参数表（CSV）。存在时其中 `enabled=1` 的行会覆盖代码默认值；显式 CLI 参数仍然优先 |
| `--llm-formula-generation` / `--no-llm-formula-generation` | 配置默认（开） | 是否让 LLM 从 `llm_formula_generation.start_iteration`（默认第 2 轮）起提出新的候选化学式。关闭时不做任何 proposer 调用，但仍写出空的 `llm_formula_proposals.json`（含 `parents`/`parent_stats`）以保持续跑契约。**关闭会切换到 `llm_nogen_*` 目录** |
| `--chemistry-diversity` / `--no-chemistry-diversity` | 配置默认（开） | 筛选策略。开 = chemistry-diverse 重排（候选带 `candidate_id`，评测用 `candidate_scores` 模式，只给 formula 级证据）；关 = 原始 `llm_full_rerank`（评测用 `selected_materials` 模式，恢复 BO 指标证据：`k_pred`/`mu_log`/`sigma_log`/95% CI/EI/rank）。**关闭会切换到 `llm_*_legacy` 目录** |
| `--websearch-enabled` / `--no-websearch-enabled` | 开 | 评测阶段是否用 WebSearch 证据补充材料判断。关闭可省时间与外网访问 |
| `--websearch-top-n N` | 5（参数表当前为 10） | 进入评测的候选中，前 N 个做 WebSearch 检索（其余只用本地证据） |
| `--skip-doc-update` / `--no-skip-doc-update` | 关闭 | 跳过本轮的“理论文档更新”步骤，不生成新的 `doc/v0.0.<迭代号>/` |
| `--agentos-host HOST` | 127.0.0.1 | `--runtime agentos` 时的监听地址 |
| `--agentos-port PORT` | 7777 | `--runtime agentos` 时的监听端口 |

隐藏兼容参数（`argparse.SUPPRESS`，不在 `--help` 中显示，供旧脚本使用）：

| 参数 | 等价于 |
|---|---|
| `--n-top-candidates N` | `--top-k-bayes N` |
| `--n-select N` | `--top-k-screen N` |

### 4.3 只在参数表 / 配置里生效的调参项

这些没有 CLI 开关，请改 `config/agentos_params.csv`（或 `config/config.yaml`）：

| Key | 默认 | 作用 |
|---|---|---|
| `relax_timeout_sec` | 900 | 单个弛豫+声子任务超时（秒）。`main.py` 只能通过参数表设置 |
| `xi` | 0.01 | EI 采集函数的探索系数，越大越偏向高不确定性（探索） |
| `k_threshold` | 1.0 | κ_L 成功阈值（W/(m·K)）。低于该值才算"命中"低热导率材料 |
| `pressure` | 0.0 | 弛豫施加的压力（GPa） |
| `seed_stride` | 1000 | 轮间种子步长（见 `--seed`） |
| `max_workers` | 4 | 每张 GPU 的结构生成并发行数（单卡时为 1） |
| `relax_workers` | 1 | 每张 GPU 的弛豫并发数 |
| `phonon_workers` | 1 | 声子并发数（当前与弛豫合并执行，保留参数） |
| `prefer_isolated_relax_process` | true | 优先用独立子进程执行弛豫（隔离显存与崩溃） |
| `allow_in_process_relax_fallback` | true | 独立子进程不可用时允许退化为同进程执行 |
| `websearch_strategy` | hybrid | WebSearch 检索策略 |
| `websearch_queries_per_candidate` | 2 | 每个候选的检索 query 数量 |
| `websearch_theory_template` | none | 检索用理论模板（可留空） |
| `llm_formula_generation_start_iteration` | 2 | 从第几轮开始让 LLM 生成候选（第 1 轮不生成） |
| `llm_formula_proposal_count` | 20 | 每轮 LLM 提出的候选化学式数量 |
| `llm_formula_parent_count` | 10 | 供 LLM 参考的父代候选数量 |
| `llm_formula_generation_max_tokens` | 12000 | 候选生成请求的最大输出 token |
| `llm_formula_generation_max_retries` | 3 | 候选生成失败重试次数 |
| `chemistry_evaluator_evidence` | chemistry_only | chemistry-diverse 路径下评测只给 formula 级证据（关闭该路径时自动变为 `full`） |
| `max_same_element_tuple` | 2 | 同一元素集合最多允许出现 2 次（多样性约束） |
| `max_same_stoichiometry_pattern` | 4 | 同一化学计量比模式最多允许 4 次（多样性约束） |
| `high_symmetry_parent_enabled` | true | 是否把高对称性历史结构作为父代种子 |
| `high_symmetry_min_crystal_system` | orthorhombic | 高对称性父代的最低晶系 |
| `high_symmetry_parent_scope` | historical_seed_only | 高对称性父代的适用范围 |
| `agentos_default_iterations` | 50 | **`main.py` 不带任何迭代参数时的默认总迭代目标**（参数表当前为 50） |
| `agentos_max_iterations_cap` | 20 | AgentOS 表单允许的最大迭代数上限 |
| `agentos_allow_text_iteration_override` | true | 允许在 AgentOS 对话文本里覆盖迭代数 |
| `agentos_ws_ping_interval` / `agentos_ws_ping_timeout` | none | AgentOS WebSocket 心跳间隔/超时 |

### 4.4 参数表 `config/agentos_params.csv`

格式：`key,value,enabled,notes`。`enabled=1`（或 `true/yes/on`）的行才会生效；
`enabled=0` 的行仅作为 UI 预填值，不覆盖配置。

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

> 上表就是仓库当前实际内容（会随运行缓慢变化）。注意 `enabled=1` 的 `websearch_top_n=10`
> 会覆盖代码默认值 5；`seed=42` 的 `enabled=0`，所以不作为覆盖生效。
> `agentos_default_iterations=50` 是 `main.py` 不带迭代参数时的默认总迭代目标。

说明：

- 只接受**已存在于配置中的 key**，未知 key 会打印 `Unknown config key skipped` 警告并被忽略。
- 每次 `main.py` 运行结束，会把 `PARAM_MEMORY_KEYS`（`samples`、`n_structures`、
  `top_k_bayes`、`top_k_screen`、`postprocess_workers`、`novelty_workers`、
  `websearch_enabled`、`websearch_top_n`、`phonon_imag_tol`、`seed`、`relax_timeout_sec`、
  `skip_doc_update`、`agentos_default_iterations`、`agentos_ws_ping_interval`、
  `agentos_ws_ping_timeout`）的**生效值写回**该 CSV（新增行默认 `enabled=0`）。
- `llm_formula_generation_enabled` 与 `chemistry_diversity_enabled` 也可以手工加进该表
  （它们已在配置中），此时**不加 CLI 参数**也会按表里的值决定 `run_mode` 与结果目录。

### 4.5 `config/config.yaml`

被 `src/utils/config_loader.py` 读取，主要覆盖 BO 搜索空间、阈值与工具行为：

| 位置 | 关键值 | 作用 |
|---|---|---|
| `bayesian_optimization.acquisition` | `function: EI`, `xi: 0.01` | 采集函数与探索系数 |
| `bayesian_optimization.sampling` | `method: mcmc`, `n_samples: 100`, `max_atoms: 20` | 采样方式、样本数、最大原子数 |
| `sampling.allowed_elements` | 14 种元素 | 允许出现的元素集合 |
| `sampling.hard_constraints.schema` | A/B/Ch 分组 | A-B-Ch 三元约束（A: Ag/Cu/In/Sn/Pb；B: As/Sb/Ge/Bi/Ti/V；Ch: S/Se/Te） |
| `sampling.hard_constraints.stoichiometry` | A/B/Ch 各 1–10（A 为 2–10） | 各组原子数范围 |
| `sampling.mcmc` | `w_ei/w_k/w_sigma/temperature/burn_in/thin` | MCMC 采样权重与迭代参数 |
| `model.retrain_from_round` | 2 | 从第几轮起改用当前实验的 GPR 而非初始模型 |
| `thresholds.thermal_conductivity` | 1.0 | κ_L 成功阈值（与 `k_threshold` 联动） |
| `thresholds.dynamic_min_frequency` | -0.1 | 声子虚频阈值（与 `--phonon-imag-tol` 联动） |
| `tools.crystallm.model_path` | `src/tools/crystallm/pre-trained-model/crystallm_v1_small` | CrystaLLM 权重目录 |
| `tools.mattersim.phonon_supercell` | `[2, 2, 2]` | 声子超胞尺寸 |
| `tools.mattersim.displacement` | 0.01 | 有限位移法的位移量（Å） |
| `tools.mattersim.timeout` | 600 | MatterSim 单次计算超时（秒） |
| `tools.ai4kappa.temperature` | 300 | κ_L 计算温度（K） |
| `loop.*` | `max_iterations: 20` 等 | **仅信息性**，实际迭代数由 CLI / 参数表决定 |

另注：`tools.crystallm.num_samples/top_k/max_new_tokens` 目前由代码固定
（`num_samples = --n-structures`、`top_k = 10`、`max_new_tokens = 2000`），改配置不生效。

### 4.6 环境变量

LLM 与数据库相关的变量见 1.5 的表格（`.env`）。会改变**运行期行为**（而非仅访问凭据）的是：

| 变量 | 作用 |
|---|---|
| `PHONON_GPUS` | 覆盖声子计算使用的 GPU 列表（如 `0,1`）；未设置时与工作流使用同一组 GPU |
| `PHONON_PARALLEL_WORKERS` | 声子并行进程上限（缺省回退到配置值） |
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA 显存分配器调优；非 Windows 自动设置，Windows 上自动移除 |

---

## 5. 输出目录与产物

### 5.1 目录布局

以 `llm_gen_diverse`（默认组合）为例，`<run_mode>` 可为
`llm_gen_diverse` / `llm_gen_legacy` / `llm_nogen_diverse` / `llm_nogen_legacy`；
纯 BO 基线为 `bo`：

```text
<run_mode>/
├── data/
│   └── iteration_<N>/data.csv              # 迭代 N 结束时的数据集（iteration_0 = bootstrap 初始数据）
├── doc/
│   └── v0.0.<N>/Theoretical_principle_document.md   # 每轮更新后的理论文档（v0.0.0 = 初始文档）
├── models/
│   └── GPR/iteration_<N>/
│       ├── gpr_thermal_conductivity.joblib  # 在第 N+1 轮训练得到的 GPR（iteration_0 = 首轮训练的初始模型）
│       ├── gpr_scaler.joblib                # 特征缩放器
│       ├── model_metadata.json              # 训练数据 hash、样本数等元信息
│       ├── Best_Model_Prediction.png
│       └── Final_Model_Comparison.png
└── results/
    ├── progress.json                        # 逐轮、逐步骤的完成状态（续跑依据）
    ├── workflow.db                          # 运行时工作流数据库
    ├── screening_summary.csv                # 逐轮的筛选汇总指标（见 5.2）
    ├── summary/                             # 跨轮材料汇总
    │   ├── all_materials_summary.csv
    │   ├── stable_materials_summary.csv
    │   ├── success_materials_summary.csv
    │   └── final_materials_db_novelty.csv
    ├── run_<时间戳>.log                      # 每次运行的完整日志
    └── iteration_<N>/
        ├── logs/ei_acquisition.log          # BO/EI 采样与排序日志
        ├── reports/                         # LLM 交互留档（input/output 成对）
        │   ├── llm_candidate_scoring_input.md / _output.md
        │   ├── llm_formula_generation_input.md / _output.md
        │   └── llm_theory_update_input.md / _output.md
        ├── selected_results/                # 候选筛选与选择的全部中间/最终结果
        ├── processed_structures/<formula>/   # 每材料的生成结构、prompt、后处理结果
        ├── MyRelaxStructure/<formula>/       # 弛豫 + 声子产物
        │   ├── sample_<i>.cif
        │   ├── sample_<i>_phonon/
        │   ├── relax_phonon_results.csv
        │   └── thermal_conductivity.csv      # 该材料的 κ_L 计算结果
        └── success_examples/                 # 成功材料汇总
            ├── cif_files_stable/             # 动力学稳定结构
            ├── cif_files_success/            # 命中 κ_L 阈值的结构
            ├── extraction_status.json
            └── final_materials_db_novelty.csv / .json   # MP/AFLOW 新颖性核对结果
```

### 5.2 每轮关键文件（`results/iteration_<N>/selected_results/`）

| 文件 | 内容 |
|---|---|
| `raw_samples.csv` | BO 采样器原始输出的全部候选与预测（`formula`、`k_pred`、`mu_log`、`sigma_log`、`ei`、95% CI），未排序 |
| `all_samples.csv` | 同上但按 EI 排序并附 `rank`，是 EI 排序的完整视图 |
| `top20_materials.json` | BO 采集步骤输出：`top20` 列表（`rank`/`formula`/`k_pred`/`mu_log`/`sigma_log`/`ei`/`k_lower`/`k_upper`）与本次采样的元信息（`xi`、`f_min`、`n_samples` 等） |
| `llm_formula_proposals.json` | LLM 提出的候选化学式与打分。**关闭候选生成时仍然写出**，其中 `proposals` 为空数组，`parents`/`parent_stats` 保留 |
| `novel_candidates.csv` | 合并 BO 候选与 LLM 候选后按 EI 排序的完整候选池（含 `candidate_source`、`original_bo_rank`、`bo_prediction_status`） |
| `websearch_enriched_candidates.csv` | 对候选做 WebSearch 补充后的表（含 `websearch_queries`/`websearch_summary`/`websearch_sources`） |
| `merged_screening_candidates.csv` | 送入 LLM 评测的最终候选表：在上一文件基础上补齐 `candidate_id`，chemistry-diverse 路径就是它作为评测输入 |
| `selection_trace.csv` / `.json` | 逐候选的筛选轨迹，含 `screening_mode`（`chemistry_diverse_rerank` 或 `llm_full_rerank`）、淘汰原因等 |
| `ai_candidate_scores.csv` | chemistry-diverse 路径下的逐候选打分（`candidate_scores` 模式） |
| `ai_selected_materials.csv` | 最终入选材料（`top_k_screen` 个），两条路径都会写 |

`results/screening_summary.csv`（每次运行追加一行，跨轮可比较）中的关键列：

| 列 | 含义 |
|---|---|
| `iteration` / `screening_mode` | 轮次 / 本轮使用的筛选模式（`chemistry_diverse_rerank` 或 `llm_full_rerank`） |
| `selected_count` | 本轮实际送入结构计算的材料数 |
| `success_count` / `stable_count` | 命中 κ_L 阈值的材料数 / 动力学稳定材料数 |
| `success_rate_at_k` / `stable_rate_at_k` | 上述两项 / `selected_count`（消融对比的核心指标） |
| `best_kappa_in_selected_success` | 本轮成功材料中的最低 κ_L |
| `selected_from_bo_top3_count` / `selected_from_bo_13_20_count` | 入选材料来自 BO 前 3 / 第 13–20 名的数量（衡量是否利用了长尾候选） |
| `tail_promotions_count` / `tail_promotions_success_count` | 长尾候选被提升入选的数量 / 其中成功的数量 |
| `protected_top3_dropped_count` | BO 前 3 名中被淘汰的数量 |
| `candidate_count` / `bo_candidate_count` / `llm_candidate_count` | 本轮候选池规模 / 其中 BO 候选数 / 其中 LLM 生成候选数 |
| `llm_selected_count` / `llm_stable_count` / `llm_success_count` / `llm_proposal_count` | 入选材料中由 LLM 提出的数量 / 其中稳定数 / 其中成功数 / 本轮 LLM 提出的候选总数（关闭候选生成时均为 0） |
| `high_symmetry_parent_count` / `high_symmetry_parent_stats` | 高对称性父代种子数量与分布 |
| `final_p1_count` / `final_low_symmetry_count` / `final_high_symmetry_count` / `unresolved_symmetry_count` | 最终结构按对称性的分类统计（P1 / 低对称 / 高对称 / 未解析） |

### 5.3 归档目录（`--reset` / `--rebuild-from`）

| 触发 | 归档路径 | 内容 |
|---|---|---|
| `--reset` | `<run_mode>_old_<时间戳>/` | 整个 `<run_mode>/` 目录 |
| `--rebuild-from N` | `<run_mode>_rebuild_<时间戳>/` | 被失效的后缀产物：`results/iteration_{N..}`、`data/iteration_{N..}`、`models/GPR/iteration_{N-1..}`、`doc/v0.0.{N..}`，以及原 `progress.json`；并附 `rebuild_manifest.json`。要求第 `1..N-1` 轮已全部完成 |

归档目录同样被 `.gitignore` 的 `llm_*/` / `bo_*/` 规则覆盖。
归档产物**只用于事后审计**，永远不会被当作续跑的输入；程序只从 `<run_mode>/` 下的活动目录读取状态。

---

## 6. 核心规则

1. **两个入口互不干扰**：`main.py` 只操作 `llm_*` 目录，`main_bo_only.py` 只操作 `bo/`。
2. **开关决定目录**：两个开关的四种组合各自独立 bootstrap、独立续跑；续跑必须沿用同一组合。
3. **续跑先校验**：`progress.json` 与磁盘产物必须一致，缺前序产物会拒绝续跑（用 `--reset`
   或 `--rebuild-from` 修复），避免产生不可信的历史链。 续跑从**第一个未完成的步骤**继续，只有当前轮全部完成才会进入下一轮。
4. **每轮种子可复现**：`seed` 固定 + `seed_stride` 递增，同一轮重跑得到相同结果；
   加 `--non-deterministic-torch` 会放弃这项保证。
5. **阶段失败边界**：结构生成/弛豫/声子任务失败会记录到 `relax_phonon_results.csv` 并在
   提取阶段被过滤；只有整批结构产物不可用才会中断本轮（`--allow-partial-structure` 可放宽）。
6. **LLM 只做建议，判据仍是计算**：LLM 负责候选生成、筛选与文档更新；
   动力学稳定性来自声子，κ_L 来自弹性模量 + Slack/PINK，最终命中判定基于 `k_threshold`。
7. **密钥只在 `.env`**：不要把 API Key 写进代码、配置或日志。

## 7. 常见问题排查

| 现象 | 原因 / 处理 |
|---|---|
| `ModuleNotFoundError: No module named 'agno'` | 未激活正确环境或未装依赖；`conda activate kappap && pip install -r requirements.txt` |
| `CrystaLLM model not found` | 缺 `src/tools/crystallm/pre-trained-model/crystallm_v1_small/ckpt.pt`，见 1.3.1 |
| κ_L 全为 0 / 弹性模量缺失 | 缺 `src/tools/kappa_lib/model/*-pre-trained.pth.tar` 或改名，见 1.3.3 |
| MatterSim 反复下载失败 | 外网受限；按 1.3.2 预先把 `MatterSim-v1.0.0-1M.pth` 放进 `~/.local/mattersim/pretrained_models/` |
| `Requested N GPU(s) but only M visible` | `--num-gpus` 超过可见设备数；改小或设置 `CUDA_VISIBLE_DEVICES` |
| 续跑时提示前序产物缺失 | 用了不同的开关组合（落到了另一个目录），或历史产物被清理；加 `--reset`/`--rebuild-from` |
| “怎么从第 1 轮重新开始” | 加 `--reset`（会先归档，不会直接删数据） |
| 想复查某一轮的 LLM 到底说了什么 | 看 `results/iteration_<N>/reports/` 下的 `*_input.md` / `*_output.md` |
| Windows 上 CrystaLLM 子进程崩在显存分配 | `PYTORCH_CUDA_ALLOC_CONF` 中含 `expandable_segments`（Windows 不支持），程序已自动移除；自定义环境变量时不要加回 |
