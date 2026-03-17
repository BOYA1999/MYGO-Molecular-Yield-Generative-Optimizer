# Molecular Yield Generative Optimizer (MYGO)（SBDD 分子生成）+ ADMET 预测评估（本仓库整理版）

本仓库包含两条相对独立的链路：

- **分子生成（SBDD 条件生成）**：以蛋白口袋为条件，使用扩散/去噪模型在 3D 空间生成配体结构（输出 SDF/SMILES）。
- **ADMET 预测评估**：对生成的分子做 ADMET 性质预测，并生成分段报告（Markdown/JSON/CSV/HTML）。

> 重要说明：本仓库目前 **未内置** 训练数据、权重 checkpoint、以及完整的训练/采样配置样例（仓库提供最小 YAML 结构样例，但需要你按本地实际路径与数据进行填写）。因此“生成链路”的可复现性取决于你是否具备对应数据与 checkpoint。

---

## 目录结构（核心）

- `mygo/scripts/`
  - `train.py`：训练入口（不依赖 PyTorch Lightning 的训练器）
  - `sample.py`：采样/生成入口（Molecular Yield Generative Optimizer (MYGO) Molecular Generation）
  - `sample_llm.py`：**实验性**（experimental）LLM 引导采样脚本
- `mygo/models/`：扩散/去噪网络、采样循环等核心实现（`maskfill.py`、`diffusion.py`、`sample.py` 等）
- `mygo/process/`：数据处理（口袋提取、LMDB 构建等；部分数据库支持已移除）
- `mygo/utils/`：数据结构、变换、噪声调度、重建、训练工具等
- `预测评估/`：ADMET 预测评估模块（相对独立，可单独使用）

---

## 快速开始：ADMET 预测评估（推荐先跑通）

### 安装依赖

此模块的依赖文件在 `预测评估/requirements.txt`：

```bash
cd 预测评估
pip install -r requirements.txt
```

> RDKit 在 Windows 上通常建议使用 conda 安装（见下方“环境与依赖”）。

### 命令行评估

```bash
python evaluate_molecules.py -s "CCO" -o ./reports --name ethanol --no-ml
```

- `--no-ml`：不依赖预训练模型文件，走规则法回退。
- 若你有模型文件（如 `metabolism_model.pkl` 等），可通过 `--model-dir` 指定目录。

---

## 分子生成：采样（需要 YAML + checkpoint + 输入蛋白/参考配体）

### 关键输入

`mygo/scripts/sample.py` 需要：

- `--config`：统一采样配置（YAML）
- `config.model.checkpoint`：训练得到的 checkpoint 路径
- `config.data.protein_path`：蛋白结构文件路径（PDB）
- `config.data.reference_ligand` 或 `pocket_args.pocket_coord`：用于定义口袋（参考配体 SDF/PDB 或口袋中心坐标）

### 最小采样命令

```bash
python mygo/scripts/sample.py --config mygo/configs/sample_min.yaml --outdir ./outputs_use
```

你需要编辑 `mygo/configs/sample_min.yaml` 中的：

- `model.checkpoint`
- `data.protein_path`
- `data.reference_ligand`（或改用 `pocket_args.pocket_coord`）

---

## 分子生成：训练（需要数据集准备）

训练入口为 `mygo/scripts/train.py`，依赖 `ForeverTaskDataset`/LMDB 数据结构与配置。

```bash
python mygo/scripts/train.py --config mygo/configs/train_min.yaml --outdir ./outputs_train
```

> `mygo/configs/train_min.yaml` 是“字段结构样例”，你需要用自己实际的数据集路径/任务权重等替换其中占位内容。

---

## 生成 → 评估 如何串联

1. 用 `mygo/scripts/sample.py` 生成分子，输出目录中会有：
   - `*_SDF/` 或 `SDF/`：生成的 `.sdf` 文件
   - `gen_info.csv`：每个分子的 SMILES、标签等信息（视脚本版本而定）
2. 将生成的 `.sdf` 作为评估输入：

```bash
python 预测评估/evaluate_molecules.py -f path/to/generated.sdf -o ./admet_reports --no-ml
```

---

## Windows 兼容性与 OpenBabel 依赖策略（已做的修复）

### SIGALRM 超时（Windows 不支持）

- `utils/misc.py` 中的 `time_limit()` 在 Windows 上不再依赖 `signal.SIGALRM`。
- 对需要硬超时的重建操作，改用 `run_with_timeout()`（线程超时）来保证 Windows 可运行。

### OpenBabel

- `utils/reconstruct.py` 已改为 **优先使用 RDKit** 将分子转为 PDB（`Chem.MolToPDBBlock`），避免 OpenBabel 作为硬依赖。
- 仍有部分“补键/连通性修复”等工具函数可能依赖 OpenBabel（安装见下方）。

---

## 实验性功能（experimental）

- `mygo/scripts/sample_llm.py` / `mygo/models/llm_guided_maskfill.py` / `mygo/llm_agents/*` are experimental:
  - Demonstrates an interface for optional model-assisted guidance during sampling and post-hoc evaluation
  - May require adaptation to match your training config and decoding/reconstruction pipeline

---

## 环境与依赖（根目录）

本仓库根目录提供 `requirements.txt`（偏“开发依赖清单”）；**强烈建议你用 conda/mamba 管理 RDKit**：

- Windows + RDKit：`conda install -c conda-forge rdkit`
- PyTorch/torch-geometric：建议按官方说明选择与你 CUDA/PyTorch 版本匹配的 wheel

---

## Smoke tests（最小冒烟测试）

在 `tests/` 下提供了最小 smoke tests，用于快速验证：

- Python 模块能否导入
- `预测评估` 的 CLI 是否能跑通一个最小示例（无模型、规则法）

运行方式（推荐，避免 unittest 扫描整个仓库导致导入可选依赖失败）：

```bash
python run_smoke_tests.py
通过网盘分享的文件：mygo.ckpt
链接: https://pan.baidu.com/s/1Mc1wLwMtKCXkmubqsNQ1sA?pwd=eu5g 提取码: eu5g 
--来自百度网盘超级会员v4的分享 this is checkpoint


