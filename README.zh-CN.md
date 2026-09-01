# 用分数阶特征进行 TDE 光变曲线分类

[English](README.md)

这是一个面向 Vera C. Rubin 天文台 LSST 模拟光变曲线的数据科学项目。项目将常规统计描述量与基于 FFT 的分数阶特征相结合，并使用 LightGBM、XGBoost 和随机森林识别潮汐瓦解事件（Tidal Disruption Event，TDE）。

## 项目概览

TDE 是恒星被大质量黑洞瓦解时产生的稀有天文暂现源。由于正类占比较低，仅依靠光度数据进行分类是一个典型的不平衡学习问题。本项目探索分数阶微分能否在常规光变统计量之外提取有价值的时间结构。

主要流程包括：

1. 加载并合并带标签的 MALLORN 光变曲线分片。
2. 进行消光校正、异常值过滤、基线扣除和均匀插值。
3. 按天体和测光波段提取常规统计特征。
4. 在频域中通过 `(jω)^α` 计算 `α ∈ {0.5, 1.0, 1.5}` 的分数阶导数。
5. 使用三种树模型比较常规特征、分数阶特征和融合特征。
6. 生成消融、波段重要性、统计检验、阈值、错误分析和 SHAP 解释结果。

## 参考结果

以下数值来自原始实验记录，不保证在不同软硬件或依赖版本下完全一致。

| 实验 | 结果 |
| --- | ---: |
| LightGBM + 融合特征 | Accuracy 94.02%，AUC 0.8850 |
| 最佳分数阶消融组合（`alpha = 0.5 + 1.0`，110 维） | AUC 0.8953 |
| 融合特征五折交叉验证 | AUC 标准差约 0.03 |
| 移除贡献最大的 `r` 波段 | AUC 下降 0.0361 |

## 仓库结构

```text
.
├── scripts/
│   ├── run_experiments.py         # 主模型对比与错误分析
│   ├── run_ablation_analysis.py   # alpha、波段、阈值和统计分析
│   ├── run_interpretability.py    # 分数阶示例、相关性和 SHAP
│   ├── 00_preprocess.py           # 独立的分步预处理工具
│   ├── 01_eda.py                  # 基于分步预处理结果的 EDA
│   └── 02_feature_extraction.py   # 实验性 GL 分数阶特征提取器
├── archive/                        # 用于追溯的早期实验脚本
├── result/                         # 生成结果；Git 仅追踪目录说明
├── DATA.md                         # 数据来源、目录和校验说明
├── CITATION.cff                    # 引用元数据
├── LICENSE                         # 当前使用条款
└── requirements.txt                # Python 依赖
```

## 环境安装

建议使用 64 位 Python 3.10 或更高版本，并从仓库根目录运行所有命令。

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS 或 Linux：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

实验会加载完整光变数据、并行训练多个模型并生成高分辨率图片，请预留足够的内存、磁盘空间和运行时间。

## 数据准备

项目使用 MALLORN Astronomical Classification Challenge 数据。本仓库不重新分发数据，请从官方来源下载，并按照 [DATA.md](DATA.md) 建立以下目录：

```text
mallorn-dataset/train_log.csv
mallorn-dataset/split_01/train_full_lightcurves.csv
...
mallorn-dataset/split_20/train_full_lightcurves.csv
```

在 PowerShell 中快速检查：

```powershell
Test-Path mallorn-dataset/train_log.csv
(Get-ChildItem mallorn-dataset/split_*/train_full_lightcurves.csv).Count
```

预期输出分别为 `True` 和 `20`。

## 运行分析

三个主要入口彼此独立，都会自行加载和预处理原始数据，因此不需要按顺序全部执行。

| 命令 | 主要输出 |
| --- | --- |
| `python scripts/run_experiments.py` | `result/eda/`、`result/viz/` 和 `result/error_analysis/` 中的模型指标与图表 |
| `python scripts/run_ablation_analysis.py` | `result/analysis/` 中的消融、波段、统计检验和阈值结果 |
| `python scripts/run_interpretability.py` | `result/analysis/` 中的分数阶、相关性和 SHAP 图表 |

解释性分析会将可复用特征保存到 `result/analysis/feature_cache.pkl`。修改数据或特征生成逻辑后，应先删除该缓存再重新运行。

### 分步工具流程

仓库还保留了一套较早的分步工具：

```bash
python scripts/00_preprocess.py
python scripts/01_eda.py
python scripts/02_feature_extraction.py
```

`02_feature_extraction.py` 使用离散 Grünwald–Letnikov（GL）分数阶导数，并采用不同的 `alpha` 集合。它是方法对照工具，不是 FFT 主流程的等价拆分，也不会被 `run_*.py` 自动调用。

## 可复现性与限制

- 训练/测试划分和模型训练使用固定随机种子 `42`，绘图抽样也使用固定种子。
- `requirements.txt` 给出兼容下限，而非完全锁定的环境。正式实验建议同时记录 `python -m pip freeze`、操作系统和 Python 版本。
- 带标签训练集包含 3,043 个天体，其中 148 个为 TDE，类别明显不平衡。因此应结合 AUC 和混淆矩阵解读 Accuracy。
- 脚本使用类别权重、固定的分层留出划分和五折交叉验证。
- 当前属于探索性数据科学代码，尚无命令行配置层、自动数据下载器、单元测试或持续集成流程。
- 相对路径以当前工作目录为基准，请从仓库根目录运行脚本。

## 常见问题

- **`FileNotFoundError: mallorn-dataset/...`**：检查 [DATA.md](DATA.md) 中的目录结构，并确认命令从仓库根目录启动。
- **`No objects to concatenate`**：脚本没有找到 20 个 `split_*/train_full_lightcurves.csv` 文件。
- **LightGBM 或 XGBoost 安装失败**：升级 `pip`，确认使用受支持的 64 位 Python，并在干净的虚拟环境中重试。
- **辅助 EDA 图提示缺少中文字体**：`01_eda.py` 使用 SimHei 字体，仅影响图中文字，不影响模型结果。
- **解释性分析结果未更新**：删除 `result/analysis/feature_cache.pkl` 后重新运行 `run_interpretability.py`。

## 引用

如果本仓库对你的工作有帮助，请使用 GitHub 的 **Cite this repository** 功能，并引用 MALLORN 数据论文。仓库引用元数据见 [CITATION.cff](CITATION.cff)，数据引用见 [DATA.md](DATA.md)。

## 许可

本仓库目前仅供查看、学术交流和引用，尚未授予开源许可。在复制、修改、重新分发代码或基于本项目提交贡献前，请先阅读 [LICENSE](LICENSE)。
