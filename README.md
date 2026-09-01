# Fractional-Order Features for TDE Light-Curve Classification

[简体中文](README.zh-CN.md)

A data science project for classifying tidal disruption events (TDEs) in simulated Vera C. Rubin Observatory LSST light curves. The workflow combines conventional statistical descriptors with FFT-based fractional-order features and evaluates LightGBM, XGBoost, and Random Forest classifiers.

## Project overview

TDEs are rare astronomical transients produced when stars are disrupted by massive black holes. Their low prevalence makes photometric classification a challenging imbalanced-learning problem. This repository explores whether fractional-order differentiation can reveal useful temporal structure beyond conventional light-curve statistics.

The main workflow:

1. Loads and merges the labelled MALLORN light-curve splits.
2. Applies extinction correction, outlier filtering, baseline subtraction, and uniform interpolation.
3. Extracts conventional statistics for each object and photometric band.
4. Computes fractional derivatives in the frequency domain with `(jω)^α`, where `α ∈ {0.5, 1.0, 1.5}`.
5. Compares conventional, fractional, and fused feature sets with three tree-based classifiers.
6. Produces ablation studies, band-importance analysis, statistical tests, threshold analysis, error analysis, and SHAP explanations.

## Reported results

The following values are reference results from the original experiment. They are not guaranteed across different environments or dependency versions.

| Experiment | Result |
| --- | ---: |
| LightGBM with fused features | Accuracy 94.02%, AUC 0.8850 |
| Best fractional-order ablation (`alpha = 0.5 + 1.0`, 110 features) | AUC 0.8953 |
| Five-fold cross-validation with fused features | AUC standard deviation about 0.03 |
| Removing the most informative `r` band | AUC decrease of 0.0361 |

## Repository structure

```text
.
├── scripts/
│   ├── run_experiments.py         # Main model comparison and error analysis
│   ├── run_ablation_analysis.py   # Alpha, band, threshold, and statistical analyses
│   ├── run_interpretability.py    # Fractional examples, correlations, and SHAP
│   ├── 00_preprocess.py           # Standalone staged preprocessing utility
│   ├── 01_eda.py                  # EDA based on staged preprocessing output
│   └── 02_feature_extraction.py   # Experimental GL fractional feature extractor
├── archive/                        # Earlier experimental scripts for provenance
├── result/                         # Generated outputs; only its README is tracked
├── DATA.md                         # Data source, layout, and validation instructions
├── CITATION.cff                    # Citation metadata
├── LICENSE                         # Current reuse terms
└── requirements.txt                # Python dependencies
```

## Installation

Use a 64-bit installation of Python 3.10 or later. Run all commands from the repository root.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The experiments load the complete light-curve data, train several models in parallel, and generate high-resolution figures. Ensure that the machine has sufficient memory, disk space, and execution time.

## Data setup

This project uses the MALLORN Astronomical Classification Challenge data. The data is not redistributed in this repository. Download it from the official source and follow [DATA.md](DATA.md) to create this layout:

```text
mallorn-dataset/train_log.csv
mallorn-dataset/split_01/train_full_lightcurves.csv
...
mallorn-dataset/split_20/train_full_lightcurves.csv
```

Quick validation in PowerShell:

```powershell
Test-Path mallorn-dataset/train_log.csv
(Get-ChildItem mallorn-dataset/split_*/train_full_lightcurves.csv).Count
```

The expected outputs are `True` and `20`.

## Running the analyses

The three main entry points are independent. Each one loads and preprocesses the raw data, so they do not need to be run in sequence.

| Command | Main outputs |
| --- | --- |
| `python scripts/run_experiments.py` | Model metrics and plots in `result/eda/`, `result/viz/`, and `result/error_analysis/` |
| `python scripts/run_ablation_analysis.py` | Ablation, band, statistical-test, and threshold outputs in `result/analysis/` |
| `python scripts/run_interpretability.py` | Fractional-derivative, correlation, and SHAP figures in `result/analysis/` |

The interpretability script stores reusable features in `result/analysis/feature_cache.pkl`. Delete this cache before rerunning the script after changing the data or feature-generation logic.

### Staged utility workflow

The repository also includes an earlier staged workflow:

```bash
python scripts/00_preprocess.py
python scripts/01_eda.py
python scripts/02_feature_extraction.py
```

`02_feature_extraction.py` uses the discrete Grünwald–Letnikov (GL) fractional derivative and a different set of `alpha` values. It is an experimental comparison utility, not an equivalent staged implementation of the FFT-based main workflow, and it is not called by the `run_*.py` scripts.

## Reproducibility and limitations

- Train/test splitting and model training use a fixed random seed of `42`; plotting samples also use fixed seeds.
- `requirements.txt` specifies compatible lower bounds rather than a fully locked environment. For formal experiments, record the output of `python -m pip freeze` together with the operating system and Python version.
- The labelled training set contains 3,043 objects, including 148 TDEs. Because the classes are highly imbalanced, accuracy should be interpreted alongside AUC and confusion matrices.
- The scripts use class weighting, a fixed stratified holdout split, and five-fold cross-validation.
- This is exploratory research code. It currently has no command-line configuration layer, automated data downloader, unit-test suite, or continuous-integration workflow.
- Relative paths are resolved from the current working directory. Run scripts from the repository root.

## Troubleshooting

- **`FileNotFoundError: mallorn-dataset/...`**: verify the directory layout in [DATA.md](DATA.md) and run the command from the repository root.
- **`No objects to concatenate`**: the script did not find the 20 `split_*/train_full_lightcurves.csv` files.
- **LightGBM or XGBoost installation fails**: upgrade `pip`, confirm that Python is 64-bit and supported, and retry in a clean virtual environment.
- **Missing Chinese font warnings in auxiliary EDA plots**: `01_eda.py` requests the SimHei font. This affects plot labels only, not model results.
- **Interpretability results appear stale**: delete `result/analysis/feature_cache.pkl` and rerun `run_interpretability.py`.

## Citation

If this repository supports your work, use GitHub's **Cite this repository** feature and cite the MALLORN data paper. Repository metadata is available in [CITATION.cff](CITATION.cff), and the data citation is provided in [DATA.md](DATA.md).

## License

The repository is currently source-available for viewing, academic discussion, and citation, but it does not grant an open-source license. See [LICENSE](LICENSE) before copying, modifying, or redistributing the code or opening a contribution based on it.
