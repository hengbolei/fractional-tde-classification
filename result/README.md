# Generated results

Experiment scripts write figures, tables, JSON summaries, caches and text reports to this directory. Generated files are ignored by Git so that reruns do not bloat the repository or accidentally publish derived artifacts.

Expected subdirectories include:

- `eda/` - exploratory plots
- `viz/` - model-comparison plots
- `error_analysis/` - misclassification and calibration plots
- `analysis/` - ablation, band-importance, threshold and SHAP outputs

The main scripts create these directories automatically. `run_interpretability.py` also stores `analysis/feature_cache.pkl`; delete that cache before rerunning after a data or feature-code change.

Do not use generated output as the only record of a formal experiment. Record the command, commit, Python version, installed package versions and platform alongside any results that will be reported.
