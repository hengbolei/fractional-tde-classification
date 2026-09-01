# -*- coding: utf-8 -*-
"""
mallorn5_analysis.py — 补充分析脚本
在 mallorn4 基础上新增：
  1. Alpha 消融实验（各 alpha 单独 + 两两组合）
  2. 波段重要性分析
  3. 统计显著性检验（McNemar + 配对t检验）
  4. 阈值选择分析（Precision/Recall/F1曲线）
"""

import pandas as pd
import numpy as np
import os
import json
from scipy.stats import skew, kurtosis, ttest_rel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                              confusion_matrix, precision_recall_curve,
                              precision_score, recall_score, f1_score)
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from scipy.interpolate import interp1d
import glob
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ===================== 全局配色 (同 mallorn4) =====================
PALETTE = {
    'bg': '#FFFFFF', 'panel': '#F5F5F5', 'border': '#B0B0B0',
    'conv': '#1F77B4', 'frac': '#2CA02C', 'fuse': '#D62728',
    'lgbm': '#D62728', 'xgb': '#1F77B4', 'rf': '#2CA02C',
    'accent': '#E3B341', 'text': '#333333', 'subtext': '#666666',
    'grid': '#DDDDDD',
}
BAND_COLORS = {'u': '#9467BD', 'g': '#2CA02C', 'r': '#D62728',
               'i': '#1F77B4', 'z': '#BC8F1F', 'y': '#17BECF'}
HIST_EDGE = '#CCCCCC'

def apply_academic_style():
    plt.rcParams.update({
        'figure.facecolor': PALETTE['bg'], 'axes.facecolor': PALETTE['panel'],
        'axes.edgecolor': PALETTE['border'], 'axes.labelcolor': PALETTE['text'],
        'axes.titlecolor': '#000000', 'xtick.color': PALETTE['subtext'],
        'ytick.color': PALETTE['subtext'], 'text.color': PALETTE['text'],
        'grid.color': PALETTE['grid'], 'grid.linestyle': '--', 'grid.alpha': 0.6,
        'legend.facecolor': '#FFFFFF', 'legend.edgecolor': PALETTE['border'],
        'legend.framealpha': 0.9, 'font.family': 'DejaVu Sans',
        'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 1.0,
    })

apply_academic_style()

OUTPUT_DIR = './result'
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, 'analysis')
for d in [OUTPUT_DIR, ANALYSIS_DIR]:
    os.makedirs(d, exist_ok=True)

DATA_DIR = "./mallorn-dataset"

# ===================== 1. 数据加载 =====================
print("=" * 60)
print("Step 1: Data Loading")
print("=" * 60)

train_log = pd.read_csv(os.path.join(DATA_DIR, "train_log.csv"))

train_lc_files = glob.glob(os.path.join(DATA_DIR, "split_*", "train_full_lightcurves.csv"))
train_lc = pd.concat(
    [pd.read_csv(f) for f in tqdm(train_lc_files, desc="Loading Lightcurve Files")],
    ignore_index=True
)

train_lc.columns = train_lc.columns.str.strip()
train_lc = train_lc.rename(columns={
    'Time (MJD)': 'mjd', 'Time (MJD )': 'mjd',
    'Flux': 'flux', 'Flux_err': 'flux_err', 'Filter': 'filter'
})

df = train_lc.merge(
    train_log[['object_id', 'target', 'Z', 'EBV']],
    on='object_id', how='left'
)

# ===================== 2. 数据预处理 =====================
print("\n" + "=" * 60)
print("Step 2: Data Preprocessing")
print("=" * 60)

df.dropna(subset=['flux', 'EBV', 'flux_err'], inplace=True)

EXTINCTION_COEFF = {'u': 3.1, 'g': 3.0, 'r': 2.9, 'i': 2.8, 'z': 2.7, 'y': 2.6}

def de_extinct_flux(df):
    k = df['filter'].map(EXTINCTION_COEFF)
    factor = 10 ** (0.4 * df['EBV'] * k)
    df = df.copy()
    df['flux_corrected'] = np.where(k.isna(), df['flux'], df['flux'] / factor)
    return df

df = de_extinct_flux(df)
df.dropna(subset=['flux_corrected'], inplace=True)
df = df.groupby('object_id').filter(lambda x: len(x) >= 10)

# ===================== 3. 光变曲线预处理 =====================
print("\n" + "=" * 60)
print("Step 3: Lightcurve Preprocessing")
print("=" * 60)

def preprocess_lightcurve(group, dt=1.0):
    if len(group) < 10:
        return None
    group = group.sort_values('mjd').copy()
    t = group['mjd'].values
    flux = group['flux_corrected'].values
    flux_err = group['flux_err'].values
    n_baseline = max(3, int(0.2 * len(t)))
    weights = 1.0 / np.clip(flux_err[:n_baseline], 1e-10, None)
    baseline = np.average(flux[:n_baseline], weights=weights)
    flux = flux - baseline
    t_uniform = np.arange(t.min(), t.max() + dt, dt)
    if len(t_uniform) < 3:
        return None
    interp_func = interp1d(t, flux, kind='linear',
                           fill_value='extrapolate', bounds_error=False)
    return pd.DataFrame({
        'mjd': t_uniform,
        'flux_interp': interp_func(t_uniform),
        'object_id': group['object_id'].iloc[0],
        'filter': group['filter'].iloc[0],
        'target': group['target'].iloc[0],
    })

grouped_raw = df.groupby(['object_id', 'filter'])
processed_list = []
for key in tqdm(grouped_raw.groups.keys(), desc="Processing Lightcurves"):
    result = preprocess_lightcurve(grouped_raw.get_group(key), dt=1.0)
    if result is not None:
        processed_list.append(result)

df_processed = pd.concat(processed_list, ignore_index=True)

# ===================== 4. 特征提取 =====================
print("\n" + "=" * 60)
print("Step 4: Feature Extraction")
print("=" * 60)

COLS = ['mean', 'std', 'max', 'min', 'amplitude',
        'skew', 'kurt', 'median', 'mad', 'rise_time', 'decay_time']
ALPHAS = [0.5, 1.0, 1.5]
N_CONV_STATS = len(COLS)

def extract_unified_stats(s):
    s = np.asarray(s, dtype=float)
    valid = s[~np.isnan(s)]
    n = len(valid)
    if n < 3:
        return [np.nan] * N_CONV_STATS
    peak_idx = int(np.argmax(valid))
    min_before_peak = int(np.argmin(valid[:peak_idx + 1]))
    rise_time_val = peak_idx - min_before_peak
    peak_val = float(valid[peak_idx])
    min_val = float(np.min(valid[:peak_idx + 1]))
    half_val = (peak_val + min_val) / 2.0
    post_peak = valid[peak_idx:]
    below_half = np.where(post_peak <= half_val)[0]
    decay_time_val = float(below_half[0]) if len(below_half) > 0 else float(len(post_peak) - 1)
    return [
        float(np.mean(valid)), float(np.std(valid)),
        float(np.max(valid)), float(np.min(valid)),
        float(np.max(valid) - np.min(valid)),
        float(skew(valid)), float(kurtosis(valid)),
        float(np.median(valid)),
        float(np.median(np.abs(valid - np.median(valid)))),
        float(rise_time_val), float(decay_time_val),
    ]

def fractional_derivative_fft(y, alpha, dt=1.0):
    if alpha == 0.0:
        return np.array(y, dtype=float)
    N = len(y)
    Y = np.fft.fft(y)
    freqs = np.fft.fftfreq(N, d=dt)
    omega = 2 * np.pi * freqs
    with np.errstate(divide='ignore', invalid='ignore'):
        scaling = np.where(omega == 0, 0.0, (1j * omega) ** alpha)
    return np.real(np.fft.ifft(Y * scaling))

records = {}
grouped_proc = df_processed.groupby(['object_id', 'filter'])

for (obj_id, filt), group in tqdm(grouped_proc, desc="Extracting Features"):
    y_vals = group['flux_interp'].values
    t_vals = group['mjd'].values
    if obj_id not in records:
        records[obj_id] = {}
    stats = extract_unified_stats(y_vals)
    for i, col in enumerate(COLS):
        records[obj_id][f"{col}_{filt}"] = stats[i]
    dt_val = float(np.median(np.diff(t_vals))) if len(t_vals) > 1 else 1.0
    dt_val = max(dt_val, 1e-6)
    for alpha in ALPHAS:
        deriv = fractional_derivative_fft(y_vals, alpha, dt_val)
        frac_stats = extract_unified_stats(deriv)
        for i, col in enumerate(COLS):
            records[obj_id][f"alpha{alpha:.1f}_{filt}_{col}"] = frac_stats[i]

features_df = pd.DataFrame.from_dict(records, orient='index')
features_df.index.name = 'object_id'
features_df = features_df.reset_index()

# ===================== 5. 对齐特征与标签 =====================
print("\n" + "=" * 60)
print("Step 5: Feature-Label Alignment")
print("=" * 60)

label_df = df_processed.groupby('object_id')['target'].first().reset_index()
merged = features_df.merge(label_df, on='object_id', how='inner')
merged = merged.dropna(axis=1, thresh=int(0.8 * len(merged)))
merged = merged.dropna()

X_all = merged.drop(columns=['object_id', 'target'])
y_all = merged['target'].copy()

all_cols = list(X_all.columns)
conv_cols = [c for c in all_cols if not c.startswith('alpha')]
frac_cols = [c for c in all_cols if c.startswith('alpha')]

# Alpha subsets for ablation
alpha_subsets = {}
for alpha in ALPHAS:
    key = f"alpha_{alpha:.1f}"
    alpha_subsets[key] = [c for c in all_cols if c.startswith(f"alpha{alpha:.1f}")]
# Two-alpha combinations
from itertools import combinations
for a1, a2 in combinations(ALPHAS, 2):
    key = f"alpha_{a1:.1f}+{a2:.1f}"
    alpha_subsets[key] = [c for c in all_cols
                          if c.startswith(f"alpha{a1:.1f}") or c.startswith(f"alpha{a2:.1f}")]

# Band subsets
# Detect available bands from column names
ALL_FILTERS = ['u', 'g', 'r', 'i', 'z', 'y']
FILTERS = []
for f in ALL_FILTERS:
    test_cols = [c for c in all_cols if f'_{f}_' in c or c.endswith(f'_{f}')]
    if len(test_cols) > 0:
        FILTERS.append(f)
    else:
        print(f"  WARNING: Band '{f}' has 0 features (likely dropped during NaN filtering). Skipping.")
print(f"  Available bands: {FILTERS}")
band_subsets = {}
for f in FILTERS:
    band_subsets[f] = [c for c in all_cols if f'_{f}_' in c or c.endswith(f'_{f}')]

print(f"Features: {X_all.shape[1]} dims")
print(f"  Conv: {len(conv_cols)}, Frac: {len(frac_cols)}")
print(f"  Alpha subsets: {[(k, len(v)) for k, v in alpha_subsets.items()]}")
print(f"  Band subsets: {[(k, len(v)) for k, v in band_subsets.items()]}")

# ===================== 6. 数据集划分 =====================
print("\n" + "=" * 60)
print("Step 6: Train/Test Split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Helper: get column indices for a subset
def get_idx(subset_cols):
    return [all_cols.index(c) for c in subset_cols if c in all_cols]

# ===================== 7. 模型定义 =====================
def make_lgbm(y_tr):
    neg = (y_tr == 0).sum(); pos = (y_tr == 1).sum()
    spw = neg / pos if pos > 0 else 1.0
    return lgb.LGBMClassifier(
        objective='binary', metric='auc', scale_pos_weight=spw,
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        random_state=42, n_jobs=-1, verbose=-1)

def make_xgb(y_tr):
    neg = (y_tr == 0).sum(); pos = (y_tr == 1).sum()
    spw = neg / pos if pos > 0 else 1.0
    return xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='auc',
        scale_pos_weight=spw, n_estimators=500, learning_rate=0.05,
        max_depth=6, random_state=42, n_jobs=-1, verbosity=0)

def make_rf(y_tr):
    return RandomForestClassifier(
        n_estimators=500, max_depth=10, class_weight='balanced',
        random_state=42, n_jobs=-1)

MODELS = {'LightGBM': make_lgbm, 'XGBoost': make_xgb, 'RandomForest': make_rf}

def quick_train(X_tr, X_te, y_tr, y_te, make_fn):
    clf = make_fn(y_tr)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    return {
        'clf': clf, 'y_pred': y_pred, 'y_prob': y_prob,
        'acc': accuracy_score(y_te, y_pred),
        'auc': roc_auc_score(y_te, y_prob),
        'cm': confusion_matrix(y_te, y_pred),
    }

# ========================================================================
# ANALYSIS 1: Alpha 消融实验
# ========================================================================
print("\n" + "=" * 60)
print("ANALYSIS 1: Alpha Ablation Study")
print("=" * 60)

ablation_results = {}

# Conv only (baseline)
conv_idx = get_idx(conv_cols)
# All frac
frac_idx = get_idx(frac_cols)

for subset_name, subset_cols in tqdm(
    [('Conv', conv_cols), ('Frac_all', frac_cols)] +
    [(k, v) for k, v in alpha_subsets.items()],
    desc="Alpha Ablation"):
    idx = get_idx(subset_cols)
    model_results = {}
    for model_name, make_fn in MODELS.items():
        r = quick_train(X_train_sc[:, idx], X_test_sc[:, idx],
                        y_train, y_test, make_fn)
        model_results[model_name] = {'acc': r['acc'], 'auc': r['auc']}
    ablation_results[subset_name] = {
        'n_features': len(subset_cols),
        'models': model_results,
    }

# Save ablation results
ablation_summary = []
for sn, res in ablation_results.items():
    row = {'subset': sn, 'n_features': res['n_features']}
    for mn in MODELS:
        row[f'{mn}_acc'] = round(res['models'][mn]['acc'], 4)
        row[f'{mn}_auc'] = round(res['models'][mn]['auc'], 4)
    ablation_summary.append(row)
ablation_df = pd.DataFrame(ablation_summary)
ablation_df.to_csv(os.path.join(ANALYSIS_DIR, 'alpha_ablation.csv'), index=False)
print("  Saved: alpha_ablation.csv")

# Visualization: AUC bar chart
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(PALETTE['bg'])
subsets = list(ablation_results.keys())
x = np.arange(len(subsets))
w = 0.25
for i, (mn, mc) in enumerate([('LightGBM', PALETTE['lgbm']),
                                ('XGBoost', PALETTE['xgb']),
                                ('RandomForest', PALETTE['rf'])]):
    vals = [ablation_results[s]['models'][mn]['auc'] for s in subsets]
    bars = ax.bar(x + w * (i - 1), vals, w, color=mc,
                  edgecolor=PALETTE['border'], linewidth=0.8, label=mn)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{val:.4f}', ha='center', fontsize=6, color=PALETTE['text'],
                fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(subsets, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Test AUC'); ax.set_title('Alpha Ablation Study — Test AUC', fontsize=13, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, 'alpha_ablation_auc.png'), dpi=300)
plt.close()
print("  Saved: alpha_ablation_auc.png")

# ========================================================================
# ANALYSIS 2: 波段重要性分析
# ========================================================================
print("\n" + "=" * 60)
print("ANALYSIS 2: Band Importance Analysis")
print("=" * 60)

band_results = {}
for band_name, band_cols in tqdm(band_subsets.items(), desc="Band Analysis"):
    idx = get_idx(band_cols)
    r = quick_train(X_train_sc[:, idx], X_test_sc[:, idx],
                    y_train, y_test, make_lgbm)
    band_results[band_name] = {'acc': r['acc'], 'auc': r['auc'], 'n_features': len(band_cols)}

# Drop-one-band analysis
drop_one_results = {}
all_bands_idx = get_idx(list(all_cols))
for drop_band in FILTERS:
    remaining = [c for c in all_cols if not (f'_{drop_band}_' in c or c.endswith(f'_{drop_band}'))]
    idx = get_idx(remaining)
    r = quick_train(X_train_sc[:, idx], X_test_sc[:, idx],
                    y_train, y_test, make_lgbm)
    drop_one_results[f'drop_{drop_band}'] = {
        'acc': r['acc'], 'auc': r['auc'], 'n_features': len(remaining)}

band_summary = []
for bn, res in band_results.items():
    band_summary.append({'band': bn, 'type': 'single', 'n_features': res['n_features'],
                         'acc': round(res['acc'], 4), 'auc': round(res['auc'], 4)})
for bn, res in drop_one_results.items():
    band_summary.append({'band': bn, 'type': 'drop_one', 'n_features': res['n_features'],
                         'acc': round(res['acc'], 4), 'auc': round(res['auc'], 4)})
# Full
all_r = quick_train(X_train_sc[:, all_bands_idx], X_test_sc[:, all_bands_idx],
                    y_train, y_test, make_lgbm)
band_summary.append({'band': 'all', 'type': 'full', 'n_features': len(all_cols),
                     'acc': round(all_r['acc'], 4), 'auc': round(all_r['auc'], 4)})

band_df = pd.DataFrame(band_summary)
band_df.to_csv(os.path.join(ANALYSIS_DIR, 'band_importance.csv'), index=False)
print("  Saved: band_importance.csv")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(PALETTE['bg'])

# Single-band AUC
ax = axes[0]
bands_ordered = sorted(FILTERS, key=lambda b: band_results[b]['auc'], reverse=True)
vals_single = [band_results[b]['auc'] for b in bands_ordered]
colors_single = [BAND_COLORS[b] for b in bands_ordered]
bars = ax.bar(bands_ordered, vals_single, color=colors_single,
              edgecolor=PALETTE['border'], linewidth=0.8)
for bar, val in zip(bars, vals_single):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', fontsize=9, color=PALETTE['text'], fontweight='bold')
ax.set_title('Single-Band Classification AUC (LightGBM)', fontsize=12, fontweight='bold')
ax.set_ylabel('Test AUC'); ax.grid(axis='y', alpha=0.4)

# Drop-one-band
ax = axes[1]
full_auc = all_r['auc']
drop_vals = [drop_one_results[f'drop_{b}']['auc'] for b in FILTERS]
drop_deltas = [full_auc - v for v in drop_vals]
bars = ax.bar(FILTERS, drop_deltas, color=[BAND_COLORS[b] for b in FILTERS],
              edgecolor=PALETTE['border'], linewidth=0.8)
for bar, val in zip(bars, drop_deltas):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
            f'{val:.4f}', ha='center', fontsize=9, color=PALETTE['text'], fontweight='bold')
ax.set_title(f'Drop-One-Band AUC Decrement (Full AUC={full_auc:.4f})', fontsize=12, fontweight='bold')
ax.set_ylabel('AUC Decrement'); ax.grid(axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, 'band_importance.png'), dpi=300)
plt.close()
print("  Saved: band_importance.png")

# ========================================================================
# ANALYSIS 3: 统计显著性检验
# ========================================================================
print("\n" + "=" * 60)
print("ANALYSIS 3: Statistical Significance Tests")
print("=" * 60)

# 3.1 McNemar test on test set predictions
from statsmodels.stats.contingency_tables import mcnemar

# Train all 9 model+feature combos and collect per-sample predictions
FEATURE_SETS_MAP = {
    'Conv': get_idx(conv_cols),
    'Frac': get_idx(frac_cols),
    'Fusion': get_idx(all_cols),
}

test_predictions = {}  # {(model, feat): y_pred_array}

for model_name, make_fn in tqdm(MODELS.items(), desc="Training for tests"):
    for feat_name, feat_idx in FEATURE_SETS_MAP.items():
        r = quick_train(X_train_sc[:, feat_idx], X_test_sc[:, feat_idx],
                        y_train, y_test, make_fn)
        test_predictions[(model_name, feat_name)] = {
            'y_pred': r['y_pred'], 'y_prob': r['y_prob'],
            'acc': r['acc'], 'auc': r['auc'],
        }

# McNemar: compare models pairwise on Fusion features
mcnemar_results = []
model_names = list(MODELS.keys())
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        m1, m2 = model_names[i], model_names[j]
        pred1 = test_predictions[(m1, 'Fusion')]['y_pred']
        pred2 = test_predictions[(m2, 'Fusion')]['y_pred']
        yt = np.array(y_test)

        # 2x2 contingency: both correct, m1 correct m2 wrong, m1 wrong m2 correct, both wrong
        c1 = (pred1 == yt)
        c2 = (pred2 == yt)
        b = int(((c1 == True) & (c2 == False)).sum())
        c = int(((c1 == False) & (c2 == True)).sum())
        table = [[0, b], [c, 0]]  # simplified — just b and c matter for McNemar
        # Use exact binomial test
        from scipy.stats import binomtest
        n_discordant = b + c
        if n_discordant > 0:
            p_value = binomtest(min(b, c), n_discordant, 0.5).pvalue
        else:
            p_value = 1.0
        mcnemar_results.append({
            'comparison': f'{m1} vs {m2} (Fusion)',
            'n_discordant': n_discordant,
            'm1_correct_m2_wrong': b,
            'm1_wrong_m2_correct': c,
            'p_value': round(p_value, 6),
            'significant_0.05': p_value < 0.05,
        })

# 3.2 Paired t-test on CV scores
cv_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, make_fn in tqdm(MODELS.items(), desc="CV for t-tests"):
    for feat_name, feat_idx in FEATURE_SETS_MAP.items():
        clf = make_fn(y_all)
        scores = cross_val_score(clf, X_all.iloc[:, feat_idx], y_all,
                                 cv=cv, scoring='roc_auc', n_jobs=-1)
        cv_results[(model_name, feat_name)] = scores

ttest_results = []
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        m1, m2 = model_names[i], model_names[j]
        for feat in FEATURE_SETS_MAP:
            s1 = cv_results[(m1, feat)]
            s2 = cv_results[(m2, feat)]
            t_stat, p_val = ttest_rel(s1, s2)
            ttest_results.append({
                'comparison': f'{m1} vs {m2} ({feat})',
                't_statistic': round(t_stat, 4),
                'p_value': round(p_val, 6),
                'significant_0.05': p_val < 0.05,
                f'{m1}_mean': round(s1.mean(), 4),
                f'{m2}_mean': round(s2.mean(), 4),
            })

mcnemar_df = pd.DataFrame(mcnemar_results)
mcnemar_df.to_csv(os.path.join(ANALYSIS_DIR, 'mcnemar_test.csv'), index=False)
print("  Saved: mcnemar_test.csv")

ttest_df = pd.DataFrame(ttest_results)
ttest_df.to_csv(os.path.join(ANALYSIS_DIR, 'paired_ttest.csv'), index=False)
print("  Saved: paired_ttest.csv")

# Visualization: CV pairwise comparison
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(PALETTE['bg'])
# Show each model+feat CV distribution
positions = []
data = []
colors = []
labels = []
for i, feat in enumerate(FEATURE_SETS_MAP):
    for j, mn in enumerate(model_names):
        pos = i * 3 + j
        positions.append(pos)
        data.append(cv_results[(mn, feat)])
        colors.append(PALETTE[{'LightGBM': 'lgbm', 'XGBoost': 'xgb', 'RandomForest': 'rf'}[mn]])
        labels.append(f'{mn}\n{feat}')

bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6,
                medianprops=dict(color=PALETTE['accent'], linewidth=2))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)
    patch.set_edgecolor(PALETTE['border'])
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=7)
ax.set_title('5-Fold CV AUC Distribution with Paired t-Test Results', fontsize=12, fontweight='bold')
ax.set_ylabel('AUC'); ax.grid(axis='y', alpha=0.4)

# Add significance bars for key comparisons
sig_pairs = [r for r in ttest_results if r['significant_0.05']]
print(f"  Significant CV differences (p<0.05): {len(sig_pairs)} pairs")
for sp in sig_pairs[:5]:
    print(f"    {sp['comparison']}: p={sp['p_value']:.6f}")

plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, 'cv_ttest.png'), dpi=300)
plt.close()
print("  Saved: cv_ttest.png")

# ========================================================================
# ANALYSIS 4: 阈值选择分析
# ========================================================================
print("\n" + "=" * 60)
print("ANALYSIS 4: Threshold Selection Analysis")
print("=" * 60)

thresholds = np.arange(0.1, 0.95, 0.05)
threshold_metrics = []

for model_name, _ in MODELS.items():
    y_prob = test_predictions[(model_name, 'Fusion')]['y_prob']
    yt = np.array(y_test)
    for thresh in thresholds:
        y_pred_t = (y_prob >= thresh).astype(int)
        threshold_metrics.append({
            'model': model_name,
            'threshold': round(thresh, 2),
            'precision': round(precision_score(yt, y_pred_t, zero_division=0), 4),
            'recall': round(recall_score(yt, y_pred_t, zero_division=0), 4),
            'f1': round(f1_score(yt, y_pred_t, zero_division=0), 4),
            'tp': int(((yt == 1) & (y_pred_t == 1)).sum()),
            'fp': int(((yt == 0) & (y_pred_t == 1)).sum()),
            'fn': int(((yt == 1) & (y_pred_t == 0)).sum()),
            'tn': int(((yt == 0) & (y_pred_t == 0)).sum()),
        })

thresh_df = pd.DataFrame(threshold_metrics)
thresh_df.to_csv(os.path.join(ANALYSIS_DIR, 'threshold_analysis.csv'), index=False)
print("  Saved: threshold_analysis.csv")

# Visualization: Precision/Recall/F1 vs Threshold
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle('Precision / Recall / F1 vs Decision Threshold', fontsize=14, fontweight='bold')

model_colors = {'LightGBM': PALETTE['lgbm'], 'XGBoost': PALETTE['xgb'],
                'RandomForest': PALETTE['rf']}

for ax, metric in zip(axes, ['precision', 'recall', 'f1']):
    for mn in MODELS:
        sub = thresh_df[thresh_df['model'] == mn]
        ax.plot(sub['threshold'], sub[metric], 'o-', color=model_colors[mn],
                linewidth=2, markersize=5, label=mn)
    ax.set_xlabel('Threshold'); ax.set_ylabel(metric.capitalize())
    ax.set_title(metric.capitalize(), fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
    ax.set_xlim(0.05, 0.95)

plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, 'threshold_analysis.png'), dpi=300)
plt.close()
print("  Saved: threshold_analysis.png")

# Find optimal thresholds for TDE candidate screening (maximize Recall subject to Precision >= 0.5)
print("\n  Optimal thresholds for TDE screening (max Recall, Precision >= 0.5):")
for mn in MODELS:
    sub = thresh_df[thresh_df['model'] == mn]
    sub_filtered = sub[sub['precision'] >= 0.5]
    if len(sub_filtered) > 0:
        best = sub_filtered.loc[sub_filtered['recall'].idxmax()]
        print(f"    {mn}: threshold={best['threshold']:.2f}, "
              f"Precision={best['precision']:.4f}, Recall={best['recall']:.4f}, F1={best['f1']:.4f}")

# ========================================================================
# SAVE ALL RESULTS AS JSON
# ========================================================================
print("\n" + "=" * 60)
print("SAVING SUMMARY")
print("=" * 60)

summary = {
    'alpha_ablation': ablation_summary,
    'band_importance': band_summary,
    'mcnemar_tests': mcnemar_results,
    'paired_ttests': ttest_results,
    'threshold_analysis': threshold_metrics,
}

# Convert numpy types for JSON
import json as json_module
class NpEncoder(json_module.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return super().default(obj)

with open(os.path.join(ANALYSIS_DIR, 'all_analysis_results.json'), 'w', encoding='utf-8') as f:
    json_module.dump(summary, f, ensure_ascii=False, indent=2, cls=NpEncoder)
print("  Saved: all_analysis_results.json")

print("\n" + "=" * 60)
print("ALL DONE!")
print(f"  Results -> {ANALYSIS_DIR}/")
print("=" * 60)
