import pandas as pd
import numpy as np
import os
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
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
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

PALETTE = {
    'bg': '#FFFFFF',
    'panel': '#F5F5F5',
    'border': '#B0B0B0',
    'conv': '#1F77B4',
    'frac': '#2CA02C',
    'fuse': '#D62728',
    'lgbm': '#D62728',
    'xgb': '#1F77B4',
    'rf': '#2CA02C',
    'accent': '#E3B341',
    'text': '#333333',
    'subtext': '#666666',
    'grid': '#DDDDDD',
}

FILTER_COLORS = {
    'u': '#9467BD', 'g': '#2CA02C', 'r': '#D62728',
    'i': '#1F77B4', 'z': '#BC8F1F', 'y': '#17BECF',
}

HIST_EDGE = '#CCCCCC'

def apply_academic_style():
    plt.rcParams.update({
        'figure.facecolor': PALETTE['bg'],
        'axes.facecolor': PALETTE['panel'],
        'axes.edgecolor': PALETTE['border'],
        'axes.labelcolor': PALETTE['text'],
        'axes.titlecolor': '#000000',
        'xtick.color': PALETTE['subtext'],
        'ytick.color': PALETTE['subtext'],
        'text.color': PALETTE['text'],
        'grid.color': PALETTE['grid'],
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'legend.facecolor': '#FFFFFF',
        'legend.edgecolor': PALETTE['border'],
        'legend.framealpha': 0.9,
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.0,
    })

apply_academic_style()

OUTPUT_DIR = './result'
EDA_DIR    = os.path.join(OUTPUT_DIR, 'eda')
VIZ_DIR    = os.path.join(OUTPUT_DIR, 'viz')
ERR_DIR    = os.path.join(OUTPUT_DIR, 'error_analysis')
for d in [OUTPUT_DIR, EDA_DIR, VIZ_DIR, ERR_DIR]:
    os.makedirs(d, exist_ok=True)

DATA_DIR = "./mallorn-dataset"

# ========================== 1. 数据加载 ==========================

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

# ========================== 2. EDA ==========================

print("\n========== EDA ==========")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle("EDA -- Dataset Overview", color=PALETTE['text'],
             fontsize=14, fontweight='bold', y=1.01)

label_counts = train_log['target'].value_counts().sort_index()
bars = axes[0].bar(
    ['Negative (0)', 'Positive (1)'],
    [label_counts.get(0, 0), label_counts.get(1, 0)],
    color=[PALETTE['conv'], PALETTE['frac']],
    edgecolor=PALETTE['border'], linewidth=1.2
)
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Count")
axes[0].grid(axis='y')
for bar, val in zip(bars, [label_counts.get(0, 0), label_counts.get(1, 0)]):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 5, f'{val}',
                 ha='center', color=PALETTE['text'], fontsize=10)

axes[1].hist(train_log['Z'].dropna(), bins=50,
             color=PALETTE['accent'], edgecolor=HIST_EDGE, linewidth=0.5)
axes[1].set_title("Redshift (Z) Distribution")
axes[1].set_xlabel("Z"); axes[1].set_ylabel("Count"); axes[1].grid(axis='y')

axes[2].hist(train_log['EBV'].dropna(), bins=50,
             color=PALETTE['fuse'], edgecolor=HIST_EDGE, linewidth=0.5)
axes[2].set_title("Dust Extinction (EBV) Distribution")
axes[2].set_xlabel("EBV"); axes[2].set_ylabel("Count"); axes[2].grid(axis='y')

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "eda_overview.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: eda_overview.png")

# ========================== 3. 数据预处理 ==========================

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

# ========================== 4. 光变曲线预处理 ==========================

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

# ========================== 5. 特征提取 ==========================

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
        float(np.mean(valid)),
        float(np.std(valid)),
        float(np.max(valid)),
        float(np.min(valid)),
        float(np.max(valid) - np.min(valid)),
        float(skew(valid)),
        float(kurtosis(valid)),
        float(np.median(valid)),
        float(np.median(np.abs(valid - np.median(valid)))),
        float(rise_time_val),
        float(decay_time_val),
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

# ========================== 6. 对齐特征与标签 ==========================

label_df = df_processed.groupby('object_id')['target'].first().reset_index()
merged = features_df.merge(label_df, on='object_id', how='inner')
merged = merged.dropna(axis=1, thresh=int(0.8 * len(merged)))
merged = merged.dropna()

X_all = merged.drop(columns=['object_id', 'target'])
y_all = merged['target']

print(f"\nFeatures shape : {X_all.shape}")
print(f"Target shape   : {y_all.shape}")
print(f"Positive rate  : {y_all.mean():.4f}")

all_cols = list(X_all.columns)
conv_cols = [c for c in all_cols if not c.startswith('alpha')]
frac_cols = [c for c in all_cols if c.startswith('alpha')]
conv_idx = [all_cols.index(c) for c in conv_cols]
frac_idx = [all_cols.index(c) for c in frac_cols]

# ========================== 7. 划分数据集 ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# 保留原始索引用于错误分析
test_indices = X_test.index

# ========================== 8. 模型定义 ==========================

def make_lgbm(y_tr):
    neg = (y_tr == 0).sum()
    pos = (y_tr == 1).sum()
    spw = neg / pos if pos > 0 else 1.0
    return lgb.LGBMClassifier(
        objective='binary', metric='auc',
        scale_pos_weight=spw,
        n_estimators=500, learning_rate=0.05,
        num_leaves=63, random_state=42,
        n_jobs=-1, verbose=-1,
    )

def make_xgb(y_tr):
    neg = (y_tr == 0).sum()
    pos = (y_tr == 1).sum()
    spw = neg / pos if pos > 0 else 1.0
    return xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='auc',
        scale_pos_weight=spw,
        n_estimators=500, learning_rate=0.05,
        max_depth=6, random_state=42,
        n_jobs=-1, verbosity=0,
    )

def make_rf(y_tr):
    neg = (y_tr == 0).sum()
    pos = (y_tr == 1).sum()
    spw = {0: 1, 1: neg / pos if pos > 0 else 1.0}
    return RandomForestClassifier(
        n_estimators=500, max_depth=10,
        class_weight='balanced',
        random_state=42, n_jobs=-1,
    )

def train_eval(X_tr, X_te, y_tr, y_te, model_name, make_clf):
    clf = make_clf(y_tr)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_pred_prob = clf.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_te, y_pred_prob)
    cm = confusion_matrix(y_te, y_pred)
    print(f"  [{model_name}]  Acc: {acc:.4f}  AUC: {auc:.4f}")
    return acc, auc, clf, fpr, tpr, cm, y_pred_prob, y_pred

def cv_auc(X_sc, y, model_name, make_clf):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = make_clf(y)
    scores = cross_val_score(clf, X_sc, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"  [{model_name}] CV AUC: {scores.mean():.4f} +- {scores.std():.4f}")
    return scores

# ========================== 9. 三模型 X 三特征组 ==========================

MODELS = {
    'LightGBM': (make_lgbm, PALETTE['lgbm']),
    'XGBoost':  (make_xgb,  PALETTE['xgb']),
    'RandomForest': (make_rf, PALETTE['rf']),
}

FEATURE_SETS = {
    'Conv':   (conv_idx, PALETTE['conv']),
    'Frac':   (frac_idx, PALETTE['frac']),
    'Fusion': (list(range(len(all_cols))), PALETTE['fuse']),
}

all_results = {}  # {model_name: {feat_name: {acc, auc, clf, fpr, tpr, cm, prob, pred}}}

print("\n========== Model Comparison ==========\n")

for model_name, (make_fn, _) in MODELS.items():
    all_results[model_name] = {}
    for feat_name, (feat_idx, _) in FEATURE_SETS.items():
        print(f"--- {model_name} + {feat_name} ---")
        acc, auc_v, clf, fpr, tpr, cm, prob, pred = train_eval(
            X_train_sc[:, feat_idx], X_test_sc[:, feat_idx],
            y_train, y_test, f"{model_name}/{feat_name}", make_fn)
        cv_scores = cv_auc(X_train_sc[:, feat_idx], y_train,
                           f"{model_name}/{feat_name} CV", make_fn)
        all_results[model_name][feat_name] = {
            'acc': acc, 'auc': auc_v, 'clf': clf,
            'fpr': fpr, 'tpr': tpr, 'cm': cm,
            'prob': prob, 'pred': pred,
            'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
        }

# ========================== 10. 模型对比可视化 ==========================

apply_academic_style()
model_names = list(MODELS.keys())
feat_names = list(FEATURE_SETS.keys())

# ---------- 10.1 综合柱状对比 (AUC + CV AUC) ----------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle("Multi-Model Performance Comparison", color=PALETTE['text'],
             fontsize=14, fontweight='bold')

x = np.arange(len(model_names))
w = 0.25
feat_colors = [PALETTE['conv'], PALETTE['frac'], PALETTE['fuse']]

for ax, metric_name in [(axes[0], 'auc'), (axes[1], 'cv_mean')]:
    for i, (feat_name, feat_color) in enumerate(zip(feat_names, feat_colors)):
        vals = [all_results[m][feat_name][metric_name] for m in model_names]
        offset = w * (i - 1)
        bars = ax.bar(x + offset, vals, width=w, color=feat_color,
                      edgecolor=PALETTE['border'], linewidth=0.8,
                      label=feat_name, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003, f'{val:.4f}',
                    ha='center', fontsize=7, color=PALETTE['text'], fontweight='bold')
    if metric_name == 'cv_mean':
        for i, m in enumerate(model_names):
            for j, feat_name in enumerate(feat_names):
                r = all_results[m][feat_name]
                offset = w * (j - 1)
                ax.errorbar(x[i] + offset, r['cv_mean'], yerr=r['cv_std'],
                           fmt='none', ecolor=PALETTE['subtext'], capsize=3, linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=11)
    title = 'Test AUC' if metric_name == 'auc' else 'CV AUC mean +- std'
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('Score')
    ax.grid(axis='y', alpha=0.4, zorder=0)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "viz_multimodel_bar.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: viz_multimodel_bar.png")

# ---------- 10.2 ROC 曲线 (每个模型的最优特征组) ----------
fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(PALETTE['bg'])
for model_name, (_, mcolor) in MODELS.items():
    best_feat = max(feat_names,
                    key=lambda f: all_results[model_name][f]['auc'])
    r = all_results[model_name][best_feat]
    ax.plot(r['fpr'], r['tpr'], color=mcolor, linewidth=2.2,
            label=f'{model_name} ({best_feat}) AUC={r["auc"]:.4f}')
    ax.fill_between(r['fpr'], r['tpr'], alpha=0.06, color=mcolor)
ax.plot([0, 1], [0, 1], '--', color=PALETTE['subtext'], linewidth=1, label='Random')
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves -- Best Feature Set per Model", fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.35)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "viz_roc_multimodel.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: viz_roc_multimodel.png")

# ---------- 10.3 混淆矩阵拼图 ----------
fig, axes = plt.subplots(3, 3, figsize=(14, 13))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle("Confusion Matrices -- All Models X Feature Sets",
             color=PALETTE['text'], fontsize=14, fontweight='bold')
for row_idx, model_name in enumerate(model_names):
    for col_idx, feat_name in enumerate(feat_names):
        ax = axes[row_idx, col_idx]
        cm = all_results[model_name][feat_name]['cm']
        acc = all_results[model_name][feat_name]['acc']
        cmap = LinearSegmentedColormap.from_list('c', [PALETTE['panel'], FEATURE_SETS[feat_name][1]])
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=cmap,
                    linewidths=1, linecolor=HIST_EDGE,
                    xticklabels=['Pred 0', 'Pred 1'],
                    yticklabels=['True 0', 'True 1'],
                    cbar=False, annot_kws={'size': 12, 'weight': 'bold'})
        ax.set_title(f"{model_name} + {feat_name} (Acc={acc:.3f})",
                     fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "viz_confusion_all.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: viz_confusion_all.png")

# ---------- 10.4 CV 箱线图 ----------
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(PALETTE['bg'])
positions = []
data_list = []
colors_list = []
labels_list = []
for i, model_name in enumerate(model_names):
    for j, feat_name in enumerate(feat_names):
        positions.append(i * 4 + j)
        r = all_results[model_name][feat_name]
        data_list.append(r['cv_scores'])
        colors_list.append(FEATURE_SETS[feat_name][1])
        labels_list.append(feat_name)

bp = ax.boxplot(data_list, positions=positions, patch_artist=True, notch=False,
                widths=0.6, medianprops=dict(color=PALETTE['accent'], linewidth=2),
                whiskerprops=dict(color=PALETTE['subtext'], linewidth=1.2),
                capprops=dict(color=PALETTE['subtext'], linewidth=1.2),
                flierprops=dict(marker='o', color=PALETTE['subtext'], markersize=4))
for patch, color in zip(bp['boxes'], colors_list):
    patch.set_facecolor(color); patch.set_alpha(0.7)
    patch.set_edgecolor(PALETTE['border'])

ax.set_xticks([i * 4 + 1 for i in range(3)])
ax.set_xticklabels(model_names, fontsize=11)
ax.set_title("5-Fold CV AUC -- All Models", fontsize=13, fontweight='bold')
ax.set_ylabel("AUC"); ax.grid(axis='y', alpha=0.4)
legend_patches = [mpatches.Patch(color=FEATURE_SETS[f][1], label=f) for f in feat_names]
ax.legend(handles=legend_patches, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "viz_cv_all_models.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: viz_cv_all_models.png")

# ========================== 11. 错误分析 ==========================

print("\n========== Error Analysis ==========")

# 使用融合特征 + LightGBM 进行错误分析
best_model_name = 'LightGBM'
best_feat_name = 'Fusion'
best_feat_idx = FEATURE_SETS[best_feat_name][0]
r_best = all_results[best_model_name][best_feat_name]
y_prob = r_best['prob']
y_pred = r_best['pred']
y_true = np.array(y_test)

# 获取测试集样本的元信息
test_meta = train_log[train_log['object_id'].isin(test_indices)].copy()
test_meta = test_meta.set_index('object_id')

# 误分类标记
is_fp = (y_true == 0) & (y_pred == 1)  # 假阳性
is_fn = (y_true == 1) & (y_pred == 0)  # 假阴性
is_correct = (y_pred == y_true)

# 获取误分类样本的 object_id
test_object_ids = merged.loc[test_indices, 'object_id'].values
fp_ids = test_object_ids[is_fp]
fn_ids = test_object_ids[is_fn]

print(f"Test samples: {len(y_true)}")
print(f"Correct: {(is_correct).sum()}")
print(f"False Positives: {is_fp.sum()}")
print(f"False Negatives: {is_fn.sum()}")

# ---------- 11.1 Z / EBV 分布对比 ----------
fig, axes = plt.subplots(4, 3, figsize=(18, 16))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle("Error Analysis -- Z & EBV Distribution by Prediction Outcome",
             color=PALETTE['text'], fontsize=14, fontweight='bold')

categories = [
    ('correct_neg', y_true == 0, y_pred == 0, PALETTE['conv'], 'Correct Negative (TN)'),
    ('correct_pos', y_true == 1, y_pred == 1, PALETTE['frac'], 'Correct Positive (TP)'),
    ('fp',         y_true == 0, y_pred == 1, PALETTE['fuse'], 'False Positive (FP)'),
    ('fn',         y_true == 1, y_pred == 0, PALETTE['accent'], 'False Negative (FN)'),
]

def get_z_ebv(mask):
    ids = test_object_ids[mask]
    sub = train_log[train_log['object_id'].isin(ids)]
    return sub['Z'].dropna(), sub['EBV'].dropna()

z_ebv_data = {}
for name, mask_y, mask_p, color, label in categories:
    z_vals, ebv_vals = get_z_ebv(mask_y & mask_p)
    z_ebv_data[name] = (z_vals, ebv_vals, color, label)

for row, (cat_name, (z_vals, ebv_vals, color, label)) in enumerate(
    [('correct_neg', z_ebv_data['correct_neg']),
     ('correct_pos', z_ebv_data['correct_pos']),
     ('fp', z_ebv_data['fp']),
     ('fn', z_ebv_data['fn'])]):
    ax_z = axes[row][0]
    ax_z.hist(z_vals, bins=25, color=color, edgecolor=HIST_EDGE, linewidth=0.5, alpha=0.8)
    ax_z.set_title(f"{label} (n={len(z_vals)}) -- Z", fontsize=10, color=color, fontweight='bold')
    ax_z.set_xlabel("Z"); ax_z.set_ylabel("Count"); ax_z.grid(axis='y', alpha=0.4)

    ax_e = axes[row][1]
    ax_e.hist(ebv_vals, bins=25, color=color, edgecolor=HIST_EDGE, linewidth=0.5, alpha=0.8)
    ax_e.set_title(f"{label} (n={len(ebv_vals)}) -- EBV", fontsize=10, color=color, fontweight='bold')
    ax_e.set_xlabel("EBV"); ax_e.set_ylabel("Count"); ax_e.grid(axis='y', alpha=0.4)

# 第三列：叠加对比 + 统计摘要
ax_z_overlay = axes[0][2]
for name in ['correct_neg', 'correct_pos', 'fp', 'fn']:
    z_vals, _, color, label = z_ebv_data[name]
    ax_z_overlay.hist(z_vals, bins=25, color=color, alpha=0.4, label=label, density=True)
ax_z_overlay.set_title("Z Distribution -- Overlay", fontsize=10, fontweight='bold')
ax_z_overlay.set_xlabel("Z"); ax_z_overlay.legend(fontsize=7); ax_z_overlay.grid(axis='y', alpha=0.4)

ax_e_overlay = axes[1][2]
for name in ['correct_neg', 'correct_pos', 'fp', 'fn']:
    _, ebv_vals, color, label = z_ebv_data[name]
    ax_e_overlay.hist(ebv_vals, bins=25, color=color, alpha=0.4, label=label, density=True)
ax_e_overlay.set_title("EBV Distribution -- Overlay", fontsize=10, fontweight='bold')
ax_e_overlay.set_xlabel("EBV"); ax_e_overlay.legend(fontsize=7); ax_e_overlay.grid(axis='y', alpha=0.4)

# 统计摘要
ax_summary = axes[2][2]
fp_z_mean = z_ebv_data['fp'][0].mean() if len(z_ebv_data['fp'][0]) > 0 else np.nan
fp_z_std = z_ebv_data['fp'][0].std() if len(z_ebv_data['fp'][0]) > 0 else np.nan
fn_z_mean = z_ebv_data['fn'][0].mean() if len(z_ebv_data['fn'][0]) > 0 else np.nan
fn_z_std = z_ebv_data['fn'][0].std() if len(z_ebv_data['fn'][0]) > 0 else np.nan
tn_z_mean = z_ebv_data['correct_neg'][0].mean() if len(z_ebv_data['correct_neg'][0]) > 0 else np.nan
tp_z_mean = z_ebv_data['correct_pos'][0].mean() if len(z_ebv_data['correct_pos'][0]) > 0 else np.nan

summary_text = (
    f"Z mean by group\n\n"
    f"TN (Correct Neg):  {tn_z_mean:.4f}\n"
    f"TP (Correct Pos):  {tp_z_mean:.4f}\n"
    f"FP (False Pos):    {fp_z_mean:.4f}\n"
    f"FN (False Neg):    {fn_z_mean:.4f}\n\n"
    f"Total Test: {len(y_true)}\n"
    f"Correct: {(is_correct).sum()} ({(is_correct).sum()/len(y_true)*100:.1f}%)\n"
    f"FP: {is_fp.sum()}  FN: {is_fn.sum()}"
)
ax_summary.text(0.5, 0.5, summary_text,
    transform=ax_summary.transAxes, ha='center', va='center',
    fontsize=10, color=PALETTE['text'], fontfamily='monospace')
ax_summary.axis('off')

# 第四行第三列留空或用图例说明
axes[3][2].axis('off')
legend_patches_err = [mpatches.Patch(color=z_ebv_data[n][2], label=z_ebv_data[n][3])
                      for n in ['correct_neg', 'correct_pos', 'fp', 'fn']]
axes[3][2].legend(handles=legend_patches_err, fontsize=10, loc='center')

plt.tight_layout()
plt.savefig(os.path.join(ERR_DIR, "err_z_ebv_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: err_z_ebv_distribution.png")

# ---------- 11.2 误分类样本光变曲线 ----------
def plot_misclassified_lightcurves(fp_ids, fn_ids, df_processed,
                                   train_log, max_samples=4):
    """展示假阳性和假阴性样本的光变曲线"""
    fp_sample = np.random.choice(fp_ids, min(max_samples, len(fp_ids)),
                                  replace=False) if len(fp_ids) > 0 else []
    fn_sample = np.random.choice(fn_ids, min(max_samples, len(fn_ids)),
                                  replace=False) if len(fn_ids) > 0 else []

    n_fp = len(fp_sample)
    n_fn = len(fn_sample)
    if n_fp == 0 and n_fn == 0:
        return

    n_total = n_fp + n_fn
    fig, axes = plt.subplots(n_total, 1, figsize=(14, 3.5 * n_total))
    fig.patch.set_facecolor(PALETTE['bg'])
    if n_total == 1:
        axes = [axes]

    row = 0
    for oid in fp_sample:
        ax = axes[row]
        sub = df_processed[df_processed['object_id'] == oid].sort_values('mjd')
        z_val = train_log[train_log['object_id'] == oid]['Z'].values[0] if oid in train_log['object_id'].values else np.nan
        ebv_val = train_log[train_log['object_id'] == oid]['EBV'].values[0] if oid in train_log['object_id'].values else np.nan
        for filt, grp in sub.groupby('filter'):
            ax.plot(grp['mjd'], grp['flux_interp'], linewidth=1.2,
                    color=FILTER_COLORS.get(filt, '#aaa'), label=filt, alpha=0.85)
        ax.set_title(f"False Positive  ID={oid}  Z={z_val:.4f}  EBV={ebv_val:.4f}",
                     fontsize=11, color=PALETTE['fuse'], fontweight='bold')
        ax.set_xlabel("MJD"); ax.set_ylabel("Flux (corrected)")
        ax.legend(fontsize=8, ncol=6); ax.grid(True, alpha=0.4)
        row += 1

    for oid in fn_sample:
        ax = axes[row]
        sub = df_processed[df_processed['object_id'] == oid].sort_values('mjd')
        z_val = train_log[train_log['object_id'] == oid]['Z'].values[0] if oid in train_log['object_id'].values else np.nan
        ebv_val = train_log[train_log['object_id'] == oid]['EBV'].values[0] if oid in train_log['object_id'].values else np.nan
        for filt, grp in sub.groupby('filter'):
            ax.plot(grp['mjd'], grp['flux_interp'], linewidth=1.2,
                    color=FILTER_COLORS.get(filt, '#aaa'), label=filt, alpha=0.85)
        ax.set_title(f"False Negative  ID={oid}  Z={z_val:.4f}  EBV={ebv_val:.4f}",
                     fontsize=11, color=PALETTE['accent'], fontweight='bold')
        ax.set_xlabel("MJD"); ax.set_ylabel("Flux (corrected)")
        ax.legend(fontsize=8, ncol=6); ax.grid(True, alpha=0.4)
        row += 1

    plt.tight_layout()
    plt.savefig(os.path.join(ERR_DIR, "err_misclassified_lightcurves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: err_misclassified_lightcurves.png")

plot_misclassified_lightcurves(fp_ids, fn_ids, df_processed, train_log)

# ---------- 11.3 模型置信度与正确性 ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle("Error Analysis -- Confidence vs Correctness",
             color=PALETTE['text'], fontsize=14, fontweight='bold')

model_names_for_err = ['LightGBM', 'XGBoost', 'RandomForest']

for ax, model_name in zip(axes, model_names_for_err):
    r = all_results[model_name]['Fusion']
    prob = r['prob']
    correct = (y_true == r['pred'])

    ax.hist(prob[correct], bins=30, alpha=0.7, color=PALETTE['frac'],
            label='Correct', density=True)
    ax.hist(prob[~correct], bins=30, alpha=0.8, color=PALETTE['fuse'],
            label='Misclassified', density=True)
    ax.axvline(0.5, linestyle='--', color=PALETTE['accent'], linewidth=1.5,
               label='Threshold = 0.5')
    ax.set_title(f"{model_name} (Fusion)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(ERR_DIR, "err_confidence_correctness.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: err_confidence_correctness.png")

# ---------- 11.4 校准曲线 ----------
fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(PALETTE['bg'])

for model_name, (_, mcolor) in MODELS.items():
    r = all_results[model_name]['Fusion']
    prob = r['prob']
    bins = np.linspace(0, 1, 11)
    bin_centers = []
    bin_true_rates = []
    for i in range(len(bins) - 1):
        mask = (prob >= bins[i]) & (prob < bins[i + 1])
        if mask.sum() > 5:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_true_rates.append(y_true[mask].mean())
    if bin_centers:
        ax.plot(bin_centers, bin_true_rates, 'o-', color=mcolor, linewidth=2,
                markersize=6, label=f'{model_name} (Fusion)')
ax.plot([0, 1], [0, 1], '--', color=PALETTE['subtext'], linewidth=1, label='Perfect Calibration')
ax.set_xlabel("Predicted Probability (binned)", fontsize=11)
ax.set_ylabel("Observed Positive Rate", fontsize=11)
ax.set_title("Calibration Curves -- Fusion Features", fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.35)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(ERR_DIR, "err_calibration_curve.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: err_calibration_curve.png")

# ========================== 12. 保存文字结果 ==========================

result_path = os.path.join(OUTPUT_DIR, "model_evaluation_results.txt")
with open(result_path, "w", encoding="utf-8") as f:
    f.write("========== Mallorn -- Multi-Model Evaluation Results ==========\n\n")
    f.write("Dataset\n")
    f.write(f"  Total samples  : {len(y_all)}\n")
    f.write(f"  Positive rate  : {y_all.mean():.4f}\n")
    f.write(f"  Feature dims   : Conv={len(conv_cols)}  Frac={len(frac_cols)}  Fusion={len(all_cols)}\n\n")
    f.write(f"{'Model':<15} {'Feat':<8} {'Test Acc':>10} {'Test AUC':>10} {'CV AUC mean':>13} {'CV AUC std':>12}\n")
    f.write("-" * 72 + "\n")
    for model_name in model_names:
        for feat_name in feat_names:
            r = all_results[model_name][feat_name]
            f.write(f"{model_name:<15} {feat_name:<8} {r['acc']:>10.4f} {r['auc']:>10.4f} "
                    f"{r['cv_mean']:>13.4f} {r['cv_std']:>12.4f}\n")
    f.write("\n--- Error Analysis (LightGBM + Fusion) ---\n")
    f.write(f"  Test samples     : {len(y_true)}\n")
    f.write(f"  Correct          : {(is_correct).sum()} ({(is_correct).sum()/len(y_true)*100:.1f}%)\n")
    f.write(f"  False Positives  : {is_fp.sum()}\n")
    f.write(f"  False Negatives  : {is_fn.sum()}\n")

print(f"\n  Results saved -> {result_path}")
print(f"\n========== Done ==========")
print(f"  EDA charts  -> {EDA_DIR}/")
print(f"  Viz charts  -> {VIZ_DIR}/")
print(f"  Error analysis -> {ERR_DIR}/")
