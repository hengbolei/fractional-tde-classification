import pandas as pd
import numpy as np
import os
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import lightgbm as lgb
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

# ─────────────────────────────────────────────
#  全局绘图风格
# ─────────────────────────────────────────────
PALETTE = {
    'bg': '#FFFFFF',          # 白底
    'panel': '#F5F5F5',       # 浅灰面板
    'border': '#B0B0B0',      # 边框线
    'conv': '#1F77B4',        # 常规特征 → 蓝（保持区分度）
    'frac': '#2CA02C',        # 分数阶特征 → 绿
    'fuse': '#D62728',        # 融合特征 → 红/橙
    'accent': '#E3B341',      # 强调色 → 金黄（不变）
    'text': '#333333',        # 正文 → 深灰
    'subtext': '#666666',     # 辅助文字 → 中灰
    'grid': '#DDDDDD',        # 网格线 → 浅灰
}


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
        'figure.dpi': 150,               # 清晰度，打印够用
        'savefig.dpi': 300,              # 保存时更高
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.spines.top': False,        # 去上右边框，更简洁
        'axes.spines.right': False,
        'axes.linewidth': 1.0,
    })


apply_academic_style()

# ─────────────────────────────────────────────
#  输出目录
# ─────────────────────────────────────────────
OUTPUT_DIR = './result'
EDA_DIR    = os.path.join(OUTPUT_DIR, 'eda')
VIZ_DIR    = os.path.join(OUTPUT_DIR, 'viz')
for d in [OUTPUT_DIR, EDA_DIR, VIZ_DIR]:
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

# 只合并需要的列，避免目标相关列污染特征
df = train_lc.merge(
    train_log[['object_id', 'target', 'Z', 'EBV']],
    on='object_id', how='left'
)

# ========================== 2. EDA ==========================

print("\n========== EDA ==========")

FILTER_COLORS = {'u': '#C678DD', 'g': '#3FB950', 'r': '#F78166',
                 'i': '#58A6FF', 'z': '#E3B341', 'y': '#79C0FF'}

# --- 2.1 类别分布 / Z 分布 / EBV 分布 ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle("EDA — Dataset Overview", color=PALETTE['text'],
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
             color=PALETTE['accent'], edgecolor=PALETTE['bg'], linewidth=0.5)
axes[1].set_title("Redshift (Z) Distribution")
axes[1].set_xlabel("Z"); axes[1].set_ylabel("Count"); axes[1].grid(axis='y')

axes[2].hist(train_log['EBV'].dropna(), bins=50,
             color=PALETTE['fuse'], edgecolor=PALETTE['bg'], linewidth=0.5)
axes[2].set_title("Dust Extinction (EBV) Distribution")
axes[2].set_xlabel("EBV"); axes[2].set_ylabel("Count"); axes[2].grid(axis='y')

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "eda_overview.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: eda_overview.png")

# --- 2.2 每个 filter 的 flux 分布 ---
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle("EDA — Flux Distribution per Filter",
             color=PALETTE['text'], fontsize=14, fontweight='bold')
axes = axes.flatten()
for ax, filt in zip(axes, ['u', 'g', 'r', 'i', 'z', 'y']):
    data = df[df['filter'] == filt]['flux'].dropna()
    q1, q99 = data.quantile(0.01), data.quantile(0.99)
    data_clip = data[(data >= q1) & (data <= q99)]
    ax.hist(data_clip, bins=60, color=FILTER_COLORS.get(filt, PALETTE['conv']),
            edgecolor=PALETTE['bg'], linewidth=0.3, alpha=0.85)
    ax.set_title(f"Filter: {filt}  (n={len(data_clip):,})")
    ax.set_xlabel("Flux"); ax.set_ylabel("Count"); ax.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "eda_flux_per_filter.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: eda_flux_per_filter.png")

# --- 2.3 示例光变曲线（负/正各3条）---
sample_neg = train_log[train_log['target'] == 0]['object_id'].sample(3, random_state=0).values
sample_pos = train_log[train_log['target'] == 1]['object_id'].sample(3, random_state=0).values
sample_ids = np.concatenate([sample_neg, sample_pos])

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle("EDA — Sample Light Curves (top: negative, bottom: positive)",
             color=PALETTE['text'], fontsize=13, fontweight='bold')
axes = axes.flatten()
for ax_idx, oid in enumerate(sample_ids):
    sub   = df[df['object_id'] == oid].sort_values('mjd')
    label = train_log[train_log['object_id'] == oid]['target'].values[0]
    for filt, grp in sub.groupby('filter'):
        axes[ax_idx].scatter(grp['mjd'], grp['flux'], s=8, alpha=0.7,
                             color=FILTER_COLORS.get(filt, '#aaa'), label=filt)
    axes[ax_idx].set_title(f"ID {oid}  |  label={label}", fontsize=9)
    axes[ax_idx].set_xlabel("MJD"); axes[ax_idx].set_ylabel("Flux")
    axes[ax_idx].legend(fontsize=7, ncol=3); axes[ax_idx].grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "eda_sample_lightcurves.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: eda_sample_lightcurves.png")

# --- 2.4 每个天体观测点数分布 ---
obs_count = df.groupby('object_id').size()
fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor(PALETTE['bg'])
ax.hist(obs_count, bins=60, color=PALETTE['accent'], edgecolor=PALETTE['bg'], linewidth=0.4)
ax.set_title("EDA — Observations per Object")
ax.set_xlabel("Number of Observations"); ax.set_ylabel("Count")
ax.axvline(obs_count.median(), color=PALETTE['fuse'], linestyle='--',
           linewidth=1.5, label=f"Median = {obs_count.median():.0f}")
ax.legend(); ax.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "eda_obs_count.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: eda_obs_count.png")

# ========================== 3. 数据预处理 ==========================

df.dropna(subset=['flux', 'EBV', 'flux_err'], inplace=True)

EXTINCTION_COEFF = {'u': 3.1, 'g': 3.0, 'r': 2.9, 'i': 2.8, 'z': 2.7, 'y': 2.6}

def de_extinct_flux(df):
    k      = df['filter'].map(EXTINCTION_COEFF)
    factor = 10 ** (0.4 * df['EBV'] * k)
    df = df.copy()
    df['flux_corrected'] = np.where(k.isna(), df['flux'], df['flux'] / factor)
    return df

df = de_extinct_flux(df)
df.dropna(subset=['flux_corrected'], inplace=True)
df = df.groupby('object_id').filter(lambda x: len(x) >= 10)

# ========================== 4. 光变曲线预处理 ==========================

def preprocess_lightcurve(group, dt=1.0):
    """时间排序 → 基线扣除 → 均匀插值"""
    if len(group) < 10:
        return None
    group    = group.sort_values('mjd').copy()
    t        = group['mjd'].values
    flux     = group['flux_corrected'].values
    flux_err = group['flux_err'].values

    n_baseline = max(3, int(0.2 * len(t)))
    weights    = 1.0 / np.clip(flux_err[:n_baseline], 1e-10, None)
    baseline   = np.average(flux[:n_baseline], weights=weights)
    flux       = flux - baseline

    t_uniform = np.arange(t.min(), t.max() + dt, dt)
    if len(t_uniform) < 3:
        return None

    interp_func = interp1d(t, flux, kind='linear',
                           fill_value='extrapolate', bounds_error=False)
    return pd.DataFrame({
        'mjd':         t_uniform,
        'flux_interp': interp_func(t_uniform),
        'object_id':   group['object_id'].iloc[0],
        'filter':      group['filter'].iloc[0],
        'target':      group['target'].iloc[0],
    })


grouped_raw    = df.groupby(['object_id', 'filter'])
processed_list = []
for key in tqdm(grouped_raw.groups.keys(), desc="Processing Lightcurves"):
    result = preprocess_lightcurve(grouped_raw.get_group(key), dt=1.0)
    if result is not None:
        processed_list.append(result)

df_processed = pd.concat(processed_list, ignore_index=True)

# ========================== 5. 特征提取 ==========================

COLS   = ['mean', 'std', 'max', 'min', 'amplitude',
          'skew', 'kurt', 'median', 'mad', 'rise_time']
ALPHAS = [0.5, 1.0, 1.5]   # alpha=0 与常规特征重叠，去除


def extract_unified_stats(s):
    s     = np.asarray(s, dtype=float)
    valid = s[~np.isnan(s)]
    if len(valid) < 3:
        return [np.nan] * 10
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
        float(np.argmax(valid)),
    ]


def fractional_derivative_fft(y, alpha, dt=1.0):
    """FFT 分数阶导数；直流分量单独置零避免 0^alpha 歧义。"""
    if alpha == 0.0:
        return np.array(y, dtype=float)
    N     = len(y)
    Y     = np.fft.fft(y)
    freqs = np.fft.fftfreq(N, d=dt)
    omega = 2 * np.pi * freqs
    with np.errstate(divide='ignore', invalid='ignore'):
        scaling = np.where(omega == 0, 0.0, (1j * omega) ** alpha)
    return np.real(np.fft.ifft(Y * scaling))


# 单次遍历，同时提取常规与分数阶特征
records      = {}
grouped_proc = df_processed.groupby(['object_id', 'filter'])

for (obj_id, filt), group in tqdm(grouped_proc, desc="Extracting Features"):
    y_vals = group['flux_interp'].values
    t_vals = group['mjd'].values

    if obj_id not in records:
        records[obj_id] = {}

    # 常规统计特征
    stats = extract_unified_stats(y_vals)
    for i, col in enumerate(COLS):
        records[obj_id][f"{col}_{filt}"] = stats[i]

    # 分数阶特征
    dt_val = float(np.median(np.diff(t_vals))) if len(t_vals) > 1 else 1.0
    dt_val = max(dt_val, 1e-6)
    for alpha in ALPHAS:
        deriv      = fractional_derivative_fft(y_vals, alpha, dt_val)
        frac_stats = extract_unified_stats(deriv)
        for i, col in enumerate(COLS):
            records[obj_id][f"alpha{alpha:.1f}_{filt}_{col}"] = frac_stats[i]

features_df = pd.DataFrame.from_dict(records, orient='index')
features_df.index.name = 'object_id'
features_df = features_df.reset_index()

# ========================== 6. 对齐特征与标签 ==========================

label_df = df_processed.groupby('object_id')['target'].first().reset_index()
merged   = features_df.merge(label_df, on='object_id', how='inner')
merged   = merged.dropna(axis=1, thresh=int(0.8 * len(merged)))
merged   = merged.dropna()

X_all = merged.drop(columns=['object_id', 'target'])
y_all = merged['target']

print(f"\nFeatures shape : {X_all.shape}")
print(f"Target shape   : {y_all.shape}")
print(f"Positive rate  : {y_all.mean():.4f}")

all_cols  = list(X_all.columns)
conv_cols = [c for c in all_cols if not c.startswith('alpha')]
frac_cols = [c for c in all_cols if c.startswith('alpha')]
conv_idx  = [all_cols.index(c) for c in conv_cols]
frac_idx  = [all_cols.index(c) for c in frac_cols]

# ========================== 7. EDA — 特征相关性热图 ==========================

print("\n  Generating feature correlation heatmap...")
corr_feats = X_all[conv_cols[:20]].corr()
fig, ax = plt.subplots(figsize=(14, 11))
fig.patch.set_facecolor(PALETTE['bg'])
sns.heatmap(
    corr_feats, ax=ax,
    cmap='coolwarm', center=0, vmin=-1, vmax=1,
    linewidths=0.3, linecolor=PALETTE['bg'],
    annot=False, square=True,
    cbar_kws={'shrink': 0.8}
)
ax.set_title("EDA — Feature Correlation (Conventional, first 20 features)",
             color=PALETTE['text'], fontsize=13, fontweight='bold', pad=12)
ax.tick_params(colors=PALETTE['subtext'], labelsize=7)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "eda_corr_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: eda_corr_heatmap.png")

# ========================== 8. 划分数据集 ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# 标准化：只在训练集 fit，防止数据泄露
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ========================== 9. 训练与评估 ==========================

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


def train_eval(X_tr, X_te, y_tr, y_te, label):
    clf = make_lgbm(y_tr)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    y_pred      = clf.predict(X_te)
    y_pred_prob = clf.predict_proba(X_te)[:, 1]
    acc         = accuracy_score(y_te, y_pred)
    auc         = roc_auc_score(y_te, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_te, y_pred_prob)
    cm          = confusion_matrix(y_te, y_pred)
    print(f"  [{label}]  Accuracy: {acc:.4f}  AUC: {auc:.4f}")
    return acc, auc, clf, fpr, tpr, cm, y_pred_prob


def cv_auc(X_sc, y, label):
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(make_lgbm(y), X_sc, y,
                             cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"  [{label}] CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


print("\n=== Model 1: Conventional Features Only ===")
acc_conv, auc_conv, mdl_conv, fpr_conv, tpr_conv, cm_conv, prob_conv = train_eval(
    X_train_sc[:, conv_idx], X_test_sc[:, conv_idx], y_train, y_test, "Conv")
cv_conv = cv_auc(X_train_sc[:, conv_idx], y_train, "Conv CV")

print("\n=== Model 2: Fractional Features Only ===")
acc_frac, auc_frac, mdl_frac, fpr_frac, tpr_frac, cm_frac, prob_frac = train_eval(
    X_train_sc[:, frac_idx], X_test_sc[:, frac_idx], y_train, y_test, "Frac")
cv_frac = cv_auc(X_train_sc[:, frac_idx], y_train, "Frac CV")

print("\n=== Model 3: Fusion (Conv + Frac) ===")
acc_fuse, auc_fuse, mdl_fuse, fpr_fuse, tpr_fuse, cm_fuse, prob_fuse = train_eval(
    X_train_sc, X_test_sc, y_train, y_test, "Fusion")
cv_fuse = cv_auc(X_train_sc, y_train, "Fusion CV")

# ========================== 10. 结果可视化 ==========================

apply_academic_style()
model_colors = [PALETTE['conv'], PALETTE['frac'], PALETTE['fuse']]
model_labels = ['Conventional', 'Fractional', 'Fusion']
y_test_arr   = np.array(y_test)

# ---------- 10.1 ROC 曲线对比 ----------
fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(PALETTE['bg'])
for fpr, tpr, auc_v, label, color in [
    (fpr_conv, tpr_conv, auc_conv, f'Conventional  AUC={auc_conv:.4f}', PALETTE['conv']),
    (fpr_frac, tpr_frac, auc_frac, f'Fractional    AUC={auc_frac:.4f}', PALETTE['frac']),
    (fpr_fuse, tpr_fuse, auc_fuse, f'Fusion        AUC={auc_fuse:.4f}', PALETTE['fuse']),
]:
    ax.plot(fpr, tpr, color=color, linewidth=2.2, label=label)
    ax.fill_between(fpr, tpr, alpha=0.06, color=color)
ax.plot([0, 1], [0, 1], '--', color=PALETTE['subtext'], linewidth=1, label='Random')
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curve — Three-Model Comparison", fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.35)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "viz_roc_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved: viz_roc_comparison.png")

# ---------- 10.2 AUC + Accuracy 柱状对比 ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle("Model Performance Comparison", color=PALETTE['text'],
             fontsize=14, fontweight='bold')

x = np.arange(3)
w = 0.45
cv_means = [cv_conv.mean(), cv_frac.mean(), cv_fuse.mean()]
cv_stds  = [cv_conv.std(),  cv_frac.std(),  cv_fuse.std()]

for ax, vals, title, ylabel, show_cv in [
    (axes[0], [auc_conv, auc_frac, auc_fuse], "AUC (Test & CV±std)", "AUC", True),
    (axes[1], [acc_conv, acc_frac, acc_fuse], "Accuracy (Test Set)",  "Accuracy", False),
]:
    bars = ax.bar(x, vals, width=w, color=model_colors,
                  edgecolor=PALETTE['border'], linewidth=1.1, zorder=3)
    if show_cv:
        ax.errorbar(x, cv_means, yerr=cv_stds, fmt='D',
                    color=PALETTE['accent'], capsize=6, linewidth=1.5,
                    markersize=5, label='CV AUC ± std', zorder=4)
        ax.legend(fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004, f'{val:.4f}',
                ha='center', color=PALETTE['text'], fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(model_labels, fontsize=10)
    ax.set_title(title, fontsize=11); ax.set_ylabel(ylabel)
    ax.set_ylim(max(0, min(vals) - 0.05), 1.02)
    ax.grid(axis='y', alpha=0.4, zorder=0)

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "viz_bar_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz_bar_comparison.png")

# ---------- 10.3 混淆矩阵（三模型并排）----------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle("Confusion Matrices", color=PALETTE['text'],
             fontsize=14, fontweight='bold')
for ax, cm, label, color in [
    (axes[0], cm_conv, 'Conventional', PALETTE['conv']),
    (axes[1], cm_frac, 'Fractional',   PALETTE['frac']),
    (axes[2], cm_fuse, 'Fusion',        PALETTE['fuse']),
]:
    cmap = LinearSegmentedColormap.from_list('custom', [PALETTE['panel'], color])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=cmap,
                linewidths=1, linecolor=PALETTE['bg'],
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'],
                cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(label, fontsize=12, color=color, fontweight='bold')
    ax.tick_params(colors=PALETTE['text'])
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "viz_confusion_matrices.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz_confusion_matrices.png")

# ---------- 10.4 预测概率分布（每个模型对数据的拟合图）----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle("Predicted Probability Distribution — Model Fit",
             color=PALETTE['text'], fontsize=13, fontweight='bold')
for ax, probs, label, color in [
    (axes[0], prob_conv, 'Conventional', PALETTE['conv']),
    (axes[1], prob_frac, 'Fractional',   PALETTE['frac']),
    (axes[2], prob_fuse, 'Fusion',        PALETTE['fuse']),
]:
    ax.hist(probs[y_test_arr == 0], bins=40, alpha=0.65,
            color=PALETTE['subtext'], label='Negative (0)', density=True)
    ax.hist(probs[y_test_arr == 1], bins=40, alpha=0.80,
            color=color, label='Positive (1)', density=True)
    ax.axvline(0.5, linestyle='--', color=PALETTE['accent'],
               linewidth=1.5, label='Threshold = 0.5')
    ax.set_title(label, fontsize=11, color=color, fontweight='bold')
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "viz_prob_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz_prob_distribution.png")

# ---------- 10.5 特征重要性（融合模型 Top-30）----------
importance = pd.DataFrame({
    'feature':    X_all.columns,
    'importance': mdl_fuse.feature_importances_,
}).sort_values('importance', ascending=False).head(30)

bar_colors = [
    PALETTE['frac'] if f.startswith('alpha') else PALETTE['conv']
    for f in importance['feature']
]

fig, ax = plt.subplots(figsize=(10, 9))
fig.patch.set_facecolor(PALETTE['bg'])
ax.barh(importance['feature'][::-1], importance['importance'][::-1],
        color=bar_colors[::-1], edgecolor=PALETTE['border'], linewidth=0.6)
ax.set_title("Feature Importance — Fusion Model (Top 30)",
             fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel("Importance Score")
ax.grid(axis='x', alpha=0.35)
ax.legend(handles=[
    mpatches.Patch(color=PALETTE['conv'], label='Conventional'),
    mpatches.Patch(color=PALETTE['frac'], label='Fractional'),
], fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "viz_feature_importance.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz_feature_importance.png")

# ---------- 10.6 交叉验证 AUC 箱线图 ----------
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(PALETTE['bg'])
bp = ax.boxplot(
    [cv_conv, cv_frac, cv_fuse],
    labels=model_labels,
    patch_artist=True, notch=False,
    medianprops=dict(color=PALETTE['accent'], linewidth=2.2),
    whiskerprops=dict(color=PALETTE['subtext'], linewidth=1.2),
    capprops=dict(color=PALETTE['subtext'], linewidth=1.2),
    flierprops=dict(marker='o', color=PALETTE['subtext'], markersize=4),
)
for patch, color in zip(bp['boxes'], model_colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)
    patch.set_edgecolor(PALETTE['border'])
ax.set_title("5-Fold Cross-Validation AUC", fontsize=13, fontweight='bold')
ax.set_ylabel("AUC"); ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "viz_cv_boxplot.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz_cv_boxplot.png")

# ========================== 11. 保存文字结果 ==========================

result_path = os.path.join(OUTPUT_DIR, "model_evaluation_results.txt")
with open(result_path, "w", encoding="utf-8") as f:
    f.write("========== Mallorn — Model Evaluation Results ==========\n\n")
    f.write("Dataset\n")
    f.write(f"  Total samples  : {len(y_all)}\n")
    f.write(f"  Positive rate  : {y_all.mean():.4f}\n")
    f.write(f"  Feature dims   : Conv={len(conv_cols)}  Frac={len(frac_cols)}  Fusion={len(all_cols)}\n\n")
    f.write(f"{'Model':<25} {'Test Acc':>10} {'Test AUC':>10} {'CV AUC mean':>13} {'CV AUC std':>12}\n")
    f.write("-" * 74 + "\n")
    for name, acc, auc_v, cv in [
        ("Conventional", acc_conv, auc_conv, cv_conv),
        ("Fractional",   acc_frac, auc_frac, cv_frac),
        ("Fusion",       acc_fuse, auc_fuse, cv_fuse),
    ]:
        f.write(f"{name:<25} {acc:>10.4f} {auc_v:>10.4f} {cv.mean():>13.4f} {cv.std():>12.4f}\n")

print(f"\n  Results saved → {result_path}")
print("\n========== Done ==========")
print(f"  EDA charts  → {EDA_DIR}/")
print(f"  Viz charts  → {VIZ_DIR}/")