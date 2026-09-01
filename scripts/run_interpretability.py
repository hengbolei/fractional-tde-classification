# -*- coding: utf-8 -*-
"""
mallorn6_viz.py — 三张高质量补充图表
  1. 分数阶微分效果示意图
  2. 特征相关性聚类热力图
  3. SHAP 特征重要性分析
"""

import pandas as pd
import numpy as np
import os
import pickle
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from scipy.interpolate import interp1d
import glob
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ===================== 全局风格 =====================
PALETTE = {
    'bg': '#FFFFFF', 'panel': '#F5F5F5', 'border': '#B0B0B0',
    'conv': '#1F77B4', 'frac': '#2CA02C', 'fuse': '#D62728',
    'lgbm': '#D62728', 'xgb': '#1F77B4', 'rf': '#2CA02C',
    'accent': '#E3B341', 'text': '#333333', 'subtext': '#666666',
    'grid': '#DDDDDD', 'tde': '#D62728', 'nontde': '#1F77B4',
}
BAND_COLORS = {'u': '#9467BD', 'g': '#2CA02C', 'r': '#D62728',
               'i': '#1F77B4', 'z': '#BC8F1F', 'y': '#17BECF'}
ALPHA_COLORS = {0.0: '#333333', 0.5: '#1F77B4', 1.0: '#2CA02C', 1.5: '#D62728'}

def apply_academic_style():
    plt.rcParams.update({
        'figure.facecolor': PALETTE['bg'], 'axes.facecolor': PALETTE['panel'],
        'axes.edgecolor': PALETTE['border'], 'axes.labelcolor': PALETTE['text'],
        'axes.titlecolor': '#000000', 'xtick.color': PALETTE['subtext'],
        'ytick.color': PALETTE['subtext'], 'text.color': PALETTE['text'],
        'grid.color': PALETTE['grid'], 'grid.linestyle': '--', 'grid.alpha': 0.6,
        'legend.facecolor': '#FFFFFF', 'legend.edgecolor': PALETTE['border'],
        'legend.framealpha': 0.9, 'font.family': 'DejaVu Sans',
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
    })

apply_academic_style()

OUTPUT_DIR = './result/analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_DIR = "./mallorn-dataset"
FEATURE_CACHE = os.path.join(OUTPUT_DIR, 'feature_cache.pkl')

COLS = ['mean','std','max','min','amplitude','skew','kurt','median','mad','rise_time','decay_time']
ALPHAS = [0.5, 1.0, 1.5]

def extract_unified_stats(s):
    s = np.asarray(s, float); valid = s[~np.isnan(s)]; n = len(valid)
    if n<3: return [np.nan]*11
    pi = int(np.argmax(valid)); mbp = int(np.argmin(valid[:pi+1]))
    rtv = pi-mbp; pv=float(valid[pi]); mv=float(np.min(valid[:pi+1]))
    hv=(pv+mv)/2.0; pp=valid[pi:]; bh=np.where(pp<=hv)[0]
    dtv=float(bh[0]) if len(bh)>0 else float(len(pp)-1)
    return [float(np.mean(valid)),float(np.std(valid)),float(np.max(valid)),
            float(np.min(valid)),float(np.max(valid)-np.min(valid)),
            float(skew(valid)),float(kurtosis(valid)),float(np.median(valid)),
            float(np.median(np.abs(valid-np.median(valid)))),float(rtv),float(dtv)]

def fractional_derivative_fft(y, alpha, dt=1.0):
    if alpha==0.0: return np.array(y, float)
    N=len(y); Y=np.fft.fft(y); freqs=np.fft.fftfreq(N,d=dt)
    omega=2*np.pi*freqs
    with np.errstate(divide='ignore', invalid='ignore'):
        scaling=np.where(omega==0,0.0,(1j*omega)**alpha)
    return np.real(np.fft.ifft(Y*scaling))

# ===================== 1. 数据加载与特征提取（带缓存） =====================
if os.path.exists(FEATURE_CACHE):
    print("Loading cached features...")
    with open(FEATURE_CACHE, 'rb') as f:
        cache = pickle.load(f)
    df_processed = cache['df_processed']
    features_df = cache['features_df']
    X_all = cache['X_all']
    y_all = cache['y_all']
    all_cols = cache['all_cols']
    train_log = cache['train_log']
    print(f"  Loaded: {X_all.shape[1]} features, {X_all.shape[0]} samples")
else:
    print("No cache found. Running full pipeline...")

    train_log = pd.read_csv(os.path.join(DATA_DIR, "train_log.csv"))
    lc_files = glob.glob(os.path.join(DATA_DIR, "split_*", "train_full_lightcurves.csv"))
    train_lc = pd.concat([pd.read_csv(f) for f in tqdm(lc_files, desc="Loading")], ignore_index=True)
    train_lc.columns = train_lc.columns.str.strip()
    train_lc = train_lc.rename(columns={
        'Time (MJD)': 'mjd', 'Time (MJD )': 'mjd',
        'Flux': 'flux', 'Flux_err': 'flux_err', 'Filter': 'filter'
    })
    df = train_lc.merge(train_log[['object_id','target','Z','EBV']], on='object_id', how='left')
    df.dropna(subset=['flux','EBV','flux_err'], inplace=True)

    EXT = {'u':3.1,'g':3.0,'r':2.9,'i':2.8,'z':2.7,'y':2.6}
    def de_ext(df_in):
        k = df_in['filter'].map(EXT)
        f = 10**(0.4*df_in['EBV']*k)
        df_in = df_in.copy()
        df_in['flux_corrected'] = np.where(k.isna(), df_in['flux'], df_in['flux']/f)
        return df_in
    df = de_ext(df); df.dropna(subset=['flux_corrected'], inplace=True)
    df = df.groupby('object_id').filter(lambda x: len(x)>=10)

    def preprocess_lightcurve(group, dt=1.0):
        if len(group)<10: return None
        group = group.sort_values('mjd').copy()
        t = group['mjd'].values; flux = group['flux_corrected'].values
        flux_err = group['flux_err'].values
        n_b = max(3, int(0.2*len(t)))
        w = 1.0/np.clip(flux_err[:n_b], 1e-10, None)
        bl = np.average(flux[:n_b], weights=w); flux = flux - bl
        tu = np.arange(t.min(), t.max()+dt, dt)
        if len(tu)<3: return None
        f_interp = interp1d(t, flux, kind='linear', fill_value='extrapolate', bounds_error=False)
        return pd.DataFrame({
            'mjd': tu, 'flux_interp': f_interp(tu),
            'object_id': group['object_id'].iloc[0],
            'filter': group['filter'].iloc[0],
            'target': group['target'].iloc[0],
        })

    grouped_raw = df.groupby(['object_id','filter'])
    pl = []
    for key in tqdm(grouped_raw.groups.keys(), desc="Preprocessing"):
        r = preprocess_lightcurve(grouped_raw.get_group(key), dt=1.0)
        if r is not None: pl.append(r)
    df_processed = pd.concat(pl, ignore_index=True)

    records = {}
    grouped_proc = df_processed.groupby(['object_id','filter'])
    for (oid,flt), grp in tqdm(grouped_proc, desc="Extracting Features"):
        yv=grp['flux_interp'].values; tv=grp['mjd'].values
        if oid not in records: records[oid]={}
        ss=extract_unified_stats(yv)
        for i,c in enumerate(COLS): records[oid][f'{c}_{flt}']=ss[i]
        dtv=float(np.median(np.diff(tv))) if len(tv)>1 else 1.0; dtv=max(dtv,1e-6)
        for a in ALPHAS:
            d=fractional_derivative_fft(yv,a,dtv); fs=extract_unified_stats(d)
            for i,c in enumerate(COLS): records[oid][f'alpha{a:.1f}_{flt}_{c}']=fs[i]

    features_df = pd.DataFrame.from_dict(records, orient='index')
    features_df.index.name='object_id'; features_df=features_df.reset_index()

    ldf=df_processed.groupby('object_id')['target'].first().reset_index()
    merged=features_df.merge(ldf, on='object_id', how='inner')
    merged=merged.dropna(axis=1, thresh=int(0.8*len(merged))); merged=merged.dropna()
    X_all=merged.drop(columns=['object_id','target']); y_all=merged['target']
    all_cols=list(X_all.columns)

    # Cache
    with open(FEATURE_CACHE, 'wb') as f:
        pickle.dump({
            'df_processed': df_processed, 'features_df': features_df,
            'X_all': X_all, 'y_all': y_all, 'all_cols': all_cols,
            'train_log': train_log,
        }, f)
    print(f"  Features cached to {FEATURE_CACHE}")

# ===================== 2. 图表1: 分数阶微分效果示意 =====================
print("\n" + "="*60)
print("Figure 1: Fractional Derivative Demonstration")
print("="*60)

# Select one representative TDE light curve (r-band, high S/N)
tde_ids = train_log[train_log['target']==1]['object_id'].values
r_band_tde = df_processed[(df_processed['object_id'].isin(tde_ids)) & (df_processed['filter']=='r')]

# Pick the TDE with most observation points
tde_counts = r_band_tde.groupby('object_id').size().sort_values(ascending=False)
demo_tde_id = tde_counts.index[0]
demo_tde = r_band_tde[r_band_tde['object_id']==demo_tde_id].sort_values('mjd')

print(f"  Demo TDE: {demo_tde_id}, {len(demo_tde)} points, Z={train_log[train_log['object_id']==demo_tde_id]['Z'].values[0]:.4f}")

y_raw = demo_tde['flux_interp'].values
t_raw = demo_tde['mjd'].values
dt_val = float(np.median(np.diff(t_raw))) if len(t_raw)>1 else 1.0; dt_val=max(dt_val,1e-6)

# Compute derivatives for alpha = 0, 0.5, 1.0, 1.5
alphas_demo = [0.0, 0.5, 1.0, 1.5]
derivatives = {}
for a in alphas_demo:
    if a == 0.0:
        derivatives[a] = y_raw.copy()
    else:
        derivatives[a] = fractional_derivative_fft(y_raw, a, dt_val)

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor(PALETTE['bg'])
gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

# Row 1: time-domain signals
for i, a in enumerate(alphas_demo):
    ax = fig.add_subplot(gs[0, i])
    color = ALPHA_COLORS[a]
    ax.plot(t_raw, derivatives[a], color=color, linewidth=1.2)
    ax.fill_between(t_raw, 0, derivatives[a], alpha=0.1, color=color)
    label = 'Original' if a == 0.0 else f'$\\alpha={a}$'
    ax.set_title(f'{label} Derivative', fontsize=12, fontweight='bold', color=color)
    ax.set_xlabel('MJD'); ax.set_ylabel('Flux / Derivative')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color=PALETTE['subtext'], linewidth=0.5, linestyle='--')

# Row 2: frequency-domain magnitude
for i, a in enumerate(alphas_demo):
    ax = fig.add_subplot(gs[1, i])
    y_use = derivatives[a]
    N = len(y_use)
    Y = np.fft.fft(y_use)
    freqs = np.fft.fftfreq(N, d=dt_val)
    # Only positive frequencies
    pos_mask = freqs > 0
    ax.loglog(freqs[pos_mask], np.abs(Y[pos_mask]), color=ALPHA_COLORS[a], linewidth=1.0)
    ax.set_xlabel('Frequency (1/day)'); ax.set_ylabel('|FFT|')
    ax.grid(True, alpha=0.3, which='both')
    if i == 0:
        ax.set_title('Original Spectrum', fontsize=11, fontweight='bold')
    else:
        ax.set_title(f'$\\alpha={a}$  =  |FFT| $\\times (j\\omega)^{{{a}}}$', fontsize=10, fontweight='bold', color=ALPHA_COLORS[a])

fig.suptitle('Fractional-Order Derivative of a Representative TDE Light Curve (r-band)\n'
             f'Object: {demo_tde_id}  |  Z={train_log[train_log["object_id"]==demo_tde_id]["Z"].values[0]:.4f}',
             fontsize=13, fontweight='bold', y=1.01)
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_fractional_demo.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_fractional_demo.png")

# ===================== 3. 图表2: 特征相关性聚类热力图 =====================
print("\n" + "="*60)
print("Figure 2: Feature Correlation Clustered Heatmap")
print("="*60)

# Select top features by variance (to keep heatmap readable)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# Compute variance per feature and select top 50
variances = np.var(X_scaled, axis=0)
top_idx = np.argsort(variances)[-50:][::-1]
top_cols = [all_cols[i] for i in top_idx]
X_top = X_scaled[:, top_idx]

# Compute correlation
corr = np.corrcoef(X_top.T)

# Create labels with band color info
def get_label_info(col):
    """Extract band and feature type from column name"""
    for b in ['u','g','r','i','z','y']:
        if f'_{b}_' in col or col.endswith(f'_{b}'):
            band = b; break
    else:
        band = '?'
    is_frac = col.startswith('alpha')
    return band, is_frac

band_labels = [get_label_info(c)[0] for c in top_cols]
band_colors = [BAND_COLORS.get(b, '#888888') for b in band_labels]
is_frac_flags = [get_label_info(c)[1] for c in top_cols]

# Create row/column color bars
fig = plt.figure(figsize=(18, 16))
fig.patch.set_facecolor(PALETTE['bg'])

# Use clustermap for automatic hierarchical clustering
type_colors_list = [PALETTE['frac'] if f else PALETTE['conv'] for f in is_frac_flags]
row_colors_list = [band_colors, type_colors_list]

cmap = LinearSegmentedColormap.from_list('custom', [PALETTE['conv'], '#FFFFFF', PALETTE['fuse']])

g = sns.clustermap(
    corr, method='ward', cmap=cmap, center=0,
    vmin=-1, vmax=1,
    row_colors=row_colors_list, col_colors=row_colors_list,
    figsize=(18, 16), linewidths=0,
    xticklabels=False, yticklabels=False,
    dendrogram_ratio=0.1,
    cbar_pos=(0.92, 0.25, 0.02, 0.4),
    tree_kws={'linewidths': 0.5},
)

g.ax_heatmap.set_title('Feature Correlation Clustermap (Top-50 by Variance)\n'
                        'Blue = Conventional  |  Green = Fractional  |  Side bars: Band colors',
                        fontsize=13, fontweight='bold', pad=20)

# Legend for band colors
band_patches = [mpatches.Patch(color=BAND_COLORS[b], label=f'Band {b}') for b in ['g','r','i','z','y']]
type_patches = [mpatches.Patch(color=PALETTE['conv'], label='Conventional'),
                mpatches.Patch(color=PALETTE['frac'], label='Fractional')]
leg1 = g.ax_heatmap.legend(handles=band_patches, title='Bands', loc='upper left',
                           bbox_to_anchor=(1.15, 1.0), fontsize=8, title_fontsize=9)
leg2 = g.ax_heatmap.legend(handles=type_patches, title='Feature Type', loc='upper left',
                           bbox_to_anchor=(1.15, 0.6), fontsize=8, title_fontsize=9)
g.ax_heatmap.add_artist(leg1)

plt.savefig(os.path.join(OUTPUT_DIR, 'fig_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_correlation_heatmap.png")

# ===================== 4. 图表3: SHAP 特征重要性 =====================
print("\n" + "="*60)
print("Figure 3: SHAP Feature Importance")
print("="*60)

try:
    import shap
except ImportError:
    print("  Installing shap...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "-q"])
    import shap

# Train LightGBM on fusion features
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

scaler2 = StandardScaler()
X_train_sc = scaler2.fit_transform(X_train)
X_test_sc = scaler2.transform(X_test)

neg = (y_train==0).sum(); pos = (y_train==1).sum()
spw = neg/pos if pos>0 else 1.0

clf = lgb.LGBMClassifier(
    objective='binary', metric='auc', scale_pos_weight=spw,
    n_estimators=300, learning_rate=0.05, num_leaves=63,
    random_state=42, n_jobs=-1, verbose=-1,
)
clf.fit(X_train_sc, y_train)
print(f"  LightGBM trained: test AUC = {roc_auc_score(y_test, clf.predict_proba(X_test_sc)[:,1]):.4f}")

# Compute SHAP values (use a subset for speed)
explainer = shap.TreeExplainer(clf)
X_explain = X_train_sc[:500]  # Use 500 samples
shap_values = explainer.shap_values(X_explain)

# For LightGBM binary, shap_values is a list of two arrays; take class 1
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values

# Create readable feature names
def shorten_name(col):
    """Shorten column names for readability"""
    for b in ['u','g','r','i','z','y']:
        col = col.replace(f'_{b}_', f'[{b}]').replace(f'_{b}', f'[{b}]')
    col = col.replace('alpha', 'a').replace('amplitude', 'amp').replace('decay_time', 'decay')
    col = col.replace('rise_time', 'rise').replace('kurt', 'kurt').replace('median', 'med')
    return col

short_names = [shorten_name(c) for c in all_cols]
feature_names_short = [shorten_name(c) for c in all_cols]

# --- SHAP Summary Plot ---
fig, ax = plt.subplots(figsize=(10, 14))
fig.patch.set_facecolor(PALETTE['bg'])
shap.summary_plot(
    shap_vals, X_explain, feature_names=short_names,
    max_display=30, show=False, plot_size=None,
    color_bar=True,
)
ax.set_title('SHAP Feature Importance Summary (Top-30)\nLightGBM + Fusion Features',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_shap_summary.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_shap_summary.png")

# --- SHAP Bar Plot ---
fig, ax = plt.subplots(figsize=(10, 12))
fig.patch.set_facecolor(PALETTE['bg'])
shap.summary_plot(
    shap_vals, X_explain, feature_names=short_names,
    max_display=30, show=False, plot_type='bar',
    color=PALETTE['frac'],
)
ax.set_title('SHAP Feature Importance (Mean |SHAP|) Top-30\nLightGBM + Fusion Features',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_shap_bar.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_shap_bar.png")

# --- SHAP Dependence for top feature ---
# Find the most important feature
mean_shap = np.abs(shap_vals).mean(axis=0)
top_feat_idx = np.argmax(mean_shap)
top_feat_name = short_names[top_feat_idx]

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(PALETTE['bg'])
shap.dependence_plot(
    top_feat_idx, shap_vals, X_explain,
    feature_names=short_names, show=False, ax=ax,
)
ax.set_title(f'SHAP Dependence Plot: {top_feat_name}\n(Most Important Feature)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_shap_dependence.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_shap_dependence.png")

# ===================== 5. Summary =====================
print("\n" + "="*60)
print("ALL DONE!")
print(f"  Output directory: {OUTPUT_DIR}/")
print("  Figures generated:")
print("    1. fig_fractional_demo.png      — Fractional derivative demo")
print("    2. fig_correlation_heatmap.png  — Feature correlation clustermap")
print("    3. fig_shap_summary.png         — SHAP beeswarm summary")
print("    4. fig_shap_bar.png             — SHAP mean importance bar")
print("    5. fig_shap_dependence.png      — SHAP dependence (top feature)")
