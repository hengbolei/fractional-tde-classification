import pandas as pd
import numpy as np
import os
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.interpolate import interp1d
import glob
from tqdm import tqdm

# 数据目录
DATA_DIR = "./mallorn-dataset"

# ========================== 1. 数据加载 ==========================

train_log = pd.read_csv(os.path.join(DATA_DIR, "train_log.csv"))

train_lc_files = glob.glob(os.path.join(DATA_DIR, "split_*", "train_full_lightcurves.csv"))
train_lc = pd.concat(
    [pd.read_csv(f) for f in tqdm(train_lc_files, desc="Loading Lightcurve Files")],
    ignore_index=True
)

# 清理列名（兼容列名带空格的情况）
train_lc.columns = train_lc.columns.str.strip()
train_lc = train_lc.rename(columns={
    'Time (MJD)': 'mjd', 'Time (MJD )': 'mjd',
    'Flux': 'flux', 'Flux_err': 'flux_err', 'Filter': 'filter'
})

# 合并标签（只取需要的列，避免 SpecType 等目标相关列污染特征）
df = train_lc.merge(
    train_log[['object_id', 'target', 'Z', 'EBV']],
    on='object_id', how='left'
)

# ========================== 2. 数据预处理 ==========================

# 2.1 去除 flux / EBV 缺失值
df.dropna(subset=['flux', 'EBV', 'flux_err'], inplace=True)

# 2.2 去尘处理：将观测 flux 还原为内禀 flux
#     消光公式：flux_intrinsic = flux_observed / 10^(0.4 * A_lambda)
#     其中 A_lambda = R_lambda * EBV
EXTINCTION_COEFF = {'u': 3.1, 'g': 3.0, 'r': 2.9, 'i': 2.8, 'z': 2.7, 'y': 2.6}

def de_extinct_flux(df):
    k = df['filter'].map(EXTINCTION_COEFF)
    # 对无法映射的 filter 值（k 为 NaN）保留原始 flux
    factor = 10 ** (0.4 * df['EBV'] * k)
    df['flux_corrected'] = np.where(k.isna(), df['flux'], df['flux'] / factor)
    return df

df = de_extinct_flux(df)
df.dropna(subset=['flux_corrected'], inplace=True)

# 2.3 保留至少 10 个观测点的天体
df = df.groupby('object_id').filter(lambda x: len(x) >= 10)

# ========================== 3. 光变曲线预处理（插值 + 基线扣除） ==========================

def preprocess_lightcurve(group, dt=1.0):
    """
    对单个 (object_id, filter) 组做：
      1. 时间排序
      2. 基线扣除（用最早 20% 观测点的加权均值）
      3. 均匀时间网格插值
    返回插值后的 DataFrame，若观测点不足则返回 None。
    """
    if len(group) < 10:
        return None

    group = group.sort_values('mjd').copy()
    t = group['mjd'].values
    flux = group['flux_corrected'].values
    flux_err = group['flux_err'].values

    # 基线扣除：用最早 20% 点的误差倒数加权均值
    n_baseline = max(3, int(0.2 * len(t)))
    # clip flux_err 防止零除
    weights = 1.0 / np.clip(flux_err[:n_baseline], 1e-10, None)
    baseline = np.average(flux[:n_baseline], weights=weights)
    flux = flux - baseline

    # 均匀插值
    t_uniform = np.arange(t.min(), t.max() + dt, dt)
    if len(t_uniform) < 3:
        return None

    interp_func = interp1d(t, flux, kind='linear',
                           fill_value='extrapolate', bounds_error=False)
    flux_interp = interp_func(t_uniform)

    result = pd.DataFrame({
        'mjd': t_uniform,
        'flux_interp': flux_interp,
        'object_id': group['object_id'].iloc[0],
        'filter': group['filter'].iloc[0],
        'target': group['target'].iloc[0],
    })
    return result


grouped_raw = df.groupby(['object_id', 'filter'])
processed_list = []
for key in tqdm(grouped_raw.groups.keys(), desc="Processing Lightcurves"):
    group = grouped_raw.get_group(key)
    result = preprocess_lightcurve(group, dt=1.0)
    if result is not None:
        processed_list.append(result)

df_processed = pd.concat(processed_list, ignore_index=True)

# ========================== 4. 特征提取 ==========================

COLS = ['mean', 'std', 'max', 'min', 'amplitude',
        'skew', 'kurt', 'median', 'mad', 'rise_time']
ALPHAS = [0.0, 0.5, 1.0, 1.5]


def extract_unified_stats(s):
    """从一维数组提取 10 个统计特征。"""
    s = np.asarray(s, dtype=float)
    valid = s[~np.isnan(s)]
    if len(valid) < 3:
        return [np.nan] * 10
    return [
        float(np.mean(valid)),
        float(np.std(valid)),
        float(np.max(valid)),
        float(np.min(valid)),
        float(np.max(valid) - np.min(valid)),          # amplitude
        float(skew(valid)),
        float(kurtosis(valid)),
        float(np.median(valid)),
        float(np.median(np.abs(valid - np.median(valid)))),  # MAD
        float(np.argmax(valid)),                       # rise_time（峰值位置）
    ]


def fractional_derivative_fft(y, alpha, dt=1.0):
    """
    用 FFT 计算分数阶导数。
    alpha=0 退化为原信号（恒等变换），直接返回原信号避免 0^0 歧义。
    """
    if alpha == 0.0:
        return np.array(y, dtype=float)

    N = len(y)
    Y = np.fft.fft(y)
    freqs = np.fft.fftfreq(N, d=dt)
    omega = 2 * np.pi * freqs

    # 对 omega=0 的直流分量单独处理：0^alpha = 0（alpha>0）
    with np.errstate(divide='ignore', invalid='ignore'):
        scaling = np.where(omega == 0, 0.0, (1j * omega) ** alpha)

    Y_alpha = Y * scaling
    return np.real(np.fft.ifft(Y_alpha))


# 单次遍历同时提取常规特征和分数阶特征，避免重复遍历占用内存
records = {}  # {object_id: {feature_name: value}}

grouped_proc = df_processed.groupby(['object_id', 'filter'])

for (obj_id, filt), group in tqdm(grouped_proc, desc="Extracting Features"):
    y_vals = group['flux_interp'].values
    t_vals = group['mjd'].values

    if obj_id not in records:
        records[obj_id] = {}

    # --- 常规统计特征 ---
    stats = extract_unified_stats(y_vals)
    for i, col in enumerate(COLS):
        records[obj_id][f"{col}_{filt}"] = stats[i]

    # --- 分数阶特征 ---
    # 插值后时间均匀，用 median(diff) 计算 dt
    dt_val = float(np.median(np.diff(t_vals))) if len(t_vals) > 1 else 1.0
    dt_val = max(dt_val, 1e-6)  # 防止 dt=0

    for alpha in ALPHAS:
        deriv = fractional_derivative_fft(y_vals, alpha, dt_val)
        frac_stats = extract_unified_stats(deriv)
        for i, col in enumerate(COLS):
            records[obj_id][f"alpha{alpha:.1f}_{filt}_{col}"] = frac_stats[i]

# 构建宽表（每行一个 object_id）
features_df = pd.DataFrame.from_dict(records, orient='index')
features_df.index.name = 'object_id'
features_df = features_df.reset_index()

# ========================== 5. 对齐特征与标签 ==========================

label_df = df_processed.groupby('object_id')['target'].first().reset_index()

merged = features_df.merge(label_df, on='object_id', how='inner')
merged = merged.dropna(axis=1, thresh=int(0.8 * len(merged)))  # 删去缺失率 >20% 的列
merged = merged.dropna()                                         # 删去含缺失值的行

X_all = merged.drop(columns=['object_id', 'target'])
y_all = merged['target']

print(f"Features shape: {X_all.shape}")
print(f"Target shape:   {y_all.shape}")
print(f"Positive rate:  {y_all.mean():.4f}")

# ========================== 6. 划分数据集 ==========================
# stratify 保证正负样本比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# 标准化：只在训练集上 fit，避免数据泄露
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ========================== 7. 分别训练三个模型进行对比 ==========================

# 常规特征列名
conv_cols = [c for c in X_all.columns
             if not c.startswith('alpha')]
# 分数阶特征列名（alpha0.0 是原信号的 FFT 恒等，视为分数阶组的一部分）
frac_cols = [c for c in X_all.columns
             if c.startswith('alpha')]

def get_col_indices(all_cols, selected_cols):
    return [all_cols.index(c) for c in selected_cols if c in all_cols]

all_cols = list(X_all.columns)
conv_idx = get_col_indices(all_cols, conv_cols)
frac_idx = get_col_indices(all_cols, frac_cols)


def train_eval(X_tr, X_te, y_tr, y_te, label):
    """训练 LightGBM 并返回 accuracy 和 AUC。"""
    # 类别不平衡：用 scale_pos_weight
    neg = (y_tr == 0).sum()
    pos = (y_tr == 1).sum()
    spw = neg / pos if pos > 0 else 1.0

    clf = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        scale_pos_weight=spw,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    y_pred      = clf.predict(X_te)
    y_pred_prob = clf.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_pred_prob)
    print(f"[{label}]  Accuracy: {acc:.4f}  AUC: {auc:.4f}")
    return acc, auc, clf


print("\n=== Model 1: Conventional Features Only ===")
acc_conv, auc_conv, _ = train_eval(
    X_train_scaled[:, conv_idx], X_test_scaled[:, conv_idx],
    y_train, y_test, "Conv"
)

print("\n=== Model 2: Fractional Features Only ===")
acc_frac, auc_frac, _ = train_eval(
    X_train_scaled[:, frac_idx], X_test_scaled[:, frac_idx],
    y_train, y_test, "Frac"
)

print("\n=== Model 3: Fusion (Conv + Frac) ===")
acc_fuse, auc_fuse, model_fuse = train_eval(
    X_train_scaled, X_test_scaled,
    y_train, y_test, "Fusion"
)

# ========================== 8. 保存结果 ==========================

output_dir = './result'
os.makedirs(output_dir, exist_ok=True)

result_path = os.path.join(output_dir, "model_evaluation_results.txt")
with open(result_path, "w", encoding="utf-8") as f:
    f.write("========== Model Evaluation Results ==========\n\n")
    f.write(f"Feature dimensions:\n")
    f.write(f"  Conventional : {len(conv_cols)}\n")
    f.write(f"  Fractional   : {len(frac_cols)}\n")
    f.write(f"  Fusion       : {len(all_cols)}\n\n")
    f.write(f"{'Model':<30} {'Accuracy':>10} {'AUC':>10}\n")
    f.write("-" * 52 + "\n")
    f.write(f"{'Conventional Features Only':<30} {acc_conv:>10.4f} {auc_conv:>10.4f}\n")
    f.write(f"{'Fractional Features Only':<30} {acc_frac:>10.4f} {auc_frac:>10.4f}\n")
    f.write(f"{'Fusion (Conv + Frac)':<30} {acc_fuse:>10.4f} {auc_fuse:>10.4f}\n")

print(f"\nResults saved to {result_path}")